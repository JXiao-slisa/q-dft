"""VASP input file generation (INCAR / KPOINTS / POTCAR) and output parsing.

The generated settings follow common practice for surface adsorption
catalysis calculations and can be overridden per-call by the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Default electronic / ionic convergence settings for standard DFT runs.
DEFAULT_INCAR = {
    # general
    "SYSTEM": "slisadft-calc",
    "PREC": "Accurate",
    "ENCUT": 500,
    "EDIFF": 1e-6,
    "EDIFFG": -0.02,
    "NELM": 100,
    "ISMEAR": 1,
    "SIGMA": 0.10,
    "ALGO": "Normal",
    "LREAL": "Auto",
    # ionic relaxation
    "IBRION": 2,
    "ISIF": 2,
    "NSW": 100,
    "NCORE": 8,
    "NPAR": 4,
    # spin
    "ISPIN": 1,
    # parallel / performance
    "LWAVE": False,
    "LCHARG": True,
}

# Settings for finite-difference vibrational frequency calculations
# (IBRION=5/6).  Notes on the classic pitfalls this encodes:
#   * NSW must be >= 1 — with NSW=0 VASP never enters the ionic loop and
#     produces no displaced structures (the "no displacement steps" bug);
#   * NFREE/POTIM control the finite-difference displacements;
#   * symmetry must be disabled (ISYM=0) or displaced geometries may be
#     mapped back and skipped;
#   * EDIFF has to be tight because frequencies are force derivatives.
FREQ_INCAR_DEFAULTS = {
    "IBRION": 5,
    "NFREE": 2,
    "POTIM": 0.015,
    "NSW": 1,
    "ISIF": 2,
    "ISYM": 0,
    "EDIFF": 1e-7,
}


def lattice_constant(element: str) -> float:
    """Return a reasonable lattice constant guess (Angstrom) for common
    fcc/bcc metals.  Used only when ASE cannot infer it."""
    # fmt: off
    table = {
        "Pt": 3.924, "Cu": 3.615, "Ag": 4.085, "Au": 4.078,
        "Pd": 3.890, "Al": 4.050, "Ni": 3.524, "Co": 3.545,
        "Fe": 2.866,  # bcc
        "W": 3.165,   # bcc
        "Mo": 3.147,  # bcc
        "Ru": 3.806,  # hcp a
        "Ir": 3.839,
        "Rh": 3.804,
        "Ti": 2.951,  # hcp a
    }
    # fmt: on
    return table.get(element, 4.0)


def write_incar(directory: Path, settings: Optional[Dict[str, object]] = None,
                relax: bool = True, freq: bool = False) -> Path:
    """Write a VASP INCAR file.

    Args:
        directory: target directory for the calculation.
        settings: overrides / extras merged on top of defaults.
        relax: if False, write a single-point calculation (IBRION=-1, NSW=0).
        freq: write a vibrational frequency calculation (IBRION=5 finite
              differences with safe NSW/NFREE/POTIM/ISYM/EDIFF defaults);
              takes precedence over ``relax``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    incar = dict(DEFAULT_INCAR)
    if not relax:
        incar.update({"IBRION": -1, "NSW": 0})
    if freq:
        # Frequency defaults are applied last (before explicit user settings)
        # so the NSW>=1 requirement cannot be broken by the single-point base.
        incar.update(FREQ_INCAR_DEFAULTS)
    if settings:
        incar.update(settings)

    lines = []
    for key, value in incar.items():
        lines.append(f"{key} = {value}")
    path = directory / "INCAR"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_kpoints(directory: Path, kpoints: tuple = (4, 4, 1),
                  mode: str = "Monkhorst-Pack") -> Path:
    """Write a VASP KPOINTS file.

    Args:
        kpoints: (nx, ny, nz) mesh. For slabs the z component is usually 1.
        mode: 'Monkhorst-Pack' or 'Gamma'.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shift = "0 0 0" if mode == "Monkhorst-Pack" else "0 0 0"
    content = (
        f"{mode} mesh\n"
        "0\n"
        f"{mode}\n"
        f"{kpoints[0]} {kpoints[1]} {kpoints[2]}\n"
        f"{shift}\n"
    )
    path = directory / "KPOINTS"
    path.write_text(content, encoding="utf-8")
    return path


def find_potcar_dir() -> Optional[Path]:
    """Locate the VASP POTCAR library directory.

    Checks (in order): VASP_POTCAR_DIR env var, common install locations.
    Returns None when not found.
    """
    env = __import__("os").environ.get("VASP_POTCAR_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    candidates = [
        "/opt/vasp/VASP6.5.1/PBE",
        "/opt/vasp/potpaw_PBE",
        "/opt/vasp/potpaw",
        "/opt/vasp/POTCAR",
        "/opt/vasp/potpaw_PBE.54",
        "/vasp/potpaw_PBE",
    ]
    for c in candidates:
        if Path(c).is_dir():
            return Path(c)
    return None


def write_potcar(directory: Path, symbols: List[str],
                 potcar_dir: Optional[Path] = None) -> Optional[Path]:
    """Concatenate POTCAR files for the given element symbols.

    Uses the system ``cat`` command to avoid encoding issues with VASP's
    binary-ish POTCAR format.

    Returns None if the POTCAR library is unavailable (VASP is a
    commercial code; users must provide their own potentials).
    """
    import subprocess
    directory = Path(directory)
    potcar_dir = potcar_dir or find_potcar_dir()
    if potcar_dir is None:
        return None
    parts = []
    for sym in symbols:
        # VASP stores PBE potentials under subdirs named by element symbol.
        candidates = [
            potcar_dir / f"POTCAR.{sym}",
            potcar_dir / sym / "POTCAR",
            potcar_dir / f"POTCAR_{sym}",
        ]
        found = None
        for c in candidates:
            if c.is_file():
                found = c
                break
        if found is None:
            raise FileNotFoundError(
                f"POTCAR for {sym} not found under {potcar_dir}")
        parts.append(str(found))
    path = directory / "POTCAR"
    # Use cat to concatenate — preserves binary-ish format VASP expects.
    subprocess.run(["cat"] + parts, stdout=open(path, "wb"), check=True)
    return path


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_oszicar_energies(directory: Path) -> List[float]:
    """Return the list of total energies (eV) from OSZICAR, one per SCF step."""
    osz = Path(directory) / "OSZICAR"
    if not osz.exists():
        return []
    energies = []
    for line in osz.read_text(errors="ignore").splitlines():
        m = re.search(r"F=\s*([-+]?[\d.]+)", line)
        if m:
            energies.append(float(m.group(1)))
    return energies


def parse_vasp_energy(directory: Path) -> Optional[float]:
    """Return the final electronic energy (eV) from a finished VASP run.

    Prefers the last `free  energy   TOTEN` from OUTCAR, then falls back
    to the last F= value in OSZICAR.
    """
    directory = Path(directory)
    outcar = directory / "OUTCAR"
    if outcar.exists():
        text = outcar.read_text(errors="ignore")
        matches = re.findall(
            r"free  energy\s+TOTEN\s*=\s*([-+]?[\d.]+) eV", text)
        if matches:
            return float(matches[-1])
    energies = parse_oszicar_energies(directory)
    return energies[-1] if energies else None


def parse_vasp_converged(directory: Path) -> bool:
    """Return True when the VASP run reached the `reached required accuracy`
    message in OUTCAR."""
    outcar = Path(directory) / "OUTCAR"
    if not outcar.exists():
        return False
    return "reached required accuracy" in outcar.read_text(errors="ignore")


_FORCE_TABLE_HEADER = "POSITION                                       TOTAL-FORCE (eV/Angst)"


def parse_vasp_forces(directory: Path) -> Optional[List[float]]:
    """Return the max atomic force magnitude (eV/A) after each ionic step.

    Parses the standard ``POSITION  TOTAL-FORCE (eV/Angst)`` tables VASP
    writes every ionic step (one value per table, = max |F| of that step).
    Falls back to legacy ``MAX TOTAL FORCE=`` lines for non-standard runs,
    and to ``None`` when no force information exists.
    """
    outcar = Path(directory) / "OUTCAR"
    if not outcar.exists():
        return None
    text = outcar.read_text(errors="ignore")
    lines = text.splitlines()

    max_forces: List[float] = []
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        if header.startswith("POSITION") and "TOTAL-FORCE" in header:
            # skip the dashed separator, read atom rows until a blank/dashed line
            j = i + 2
            step_max = 0.0
            found = False
            while j < len(lines):
                parts = lines[j].split()
                if len(parts) >= 6:
                    try:
                        fx, fy, fz = (float(parts[3]), float(parts[4]), float(parts[5]))
                    except ValueError:
                        break
                    import math as _math
                    step_max = max(step_max, _math.sqrt(fx * fx + fy * fy + fz * fz))
                    found = True
                    j += 1
                    continue
                break
            if found:
                max_forces.append(step_max)
            i = j
            continue
        i += 1
    if max_forces:
        return max_forces

    # Legacy fallback (e.g. custom toolchains that print a MAX line).
    matches = re.findall(r"MAX TOTAL FORCE=\s*([-+]?[\d.]+)", text)
    return [float(m) for m in matches]


_FREQ_LINE_RE = re.compile(
    r"^\s*\d+\s+(f|f/i)\s*=\s*[-+]?[\d.]+\s*THz\s+[-+]?[\d.]+\s*2PiTHz\s+"
    r"([-+]?[\d.]+)\s*cm-1\s+[-+]?[\d.]+\s*meV",
    re.IGNORECASE,
)


def parse_vasp_frequencies(directory: Path) -> Dict[str, object]:
    """Parse harmonic vibrational frequencies from a VASP OUTCAR
    (written by IBRION=5/6 finite-difference runs).

    Returns a dict:
        frequencies_cm1: list of signed frequencies in cm^-1
                         (imaginary modes are negative),
        n_imaginary:     number of imaginary modes,
        has_freq_block:  False when the OUTCAR contains no frequency block
                         (e.g. the run never produced displaced structures).
    """
    outcar = Path(directory) / "OUTCAR"
    result: Dict[str, object] = {
        "frequencies_cm1": [],
        "n_imaginary": 0,
        "has_freq_block": False,
        "source": str(outcar),
    }
    if not outcar.exists():
        result["source"] = ""
        return result
    freqs: List[float] = []
    for line in outcar.read_text(errors="ignore").splitlines():
        m = _FREQ_LINE_RE.match(line)
        if not m:
            continue
        result["has_freq_block"] = True
        value = float(m.group(2))
        if m.group(1).lower() == "f/i":
            value = -abs(value)
        freqs.append(value)
    result["frequencies_cm1"] = freqs
    result["n_imaginary"] = sum(1 for f in freqs if f < 0)
    return result


def parse_fermi_energy(directory: Path) -> Optional[float]:
    """Return the Fermi energy (eV) from EIGENVAL or OUTCAR."""
    directory = Path(directory)
    for name in ("OUTCAR", "EIGENVAL"):
        p = directory / name
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        if name == "OUTCAR":
            m = re.search(r"E-fermi\s*:\s*([-+]?[\d.]+)", text)
            if m:
                return float(m.group(1))
        else:
            # volume line in EIGENVAL: nspins? no - fermi not stored there
            pass
    return None


def parse_doscar(directory: Path) -> Dict[str, object]:
    """Parse a VASP DOSCAR into a dict with keys:
        efermi, energies (list), dos_up, dos_down (lists), pdos (list of dicts)
    Returns an empty dict on failure."""
    doscar = Path(directory) / "DOSCAR"
    if not doscar.exists():
        return {}
    with open(doscar, encoding="latin-1") as f:
        lines = f.readlines()
    try:
        header = lines[0].strip().split()
        nions = int(header[0])
        # Fermi energy is on line 6 (index 5), column 4 (index 3)
        dos_header = lines[5].strip().split()
        if len(dos_header) >= 4:
            efermi = float(dos_header[3])
        else:
            efermi = 0.0
        npts = int(dos_header[2])
        # total DOS block starts at line index 6
        grid = lines[6:6 + npts]
        energies, dos_up, dos_down = [], [], []
        for line in grid:
            parts = line.split()
            if len(parts) >= 3:
                energies.append(float(parts[0]))
                dos_up.append(float(parts[1]))
                dos_down.append(float(parts[2]))
        pdos = []
        # Each atom's PDOS block: a DOS-header duplicate line + npts data lines.
        # VASP repeats the DOS header (5 cols) at the start of every atom's PDOS.
        colnames = ["s", "py", "pz", "px", "dyz", "dz2", "dxz", "dxy", "dx2", "dtot"]
        idx = 6 + npts
        for iatom in range(nions):
            # Skip the DOS-header duplicate, then read npts data lines
            start = idx + 1
            block = lines[start: start + npts]
            if len(block) < 2:
                idx += 1 + npts
                continue
            try:
                data = {"atom": iatom + 1, "columns": colnames}
                ncols = min(len(block[0].split()) - 1, 10)
                for j, col in enumerate(colnames[:ncols]):
                    data[col] = [float(b.split()[j + 1]) for b in block]
                pdos.append(data)
                idx += 1 + npts
            except (IndexError, ValueError):
                idx += 1 + npts
                continue
        return {
            "efermi": efermi,
            "energies": energies,
            "dos_up": dos_up,
            "dos_down": dos_down,
            "pdos": pdos,
        }
    except (IndexError, ValueError):
        return {}