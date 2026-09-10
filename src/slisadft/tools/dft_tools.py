"""DFT calculation tools (VASP / CP2K / ABACUS).

Provides a unified CrewAI tool interface for submitting DFT calculations
and extracting results.  The actual DFT engine is selected by the
``calculator`` parameter (``vasp`` / ``cp2k`` / ``abacus``).

All environment-specific paths are read from environment variables
(see the ``SlurmConfig`` docstring).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess as sp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ase.io import read, write

from ..utils.crewai_compat import tool

from ..utils.slurm import SlurmConfig, wait_job, job_status
from ..engines import is_mock, synthetic_result
from ..utils.vasp_input import (
    write_incar,
    write_kpoints,
    write_potcar,
    find_potcar_dir,
    parse_vasp_energy,
    parse_vasp_converged,
    parse_vasp_forces,
)


# Environment variable names for DFT engine paths.
ENV_VASP_EXE = "VASP_EXE"
ENV_VASPKIT_EXE = "VASPKIT_EXE"
ENV_CP2K_EXE = "CP2K_EXE"
ENV_ABACUS_EXE = "ABACUS_EXE"

# Default paths (the server's known installation).
DEFAULT_VASP_EXE = "/opt/vasp/VASP6.5.1/vasp.6.5.1/bin/vasp_std"
DEFAULT_VASPKIT_EXE = "/opt/vasp/vaspkit.1.5.1/bin/vaspkit"
DEFAULT_VASP_POTCAR = "/opt/vasp/potpaw_PBE"  # common but not guaranteed


def _resolve_exe(env_key: str, default: str) -> Optional[str]:
    exe = os.environ.get(env_key, "") or default
    return exe if Path(exe).is_file() else None


def _check_vasp() -> Tuple[bool, str]:
    exe = _resolve_exe(ENV_VASP_EXE, DEFAULT_VASP_EXE)
    if not exe:
        potcar = find_potcar_dir()
        missing = []
        if not potcar:
            missing.append("POTCAR library (set VASP_POTCAR_DIR)")
        return False, f"VASP not found ({', '.join(missing)})"
    return True, exe


def _conda_env_bin(env_name: str, exe_name: str) -> Optional[str]:
    """Locate an executable inside a local (project-scoped) conda env.

    Looks under <project>/.conda/envs/<env_name>/bin, then the user's
    conda envs dir.  Returns the absolute path if the file exists.
    """
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / ".conda" / "envs" / env_name / "bin" / exe_name,
        Path.home() / ".conda" / "envs" / env_name / "bin" / exe_name,
        Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / exe_name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _check_cp2k() -> Tuple[bool, str]:
    exe = _resolve_exe(ENV_CP2K_EXE, "")
    if not exe:
        exe = _conda_env_bin("cp2k", "cp2k") or _conda_env_bin("cp2k", "cp2k.popt") or ""
        if exe:
            os.environ[ENV_CP2K_EXE] = exe
    if not exe:
        exe = shutil.which("cp2k") or shutil.which("cp2k.popt") or ""
        if exe:
            os.environ[ENV_CP2K_EXE] = exe
    if not exe:
        return False, "CP2K binary not found (set CP2K_EXE or install via conda)"
    return True, exe


def _check_abacus() -> Tuple[bool, str]:
    exe = _resolve_exe(ENV_ABACUS_EXE, "")
    if not exe:
        exe = _conda_env_bin("abacus", "abacus") or ""
        if exe:
            os.environ[ENV_ABACUS_EXE] = exe
    if not exe:
        exe = shutil.which("abacus") or ""
        if exe:
            os.environ[ENV_ABACUS_EXE] = exe
    if not exe:
        return False, "ABACUS binary not found (set ABACUS_EXE or compile from source)"
    return True, exe


@dataclass
class DftSetup:
    """Holds paths and settings for a single DFT calculation run."""
    workdir: Path
    calculator: str           # vasp / cp2k / abacus
    input_structure: str
    incar: Optional[Dict[str, object]] = None
    kpoints: Tuple[int, int, int] = (4, 4, 1)
    relax: bool = True
    freq: bool = False        # IBRION=5/6 vibrational frequency calculation
    ncpu: int = 16

    # Outputs populated after submission.
    job_id: Optional[int] = None
    job_state: str = ""
    final_energy: Optional[float] = None
    final_max_force: Optional[float] = None
    converged: bool = False
    error: str = ""

    def _prepare_vasp(self) -> Tuple[str, str]:
        """Write all VASP input files and return the run command + workdir."""
        wd = self.workdir / "vasp"
        wd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.input_structure, wd / "POSCAR")
        write_incar(wd, self.incar, relax=self.relax, freq=self.freq)
        write_kpoints(wd, self.kpoints)

        # POTCAR — try to find automatically (skipped in mock mode: no real
        # pseudopotential library is required for synthetic runs).
        if not is_mock():
            symbols = self._guess_symbols()
            if symbols:
                try:
                    write_potcar(wd, symbols)
                except (FileNotFoundError, Exception) as exc:
                    self.error = f"POTCAR generation failed: {exc}"

        exe = _resolve_exe(ENV_VASP_EXE, DEFAULT_VASP_EXE) or "vasp_std"
        run_cmd = f"{SlurmConfig().mpi_cmd} {exe} > vasp.out"
        return run_cmd, str(wd)

    def _prepare_cp2k(self) -> Tuple[str, str]:
        """Generate a minimal CP2K input and return the run command + workdir."""
        wd = self.workdir / "cp2k"
        wd.mkdir(parents=True, exist_ok=True)
        # Convert structure to CP2K input format via ASE.
        atoms = read(self.input_structure)
        # Write a basic CP2K input file (users customize via incar overrides).
        inp = _cp2k_input(atoms, self.incar or {}, self.kpoints, relax=self.relax)
        (wd / "input.inp").write_text(inp, encoding="utf-8")
        write(wd / "coord.xyz", atoms, format="xyz")

        exe = _resolve_exe(ENV_CP2K_EXE, "cp2k") or "cp2k"
        run_cmd = f"{SlurmConfig().mpi_cmd} {exe} -i input.inp > cp2k.out"
        return run_cmd, str(wd)

    def _prepare_abacus(self) -> Tuple[str, str]:
        """Generate ABACUS input files and return the run command + workdir."""
        wd = self.workdir / "abacus"
        wd.mkdir(parents=True, exist_ok=True)
        atoms = read(self.input_structure)
        # ABACUS needs STRU + INPUT + pseudopotential.
        _abacus_input(atoms, wd, self.incar or {}, self.kpoints, relax=self.relax)

        exe = _resolve_exe(ENV_ABACUS_EXE, "abacus") or "abacus"
        run_cmd = f"{SlurmConfig().mpi_cmd} {exe} > abacus.out"
        return run_cmd, str(wd)

    def _guess_symbols(self) -> List[str]:
        try:
            atoms = read(self.input_structure)
            return list(dict.fromkeys(atoms.get_chemical_symbols()))
        except Exception:
            return []

    def submit(self) -> Optional[int]:
        """Prepare input files, submit the job, return job/pid."""
        if self.calculator == "vasp":
            run_cmd, wd = self._prepare_vasp()
        elif self.calculator == "cp2k":
            run_cmd, wd = self._prepare_cp2k()
        elif self.calculator == "abacus":
            run_cmd, wd = self._prepare_abacus()
        else:
            self.error = f"Unsupported calculator: {self.calculator}"
            return None

        cfg = SlurmConfig()
        job_name = f"slisa_{self.calculator}"
        self.job_id = cfg.submit(job_name, Path(wd), run_cmd)
        return self.job_id

    def collect(self) -> Dict[str, object]:
        """Poll job completion and collect results."""
        if self.job_id is not None:
            state = wait_job(self.job_id, poll_interval=10, timeout=7200)
            self.job_state = state
        self._extract_results()
        return self._to_dict()

    def _extract_results(self):
        if self.calculator == "vasp":
            wd = self.workdir / "vasp"
            self.final_energy = parse_vasp_energy(wd)
            self.converged = parse_vasp_converged(wd)
            forces = parse_vasp_forces(wd)
            self.final_max_force = forces[-1] if forces else None
        elif self.calculator == "cp2k":
            wd = self.workdir / "cp2k"
            self.final_energy = _parse_cp2k_energy(str(wd / "cp2k.out"))
            self.converged = self.final_energy is not None
        elif self.calculator == "abacus":
            wd = self.workdir / "abacus"
            self.final_energy = _parse_abacus_energy(str(wd / "abacus.out"))
            self.converged = self.final_energy is not None

    def _to_dict(self) -> Dict[str, object]:
        return {
            "calculator": self.calculator,
            "workdir": str(self.workdir),
            "job_id": self.job_id,
            "job_state": self.job_state,
            "final_energy_eV": self.final_energy,
            "final_max_force_eV_per_angstrom": self.final_max_force,
            "converged": self.converged,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# CP2K input generation (minimal)
# ---------------------------------------------------------------------------

def _cp2k_input(atoms: "Atoms", settings: Dict[str, object],
                kpoints: Tuple[int, int, int], relax: bool) -> str:
    # Write a minimal CP2K input template.
    cell = atoms.get_cell()
    cell_str = "\n".join(f"  {' '.join(f'{v:.6f}' for v in row)}" for row in cell)
    coord = "\n".join(
        f"  {s}  {' '.join(f'{p:.6f}' for p in atoms.get_positions()[i])}"
        for i, s in enumerate(atoms.get_chemical_symbols())
    )
    inp = f"""&GLOBAL
  PROJECT slisadft
  RUN_TYPE {"GEO_OPT" if relax else "ENERGY"}
  PRINT_LEVEL MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME POTENTIAL
    &MGRID
      CUTOFF {settings.get("cutoff", 400)}
      REL_CUTOFF {settings.get("rel_cutoff", 60)}
    &END MGRID
    &SCF
      SCF_GUESS RESTART
      MAX_SCF {settings.get("max_scf", 100)}
      EPS_SCF {settings.get("eps_scf", 1.0e-6)}
      &OT
        T {settings.get("ot", True)}
        MINIMIZER DIIS
      &END OT
    &END SCF
    &KPOINTS
      SCHEME MONKHORST-PACK
      {kpoints[0]} {kpoints[1]} {kpoints[2]}
    &END KPOINTS
  &END DFT
  &SUBSYS
    &CELL
{cell_str}
    &END CELL
    &COORD
{coord}
    &END COORD
    &QMMM
      &QMKIND
        &BASIS
          &BASIS_SET
            &END
          &END
        &END
      &END
    &END
  &END SUBSYS
&END FORCE_EVAL
"""
    return inp


# ---------------------------------------------------------------------------
# ABACUS input generation (minimal)
# ---------------------------------------------------------------------------

def _abacus_input(atoms: "Atoms", workdir: Path, settings: Dict[str, object],
                  kpoints: Tuple[int, int, int], relax: bool):
    """Write minimal ABACUS INPUT and STRU files."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    # INPUT (ABACUS key = value lines)
    input_params = {
        "calculation": "relax" if relax else "scf",
        "ntype": len(set(symbols)),
        "nbands": settings.get("nbands", 20),
        "ecutwfc": settings.get("ecutwfc", 50),
        "scf_nmax": settings.get("scf_nmax", 100),
        "ks_solver": "genelpa",
        "basis_type": "pw",
        "pseudo_dir": "./PP",
        "mixing_type": "broyden",
        "scf_thr": settings.get("scf_thr", 1e-6),
        "dr2": 1e-3,
    }
    inp = "INPUT_PARAMETERS\n" + "\n".join(
        f"{key} {value}" for key, value in input_params.items())
    (workdir / "INPUT").write_text(inp + "\n", encoding="utf-8")

    # STRU
    from collections import Counter
    counts = Counter(symbols)
    stru_lines = ["ATOMIC_SPECIES"]
    for sym, cnt in counts.items():
        stru_lines.append(f"{sym}  {cnt}  {sym}_PSEUDO")
    stru_lines.append(f"\nNUMERICAL_ORBITAL\n")
    stru_lines.append(f"LATTICE_CONSTANT\n  1.0")
    stru_lines.append(f"LATTICE_VECTORS")
    for row in cell:
        stru_lines.append(f"  {' '.join(f'{v:.6f}' for v in row)}")
    for sym in set(symbols):
        n = counts[sym]
        sel = [i for i, s in enumerate(symbols) if s == sym]
        stru_lines.append(f"\nATOMIC_POSITIONS\n  {sym}\n  {n}")
        for i in sel:
            p = positions[i]
            stru_lines.append(f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}  0  0  0")
    (workdir / "STRU").write_text("\n".join(stru_lines) + "\n", encoding="utf-8")

    # Create a simple pseudopotential directory stub with a placeholder.
    pp_dir = workdir / "PP"
    pp_dir.mkdir(parents=True, exist_ok=True)
    for sym in set(symbols):
        # A minimal placeholder — real pseudopotentials must be provided.
        (pp_dir / f"{sym}_PSEUDO").write_text(
            "Placeholder — replace with real ABACUS pseudopotential.\n",
            encoding="utf-8")


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_cp2k_energy(out_path: str) -> Optional[float]:
    """Extract final total energy from CP2K output."""
    import re
    p = Path(out_path)
    if not p.exists():
        return None
    text = p.read_text(errors="ignore")
    m = re.findall(r"Total energy:\s+([-+]?[\d.]+)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"ENERGY\| Total FORCE_EVAL.*?([-+]?[\d.]+)", text)
    if m:
        return float(m[-1])
    return None


def _parse_abacus_energy(out_path: str) -> Optional[float]:
    """Extract final total energy from ABACUS output."""
    import re
    p = Path(out_path)
    if not p.exists():
        return None
    text = p.read_text(errors="ignore")
    m = re.findall(r"!FINAL_ETOT_IS\s+([-+]?[\d.]+)", text)
    if m:
        return float(m[-1])
    return None


# ---------------------------------------------------------------------------
# CrewAI Tools
# ---------------------------------------------------------------------------


@tool
def dft_optimize_tool(input_structure: str, calculator: str = "vasp",
                      incar_overrides: str = "",
                      kpoints: str = "4,4,1",
                      relax: bool = True,
                      freq: bool = False) -> str:
    """Run a DFT geometry optimization (or single-point energy) with
    the selected engine: VASP, CP2K, or ABACUS.

    The tool submits the job to SLURM (or runs locally if SLURM is
    unavailable), waits for completion, and returns the energy and
    convergence status.

    Args:
        input_structure: path to the input structure (POSCAR format).
        calculator: 'vasp', 'cp2k', or 'abacus'.
        incar_overrides: optional JSON string of INCAR/input parameters
                         to override defaults (e.g. '{"ENCUT": 600}').
        kpoints: k-point mesh as 'nx,ny,nz' (e.g. '4,4,1').
        relax: whether to perform geometry relaxation (True) or
               single-point energy calculation (False).
        freq: run a vibrational frequency calculation instead
              (VASP only: IBRION=5 finite differences; implies relax=False).

    Returns:
        A JSON string with the calculation results.
    """
    import json as _json
    import tempfile

    if freq:
        relax = False

    # Mock engine mode: deterministic synthetic run without any real binary.
    if is_mock():
        from ..engines.mock import run_mock_dft
        workdir = Path(tempfile.mkdtemp(prefix=f"slisa_{calculator}_mock_", dir="."))
        try:
            result = run_mock_dft(calculator, input_structure, workdir,
                                  relax=relax, freq=freq,
                                  incar=_parse_overrides(incar_overrides),
                                  kpoints=_parse_kpoints(kpoints))
        except ValueError:
            return _json.dumps({"error": f"Unsupported calculator: {calculator}",
                                "calculator": calculator})
        return _json.dumps(result, ensure_ascii=False)

    # Validate calculator availability.
    checkers = {"vasp": _check_vasp, "cp2k": _check_cp2k, "abacus": _check_abacus}
    ok, msg = checkers[calculator]()
    if not ok:
        from ..utils.failure_diagnosis import diagnose, summarize_diagnosis
        return _json.dumps({"error": msg, "calculator": calculator,
                            "diagnosis": summarize_diagnosis(diagnose(msg))})

    if freq and calculator != "vasp":
        return _json.dumps({
            "error": "frequency calculations are currently only supported for VASP",
            "calculator": calculator,
        })

    # Create a unique workdir.
    workdir = Path(tempfile.mkdtemp(prefix=f"slisa_{calculator}_", dir="."))

    setup = _make_setup(workdir)
    job_id = setup.submit()
    if job_id is None:
        return _json.dumps(setup._to_dict())
    result = setup.collect()
    return _json.dumps(result, ensure_ascii=False)


def _parse_kpoints(kpoints: str) -> Tuple[int, int, int]:
    """Parse 'nx,ny,nz' into a tuple, falling back to (4, 4, 1)."""
    try:
        kp = tuple(int(x) for x in kpoints.split(","))
        if len(kp) == 3:
            return kp  # type: ignore[return-value]
    except Exception:
        pass
    return (4, 4, 1)


def _parse_overrides(incar_overrides: str) -> Dict[str, object]:
    if not incar_overrides:
        return {}
    try:
        return json.loads(incar_overrides)
    except Exception:
        return {}


@tool
def dft_single_point_tool(input_structure: str, calculator: str = "vasp",
                          incar_overrides: str = "",
                          kpoints: str = "4,4,1") -> str:
    """Run a single-point DFT energy calculation (no geometry relaxation).

    Identical to ``dft_optimize_tool`` with ``relax=False``.  Use this
    for electronic energy of an already-optimized structure.
    """
    return dft_optimize_tool.func(input_structure, calculator, incar_overrides,
                                  kpoints, relax=False)


@tool
def check_dft_environment_tool() -> str:
    """Check which DFT engines (VASP, CP2K, ABACUS) are available on
    the current system and return a report.

    Useful for diagnosing the environment before running calculations.
    In mock engine mode all three engines are reported as synthetic.
    """
    import json as _json

    if is_mock():
        from ..engines import SYNTHETIC_MARK
        return _json.dumps(
            {name: {"available": True, "detail": f"mock engine mode — {SYNTHETIC_MARK}"}
             for name in ("vasp", "cp2k", "abacus")},
            ensure_ascii=False, indent=2)

    results = {}
    for name, check in [("vasp", _check_vasp), ("cp2k", _check_cp2k),
                        ("abacus", _check_abacus)]:
        ok, msg = check()
        results[name] = {"available": ok, "detail": msg}
    return _json.dumps(results, ensure_ascii=False, indent=2)