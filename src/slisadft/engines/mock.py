"""Mock 引擎后端：确定性合成数据，仅用于演示 / 开发 / CI。

实现要点：

* 优先使用 ASE 的 EMT 势（支持 H, C, N, O, Al, Ni, Cu, Pd, Ag, Pt, Au），
  能给出物理上定性合理（且对相同体系完全可复现）的能量；
  不支持的元素自动退回 Lennard-Jones 势。
* ``run_mock_dft`` 先用真实代码路径生成引擎输入文件（INCAR / KPOINTS /
  CP2K / ABACUS 输入），再用 mock 能量写出与真实引擎同格式的输出文件，
  因此下游解析、报告与溯源流程被完整 exercised。
* 所有输出文件都写入 ``SYNTHETIC`` 标记行（Rule Zero）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..engines import SYNTHETIC_MARK, synthetic_result

# EMT 支持的元素（ASE 实现）；其余元素退回 LJ。
_EMT_ELEMENTS = {"H", "C", "N", "O", "Al", "Ni", "Cu", "Pd", "Ag", "Pt", "Au"}


def get_mock_calculator(atoms):
    """Return a deterministic ASE calculator for ``atoms`` (EMT or LJ)."""
    symbols = set(atoms.get_chemical_symbols())
    if symbols <= _EMT_ELEMENTS:
        from ase.calculators.emt import EMT
        return EMT()
    from ase.calculators.lj import LennardJones
    return LennardJones(rc=6.0, smooth=True)


def mock_relax(input_structure: str, fmax: float = 0.05, steps: int = 60):
    """Relax ``input_structure`` with the mock calculator; returns (atoms, info)."""
    from ase.io import read
    from ase.optimize import BFGS

    atoms = read(input_structure)
    atoms.calc = get_mock_calculator(atoms)
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    import numpy as np
    forces = atoms.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0
    info = {
        "energy": float(atoms.get_potential_energy()),
        "max_force": max_force,
        "converged": bool(max_force <= fmax),
        "n_atoms": len(atoms),
    }
    return atoms, info


# ---------------------------------------------------------------------------
# Mock engine output writers（格式与真实引擎一致，便于复用解析器）
# ---------------------------------------------------------------------------

def _header_lines(engine: str) -> str:
    return "\n".join([
        f"# {engine} output — {SYNTHETIC_MARK}",
        "# E_ad / G / DOS figures derived from this file are synthetic.",
    ])


def write_mock_vasp_outputs(directory: Path, energy: float, max_force: float,
                            converged: bool) -> Path:
    """Write OSZICAR / OUTCAR files that the project's parsers understand."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    osz = directory / "OSZICAR"
    osz.write_text(
        _header_lines("VASP (mock)") + "\n"
        "       N       E                     dE       d eps   ncg     rms          rms(c)\n"
        "DAV:   1    " f"{energy:+.6f}   " f"{energy:+.6f}\n"
        f"F=  {energy:.6f}     E0= {energy:.6f}  d E ={0.0:+.6f}\n",
        encoding="utf-8",
    )

    outcar = directory / "OUTCAR"
    lines = _header_lines("VASP (mock)").splitlines()
    lines += [
        "  Timing: mock run, no real electronic steps",
        f"  free  energy   TOTEN        =     {energy:.6f} eV",
        "",
        "  POSITION                                       TOTAL-FORCE (eV/Angst)",
        "  -----------------------------------------------------------------------------------",
        f"     0.00000     0.00000     0.00000     0.000000   0.000000  {max_force:12.6f}",
        "  -----------------------------------------------------------------------------------",
    ]
    if converged:
        lines.append("  reached required accuracy - stopping structural energy minimisation")
    outcar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcar


def write_mock_cp2k_output(directory: Path, energy: float) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "cp2k.out"
    out.write_text(
        _header_lines("CP2K (mock)") + "\n"
        "  SCF run converged in  12 steps\n"
        f" Total energy: {energy:.6f}\n",
        encoding="utf-8",
    )
    return out


def write_mock_abacus_output(directory: Path, energy: float) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "abacus.out"
    out.write_text(
        _header_lines("ABACUS (mock)") + "\n"
        " * * * * * * * *\n"
        f"!FINAL_ETOT_IS {energy:.6f} eV\n"
        " * * * * * * * *\n",
        encoding="utf-8",
    )
    return out


def run_mock_dft(calculator: str, input_structure: str, workdir: Path,
                 relax: bool = True, freq: bool = False,
                 incar: Optional[dict] = None,
                 kpoints=(4, 4, 1)) -> Dict[str, object]:
    """Run the mock DFT workflow for one calculator and collect results.

    Real input-generation code paths are exercised first (INCAR / KPOINTS /
    CP2K / ABACUS inputs), then mock engine outputs are written and parsed
    back through the project's own parsers.

    ``workdir`` is the run directory (engine files land in ``workdir/<calc>``).
    Returns a result dict already stamped with the SYNTHETIC markers.
    """
    # Generate the same input files the real engine would receive.
    from ..tools.dft_tools import DftSetup
    setup = DftSetup(
        workdir=Path(workdir),
        calculator=calculator,
        input_structure=input_structure,
        incar=incar,
        kpoints=tuple(kpoints),
        relax=relax,
        freq=freq,
    )
    preparers = {"vasp": setup._prepare_vasp, "cp2k": setup._prepare_cp2k,
                 "abacus": setup._prepare_abacus}
    if calculator not in preparers:
        raise ValueError(f"unsupported calculator: {calculator}")
    preparers[calculator]()

    atoms, info = mock_relax(input_structure)
    if not relax:
        # Single-point: use the unrelaxed structure energy for honesty.
        from ase.io import read as _read
        sp = _read(input_structure)
        sp.calc = get_mock_calculator(sp)
        info = {
            "energy": float(sp.get_potential_energy()),
            "max_force": info["max_force"],
            "converged": True,
            "n_atoms": len(sp),
        }

    wd = Path(workdir) / calculator
    if calculator == "vasp":
        write_mock_vasp_outputs(wd, info["energy"], info["max_force"], True)
    elif calculator == "cp2k":
        write_mock_cp2k_output(wd, info["energy"])
    elif calculator == "abacus":
        write_mock_abacus_output(wd, info["energy"])

    return synthetic_result({
        "calculator": calculator,
        "workdir": str(wd),
        "job_id": None,
        "job_state": "COMPLETED_MOCK",
        "final_energy_eV": round(info["energy"], 6),
        "final_max_force_eV_per_angstrom": round(info["max_force"], 6),
        "converged": True,
        "error": "",
    })
