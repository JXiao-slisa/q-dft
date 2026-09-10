"""跨引擎一致性校验（cross-engine parity QA）。

slisaDFT 的经验规则多次记录"某 MLIP 对某类化学体系误差过大"
（如某通用势对特定化学体系误差偏大）。本模块把这类经验固化为
例行 QA：对同一结构用两个计算器各做一次单点，报告能量/受力偏差并对照
精度预算（accuracy budget）判定是否可接受。

mock 模式提供 EMT 与 Lennard-Jones 两个确定性计算器用于测试/演示；
real 模式下 calculators 可指定 mace / dpa（要求相应环境已安装），
DFT 参照值请通过 dft_single_point_tool 获取后自行对比。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from . import is_mock


def _calculator(name: str, atoms):
    """Resolve a calculator by name (EMT/LJ in mock; MACE/DPA in real)."""
    key = name.strip().lower()
    if key in ("emt", "lj"):
        from .mock import get_mock_calculator
        calc = get_mock_calculator(atoms) if key == "emt" else None
        if key == "lj":
            from ase.calculators.lj import LennardJones
            calc = LennardJones(rc=6.0, smooth=True)
        if calc is None:
            raise ValueError(f"calculator '{name}' unavailable for this system")
        return calc, f"{key}(mock)"
    if key in ("mace", "mace-mp", "dpa", "dpa4", "deepmd"):
        if is_mock():
            # mock 模式不加载 torch 权重，退回 EMT 并显式标注
            from .engines.mock import get_mock_calculator
            return (get_mock_calculator(atoms),
                    f"{key}->EMT(mock substitute, SYNTHETIC)")
        from .tools.mlip_tools import (
            _get_mace_calculator, _get_dpa_calculator)
        if key.startswith("mace"):
            return _get_mace_calculator(), "MACE"
        return _get_dpa_calculator(), "DPA-4"
    raise ValueError(f"unknown calculator '{name}' "
                     "(available: emt, lj, mace, dpa)")


def _single_point(atoms, calc_name: str) -> Dict[str, object]:
    import numpy as np

    work = atoms.copy()
    work.calc = _calculator(calc_name, work)[0]
    energy = float(work.get_potential_energy())
    forces = work.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0
    return {"calculator": calc_name, "energy_eV": energy,
            "max_force_eV_per_angstrom": max_force, "n_atoms": len(work)}


def parity_report(structure: str, calc_a: str = "emt", calc_b: str = "lj",
                  budget_ev: Optional[float] = None,
                  budget_force: Optional[float] = None) -> Dict[str, object]:
    """Single-point two calculators on one structure and compare.

    Args:
        structure: path to POSCAR/CIF/XYZ.
        calc_a / calc_b: calculator names (emt, lj, mace, dpa).
        budget_ev: |ΔE| tolerance in eV (default: 0.2 eV per system).
        budget_force: max-force difference tolerance eV/Å (default 0.5).

    Returns:
        {"points": {a: {...}, b: {...}}, "delta_eV", "delta_max_force",
         "within_budget", "budget_ev", "synthetic", ...}
    """
    from ase.io import read

    atoms = read(structure)
    budget_ev = 0.2 if budget_ev is None else float(budget_ev)
    budget_force = 0.5 if budget_force is None else float(budget_force)

    pa = _single_point(atoms, calc_a)
    pb = _single_point(atoms, calc_b)
    d_e = abs(pa["energy_eV"] - pb["energy_eV"])
    d_f = abs(pa["max_force_eV_per_angstrom"] - pb["max_force_eV_per_angstrom"])

    report = {
        "structure": str(structure),
        "points": {calc_a: pa, calc_b: pb},
        "delta_energy_eV": round(d_e, 6),
        "delta_max_force_eV_per_angstrom": round(d_f, 6),
        "budget_ev": budget_ev,
        "budget_force": budget_force,
        "within_budget": bool(d_e <= budget_ev and d_f <= budget_force),
        "engine_mode": "mock" if is_mock() else "real",
        "synthetic": is_mock() or "EMT(mock substitute)" in str(pa) + str(pb),
    }
    report["verdict"] = (
        "PASS — calculators agree within the accuracy budget"
        if report["within_budget"] else
        "REVIEW — deviation exceeds the accuracy budget; prefer the "
        "higher-fidelity calculator for this chemistry (see knowledge rules)")
    return report
