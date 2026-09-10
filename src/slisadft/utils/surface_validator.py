"""Surface Validator Agent — 表面参数收敛性测试 + 数据归档规则。

Rule（用户确认）：
  R-SURF-1: 当出现以下任一情况时，必须对表面参数做收敛性测试：
            (a) 新项目/新体系；(b) 新空间群的晶体切表面；
            (c) 新晶胞参数；(d) 新中间体（原子数或直径超过已有中间体数量级，
            或直径大于晶胞参数 1/2）。
  收敛判据：相邻两档参数（超胞大小/层数）吸附能差异 < 0.1 eV 视为收敛。

  R-ARCH-1: 每次涉及能量/优化的计算必须有独立文件夹（含 init.vasp/opt.vasp/
            opt.traj/optimization.log）；数据 JSON 每条记录必须含 `dir` 字段
            指向对应文件夹，可回溯到结构+能量+轨迹。

本 Agent 实现：
  1. 超胞/层数收敛性测试
  2. 中间体尺寸 vs 晶胞参数检查（触发收敛测试的依据）
  3. 吸附位点合理性检查
  4. 数据归档完整性验证（archiver）
"""

from __future__ import annotations

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.io import read, write


# ── Rule definitions (shared) ──────────────────────────────────────────────

RULES = {
    "R-SURF-1": (
        "表面参数收敛性测试触发条件：新项目/新空间群晶体切表面/新晶胞参数/"
        "新中间体(原子数或直径超已有中间体数量级, 或直径>晶胞参数1/2)时，"
        "必须对超胞大小、层数、真空层、k-points做收敛性测试。"
    ),
    "R-SURF-2": "收敛判据：相邻两档参数吸附能差异 < 0.1 eV 视为收敛。",
    "R-ARCH-1": (
        "每次涉及能量/优化的计算必须独立文件夹(init.vasp/opt.vasp/opt.traj/"
        "optimization.log)；数据JSON每条记录含dir字段可回溯。"
    ),
    "R-ARCH-2": "每次迭代必须有迭代日志(iteration_log.md)记录变更、决策、台账引用。",
}


# ── 1. Intermediate size check (triggers convergence test) ────────────────

def intermediate_diameter(atoms: Atoms) -> float:
    """Estimate max lateral extent of an adsorbate (Å)."""
    if len(atoms) <= 1:
        return 1.0
    pos = atoms.get_positions()
    center = pos.mean(axis=0)
    # max distance from center among atoms
    return float(2.0 * np.max(np.linalg.norm(pos - center, axis=1)))


def check_intermediate_vs_cell(adsorbate: Atoms, cell_vectors, 
                               warning_ratio: float = 0.5) -> Dict:
    """Check if intermediate diameter exceeds half the in-plane cell size.
    Returns risk assessment."""
    d = intermediate_diameter(adsorbate)
    cell = np.array(cell_vectors)
    # in-plane cell lengths (a, b)
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    min_cell = min(a_len, b_len)
    risk = "none"
    if d > warning_ratio * min_cell:
        risk = "HIGH (需收敛性测试)"
    elif d > 0.4 * min_cell:
        risk = "MEDIUM (建议测试)"
    return {
        "adsorbate_diameter_A": round(d, 2),
        "in_plane_cell_A": round(min_cell, 2),
        "ratio": round(d / min_cell, 2) if min_cell else None,
        "risk": risk,
        "rule": RULES["R-SURF-1"],
    }


# ── 2. Convergence test (supercell size / layers) ──────────────────────────

def convergence_test_adsorption(clean_slab: Atoms, adsorbate: Atoms,
                                 site_xy, size_options: List[tuple] = [(2, 2), (3, 3)],
                                 layer_options: List[int] = [3, 5],
                                 calculator=None, fmax: float = 0.05) -> Dict:
    """Run supercell-size / layer convergence test for an adsorbate.

    Returns adsorption energies at each (size, layers) and convergence verdict.
    """
    from ase.build import add_adsorbate
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms

    results = []
    for layers in layer_options:
        for size in size_options:
            slab = clean_slab.copy()
            # rebuild with target size/layers is complex; here we rely on the
            # slab being rebuilt externally. For a self-contained test we
            # assume clean_slab already at (2,2,3); expansion requires builder.
            # This function expects pre-built slabs passed in `slab_variants`.
            pass
    # NOTE: full implementation requires a slab builder that varies size/layers.
    # We provide the framework; the caller supplies pre-built slabs.
    return {"framework": "convergence_test_adsorption",
            "note": "需要外部提供不同size/layers的slab变体"}


def convergence_test_slab(clean_variants: Dict[str, Atoms],
                          adsorbate: Atoms, site_xy,
                          calculator, fmax=0.05) -> Dict:
    """Run convergence test on provided clean-slab variants.

    Args:
        clean_variants: {label: clean_slab_atoms}, e.g. {'2x2_3L': slab, '3x3_3L': slab}
        adsorbate: molecule to adsorb (will be placed at site_xy)
        site_xy: (x, y) adsorption position
    """
    from ase.build import add_adsorbate
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms

    eads = {}
    for label, slab in clean_variants.items():
        # fix bottom
        z = slab.get_positions()[:, 2]
        zmin, zmax = z.min(), z.max()
        slab_fixed = slab.copy()
        slab_fixed.set_constraint(FixAtoms(
            indices=[i for i, zi in enumerate(z) if zi <= zmin + (zmax - zmin) * 0.3]))
        e_clean = _relax_energy(slab_fixed, calculator, fmax)

        ads = slab_fixed.copy()
        mol = adsorbate.copy()
        # position molecule at site
        h = 1.8
        mz = min(mol.get_positions()[:, 2])
        mol.translate([site_xy[0], site_xy[1], zmax + h - mz])
        ads += mol
        z2 = ads.get_positions()[:, 2]
        zmin2, zmax2 = z2.min(), z2.max()
        ads.set_constraint(FixAtoms(
            indices=[i for i, zi in enumerate(z2) if zi <= zmin2 + (zmax2 - zmin2) * 0.25]))
        e_ads = _relax_energy(ads, calculator, fmax)
        eads[label] = e_ads - e_clean

    # Convergence: compare adjacent variants
    labels = list(eads.keys())
    verdicts = []
    converged = True
    for i in range(len(labels) - 1):
        diff = abs(eads[labels[i]] - eads[labels[i + 1]])
        verdicts.append({"pair": f"{labels[i]}-{labels[i+1]}",
                         "diff_eV": round(diff, 3),
                         "converged": diff < 0.1})
        if diff >= 0.1:
            converged = False
    return {"adsorption_energies": {k: round(v, 3) for k, v in eads.items()},
            "pairwise": verdicts,
            "overall_converged": converged,
            "rule": RULES["R-SURF-2"]}


def _relax_energy(atoms, calculator, fmax=0.05):
    from ase.optimize import BFGS
    atoms.calc = calculator
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=80)
    return atoms.get_potential_energy()


# ── 3. Adsorption site sanity check ───────────────────────────────────────

def check_adsorption_site(slab: Atoms, ads: Atoms,
                          min_height: float = 1.0,
                          max_height: float = 3.5) -> Dict:
    """Check adsorbate-surface distance sanity."""
    slab_z = slab.get_positions()[:, 2]
    top = slab_z.max()
    ads_z = ads.get_positions()[:, 2]
    lowest = ads_z.min()
    height = lowest - top
    ok = min_height <= height <= max_height
    return {
        "adsorbate_lowest_z": round(lowest, 2),
        "surface_top_z": round(top, 2),
        "adsorption_height_A": round(height, 2),
        "reasonable": ok,
        "note": "合理吸附高度 1.0-3.5 Å" if not ok else "",
    }


# ── 4. Archiver: validate data traceability ───────────────────────────────

def validate_archive(data_json, base_dir: str = ".") -> Dict:
    """Validate that every data record has a traceable directory with
    init.vasp / opt.vasp / opt.traj / optimization.log."""
    if isinstance(data_json, (str, os.PathLike)) and Path(data_json).exists():
        data = json.loads(Path(data_json).read_text())
    else:
        data = data_json
    if isinstance(data, dict):
        records = []
        for k, v in data.items():
            if isinstance(v, dict) and "dir" in v:
                records.append(v)
    elif isinstance(data, list):
        records = [r for r in data if isinstance(r, dict)]
    else:
        records = []

    results = []
    for r in records:
        d = r.get("dir")
        if not d:
            results.append({"record": r.get("name", "?"), "status": "MISSING dir"})
            continue
        p = Path(base_dir) / d
        files = {f: (p / f).exists() for f in ["init.vasp", "opt.vasp", "opt.traj", "optimization.log"]}
        complete = all(files.values())
        results.append({"dir": d, "files": files, "complete": complete})
    return {
        "n_records": len(records),
        "n_complete": sum(1 for r in results if r.get("complete")),
        "details": results,
        "rule": RULES["R-ARCH-1"],
    }


if __name__ == "__main__":
    # Example usage
    print("Rules:")
    for k, v in RULES.items():
        print(f"  {k}: {v[:60]}...")