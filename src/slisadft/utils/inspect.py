"""v0.4.0 usability utilities: structure inspection, run history, citation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Approximate DFT cost model for feasibility hints (very rough, deliberately
# conservative): cost scales ~ O(N_atoms^3) for diagonalization-dominated runs.
_DFT_COMFORT_ATOMS = 150
_MLIP_COMFORT_ATOMS = 5000


def inspect_structure(path: str) -> Dict[str, object]:
    """Inspect a structure file (POSCAR/CIF/XYZ/...): composition, cell, cost hints.

    Raises FileNotFoundError / ValueError on bad input.
    """
    from ase.io import read
    import numpy as np

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"structure not found: {path}")
    atoms = read(str(p))
    symbols = atoms.get_chemical_symbols()
    counts: Dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    cell = atoms.get_cell()
    volume = float(atoms.get_volume()) if atoms.cell.rank >= 3 else 0.0
    n = len(atoms)
    return {
        "file": str(p),
        "n_atoms": n,
        "composition": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "formula": atoms.get_chemical_formula(mode="hill"),
        "cell_lengths": [round(float(x), 4) for x in
                         (cell.lengths() if atoms.cell.rank >= 3 else [0, 0, 0])],
        "cell_angles": [round(float(x), 2) for x in
                        (cell.angles() if atoms.cell.rank >= 3 else [0, 0, 0])],
        "volume_angstrom3": round(volume, 3),
        "pbc": [bool(x) for x in atoms.pbc],
        "feasibility": {
            "dft": "comfortable" if n <= _DFT_COMFORT_ATOMS else (
                "expensive" if n <= 3 * _DFT_COMFORT_ATOMS else "very expensive"),
            "mlip": "comfortable" if n <= _MLIP_COMFORT_ATOMS else "expensive",
            "hint": (f"DFT reference comfort zone ≈ ≤{_DFT_COMFORT_ATOMS} atoms "
                     f"per cell; MLIP screening ≈ ≤{_MLIP_COMFORT_ATOMS}."),
        },
    }


# ---------------------------------------------------------------------------
# Run history: scan manifests written under runs/ (and jobs/<id> manifests)
# ---------------------------------------------------------------------------


def _fmt_row(m: Dict[str, object]) -> Dict[str, object]:
    e_ad = (m.get("result_summary") or {}).get("adsorption_energy_eV") \
        if isinstance(m.get("result_summary"), dict) else None
    return {
        "created_at": str(m.get("created_at", ""))[:19],
        "workflow": m.get("workflow", "?"),
        "engine_mode": m.get("engine_mode", "?"),
        "status": m.get("status", "running"),
        "element": (m.get("inputs") or {}).get("element", "")
        if isinstance(m.get("inputs"), dict) else "",
        "adsorbate": (m.get("inputs") or {}).get("adsorbate", "")
        if isinstance(m.get("inputs"), dict) else "",
        "e_ad_eV": e_ad,
        "manifest": str(m.get("_path", "")),
    }


def list_runs(runs_dir: Optional[str] = None, limit: int = 20) -> List[Dict[str, object]]:
    """List recent run manifests (newest first) as summary rows."""
    import os
    root = Path(runs_dir) if runs_dir else Path("runs")
    candidates: List[Path] = []
    if root.is_dir():
        candidates.extend(sorted(root.glob("**/run_manifest.json"),
                                 key=lambda p: p.stat().st_mtime, reverse=True))
    jobs = Path(os.environ.get("SLISA_JOBS_DIR", "jobs"))
    if jobs.is_dir():
        candidates.extend(sorted(jobs.glob("*/run_manifest.json"),
                                 key=lambda p: p.stat().st_mtime, reverse=True))
    rows: List[Dict[str, object]] = []
    seen = set()
    for mf in candidates:
        if mf in seen or len(rows) >= limit:
            continue
        seen.add(mf)
        try:
            m = json.loads(mf.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, ValueError):
            continue
        m = dict(m)
        m["_path"] = str(mf)
        rows.append(_fmt_row(m))
    return rows


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

CITATION_BIBTEX = """@software{qdft_agent,
  title  = {q-dft agent: an AI multi-agent platform for autonomous
            first-principles computational catalysis research},
  author = {q-dft team},
  year   = {2026},
  url    = {https://github.com/JXiao-slisa/q-dft},
  note   = {v0.4.0; engine-validated release, Apache-2.0}
}"""


def citation() -> Dict[str, str]:
    """Return citation info (plain text + BibTeX)."""
    return {
        "text": ("q-dft team, \"q-dft agent: an AI multi-agent platform for "
                 "autonomous first-principles computational catalysis "
                 "research\", v0.4.0, 2026. https://github.com/JXiao-slisa/q-dft"),
        "bibtex": CITATION_BIBTEX,
    }
