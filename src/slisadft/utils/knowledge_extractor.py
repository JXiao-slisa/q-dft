"""Automatic knowledge extraction from DFT calculation results.

After each calculation (VASP/CP2K/ABACUS/MLIP), this module extracts:
  - System info (formula, elements, atoms, cell)
  - Calculation parameters (engine, functional, encut, kpoints)
  - Convergence info (steps, time, final energy, final force)
  - Result summary (adsorption energy, d-band, etc.)
  - Issues / warnings

The extracted knowledge is stored in YAML format under knowledge/auto/,
organized by chemical system (e.g., Pt_CO, C_diamond).
"""

from __future__ import annotations

import os
import re
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ── Knowledge base constants ──────────────────────────────────────────────

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[2] / "knowledge" / "auto"
# Fallback: if parents[2] is ".../src" (package layout), go up one more to project root
if Path(__file__).resolve().parents[2].name == "src":
    KNOWLEDGE_ROOT = Path(__file__).resolve().parents[3] / "knowledge" / "auto"
RULES_FILE = Path(__file__).resolve().parents[2] / "knowledge" / "computational_rules.md"

# Ensure directory structure
for d in [KNOWLEDGE_ROOT, KNOWLEDGE_ROOT / "summary"]:
    d.mkdir(parents=True, exist_ok=True)


# ── Data structures ───────────────────────────────────────────────────────

def _system_dir(system: str) -> Path:
    """Return the knowledge directory for a chemical system."""
    safe = system.replace(" ", "_").replace("/", "_").replace("(", "_").replace(")", "_")
    d = KNOWLEDGE_ROOT / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_record_id(system_dir: Path) -> str:
    """Generate a unique record ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing = len(list(system_dir.glob("*.yaml")))
    return f"{ts}_{existing:03d}"


# ── Knowledge record schema ────────────────────────────────────────────────


def make_record(
    system: str,
    engine: str,
    calculation: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a structured knowledge record (经验).

    Fields:
      system, engine, timestamp, calculation (parameters + convergence),
      result (energies + derived quantities), issues (warnings / errors),
      tags (keywords for retrieval).
    """
    return {
        "type": "experience",  # 经验 = reference / recommendation
        "system": system,
        "engine": engine,
        "timestamp": datetime.now().isoformat(),
        "calculation": calculation,
        "result": result or {},
        "issues": issues or [],
        "tags": tags or [],
    }


def make_rule(
    category: str,
    description: str,
    rule: str,
    source: str,
    severity: str = "constraint",  # constraint | recommendation
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a structured rule record (规则 = lower-level constraint).

    Rules are reviewed by DFT expert + user before being added to
    computational_rules.md.
    """
    return {
        "type": "rule",
        "category": category,
        "description": description,
        "rule": rule,
        "source": source,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
        "tags": tags or [],
    }


# ── Extraction from VASP ──────────────────────────────────────────────────


def extract_vasp(dir_path: str) -> Optional[Dict[str, Any]]:
    """Extract a knowledge record from a VASP calculation directory."""
    from ..utils.vasp_input import (
        parse_vasp_energy, parse_vasp_converged, parse_vasp_forces,
        parse_fermi_energy, parse_doscar,
    )
    from ..utils.dos_analysis import d_band_center, accumulate_pdos
    import numpy as np

    d = Path(dir_path)
    if not (d / "OUTCAR").exists():
        return None

    # Calculation parameters
    incar_text = (d / "INCAR").read_text() if (d / "INCAR").exists() else ""
    params = {}
    for line in incar_text.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()

    energy = parse_vasp_energy(dir_path)
    converged = parse_vasp_converged(dir_path)
    forces = parse_vasp_forces(dir_path)
    fermi = parse_fermi_energy(dir_path)

    # Determine system from POSCAR
    system = "unknown"
    if (d / "POSCAR").exists():
        from ase.io import read
        try:
            atoms = read(str(d / "POSCAR"))
            system = atoms.get_chemical_formula()
        except Exception:
            pass

    calc = {
        "engine": "vasp",
        "encut": params.get("ENCUT", "unknown"),
        "kpoints": params.get("KPOINTS", "unknown"),
        "functional": "PBE",
        "n_ionic_steps": forces[-1] if forces else None,
        "converged": converged,
        "final_energy": energy,
        "final_force": forces[-1] if forces else None,
    }

    # DOS analysis if available
    result = {}
    dos_data = parse_doscar(dir_path)
    if dos_data.get("pdos"):
        energies = np.array(dos_data["energies"], dtype=float)
        ef = dos_data.get("efermi", 0.0)
        d_cols = ["dxy", "dyz", "dz2", "dxz", "dx2"]
        _, d_dos = accumulate_pdos(dos_data["pdos"], columns=d_cols)
        if len(d_dos) > 0:
            result["d_band_center"] = d_band_center(energies, d_dos, ef)

    # Issues
    issues = []
    if not converged:
        # Single-point (NSW=0, IBRION=-1) doesn't have "reached required accuracy"
        is_single_point = params.get("IBRION", "") == "-1" and params.get("NSW", "") == "0"
        if not is_single_point and energy is None:
            issues.append("Calculation did not converge")
    if energy is None:
        issues.append("Could not extract final energy")

    return make_record(
        system=system,
        engine="vasp",
        calculation=calc,
        result=result,
        issues=issues,
        tags=["vasp", system],
    )


# ── Extraction from calculation logs ──────────────────────────────────────


def save_record(record: Dict[str, Any]) -> str:
    """Save a knowledge record to the YAML knowledge base."""
    system = record.get("system", "unknown")
    sys_dir = _system_dir(system)
    record_id = _next_record_id(sys_dir)
    path = sys_dir / f"{record_id}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(record, f, default_flow_style=False, allow_unicode=True)
    return str(path)


def save_rule(rule: Dict[str, Any]) -> str:
    """Save a proposed rule for expert review."""
    import re
    rules_dir = KNOWLEDGE_ROOT / "pending_rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize filename: replace non-alnum chars with _
    safe_cat = re.sub(r"[^A-Za-z0-9_-]+", "_", rule["category"]).strip("_")
    safe_desc = re.sub(r"[^A-Za-z0-9_-]+", "_", rule["description"][:30]).strip("_")
    path = rules_dir / f"{safe_cat}_{safe_desc}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(rule, f, default_flow_style=False, allow_unicode=True)
    return str(path)


def list_systems() -> List[str]:
    """List all chemical systems with knowledge records."""
    return sorted(d.name for d in KNOWLEDGE_ROOT.iterdir() if d.is_dir() and d.name != "summary")


def list_records(system: str) -> List[Dict[str, Any]]:
    """List all knowledge records for a system."""
    sys_dir = _system_dir(system)
    records = []
    for f in sorted(sys_dir.glob("*.yaml")):
        try:
            records.append(yaml.safe_load(f.read_text()))
        except Exception:
            pass
    return records


def get_summary(system: str) -> Dict[str, Any]:
    """Compute a summary of all records for a system."""
    records = list_records(system)
    if not records:
        return {"system": system, "n_records": 0}
    summary = {
        "system": system,
        "n_records": len(records),
        "engines": list(set(r.get("engine", "?") for r in records)),
        "last_updated": max(r.get("timestamp", "") for r in records),
        "best_energy": min((r.get("result", {}).get("adsorption_energy", float("inf"))
                           for r in records if r.get("result", {}).get("adsorption_energy")),
                          default=None),
        "issues": list(set(
            i for r in records for i in r.get("issues", [])
        )),
    }
    return summary