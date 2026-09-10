"""Self-healing runs (v0.4.0): diagnosis-driven automatic remediation.

Bridges failure intelligence (``utils/failure_diagnosis.py``) and actual
recalculation: given the diagnosis codes of a failed DFT run, produce a
concrete, minimal set of calculation-parameter remedies and — at the workflow
level — automatically retry once with the remedied parameters.

Every remediation is recorded in the run result (``remediation`` field), so
the automatic intervention is auditable (Rule Zero).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# code -> (INCAR-level remedies, human note)
_INCAR_REMEDIES: Dict[str, Tuple[Dict[str, object], str]] = {
    "SCF_NOT_CONVERGED": (
        {"NELM": 200, "ALGO": "All", "SIGMA": 0.05},
        "tightened SCF: NELM doubled, ALGO=All, smaller smearing",
    ),
    "ZBRENT_FATAL": (
        {"POTIM": 0.15, "EDIFFG": -0.03},
        "smaller ionic step: POTIM=0.15, relaxed EDIFFG",
    ),
    "TETRAHEDRON_FAIL": (
        {"ISMEAR": 0, "SIGMA": 0.05},
        "switched to Gaussian smearing (ISMEAR=0)",
    ),
    "OUT_OF_MEMORY": (
        {"NCORE": 4, "NPAR": 1},
        "reduced parallelization: NCORE=4, NPAR=1",
    ),
}


def remediate_incar(incar: Optional[Dict[str, object]],
                    diagnosis_codes: List[str]) -> Tuple[Dict[str, object], List[str], List[str]]:
    """Apply known remedies for diagnosis codes to an INCAR-like parameter dict.

    Args:
        incar: current calculation parameters (may be empty).
        diagnosis_codes: codes from ``failure_diagnosis.diagnose``.

    Returns:
        (new_params, applied_codes, notes) — ``applied_codes`` lists the codes
        that had a known remedy and were applied.
    """
    params = dict(incar or {})
    applied: List[str] = []
    notes: List[str] = []
    for code in diagnosis_codes:
        remedy = _INCAR_REMEDIES.get(code)
        if not remedy:
            continue
        overrides, note = remedy
        params.update(overrides)
        applied.append(code)
        notes.append(f"{code}: {note}")
    return params, applied, notes


def remediation_note(applied: List[str], notes: List[str]) -> Optional[str]:
    """One-line audit string for run records (None when nothing applied)."""
    if not applied:
        return None
    return "auto-remediated (" + "; ".join(notes) + ")"
