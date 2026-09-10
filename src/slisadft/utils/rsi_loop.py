"""RSI Scientific Hypothesis Loop — fast MLIP iteration, slow DFT verification.

Implements the closed-loop workflow:
  Hypothesis → Design → Calculate (MLIP) → Analyze → Insight → Validate → New Hypothesis

Uses MACE/DPA-4 for rapid screening (seconds/calculation) and VASP for
final verification (hours/calculation).
"""

from __future__ import annotations

import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from ase import Atoms
from ase.io import read, write
from ase.build import fcc111, molecule, add_adsorbate
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import numpy as np

from ..utils.knowledge_extractor import (
    save_record, make_record, list_systems, get_summary, KNOWLEDGE_ROOT
)

# ── Hypothesis data structure ──────────────────────────────────────────────


class Hypothesis:
    """A scientific hypothesis about a catalytic system."""

    def __init__(self, description: str, system: str, prediction: str,
                 experiment: Dict[str, Any], tags: Optional[List[str]] = None):
        self.description = description
        self.system = system
        self.prediction = prediction
        self.experiment = experiment
        self.tags = tags or []
        self.status = "proposed"  # proposed | running | verified | rejected
        self.results: List[Dict] = []
        self.insight: str = ""

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "system": self.system,
            "prediction": self.prediction,
            "experiment": self.experiment,
            "tags": self.tags,
            "status": self.status,
            "results": self.results,
            "insight": self.insight,
        }


# ── MLIP calculators (fast) ────────────────────────────────────────────────


def get_mace_calculator():
    """Load MACE model (auto-discovers local path)."""
    from mace.calculators import mace_mp
    model_path = None
    for p in [Path(os.environ.get("XDG_CACHE_HOME", ".cache")) / "mace",
              Path.home() / ".cache" / "mace"]:
        if p.is_dir():
            models = sorted(p.glob("2023-12-03-mace-128-L1_epoch-*.model"))
            if models:
                model_path = str(models[0])
                break
    return mace_mp(model=model_path if model_path else "medium",
                   device="cpu", default_dtype="float64")


def get_dpa_calculator():
    """Load DPA-4 model (auto-discovers local path)."""
    from deepmd.calculator import DP
    for p in [Path("models/mlip"), Path(os.environ.get("MLIP_MODEL_DIR", "models/mlip"))]:
        if p.is_dir():
            for f in sorted(p.glob("DPA4-*.pt")):
                if f.stat().st_size > 1024 * 1024:
                    return DP(model=str(f))
    raise RuntimeError("No DPA-4 model found. Place a DPA4-*.pt under models/mlip/")


# ── Fast structure builder ─────────────────────────────────────────────────


def build_system(element: str, miller: str = "(111)", layers: int = 3,
                 size: tuple = (2, 2), vacuum: float = 10.0,
                 adsorbate: str = "") -> Atoms:
    """Build a slab or adsorption system for rapid MLIP screening."""
    slab = fcc111(element, size=(size[0], size[1], layers), vacuum=vacuum)

    if adsorbate and adsorbate.strip():
        co = molecule(adsorbate)
        z_max = max(slab.get_positions()[:, 2])
        # Place on top site
        top_layer = [i for i, p in enumerate(slab.get_positions())
                     if abs(p[2] - z_max) < 0.5]
        center = np.mean([slab.get_positions()[i][:2] for i in top_layer], axis=0)
        dists = [np.linalg.norm(slab.get_positions()[i][:2] - center) for i in top_layer]
        top_pt = top_layer[np.argmin(dists)]
        pt_pos = slab.get_positions()[top_pt]
        co.translate([pt_pos[0], pt_pos[1], z_max + 1.85 + 0.657])
        system = slab.copy()
        system += co
        return system

    # Fix bottom layer
    z = slab.get_positions()[:, 2]
    zmin, zmax = z.min(), z.max()
    fix_idx = [i for i, zi in enumerate(z) if zi <= zmin + (zmax - zmin) * 0.3]
    slab.set_constraint(FixAtoms(indices=fix_idx))
    return slab


# ── Fast MLIP calculation ──────────────────────────────────────────────────


def fast_optimize(atoms: Atoms, calculator: str = "mace",
                  fmax: float = 0.05, steps: int = 200) -> Dict[str, Any]:
    """Run a fast MLIP geometry optimization (seconds).

    Args:
        atoms: ASE Atoms object.
        calculator: 'mace' or 'dpa'.
        fmax: force convergence criterion.
        steps: max optimization steps.

    Returns:
        dict with energy, max_force, converged, n_steps.
    """
    if calculator == "mace":
        atoms.calc = get_mace_calculator()
    elif calculator == "dpa":
        atoms.calc = get_dpa_calculator()
    else:
        raise ValueError(f"Unknown calculator: {calculator}")

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))

    return {
        "energy_eV": energy,
        "max_force_eV_per_angstrom": max_force,
        "converged": max_force <= fmax,
        "n_steps": opt.get_number_of_steps(),
        "calculator": calculator,
    }


# ── Hypothesis executor ────────────────────────────────────────────────────


def run_hypothesis(hypothesis: Hypothesis, calculator: str = "mace",
                   save_results: bool = True) -> Hypothesis:
    """Execute a hypothesis: build system → MLIP optimize → analyze.

    The results are saved to the knowledge base.
    """
    hypothesis.status = "running"
    atoms = build_system(**hypothesis.experiment)
    result = fast_optimize(atoms, calculator=calculator)

    # Analyze: compute adsorption energy if applicable
    e_ads = result["energy_eV"]
    if hypothesis.experiment.get("adsorbate"):
        # Need reference energies
        ref_atoms = build_system(hypothesis.experiment["element"])
        ref_result = fast_optimize(ref_atoms, calculator=calculator)
        e_slab = ref_result["energy_eV"]

        # Free molecule reference
        co = molecule(hypothesis.experiment.get("adsorbate", "CO"))
        co.set_cell([15, 15, 15])
        co.center()
        co_result = fast_optimize(co, calculator=calculator)
        e_free = co_result["energy_eV"]

        result["adsorption_energy"] = e_ads - e_slab - e_free
        result["reference_energies"] = {"slab": e_slab, "free": e_free}

    hypothesis.results.append(result)
    hypothesis.status = "completed"

    # Save to knowledge base
    if save_results:
        record = make_record(
            system=hypothesis.system,
            engine=calculator,
            calculation=result,
            result={"adsorption_energy": result.get("adsorption_energy", None)},
            tags=hypothesis.tags + ["rsi", "hypothesis"],
        )
        save_record(record)

    return hypothesis


# ── Hypothesis comparison ──────────────────────────────────────────────────


def compare_hypotheses(system: str, property: str = "adsorption_energy") -> Dict:
    """Compare results from different calculators/parameter sets for a system."""
    records = []
    sys_dir = KNOWLEDGE_ROOT / system
    if sys_dir.is_dir():
        for f in sorted(sys_dir.glob("*.yaml")):
            try:
                records.append(yaml.safe_load(f.read_text()))
            except Exception:
                pass

    # Extract relevant values
    values = {}
    for r in records:
        engine = r.get("calculation", {}).get("engine", "?")
        val = r.get("result", {}).get(property, None)
        if val is not None:
            values[engine] = val

    return {
        "system": system,
        "property": property,
        "values": values,
        "n_records": len(records),
    }


# ── Example: testing d-band center vs adsorption energy correlation ─────────


def test_dband_correlation(elements: List[str], adsorbate: str = "CO",
                           calculator: str = "mace") -> List[Dict]:
    """Test the d-band center theory: vary the metal surface and compute
    d-band center vs adsorption energy correlation.

    This is a classic catalytic hypothesis: Nørskov's d-band model predicts
    that a higher d-band center (closer to E_F) leads to stronger adsorption.
    """
    results = []
    for element in elements:
        print(f"  Testing {element}...")
        atoms = build_system(element, layers=3, size=(2, 2))
        atoms.calc = get_mace_calculator()
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.05, steps=100)
        e_slab = atoms.get_potential_energy()

        # Adsorption
        ads = build_system(element, layers=3, size=(2, 2), adsorbate=adsorbate)
        # Fix bottom layer
        z = ads.get_positions()[:, 2]
        zmin, zmax = z.min(), z.max()
        fix_idx = [i for i, zi in enumerate(z) if zi <= zmin + (zmax - zmin) * 0.25]
        ads.set_constraint(FixAtoms(indices=fix_idx))
        ads.calc = get_mace_calculator()
        opt2 = BFGS(ads, logfile=None)
        opt2.run(fmax=0.05, steps=100)
        e_ads = ads.get_potential_energy()

        # Free molecule
        co = molecule(adsorbate)
        co.set_cell([15, 15, 15])
        co.center()
        co.calc = get_mace_calculator()
        opt3 = BFGS(co, logfile=None)
        opt3.run(fmax=0.05, steps=50)
        e_free = co.get_potential_energy()

        e_ad = e_ads - e_slab - e_free
        results.append({
            "element": element,
            "e_ad": e_ad,
            "e_slab": e_slab,
            "e_ads": e_ads,
            "e_free": e_free,
        })
        print(f"    E_ad = {e_ad:.3f} eV")

    return results