"""MLIP (machine-learning interatomic potential) optimization tools.

Provides CrewAI tools that optimize atomistic structures using either
MACE or DPA-4 (DeepMD-kit).  Both are registered with ASE as calculators
so the rest of the pipeline stays calculator-agnostic.

Environment notes:
  * MACE  -> pip package `mace-torch`; models downloaded on first use.
  * DPA-4 -> pip package `deepmd-kit` (or `deepmd-kit-core` for CPU);
             the DPA-4 model file is referenced by DPA4_MODEL env var or
             the `model_path` argument.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# PyTorch >=2.6 defaults to weights_only=True for torch.load; the MACE and
# e3nn checkpoints are trusted and rely on full unpickling.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS

from ..utils.crewai_compat import tool

from ..utils.slurm import SlurmConfig
from ..engines import is_mock, synthetic_result


def _default_model_dir() -> Path:
    """Directory used to cache downloaded MLIP model files."""
    p = Path(os.environ.get("MLIP_MODEL_DIR", "models/mlip"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _discover_local_mace_model() -> Optional[str]:
    """Find a MACE model already downloaded on disk.

    Checks (in order): explicit MACE_MODEL env var, common cache locations.
    """
    explicit = os.environ.get("MACE_MODEL", "")
    if explicit and Path(explicit).is_file():
        return explicit
    candidates = [
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "mace",
        Path.home() / ".cache" / "mace",
        Path(".cache") / "mace",
    ]
    for d in candidates:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("2023-12-03-mace-128-L1_epoch-*.model")):
            return str(f)
        for f in sorted(d.glob("MACE-MP-0*.model")):
            if f.stat().st_size > 1024 * 1024:  # ignore error stubs
                return str(f)
    return None


def _get_mace_calculator(model_path: Optional[str] = None,
                         device: str = "cpu", default_dtype: str = "float64"):
    """Build a MACE ASE calculator.

    MACE uses PyTorch; on CPU we default to float64 for accuracy.
    When ``model_path`` is empty, a locally-downloaded MACE-MP-0 model
    is used if available, otherwise the `medium` MACE-MP-0 checkpoint is
    downloaded on first use.
    """
    from mace.calculators.foundations_models import mace_mp
    path = model_path or _discover_local_mace_model()
    model_arg = path if path else "medium"
    calc = mace_mp(
        model=model_arg,
        device=device,
        default_dtype=default_dtype,
        return_raw_model=False,
    )
    return calc


def _get_dpa_calculator(model_path: Optional[str] = None):
    """Build a DPA-4 (DeepMD-kit) ASE calculator.

    Requires a DPA model file (e.g. dpa-4.pth / DPA4-*.pt).  The model
    path can be given explicitly, via the DPA4_MODEL environment
    variable, or auto-discovered under the default model directory.
    If none is found an informative error is raised.
    """
    import deepmd
    from deepmd.calculator import DP

    path = model_path or os.environ.get("DPA4_MODEL", "") or os.environ.get("DPA_MODEL", "")
    if not path:
        # Auto-discovery under models/mlip — prefer the DPA4-* series
        # (newer) over the legacy dpa-4 name, and skip non-zip stubs.
        for pattern in ("DPA4-*.pt", "DPA4-*.pth", "dpa-4.pth", "dpa4*.pth"):
            matches = sorted(Path("models/mlip").glob(pattern))
            for m in matches:
                if m.stat().st_size < 1024 * 1024:  # ignore error stubs
                    continue
                path = str(m)
                break
            if path:
                break
    if not path:
        raise RuntimeError(
            "DPA-4 model file not specified. Pass model_path or set the "
            "DPA4_MODEL environment variable to the DPA model file (e.g. "
            "dpa-4.pth), or place a DPA4-*.pt file under models/mlip/."
        )
    if not Path(path).exists():
        raise FileNotFoundError(f"DPA-4 model file not found: {path}")
    return DP(model=str(path))


def _optimize(atoms: Atoms, fmax: float, steps: int,
              output_file: str, calculator_name: str) -> dict:
    """Shared optimization driver: relax, save, report energy and forces."""
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    write(output_file, atoms, format="vasp")
    energy = atoms.get_potential_energy()
    fmax_achieved = _max_force(atoms)
    return {
        "calculator": calculator_name,
        "output_file": os.path.abspath(output_file),
        "energy_eV": float(energy),
        "max_force_eV_per_angstrom": float(fmax_achieved),
        "n_atoms": len(atoms),
        "converged": bool(fmax_achieved is not None and fmax_achieved <= fmax),
    }


def _max_force(atoms: Atoms) -> float:
    forces = atoms.get_forces()
    import numpy as np
    return float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0


@tool
def mace_optimize_tool(input_file: str, fmax: float = 0.05, steps: int = 200,
                       model_path: str = "", device: str = "cpu",
                       output_file: str = "", keep_slabs_fixed: bool = True) -> str:
    """Optimize an atomistic structure with the MACE machine-learning potential.

    Args:
        input_file: path to the input structure (POSCAR/CIF/XYZ/...).
        fmax: force convergence criterion (eV/Angstrom).
        steps: maximum number of relaxation steps.
        model_path: optional path to a MACE model (.model / .pt). When empty,
                    the MACE-MP-0 middle model (`medium`) is used.
        device: 'cpu' (default) or 'cuda'.
        output_file: optional output path (defaults to <input>_mace_opt.vasp).
        keep_slabs_fixed: if True, the z positions (or lower layers) of the
                          slab are held fixed where sensible; implemented as a
                          simple heuristic: fix atoms whose z is below the
                          geometric middle of the cell for slab-like systems.

    Returns:
        A JSON string describing the optimized structure path, final energy,
        maximum force and convergence status.
    """
    import json

    atoms = read(input_file)
    if keep_slabs_fixed and _looks_like_slab(atoms):
        _fix_slab_bottom(atoms)

    out = output_file or str(Path(input_file).with_suffix("")) + "_mace_opt.vasp"

    if is_mock():
        from ..engines.mock import get_mock_calculator
        atoms.calc = get_mock_calculator(atoms)
        result = _optimize(atoms, fmax, steps, out, "MACE-mock(EMT/LJ)")
        return json.dumps(synthetic_result(result), ensure_ascii=False)

    calc = _get_mace_calculator(model_path if model_path else None, device=device)
    atoms.calc = calc

    result = _optimize(atoms, fmax, steps, out, f"MACE{f' ({model_path})' if model_path else ''}")
    return json.dumps(result, ensure_ascii=False)


@tool
def dpa_optimize_tool(input_file: str, fmax: float = 0.05, steps: int = 200,
                      model_path: str = "", output_file: str = "",
                      keep_slabs_fixed: bool = True) -> str:
    """Optimize an atomistic structure with the DPA-4 (DeepMD-kit) potential.

    Args:
        input_file: path to the input structure (POSCAR/CIF/XYZ/...).
        fmax: force convergence criterion (eV/Angstrom).
        steps: maximum number of relaxation steps.
        model_path: path to the DPA-4 model file (e.g. dpa-4.pth). When empty,
                    the DPA4_MODEL environment variable is used.
        output_file: optional output path (defaults to <input>_dpa_opt.vasp).
        keep_slabs_fixed: fix the bottom layers of slab-like structures.

    Returns:
        A JSON string with the optimized structure path, final energy,
        maximum force and convergence status.
    """
    import json

    atoms = read(input_file)
    if keep_slabs_fixed and _looks_like_slab(atoms):
        _fix_slab_bottom(atoms)

    out = output_file or str(Path(input_file).with_suffix("")) + "_dpa_opt.vasp"

    if is_mock():
        from ..engines.mock import get_mock_calculator
        atoms.calc = get_mock_calculator(atoms)
        result = _optimize(atoms, fmax, steps, out, "DPA-4-mock(EMT/LJ)")
        return json.dumps(synthetic_result(result), ensure_ascii=False)

    calc = _get_dpa_calculator(model_path if model_path else None)
    atoms.calc = calc

    result = _optimize(atoms, fmax, steps, out, f"DPA-4{f' ({model_path})' if model_path else ''}")
    return json.dumps(result, ensure_ascii=False)


@tool
def mlip_single_point_tool(input_file: str, calculator: str = "mace",
                           model_path: str = "", device: str = "cpu") -> str:
    """Compute a single-point energy (and forces) with MACE or DPA-4.

    Args:
        input_file: structure file path.
        calculator: 'mace' or 'dpa'.
        model_path: optional model path (MACE model or DPA model file).
        device: 'cpu' or 'cuda' (MACE only).

    Returns:
        JSON with energy, forces, and structure summary.
    """
    import json

    atoms = read(input_file)

    if is_mock():
        from ..engines.mock import get_mock_calculator
        calc = get_mock_calculator(atoms)
        calc_name = f"{calculator}-mock(EMT/LJ)"
        synthetic = True
    elif calculator.lower() in ("mace", "mace-mp"):
        calc = _get_mace_calculator(model_path if model_path else None, device=device)
        calc_name = calculator
        synthetic = False
    elif calculator.lower() in ("dpa", "dpa4", "deepmd"):
        calc = _get_dpa_calculator(model_path if model_path else None)
        calc_name = calculator
        synthetic = False
    else:
        return json.dumps({"error": f"unknown calculator {calculator}"})

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    import numpy as np
    result = {
        "calculator": calc_name,
        "energy_eV": float(energy),
        "max_force_eV_per_angstrom": float(np.max(np.linalg.norm(forces, axis=1))),
        "n_atoms": len(atoms),
        "input_file": os.path.abspath(input_file),
    }
    if synthetic:
        result = synthetic_result(result)
    return json.dumps(result, ensure_ascii=False)


def _looks_like_slab(atoms: Atoms) -> bool:
    """Heuristic: a slab has a vacuum gap (max z spacing >> typical bond)."""
    import numpy as np
    if len(atoms) < 4:
        return False
    z = atoms.get_positions()[:, 2]
    if z.max() - z.min() < 3.0:
        return False
    return True


def _fix_slab_bottom(atoms: Atoms, fraction: float = 0.35):
    """Freeze the bottom `fraction` of the slab layers (by z)."""
    import numpy as np
    z = atoms.get_positions()[:, 2]
    zmin = z.min()
    zmax = z.max()
    threshold = zmin + (zmax - zmin) * fraction
    mask = z <= threshold
    atoms.set_constraint(
        [a for a in atoms.constraints]  # keep existing constraints
        if atoms.constraints else []
    )
    from ase.constraints import FixAtoms
    n = len(atoms)
    fix = FixAtoms(indices=[i for i in range(n) if mask[i]])
    atoms.set_constraint(fix)