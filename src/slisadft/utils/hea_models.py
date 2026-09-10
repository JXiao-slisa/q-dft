"""High-entropy alloy (HEA) surface generator.

Builds fcc(111) slab models with multiple metal elements distributed
randomly (or with controlled ordering) across lattice sites.

Supports arbitrary fcc-based HEA compositions (pass ``elements``).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.io import write
from typing import Dict, List, Optional, Tuple

# fcc lattice constants (Å) for common elements
LATTICE = {
    "Pt": 3.924, "Fe": 3.545, "Co": 3.545, "Ni": 3.524,
    "Cu": 3.615, "Pd": 3.890, "Au": 4.078, "Ag": 4.085,
    "Al": 4.050, "Ru": 3.806, "Ir": 3.839, "Rh": 3.804,
    "Mn": 3.080, "Cr": 3.595, "V": 3.040, "Zn": 2.665,
    "Ti": 2.951, "Mg": 3.209,
}

# Default HEA elements
# Neutral example composition; pass your own ``elements`` list.
HEA_ELEMENTS = ["Cu", "Ni", "Ag"]


def hea_lattice_constant(elements: List[str]) -> float:
    """Vegard's law: average lattice constant weighted by composition."""
    vals = [LATTICE.get(e, 3.9) for e in elements]
    return float(np.mean(vals))


def build_hea_slab(
    elements: Optional[List[str]] = None,
    layers: int = 3,
    size: Tuple[int, int] = (2, 2),
    vacuum: float = 10.0,
    seed: int = 0,
    a: Optional[float] = None,
) -> Atoms:
    """Build an fcc(111) HEA slab with random element distribution.

    Args:
        elements: list of element symbols (default: HEA_ELEMENTS example).
        layers: number of atomic layers.
        size: surface supercell (x, y).
        vacuum: vacuum thickness (Å).
        seed: random seed for element distribution.
        a: lattice constant; if None, use Vegard average.

    Returns:
        ASE Atoms with elements randomly distributed.
    """
    elements = elements or HEA_ELEMENTS
    a = a or hea_lattice_constant(elements)

    # Build pure Pt scaffold then substitute elements
    slab = fcc111("Pt", size=(size[0], size[1], layers), a=a, vacuum=vacuum)

    rng = np.random.default_rng(seed)
    symbols = list(slab.get_chemical_symbols())
    # Replace each Pt with a random HEA element
    new_symbols = [elements[i % len(elements)] for i in range(len(symbols))]
    # Random reshuffle with controlled composition (near-equal)
    rng.shuffle(new_symbols)
    slab.set_chemical_symbols(new_symbols)
    return slab


def build_ordered_hea_slab(
    elements: List[str] = HEA_ELEMENTS,
    layers: int = 3,
    size: Tuple[int, int] = (2, 2),
    vacuum: float = 10.0,
    seed: int = 0,
) -> Atoms:
    """Build an HEA slab with layer-resolved ordering:
       - bottom layer = one element (stabilising),
       - surface layers = mixed HEA.

    This mimics realistic HEA surfaces where the top layers carry the
    multi-element character.
    """
    slab = build_hea_slab(elements, layers, size, vacuum, seed)
    # Identify layers by z-coordinate
    z = slab.get_positions()[:, 2]
    zmin, zmax = z.min(), z.max()
    n_per_layer = len(slab) // layers

    symbols = list(slab.get_chemical_symbols())
    zsorted = np.argsort(z)
    # Bottom layer: use first element (e.g., Pt as stabiliser)
    for i in zsorted[:n_per_layer]:
        symbols[i] = elements[0]

    # Optional: enrich the top layer with a chosen element
    top_layer_idx = zsorted[-n_per_layer:]
    rng = np.random.default_rng(seed + 999)
    n_cu = max(1, n_per_layer // 3)
    cu_idx = rng.choice(top_layer_idx, n_cu, replace=False)
    for i in cu_idx:
        symbols[i] = "Cu"

    slab.set_chemical_symbols(symbols)
    return slab


def make_adsorption_models(slab: Atoms, adsorbates: List[str],
                           fmax_fixed_layers: int = 1) -> List[Atoms]:
    """Generate adsorption models for multiple adsorbates on an HEA slab.

    Adsorbates are placed at the hollow (fcc) site near the surface center.
    Returns a list of Atoms objects.
    """
    from ase.build import add_adsorbate, molecule

    models = []
    z_max = max(slab.get_positions()[:, 2])
    top_layer = [i for i, p in enumerate(slab.get_positions())
                 if abs(p[2] - z_max) < 0.5]
    center = np.mean([slab.get_positions()[i][:2] for i in top_layer], axis=0)

    for ads in adsorbates:
        try:
            mol = molecule(ads)
        except Exception:
            continue
        # Place at fcc-type hollow (offset from center slightly)
        height = {"H": 1.0, "O": 1.2, "N": 1.2, "C": 1.3, "CO": 1.8,
                  "OH": 1.5, "COOH": 1.9, "CHO": 1.9, "CHOH": 1.9,
                  "CH2O": 2.0, "OCH3": 2.0, "CH3OH": 2.2, "CO2": 2.2,
                  "H2O": 2.0}.get(ads, 1.8)
        model = slab.copy()
        add_adsorbate(model, mol, height=height, position=center)
        models.append(model)
    return models


def save_hea(path: str, atoms: Atoms) -> str:
    """Save an HEA structure (POSCAR format)."""
    write(path, atoms, format="vasp")
    return path


if __name__ == "__main__":
    # Quick demo
    for seed in range(3):
        slab = build_hea_slab(seed=seed)
        print(f"HEA slab seed={seed}: {slab.get_chemical_formula()}, {len(slab)} atoms")
        save_hea(f"hea_seed{seed}.vasp", slab)