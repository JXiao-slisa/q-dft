"""Surface, adsorption, and molecular model building tools.

Uses ASE to construct common DFT initial structures:
  * Metal slabs (fcc / bcc / hcp, arbitrary Miller indices where ASE supports)
  * Adsorption systems (slab + adsorbate molecule)
  * Isolated molecules (gas-phase reference)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ase import Atoms
from ase.build import fcc111, bcc111, hcp0001, add_adsorbate, molecule as ase_molecule
from ase.io import write, read

from ..utils.crewai_compat import tool


# Cache of common lattice constants (Angstrom).
LATTICE_CONSTANTS = {
    "Pt": 3.924, "Cu": 3.615, "Ag": 4.085, "Au": 4.078,
    "Pd": 3.890, "Ni": 3.524, "Co": 3.545, "Fe": 2.866,
    "W": 3.165, "Mo": 3.147, "Ru": 3.806, "Ir": 3.839,
    "Rh": 3.804, "Al": 4.050, "Ti": 2.951, "Zn": 2.665,
    "Mg": 3.209, "V": 3.040, "Cr": 2.880, "Mn": 3.080,
    "Sn": 3.248, "Pb": 4.950, "Si": 5.431,
}

# Typical adsorption heights above the surface (Angstrom).
ADSORBATE_HEIGHT: Dict[str, float] = {
    "H": 1.0, "O": 1.2, "N": 1.2, "C": 1.3, "S": 1.6,
    "CO": 1.8, "NO": 1.8, "OH": 1.5, "H2O": 2.0, "NH3": 2.0,
    "CH4": 2.2, "CO2": 2.2, "O2": 1.5, "N2": 1.8,
}


def _guess_a(element: str) -> Optional[float]:
    """Return the lattice constant for a known element, or None."""
    return LATTICE_CONSTANTS.get(element)


def _build_slab(element: str, miller: Tuple[int, int, int] = (1, 1, 1),
                layers: int = 3, size: Tuple[int, int] = (2, 2),
                vacuum: float = 10.0) -> Atoms:
    """Build a clean metal slab using ASE built-in functions.

    Supports fcc(111), fcc(100), bcc(110), bcc(111), hcp(0001) based
    on the lattice type and Miller index.
    """
    a = _guess_a(element)
    if miller == (1, 1, 1):
        # Try fcc111 first (most common).
        slab = fcc111(element, size=(size[0], size[1], layers), a=a, vacuum=vacuum)
    elif miller == (1, 0, 0):
        from ase.build import fcc100
        slab = fcc100(element, size=(size[0], size[1], layers), a=a, vacuum=vacuum)
    elif miller == (1, 1, 0):
        from ase.build import bcc110
        slab = bcc110(element, size=(size[0], size[1], layers), a=a, vacuum=vacuum)
    elif miller == (0, 0, 0, 1):
        slab = hcp0001(element, size=(size[0], size[1], layers), a=a, vacuum=vacuum)
    else:
        # Generic fallback: use fcc111 as default.
        slab = fcc111(element, size=(size[0], size[1], layers), a=a, vacuum=vacuum)
    return slab


@tool
def build_surface_tool(element: str, miller_index: str = "(111)",
                       layers: int = 3, size: str = "2,2",
                       vacuum: float = 10.0) -> str:
    """Build a clean metal surface model (slab).

    Args:
        element: metal element symbol (e.g. 'Pt', 'Cu', 'Ni').
        miller_index: Miller index, e.g. '(111)', '(100)', '(110)'.
        layers: number of atomic layers.
        size: surface supercell size as 'nx,ny' (e.g. '2,2', '3,3').
        vacuum: vacuum thickness in Angstroms.

    Returns:
        Absolute path to the created POSCAR file.
    """
    # Parse miller index.
    import re
    m = re.findall(r"\d+", miller_index)
    if len(m) == 3:
        miller = tuple(int(x) for x in m)
    elif len(m) == 4:
        miller = tuple(int(x) for x in m)
    else:
        miller = (1, 1, 1)

    size_tuple = tuple(int(x) for x in size.split(","))
    if len(size_tuple) == 1:
        size_tuple = (size_tuple[0], size_tuple[0])

    slab = _build_slab(element, miller, layers, size_tuple, vacuum)
    # Write out.
    safe_miller = miller_index.replace("(", "").replace(")", "").replace(",", "")
    filepath = f"{element}_{safe_miller}_{layers}l.vasp"
    write(filepath, slab, format="vasp")
    return os.path.abspath(filepath)


@tool
def build_adsorption_tool(slab_file: str, adsorbate: str = "CO",
                          site: str = "top", height: Union[float, None] = None,
                          offset: str = "0.0,0.0",
                          output_file: str = "") -> str:
    """Add an adsorbate molecule onto a pre-built slab surface.

    Args:
        slab_file: path to the slab POSCAR file.
        adsorbate: adsorbate formula or ASE molecule name
                   (e.g. 'CO', 'OH', 'H', 'O', 'NH3', 'H2O').
        site: adsorption site type:
              - 'top': above a surface atom
              - 'fcc': fcc hollow site
              - 'hcp': hcp hollow site
              - 'bridge': bridge site
              - 'center': center of the surface cell
              Default 'top'.  The offset parameter can fine-tune the
              lateral position.
        height: vertical distance from the adsorbate to the surface (Å).
                Defaults are looked up from built-in table (1.8 Å for CO).
        offset: lateral offset (x,y) in Angstroms relative to the site.
        output_file: optional output path. Defaults to
                     <slab_basename>_<adsorbate>_<site>.vasp.

    Returns:
        Absolute path to the created adsorption structure file.
    """
    import numpy as np

    slab = read(slab_file)

    # Build the adsorbate molecule.
    try:
        ads = ase_molecule(adsorbate)
    except Exception:
        # For atomic adsorbates, create a single atom.
        try:
            ads = Atoms(adsorbate, positions=[(0, 0, 0)])
        except Exception as exc:
            return f"Error: cannot create adsorbate {adsorbate}: {exc}"

    height = height or ADSORBATE_HEIGHT.get(adsorbate, 1.8)

    # Determine the adsorption site position on the topmost layer.
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    z_max = max(positions[:, 2])
    top_layer_idx = [i for i, p in enumerate(positions) if abs(p[2] - z_max) < 0.5]

    if len(top_layer_idx) == 0:
        return "Error: no surface atoms found."

    top_positions = positions[top_layer_idx]
    top_center = np.mean(top_positions, axis=0)

    # Compute site position.
    if site == "top":
        # Pick the top layer atom closest to the center.
        center_xy = np.array([top_center[0], top_center[1]])
        dists = [np.linalg.norm(positions[i][:2] - center_xy) for i in top_layer_idx]
        anchor = top_layer_idx[np.argmin(dists)]
        xy = positions[anchor][:2]
    elif site == "fcc":
        # FCC hollow: find the centroid of 3 surface atoms forming a triangle.
        # Approximate: use the center of the first 3 top layer atoms.
        tri = top_layer_idx[:3]
        xy = np.mean(positions[tri][:, :2], axis=0)
    elif site == "hcp":
        # Similar to fcc but offset.
        tri = top_layer_idx[:3]
        xy = np.mean(positions[tri][:, :2], axis=0) + np.array([0.3, 0.3])
    elif site == "bridge":
        # Midpoint of two nearest surface atoms.
        if len(top_layer_idx) >= 2:
            idx1, idx2 = top_layer_idx[0], top_layer_idx[1]
            xy = (positions[idx1][:2] + positions[idx2][:2]) / 2.0
        else:
            xy = positions[top_layer_idx[0]][:2]
    elif site == "center":
        xy = top_center[:2]
    else:
        xy = top_center[:2]

    # Apply offset.
    offset_xy = np.array([float(x) for x in offset.split(",")])
    xy += offset_xy

    # Place the adsorbate.
    # For ASE add_adsorbate, we need the adsorbate's lowest atom to be at `height`.
    # The add_adsorbate function places the molecule's bottom at the given height.
    # We modify the molecule's positions so its lowest z is at 0.
    ads_pos = ads.get_positions()
    ads_z_min = min(ads_pos[:, 2])
    ads.translate([xy[0], xy[1], height - ads_z_min])

    # Add adsorbate to slab.
    # Use ASE add_adsorbate for consistency.
    system = slab.copy()
    add_adsorbate(system, ads, height=height, position=xy)

    # Write output.
    if not output_file:
        base = Path(slab_file).stem
        output_file = f"{base}_{adsorbate}_{site}.vasp"
    write(output_file, system, format="vasp")
    return os.path.abspath(output_file)


@tool
def build_molecule_tool(formula: str, output_file: str = "") -> str:
    """Build an isolated molecule for gas-phase reference calculations.

    Uses ASE's built-in molecule database.  For molecules not in the
    database, a simple linear/planar approximate geometry is generated.

    Args:
        formula: chemical formula (e.g. 'CO', 'H2O', 'NH3', 'CH4',
                 'O2', 'N2', 'H2', 'CO2', 'OH').
        output_file: optional output path. Defaults to <formula>.vasp.

    Returns:
        Absolute path to the created molecule structure file.
    """
    try:
        mol = ase_molecule(formula)
    except Exception:
        # Fallback: create a simple diatomic or single atom.
        if len(formula) <= 2:
            # Simple diatomic.
            from ase.atoms import Atoms
            mol = Atoms(formula, positions=[(0, 0, 0), (0, 0, 1.2)])
        else:
            return f"Error: molecule {formula} not in ASE database and no fallback available."

    out = output_file or f"{formula}.vasp"
    # Add a large vacuum box for VASP-format output (molecules need a cell).
    from ase.calculators.calculator import Calculator
    box_size = 15.0
    mol.set_cell([box_size, box_size, box_size])
    mol.center()
    write(out, mol, format="vasp")
    return os.path.abspath(out)


@tool
def build_slab_with_adsorbate_tool(element: str, miller_index: str = "(111)",
                                    layers: int = 3, size: str = "2,2",
                                    vacuum: float = 10.0,
                                    adsorbate: str = "CO",
                                    site: str = "top",
                                    height: Union[float, None] = None) -> str:
    """Convenience tool: build a slab and immediately add an adsorbate.

    This is a two-in-one tool that calls both build_surface_tool and
    build_adsorption_tool in sequence.

    Args:
        element: metal element symbol.
        miller_index: Miller index.
        layers: number of atomic layers.
        size: surface supercell size as 'nx,ny'.
        vacuum: vacuum thickness in Angstroms.
        adsorbate: adsorbate formula.
        site: adsorption site.
        height: adsorption height (defaults to built-in table).

    Returns:
        Absolute path to the adsorption structure file.
    """
    slab_path = build_surface_tool.func(element, miller_index, layers, size, vacuum)
    if slab_path.startswith("Error"):
        return slab_path
    return build_adsorption_tool.func(slab_path, adsorbate, site, height)