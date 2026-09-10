"""Structure visualization utilities.

Generates images (static PNG via matplotlib, suitable for reports and
chat) from atomistic structure files, plus exports atomic coordinates
in a compact JSON format consumed by the DSH 3D viewer plugin.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.io import read

# CPK-ish colors for common elements (RGB 0-255).
ELEMENT_COLORS = {
    "H": (255, 255, 255), "C": (80, 80, 80), "N": (60, 60, 220),
    "O": (230, 20, 20), "F": (0, 200, 0), "Cl": (0, 255, 0),
    "S": (200, 180, 0), "P": (255, 130, 0), "Pt": (120, 120, 130),
    "Cu": (200, 120, 60), "Ag": (190, 190, 190), "Au": (255, 200, 0),
    "Pd": (140, 140, 140), "Ni": (90, 140, 90), "Fe": (180, 90, 90),
    "Co": (60, 100, 160), "W": (100, 100, 100), "Mo": (90, 120, 160),
    "Ru": (110, 120, 150), "Ir": (130, 130, 150), "Rh": (150, 130, 120),
    "Ti": (150, 150, 150), "Al": (180, 180, 200), "Zn": (160, 160, 140),
    "Mg": (140, 170, 140), "K": (140, 90, 180), "Na": (120, 120, 180),
    "Si": (120, 130, 140), "Ge": (120, 130, 140), "B": (140, 170, 170),
    "Br": (150, 60, 40), "I": (130, 40, 130),
}

# Van der Waals radii (Angstrom) for drawing scale.
VDW_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Cl": 1.75, "S": 1.80, "P": 1.80, "Pt": 1.75, "Cu": 1.40,
    "Ag": 1.72, "Au": 1.66, "Pd": 1.63, "Ni": 1.63, "Fe": 1.80,
    "Co": 1.75, "W": 2.00, "Mo": 1.90, "Ru": 1.78, "Ir": 1.80,
    "Rh": 1.80, "Ti": 1.76, "Al": 1.84, "Zn": 1.39, "Mg": 1.73,
}


def _atom_radius(symbol: str) -> float:
    return VDW_RADII.get(symbol, 1.5)


def read_structure(path: str, index: int = 0) -> Optional[Atoms]:
    """Read an atomistic structure file (POSCAR/CIF/XYZ/...)."""
    try:
        if index is not None and os.path.splitext(path)[1] not in (".traj",):
            try:
                atoms = read(path, index=index)
                return atoms
            except (IndexError, ValueError):
                atoms = read(path)
                return atoms
        atoms = read(path)
        return atoms
    except Exception:
        return None


def structure_to_json(path: str) -> Dict[str, object]:
    """Serialize a structure into compact JSON for the 3D viewer.

    Keys: elements, positions (nested lists), cell (optional), labels.
    """
    atoms = read_structure(path)
    if atoms is None:
        return {"error": f"cannot read structure: {path}"}
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions().tolist()
    cell = atoms.get_cell().tolist() if atoms.get_cell() is not None else None
    radii = [_atom_radius(s) for s in symbols]
    colors = [ELEMENT_COLORS.get(s, (180, 180, 180)) for s in symbols]
    return {
        "path": os.path.abspath(path),
        "natoms": len(symbols),
        "elements": symbols,
        "positions": positions,
        "cell": cell,
        "radii": radii,
        "colors": colors,
        "formula": atoms.get_chemical_formula(),
    }


def plot_structure_2d(path: str, output: str,
                      show_cell: bool = True,
                      title: Optional[str] = None) -> Optional[str]:
    """Render a static ball-and-stick 2D projection of a structure.

    Returns the output path on success, else None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    atoms = read_structure(path)
    if atoms is None:
        return None
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()

    for sym, p in zip(symbols, pos):
        color = tuple(c / 255.0 for c in ELEMENT_COLORS.get(sym, (180, 180, 180)))
        ax.scatter(p[0], p[1], p[2], color=color, s=220,
                   edgecolors="black", linewidths=0.4, depthshade=True)

    # draw bonds for pairs closer than sum of covalent radii * 1.2
    cov_r = {s: _atom_radius(s) * 0.75 for s in set(symbols)}
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < (cov_r.get(symbols[i], 1.0) + cov_r.get(symbols[j], 1.0)) * 1.35:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        [pos[i][2], pos[j][2]], color="gray", linewidth=0.6)

    if show_cell and atoms.get_cell() is not None:
        cell = atoms.get_cell()
        origin = np.zeros(3)
        for i in range(3):
            for sign in (-1, 1):
                pass  # keep simple; draw the cell box edges
        corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        box = corners @ np.array(cell)
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax.plot([box[i][0], box[j][0]], [box[i][1], box[j][1]],
                    [box[i][2], box[j][2]], color="black", linewidth=0.8)

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(title or f"{atoms.get_chemical_formula()} — {os.path.basename(path)}")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def make_deliverable(path: str, out_dir: str = "deliverables",
                     base_name: Optional[str] = None) -> Dict[str, str]:
    """Produce a ready-to-deliver package for a structure file:
    JSON viewer data + static PNG + copies of the original file.

    Returns a dict of produced artifact paths.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = base_name or os.path.splitext(os.path.basename(path))[0]
    json_path = os.path.join(out_dir, f"{base}.struct.json")
    png_path = os.path.join(out_dir, f"{base}.png")
    data = structure_to_json(path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    png = plot_structure_2d(path, png_path)
    return {
        "json": json_path,
        "png": png or "",
        "source": os.path.abspath(path),
        "formula": data.get("formula", ""),
        "natoms": data.get("natoms", 0),
    }