"""Publication-quality structure figures: 4-panel composite (top/side/side/3D).
Smaller atom radii for clear rendering.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from ase.io import read

# Smaller radii for clear rendering (fit within 40% of nearest-neighbor distance)
RADIUS = {
    "Pt": 0.42, "Fe": 0.35, "Co": 0.35, "Ni": 0.33, "Cu": 0.38,
    "C": 0.28, "O": 0.26, "H": 0.16, "N": 0.28,
}
CPK = {
    "Pt": (0.62, 0.62, 0.67), "Fe": (0.82, 0.56, 0.56), "Co": (0.36, 0.56, 0.82),
    "Ni": (0.56, 0.76, 0.56), "Cu": (0.78, 0.56, 0.36), "C": (0.36, 0.36, 0.36),
    "O": (0.95, 0.16, 0.16), "H": (0.92, 0.92, 0.92), "N": (0.26, 0.26, 0.76),
}
# Atom scale for scatter: s = (r*scale)^2
SCALE = 180.0


def _draw_2d(ax, view, pos, syms, cell):
    """Draw 2D projection for a view."""
    if view == "top":
        proj = pos[:, :2]
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    elif view == "side_x":
        proj = np.stack([pos[:, 2], pos[:, 1]], axis=1)
        ax.set_xlabel("z (Å)"); ax.set_ylabel("y (Å)")
    else:
        proj = np.stack([pos[:, 0], pos[:, 2]], axis=1)
        ax.set_xlabel("x (Å)"); ax.set_ylabel("z (Å)")
    for i, s in enumerate(syms):
        c = CPK.get(s, (0.6, 0.6, 0.6)); r = RADIUS.get(s, 0.30)
        ax.scatter(proj[i, 0], proj[i, 1], s=(r*SCALE)**2, color=c,
                   edgecolors="k", linewidths=0.2, zorder=3, alpha=0.92)
    cov = {s: RADIUS.get(s, 0.3) for s in set(syms)}
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            d = np.linalg.norm(pos[i]-pos[j])
            if d < (cov[syms[i]]+cov[syms[j]])*1.45+0.15:
                ax.plot([proj[i,0],proj[j,0]],[proj[i,1],proj[j,1]],
                        color="0.6", lw=0.7, zorder=1, alpha=0.7)
    # cell projection
    if cell is not None:
        corners = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],
                            [1,0,1],[1,1,1],[0,1,1]]) @ np.array(cell)
        for i,j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            if view == "top":
                ax.plot([corners[i][0],corners[j][0]],[corners[i][1],corners[j][1]],
                        color="k", lw=0.6, alpha=0.4)
            elif view == "side_x":
                ax.plot([corners[i][2],corners[j][2]],[corners[i][1],corners[j][1]],
                        color="k", lw=0.6, alpha=0.4)
            else:
                ax.plot([corners[i][0],corners[j][0]],[corners[i][2],corners[j][2]],
                        color="k", lw=0.6, alpha=0.4)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)


def _draw_3d(ax, pos, syms, cell):
    for i, s in enumerate(syms):
        c = CPK.get(s, (0.6,0.6,0.6)); r = RADIUS.get(s, 0.30)
        ax.scatter(pos[i,0], pos[i,1], pos[i,2], s=(r*SCALE)**2, color=c,
                   edgecolors="k", linewidths=0.2, depthshade=True, zorder=3)
    cov = {s: RADIUS.get(s, 0.3) for s in set(syms)}
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            d = np.linalg.norm(pos[i]-pos[j])
            if d < (cov[syms[i]]+cov[syms[j]])*1.45+0.15:
                ax.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]],[pos[i,2],pos[j,2]],
                        color="0.6", lw=0.7, alpha=0.7)
    if cell is not None:
        corners = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],
                            [1,0,1],[1,1,1],[0,1,1]]) @ np.array(cell)
        for i,j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot([corners[i][0],corners[j][0]],[corners[i][1],corners[j][1]],
                    [corners[i][2],corners[j][2]], color="k", lw=0.5, alpha=0.35)
    ax.view_init(elev=25, azim=-60)
    ax.set_axis_off()


def render_composite(path: str, output: str, title: str = "", dpi: int = 200) -> str:
    """Render 4-panel composite: 3D / top / side_x / side_y."""
    atoms = read(path)
    pos = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    syms = atoms.get_chemical_symbols()
    cell = atoms.get_cell()

    fig = plt.figure(figsize=(13, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_3d(ax3d, pos, syms, cell)
    ax3d.set_title("3D perspective", fontsize=10)

    ax_top = fig.add_subplot(2, 2, 2)
    _draw_2d(ax_top, "top", pos, syms, cell)
    ax_top.set_title("Top view", fontsize=10)

    ax_sx = fig.add_subplot(2, 2, 3)
    _draw_2d(ax_sx, "side_x", pos, syms, cell)
    ax_sx.set_title("Side view (x)", fontsize=10)

    ax_sy = fig.add_subplot(2, 2, 4)
    _draw_2d(ax_sy, "side_y", pos, syms, cell)
    ax_sy.set_title("Side view (y)", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def render_surface_and_adsorbates(surface_path: str, adsorbate_paths: dict,
                                   outdir: str, title: str = "") -> dict:
    """Render surface + adsorbates as 4-panel composites."""
    import os
    os.makedirs(outdir, exist_ok=True)
    figs = {}
    figs["surface"] = render_composite(surface_path, f"{outdir}/fig_surface_composite.png",
                                        title or "HEA Surface")
    for name, p in adsorbate_paths.items():
        figs[name] = render_composite(p, f"{outdir}/fig_{name}_composite.png",
                                       f"{name} adsorption")
    return figs


if __name__ == "__main__":
    import glob, os
    fs = glob.glob("results/<project>/surfaces/surface_0_clean.vasp")
    if fs:
        render_composite(fs[0], "results/<project>/reports/fig_surface_composite.png",
                         "HEA Surface")
        print("surface composite OK")
    for im, title in [("CO", "CO/Pt top"), ("COOH", "COOH/Pt top")]:
        d = glob.glob(f"results/<project>/adsorption/{im}")
        if d and os.path.exists(f"{d[0]}/opt.vasp"):
            render_composite(f"{d[0]}/opt.vasp",
                             f"results/<project>/reports/fig_{im}_composite.png", title)
            print(f"{im} composite OK")