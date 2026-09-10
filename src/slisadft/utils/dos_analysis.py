"""DOS / PDOS analysis and d-band center calculation.

Supports both VASP DOSCAR parsing (via ``utils.vasp_input.parse_doscar``)
and generic data (energies + projected DOS as numpy arrays).

The d-band center is the first moment of the projected d-DOS:

    eps_d = int(E * n_d(E) dE) / int(n_d(E) dE)
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Optional, Tuple


def d_band_center(energies: np.ndarray, dos: np.ndarray,
                  e_fermi: float = 0.0, e_min: float = -10.0,
                  e_max: float = 12.0) -> float:
    """Compute the d-band center (eV) relative to the Fermi level.

    Args:
        energies: energy grid (eV), arbitrary origin.
        dos: projected d-DOS values (numpy array, same length).
        e_fermi: Fermi energy in the same energy units (eV).
        e_min/e_max: integration window relative to e_fermi (eV).
    """
    energies = np.asarray(energies, dtype=float)
    dos = np.asarray(dos, dtype=float)
    mask = (energies >= e_fermi + e_min) & (energies <= e_fermi + e_max)
    e = energies[mask] - e_fermi
    d = dos[mask]
    denom = np.trapezoid(d, e)
    if abs(denom) < 1e-12:
        return float("nan")
    num = np.trapezoid(e * d, e)
    return float(num / denom)


def d_band_width(energies: np.ndarray, dos: np.ndarray,
                 e_fermi: float = 0.0, e_min: float = -10.0,
                 e_max: float = 12.0) -> float:
    """Second central moment (width) of the d-band, in eV."""
    center = d_band_center(energies, dos, e_fermi, e_min, e_max)
    energies = np.asarray(energies, dtype=float)
    dos = np.asarray(dos, dtype=float)
    mask = (energies >= e_fermi + e_min) & (energies <= e_fermi + e_max)
    e = energies[mask] - e_fermi
    d = dos[mask]
    denom = np.trapezoid(d, e)
    if abs(denom) < 1e-12:
        return float("nan")
    num = np.trapezoid((e - center) ** 2 * d, e)
    return float(np.sqrt(max(num / denom, 0.0)))


def d_band_filling(energies: np.ndarray, dos: np.ndarray,
                   e_fermi: float = 0.0, e_min: float = -10.0,
                   e_max: float = 12.0) -> float:
    """Fraction of occupied d states: integral of d-DOS below E_F over
    the window."""
    energies = np.asarray(energies, dtype=float)
    dos = np.asarray(dos, dtype=float)
    mask = (energies >= e_fermi + e_min) & (energies <= e_fermi + e_max)
    e = energies[mask]
    d = dos[mask]
    total = np.trapezoid(d, e)
    occ = np.trapezoid(d[e <= e_fermi], e[e <= e_fermi])
    if abs(total) < 1e-12:
        return float("nan")
    return float(occ / total)


def integrate_dos(energies: np.ndarray, dos: np.ndarray,
                  e_fermi: float = 0.0, e_min: float = -10.0,
                  e_max: float = 12.0) -> float:
    """Integral of DOS over a window (eV -> states)."""
    energies = np.asarray(energies, dtype=float)
    dos = np.asarray(dos, dtype=float)
    mask = (energies >= e_fermi + e_min) & (energies <= e_fermi + e_max)
    return float(np.trapezoid(dos[mask], energies[mask]))


def accumulate_pdos(pdos_list: List[Dict[str, object]],
                    atom_indices: Optional[List[int]] = None,
                    columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sum partial DOS over selected atoms and angular-momentum channels.

    Args:
        pdos_list: parsed PDOS blocks from parse_doscar().
        atom_indices: 1-based atom indices to sum (None = all).
        columns: column names in each block (e.g. ['dxy','dyz','dz2','dxz','dx2']
                 for the d-band of a transition metal).
    Returns:
        (energies, summed_dos) arrays.
    """
    if not pdos_list:
        return np.array([]), np.array([])
    if columns is None:
        columns = ["dtot"]
    if atom_indices is None:
        atom_indices = [p["atom"] for p in pdos_list]
    selected = [p for p in pdos_list if p["atom"] in atom_indices]
    if not selected:
        return np.array([]), np.array([])
    # energies come from the total DOS, take first block's energies if present
    energies = None
    total = None
    for p in selected:
        acc = None
        for col in columns:
            if col in p:
                arr = np.asarray(p[col], dtype=float)
                acc = arr if acc is None else acc + arr
        if acc is None:
            continue
        # spin-down channels contribute equally
        for col in columns:
            down_key = f"{col}_down"
            if down_key in p:
                acc = acc + np.asarray(p[down_key], dtype=float)
        if total is None:
            total = acc
            energies = np.arange(len(acc))
        else:
            total = total + acc
    if total is None:
        return np.array([]), np.array([])
    return energies.astype(float), total


def plot_dos(energies: np.ndarray, dos_up: np.ndarray,
             dos_down: Optional[np.ndarray] = None,
             pdos_series: Optional[Dict[str, np.ndarray]] = None,
             e_fermi: float = 0.0, title: str = "DOS",
             output: str = "dos.png",
             d_band: Optional[Tuple[float, float]] = None) -> str:
    """Plot total DOS (+ PDOS + d-band center marker) and save to file.

    Returns the output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    e = np.asarray(energies) - e_fermi
    ax.plot(e, np.asarray(dos_up), color="tab:blue", label="Total DOS (up)")
    if dos_down is not None:
        ax.plot(e, -np.asarray(dos_down), color="tab:orange",
                label="Total DOS (dn)")
    if pdos_series:
        for name, arr in pdos_series.items():
            ax.plot(e, np.asarray(arr), label=name, linewidth=1.2)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0,
               label=f"E_F = {e_fermi:.2f} eV")
    if d_band is not None:
        center, width = d_band
        ax.axvline(center, color="red", linestyle=":", linewidth=1.5,
                   label=f"d-center = {center:.2f} eV")
    ax.set_xlabel("E - E_F (eV)")
    ax.set_ylabel("DOS (states/eV)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(-12, 8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def analyze_dband(doscar_dir: str, atom_indices: Optional[List[int]] = None,
                  d_columns: Optional[List[str]] = None,
                  e_fermi: Optional[float] = None,
                  output_plot: Optional[str] = None) -> Dict[str, object]:
    """High-level d-band analysis on a VASP directory.

    Args:
        doscar_dir: directory containing DOSCAR (and optional OUTCAR for E_F).
        atom_indices: 1-based atom indices of the metal atoms.
        d_columns: DOSCAR column names making up the d-channel
                   (default: ['dxy','dyz','dz2','dxz','dx2']).
        e_fermi: explicit Fermi energy (eV). If None, read from OUTCAR.
        output_plot: optional PNG path for the DOS plot.
    Returns:
        dict with keys: e_fermi, d_center, d_width, d_filling,
        n_d_states, plot (path or None).
    """
    from . import vasp_input as vi

    data = vi.parse_doscar(doscar_dir)
    if not data:
        return {"error": f"no DOSCAR found in {doscar_dir}"}

    if e_fermi is None:
        ef = vi.parse_fermi_energy(doscar_dir) or data.get("efermi", 0.0)
        # fall back to the clearly-labeled E-F from DOSCAR header
        ef = ef or data.get("efermi", 0.0)
    else:
        ef = float(e_fermi)

    energies = np.asarray(data["energies"], dtype=float)
    dos_up = np.asarray(data["dos_up"], dtype=float)
    dos_down = np.asarray(data["dos_down"], dtype=float)

    pdos = data.get("pdos", [])
    cols = d_columns or ["dxy", "dyz", "dz2", "dxz", "dx2"]
    _, d_dos = accumulate_pdos(pdos, atom_indices, cols)

    result: Dict[str, object] = {"e_fermi": ef}
    if len(d_dos):
        center = d_band_center(energies, d_dos, ef)
        width = d_band_width(energies, d_dos, ef)
        filling = d_band_filling(energies, d_dos, ef)
        nd = integrate_dos(energies, d_dos, ef)
        result.update({
            "d_center": center,
            "d_width": width,
            "d_filling": filling,
            "n_d_states": nd,
        })
    if output_plot:
        result["plot"] = plot_dos(
            energies, dos_up, dos_down,
            pdos_series={"d-PDOS": d_dos} if len(d_dos) else None,
            e_fermi=ef,
            title=f"DOS of {os.path.basename(doscar_dir)}",
            output=output_plot,
            d_band=(result["d_center"], result["d_width"]) if "d_center" in result else None,
        )
    return result