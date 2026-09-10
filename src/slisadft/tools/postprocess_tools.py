"""Post-processing and analysis tools for DFT results.

Provides CrewAI tools for:
  * Adsorption energy calculation
  * Gibbs free energy / reaction step diagram
  * DOS / PDOS / d-band center analysis
  * Volcano (free-energy step) plot generation
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.crewai_compat import tool

from ..utils.thermochemistry import (
    adsorption_energy as _ads_energy,
    gibbs_reaction_energy as _gibbs_step,
    volcano_coordinates as _volcano,
    gibbs_free_energy as _gibbs_total,
    EMPIRICAL_GAS_CORRECTIONS,
)
from ..utils.vasp_input import parse_vasp_frequencies
from ..utils.dos_analysis import (
    analyze_dband as _dband,
)


@tool
def adsorption_energy_tool(e_slab_ads: float, e_slab: float,
                           e_adsorbate: float, label: str = "",
                           formula: str = "") -> str:
    """Calculate the adsorption energy of an adsorbate on a surface.

        E_ad = E(slab+ads) - E(slab) - E(ads, free)

    Negative values mean exothermic (stabilized) adsorption.

    Args:
        e_slab_ads: total energy of the slab with adsorbate (eV).
        e_slab: total energy of the clean slab (eV).
        e_adsorbate: total energy of the free adsorbate/molecule (eV).
        label: optional label for the adsorption site (e.g. 'top', 'fcc').
        formula: optional chemical formula of the adsorbate (e.g. 'CO', 'OH').

    Returns:
        A JSON string with the adsorption energy and metadata.
    """
    e_ad = _ads_energy(e_slab_ads, e_slab, e_adsorbate)
    result = {
        "label": label,
        "formula": formula,
        "e_slab_ads_eV": e_slab_ads,
        "e_slab_eV": e_slab,
        "e_adsorbate_eV": e_adsorbate,
        "adsorption_energy_eV": round(e_ad, 4),
        "exothermic": e_ad < 0,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def gibbs_free_energy_tool(electronic_energy: float, formula: str = "",
                            temperature: float = 298.15,
                            zpe_empirical: float = 0.0,
                            s_empirical: float = 0.0) -> str:
    """Compute the Gibbs free energy correction for a molecule or
    adsorbate at a given temperature.

    If the formula is recognized (H2, H2O, CO, CO2, CH4, NH3, O2, N2),
    standard empirical corrections at 298.15 K are used.  Otherwise the
    provided zpe and entropy values are applied.

    Args:
        electronic_energy: raw DFT total energy (eV).
        formula: chemical formula for empirical lookup (optional).
        temperature: temperature in Kelvin (default 298.15).
        zpe_empirical: zero-point energy correction (eV), used when
                       formula is not in the built-in table.
        s_empirical: entropy correction (J/mol/K), used when formula
                     is not in the built-in table.

    Returns:
        JSON with G(T) and the applied corrections.
    """
    from ..utils.thermochemistry import gibbs_free_energy as _gibbs

    g = _gibbs(electronic_energy, T=temperature, formula=formula if formula else None)
    # Provide the corrections used.
    if formula and formula in EMPIRICAL_GAS_CORRECTIONS:
        zpe, s = EMPIRICAL_GAS_CORRECTIONS[formula]
        zpe_used = zpe
        s_used = s
    else:
        zpe_used = zpe_empirical
        s_used = s_empirical

    result = {
        "formula": formula or "",
        "temperature_K": temperature,
        "electronic_energy_eV": electronic_energy,
        "gibbs_free_energy_eV": round(g, 4),
        "zpe_correction_eV": zpe_used,
        "entropy_correction_J_per_mol_K": s_used,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def reaction_step_diagram_tool(reaction_steps: str, temperature: float = 298.15,
                                ph: float = 0.0, u_vs_rhe: float = 0.0,
                                zero_energy: float = 0.0) -> str:
    """Build a Gibbs free-energy step diagram (volcano / reaction profile).

    The reaction steps are provided as a JSON list of step objects, each
    containing:
        label:      short name (e.g. '*CO -> *CHO')
        delta_e:    electronic energy difference of the step (eV)
        n_h:        number of (H+ + e-) transferred (0 for purely chemical)
        zpe_corr:   zero-point energy correction (eV, optional)
        ts_corr:    -T*S entropy correction (eV, optional)

    Args:
        reaction_steps: JSON string of the steps array.
        temperature: temperature in Kelvin.
        ph: pH of the electrolyte.
        u_vs_rhe: applied potential vs RHE.
        zero_energy: reference energy of the initial state (eV).

    Returns:
        JSON with cumulative free energies and the step labels suitable
        for plot generation.
    """
    steps = json.loads(reaction_steps)
    corrected = []
    for s in steps:
        dg = _gibbs_step(
            delta_e=s["delta_e"],
            n_h=s.get("n_h", 0),
            u_vs_rhe=u_vs_rhe,
            ph=ph,
            t=temperature,
            zpe_corr=s.get("zpe_corr", 0.0),
            ts_corr=s.get("ts_corr", 0.0),
        )
        corrected.append((s["label"], dg))

    coords, labels = _volcano(corrected)
    # Shift by zero_energy.
    cumulative = [zero_energy + c for c in coords]

    result = {
        "temperature_K": temperature,
        "pH": ph,
        "U_vs_RHE": u_vs_rhe,
        "steps": [
            {"label": labels[i], "delta_g_eV": round(corrected[i][1], 4)}
            for i in range(len(corrected))
        ],
        "cumulative_energy_eV": [round(c, 4) for c in cumulative],
        "overpotential_V": max(cumulative) - min(cumulative) if len(cumulative) > 1 else 0.0,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def vibrational_thermochemistry_tool(outcar_dir: str, electronic_energy: float,
                                     formula: str = "",
                                     temperature: float = 298.15) -> str:
    """Compute ZPE and Gibbs free energy from a VASP vibrational frequency
    calculation (IBRION=5/6 OUTCAR).

    Frequencies are parsed from the OUTCAR in ``outcar_dir`` and used for
    the harmonic vibrational contribution to G(T).  When the OUTCAR has no
    frequency block, the tool falls back to the empirical correction table
    (if ``formula`` is recognized) and reports ``correction_source:
    "empirical-fallback"`` so the limitation is explicit.

    Args:
        outcar_dir: directory containing the frequency-run OUTCAR.
        electronic_energy: raw DFT electronic energy of the species (eV).
        formula: optional chemical formula for the empirical fallback
                 (e.g. 'CO').
        temperature: temperature in Kelvin (default 298.15).

    Returns:
        JSON with frequencies, ZPE, G(T) and the correction provenance.
    """
    parsed = parse_vasp_frequencies(Path(outcar_dir))
    freqs = parsed.get("frequencies_cm1") or []

    if parsed.get("has_freq_block") and freqs:
        g = _gibbs_total(electronic_energy, T=temperature, nu_cm1=freqs)
        from ..utils.thermochemistry import zpe_from_frequencies
        zpe = zpe_from_frequencies(freqs)
        source = "vibrational"
    else:
        g = _gibbs_total(electronic_energy, T=temperature,
                         formula=formula if formula else None)
        zpe, _ = EMPIRICAL_GAS_CORRECTIONS.get(formula or "", (0.0, 0.0))
        source = "empirical-fallback"

    result = {
        "outcar_dir": outcar_dir,
        "formula": formula or "",
        "temperature_K": temperature,
        "electronic_energy_eV": electronic_energy,
        "frequencies_cm1": freqs,
        "n_imaginary": parsed.get("n_imaginary", 0),
        "zpe_correction_eV": round(zpe, 6),
        "gibbs_free_energy_eV": round(g, 6),
        "correction_source": source,
        "source_file": parsed.get("source", ""),
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def analyze_dos_tool(doscar_dir: str, atom_indices: str = "",
                     d_columns: str = "", output_plot: str = "") -> str:
    """Analyze VASP DOSCAR to compute the d-band center, width, and
    integrated d-states of the selected metal atoms.

    Optionally generates a DOS/PDOS plot.

    Args:
        doscar_dir: directory containing the VASP DOSCAR (and OUTCAR).
        atom_indices: comma-separated 1-based atom indices (e.g. '1,2,3').
                      If empty, all atoms are included.
        d_columns: comma-separated DOSCAR column names for the d-channel
                   (default: 'dxy,dyz,dz2,dxz,dx2').
        output_plot: optional PNG file path for the DOS plot. If empty,
                     no plot is generated.

    Returns:
        JSON with d-band center, width, filling, and (optionally) the
        plot path.
    """
    if atom_indices:
        indices = [int(x.strip()) for x in atom_indices.split(",")]
    else:
        indices = None
    cols = [c.strip() for c in d_columns.split(",")] if d_columns else None
    plot = output_plot or None

    result = _dband(
        doscar_dir=doscar_dir,
        atom_indices=indices,
        d_columns=cols,
        output_plot=plot,
    )
    return json.dumps(result, ensure_ascii=False, default=str)


@tool
def generate_volcano_plot_tool(reaction_data: str, output: str = "volcano.png",
                                title: str = "Free Energy Diagram") -> str:
    """Generate a free-energy step diagram (volcano plot) from reaction
    step data.

    The reaction_data should be the JSON output from
    ``reaction_step_diagram_tool``.

    Args:
        reaction_data: JSON string containing 'steps' (list of label, delta_g)
                       and 'cumulative_energy_eV' (list of cumulative energies).
        output: output PNG file path.
        title: plot title.

    Returns:
        Path to the generated plot image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(reaction_data)
    steps = data.get("steps", [])
    cumulative = data.get("cumulative_energy_eV", [0])
    if not steps:
        return json.dumps({"error": "no reaction steps provided"})

    labels = ["initial"] + [s["label"] for s in steps]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, cumulative, "-o", color="tab:blue", linewidth=2, markersize=8)
    ax.fill_between(x, cumulative, alpha=0.1, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Free Energy (eV)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Annotate each step with its Delta G value.
    for i, (xi, yi) in enumerate(zip(x, cumulative)):
        if i > 0:
            dg = cumulative[i] - cumulative[i - 1]
            ax.annotate(f"{dg:+.2f} eV", (xi, yi),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color="darkred")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output