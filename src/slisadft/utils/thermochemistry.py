"""Thermochemistry helpers for adsorption energy and Gibbs free energy.

Implements the standard computational electrocatalysis / heterogeneous
catalysis corrections:

  * Zero-point energy + enthalpy + entropy corrections for gas-phase
    molecules (using harmonic/translational/rotational partition
    functions, or empirical corrections when frequencies are missing).
  * Adsorption energy:  E_ad = E_slab+ads - E_slab - E_ads_free
  * Gibbs free energy of reaction steps (computational hydrogen
    electrode style, pH / U corrections).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Physical constants
KB = 8.617333262e-5      # Boltzmann constant, eV/K
H_PLANCK_EVS = 4.135667696e-15  # Planck constant, eV*s
R_GAS = 8.314462618        # J/(mol*K)
FARADAY = 96485.33212      # C/mol


@dataclass
class MoleculeRef:
    """Reference energy of a free (gas-phase) molecule or isolated atom.

    ``energy`` is the raw DFT/MLIP total energy in eV.
    """
    formula: str
    energy: float
    # empirical ZPE (eV) for common adsorbates/gases; used when no
    # vibrational frequencies are provided.
    zpe_empirical: float = 0.0
    # empirical entropy (J/mol/K) at 298.15 K and 1 bar.
    s_empirical: float = 0.0


# Empirical thermochemical corrections at 298.15 K, 1 bar (common values
# used in electrocatalysis / heterogeneous catalysis literature).
# ZPE in eV, S in J/mol/K, Cp contribution ignored (approximation).
EMPIRICAL_GAS_CORRECTIONS: Dict[str, Tuple[float, float]] = {
    # formula: (ZPE [eV], S [J/mol/K])
    "H2":    (0.27, 130.68),
    "H2O":   (0.56, 188.83),
    "CO":    (0.13, 197.66),
    "CO2":   (0.31, 213.79),
    "CH4":   (1.20, 186.26),
    "NH3":   (0.90, 192.77),
    "O2":    (0.10, 205.15),
    "N2":    (0.14, 191.61),
    "NO":    (0.11, 210.76),
    "OH":    (0.20, 183.0),   # approx
    "H":     (0.00, 114.72),
    "N":     (0.00, 153.30),
    "O":     (0.00, 161.06),
}


def zpe_from_frequencies(freqs_cm1: List[float]) -> float:
    """Zero-point energy (eV) from harmonic frequencies in cm^-1.

    Conversion: E = h * c * nu, with nu[Hz] = c[cm/s] * nu[cm^-1].
    """
    c_cms = 2.99792458e10  # speed of light in cm/s
    total = 0.0
    for nu in freqs_cm1:
        if nu > 0:
            total += 0.5 * H_PLANCK_EVS * (nu * c_cms)
    return total


def _thermal_corrections(T: float, nu_cm1: List[float]) -> Tuple[float, float]:
    """Enthalpy (H - E_elec) and entropy contributions (eV) from harmonic
    vibrational frequencies at temperature T (K)."""
    c_cms = 2.99792458e10
    h = 0.0
    s = 0.0
    kT = KB * T
    for nu in nu_cm1:
        if nu <= 0:
            continue
        hv = H_PLANCK_EVS * nu * c_cms
        x = hv / kT
        if x > 700:
            continue
        try:
            zpe = 0.5 * hv
            cvib = hv / (math.exp(x) - 1.0)
            h += zpe + cvib
            s += hv / T * math.exp(-x) / (1 - math.exp(-x)) - KB * math.log(
                1 - math.exp(-x))
        except (OverflowError, ZeroDivisionError):
            continue
    return h, s * 1.0  # s already in eV/K contributing S*T in eV units


def enthalpy_correction(T: float, formula: str,
                        nu_cm1: Optional[List[float]] = None) -> float:
    """H(T) - E_elec correction in eV for a molecule at temperature T."""
    if nu_cm1:
        h, _ = _thermal_corrections(T, nu_cm1)
        return h
    zpe, _ = EMPIRICAL_GAS_CORRECTIONS.get(formula, (0.0, 0.0))
    return zpe


def entropy_correction(T: float, formula: str,
                       nu_cm1: Optional[List[float]] = None) -> float:
    """-T*S correction in eV (S from vibrational modes or empirical)."""
    if nu_cm1:
        _, s = _thermal_corrections(T, nu_cm1)
        return -T * s
    _, s_emp = EMPIRICAL_GAS_CORRECTIONS.get(formula, (0.0, 0.0))
    # S [J/mol/K] -> eV/K via Faraday (C/mol), then scale by -T.
    return -T * s_emp / FARADAY


def gibbs_free_energy(electronic: float, T: float = 298.15,
                      formula: Optional[str] = None,
                      nu_cm1: Optional[List[float]] = None,
                      pressure_bar: float = 1.0) -> float:
    """Gibbs free energy G(T) = E_elec + ZPE + Cvib,dT - T*S for a molecule.

    Vibrational path (``nu_cm1`` given): G_vib = sum[0.5*hv + kT*ln(1-exp(-x))]
    is exactly ``h_vib - T*s_vib`` from :func:`_thermal_corrections` — the
    ZPE must NOT be added twice.

    Empirical path: ZPE/entropy from the built-in 298.15 K table, scaled to
    the given pressure via S(p) = S(1 bar) - R*ln(p).
    """
    if nu_cm1:
        h_vib, s_vib = _thermal_corrections(T, nu_cm1)
        return electronic + h_vib - T * s_vib

    zpe = EMPIRICAL_GAS_CORRECTIONS.get(formula or "", (0.0, 0.0))[0]
    if formula:
        _, s_emp = EMPIRICAL_GAS_CORRECTIONS.get(formula, (0.0, 0.0))
        # J/mol/K -> eV/K (divide by Faraday); pressure: S(p) = S(1bar) - R*ln(p)
        s_ev_per_k = s_emp / FARADAY
        s_eff = s_ev_per_k - R_GAS / FARADAY * math.log(max(pressure_bar, 1e-10))
        return electronic + zpe - T * s_eff
    return electronic


def adsorption_energy(e_slab_ads: float, e_slab: float,
                      e_ads_free: float) -> float:
    """Adsorption energy in eV.

        E_ad = E(slab+ads) - E(slab) - E(ads, free)

    Negative values indicate exothermic adsorption (stabilized).
    """
    return e_slab_ads - e_slab - e_ads_free


def gibbs_reaction_energy(delta_e: float, n_h: int = 0,
                          u_vs_rhe: float = 0.0, ph: float = 0.0,
                          t: float = 298.15,
                          zpe_corr: float = 0.0, ts_corr: float = 0.0) -> float:
    """Gibbs free-energy change of an electrochemical step (eV).

    Computational hydrogen electrode (Nørskov convention)::

        DeltaG = DeltaE + ZPE - T*S + n_H * (-eU - kT*ln(10)*pH)

    ``u_vs_rhe`` is the electrode potential; the pH term applies when the
    potential is quoted against a fixed (SHE-like) reference — on a true
    RHE scale pH is already absorbed and ``ph`` should stay 0.

    Args:
        delta_e: electronic energy difference of the step (eV).
        n_h: number of (H+ + e-) transferred (positive for reduction).
        u_vs_rhe: applied potential (V).
        ph: pH of the electrolyte (0 when using the RHE scale).
        zpe_corr: zero-point energy correction (eV).
        ts_corr: -T*S entropy correction (eV).
    """
    kT = KB * t
    chem_pot = n_h * (-u_vs_rhe - kT * math.log(10.0) * ph)
    return delta_e + zpe_corr + ts_corr + chem_pot


def volcano_coordinates(
        reactions: List[Tuple[str, float]]) -> Tuple[List[float], List[str]]:
    """Return the cumulative (step-index, DeltaG) path for a free-energy
    diagram (volcano / step plot).

    Args:
        reactions: list of (label, delta_g) for each elementary step.
    Returns:
        (x_coords, labels) where x_coords[i] is the running free energy
        at step i, starting from 0.
    """
    coords = [0.0]
    acc = 0.0
    for _, dg in reactions:
        acc += dg
        coords.append(acc)
    labels = [s for s, _ in reactions]
    return coords, labels