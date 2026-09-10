"""Thermochemistry: adsorption energy, ZPE, Gibbs, CHE corrections."""

import pytest

from slisadft.utils.thermochemistry import (
    adsorption_energy,
    gibbs_free_energy,
    gibbs_reaction_energy,
    volcano_coordinates,
    zpe_from_frequencies,
)


def test_adsorption_energy_sign_convention():
    # E_ad = E(slab+ads) - E(slab) - E(ads)
    e = adsorption_energy(e_slab_ads=-520.0, e_slab=-500.0, e_ads_free=-15.0)
    assert e == pytest.approx(-5.0)
    assert e < 0  # exothermic


def test_zpe_from_frequencies():
    # 1000 cm-1 -> 0.5 * h * c * nu in eV (~0.0619 eV per mode)
    zpe = zpe_from_frequencies([1000.0, 2000.0])
    assert 0.15 < zpe < 0.20
    assert zpe_from_frequencies([-500.0]) == 0.0  # imaginary modes ignored


def test_gibbs_empirical_path_co():
    e = -300.0
    g = gibbs_free_energy(e, T=298.15, formula="CO")
    # G = E + ZPE - T*S; CO: ZPE=0.13 eV, S=197.66 J/mol/K (~0.51 eV at 298K)
    assert g < e + 0.13  # entropy dominates
    assert e - 0.6 < g < e + 0.13


def test_gibbs_vibrational_path_matches_zpe_and_entropy():
    freqs = [2100.0, 500.0, 300.0]
    e = -100.0
    g_vib = gibbs_free_energy(e, T=500.0, nu_cm1=freqs)
    zpe = zpe_from_frequencies(freqs)
    # per-mode contribution = zpe_i + kT*ln(1-exp(-x)) which is <= zpe_i and
    # stays positive for these stiff modes; ZPE must not be double-counted.
    assert e < g_vib <= e + zpe


def test_gibbs_unknown_formula_zero_correction():
    assert gibbs_free_energy(-50.0, T=298.15) == pytest.approx(-50.0)


def test_che_reaction_energy_potential_and_ph():
    # Nørskov CHE convention: each (H+ + e-) contributes -eU - kT*ln10*pH
    d = gibbs_reaction_energy(delta_e=1.0, n_h=1, u_vs_rhe=0.0, ph=0.0)
    assert d == pytest.approx(1.0)
    # U = 0.5 V lowers a 1e- step by 0.5 eV
    d_u = gibbs_reaction_energy(delta_e=1.0, n_h=1, u_vs_rhe=0.5)
    assert d_u == pytest.approx(0.5, abs=1e-6)
    # at fixed SHE-like reference, pH=7 lowers a reduction barrier (~0.41 eV at 298K)
    d_ph = gibbs_reaction_energy(delta_e=0.0, n_h=1, ph=7.0)
    assert -0.45 < d_ph < -0.35


def test_volcano_coordinates_cumulative():
    coords, labels = volcano_coordinates([("a", 0.5), ("b", -1.0), ("c", 0.2)])
    assert coords == [0.0, 0.5, -0.5, -0.3]
    assert labels == ["a", "b", "c"]
