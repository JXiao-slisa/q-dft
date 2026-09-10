"""Mock engine layer: deterministic synthetic runs with SYNTHETIC marking."""

import json
from pathlib import Path

from slisadft.engines import SYNTHETIC_MARK, engine_mode, is_mock, synthetic_result
from slisadft.engines.mock import (
    get_mock_calculator,
    run_mock_dft,
    write_mock_vasp_outputs,
)
from slisadft.utils.vasp_input import (
    parse_vasp_converged,
    parse_vasp_energy,
    parse_vasp_forces,
)


def test_engine_mode_switch(monkeypatch):
    monkeypatch.setenv("SLISADFT_ENGINE_MODE", "mock")
    assert engine_mode() == "mock" and is_mock()
    monkeypatch.setenv("SLISADFT_ENGINE_MODE", "real")
    assert engine_mode() == "real" and not is_mock()
    monkeypatch.setenv("SLISADFT_ENGINE_MODE", "bogus")
    assert engine_mode() == "real"  # invalid values fall back to real


def test_synthetic_result_stamping():
    out = synthetic_result({"calculator": "vasp"})
    assert out["engine_mode"] == "mock"
    assert out["synthetic"] is True
    assert SYNTHETIC_MARK in out["synthetic_notice"]


def test_mock_vasp_outputs_parse_back(tmp_path):
    write_mock_vasp_outputs(tmp_path, energy=-42.5, max_force=0.01, converged=True)
    assert parse_vasp_energy(tmp_path) == pytest_approx(-42.5)
    assert parse_vasp_converged(tmp_path)
    forces = parse_vasp_forces(tmp_path)
    assert forces and forces[-1] == pytest_approx(0.01)
    assert SYNTHETIC_MARK.split(":")[0] in (tmp_path / "OUTCAR").read_text()


def pytest_approx(x):
    import pytest
    return pytest.approx(x)


def test_mock_dft_run_vasp(mock_mode, co_pt111_poscar, tmp_path):
    result = run_mock_dft("vasp", str(co_pt111_poscar), tmp_path, relax=True)
    assert result["synthetic"] is True
    assert result["engine_mode"] == "mock"
    assert result["converged"] is True
    assert isinstance(result["final_energy_eV"], float)
    assert (tmp_path / "vasp" / "OUTCAR").is_file()
    assert (tmp_path / "vasp" / "INCAR").is_file()


def test_mock_dft_all_calculators_parse(mock_mode, co_pt111_poscar, tmp_path):
    for calc, out_name in (("cp2k", "cp2k.out"), ("abacus", "abacus.out")):
        wd = tmp_path / calc
        result = run_mock_dft(calc, str(co_pt111_poscar), tmp_path, relax=True)
        assert (wd / out_name).is_file(), f"{calc} output missing"
        assert isinstance(result["final_energy_eV"], float)  # finite synthetic energy
        assert result["synthetic"] is True


def test_mock_energies_deterministic(mock_mode, co_pt111_poscar, tmp_path):
    from ase.io import read
    r1 = run_mock_dft("vasp", str(co_pt111_poscar), tmp_path / "a")
    r2 = run_mock_dft("vasp", str(co_pt111_poscar), tmp_path / "b")
    assert r1["final_energy_eV"] == r2["final_energy_eV"]


def test_mock_consistent_adsorption_energy(mock_mode, co_pt111_poscar, tmp_path):
    """Slab+ads and molecule runs give an E_ad in a sane window (mock physics)."""
    from ase.io import read as ase_read
    from ase.build import molecule as ase_molecule
    from ase.io import write as ase_write

    co = tmp_path / "CO.vasp"
    co_mol = ase_molecule("CO")
    co_mol.center(vacuum=5.0)
    ase_write(co, co_mol, format="vasp", vasp5=True)

    e_combo = run_mock_dft("vasp", str(co_pt111_poscar), tmp_path / "c")["final_energy_eV"]
    e_co = run_mock_dft("vasp", str(co), tmp_path / "m")["final_energy_eV"]

    # "slab-only" proxy: the Pt atoms of the combined structure, single-point
    atoms = ase_read(co_pt111_poscar)
    del atoms[[a.index for a in atoms if a.symbol in ("C", "O")]]
    atoms.calc = get_mock_calculator(atoms)
    e_slab = float(atoms.get_potential_energy())

    e_ad = e_combo - e_slab - e_co
    assert -20.0 < e_ad < 20.0  # mock physics stays in a sane window
