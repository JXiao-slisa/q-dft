"""v0.3.0 deterministic workflows: adsorption-full (three-point protocol)."""

import json

import pytest

from slisadft.workflows import (
    BATCH_CSV_COLUMNS,
    REPORT_NAME,
    run_adsorption_full,
    run_batch,
)


@pytest.fixture
def workflow_result(mock_mode, tmp_path):
    return run_adsorption_full(
        {"element": "Pt", "adsorbate": "CO", "dft_calculator": "vasp"},
        workdir=str(tmp_path))


def test_adsorption_full_three_point_protocol(workflow_result, tmp_path):
    r = workflow_result
    en = r["energies_eV"]
    assert set(en) == {"slab", "adsorbate", "slab_ads"}
    for name, d in r["system_details"].items():
        assert d["mlip_optimized"], f"{name}: MLIP step missing"
        assert d["dft_workdir"], f"{name}: DFT step missing"
    # E_ad consistent with the three energies (deterministic protocol)
    expected = en["slab_ads"] - en["slab"] - en["adsorbate"]
    assert r["adsorption_energy_eV"] == pytest.approx(expected, abs=1e-3)
    assert r["synthetic"] is True and r["engine_mode"] == "mock"


def test_adsorption_full_report_and_manifest(workflow_result, tmp_path):
    from pathlib import Path as _Path
    r = workflow_result
    report = _Path(r["report"])
    assert report.parent == tmp_path
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "SYNTHETIC" in text  # Rule Zero marking in the report
    assert "adsorption-full" in text


def test_adsorption_full_is_deterministic(mock_mode, tmp_path):
    a = run_adsorption_full({"element": "Pt", "adsorbate": "CO"},
                            workdir=str(tmp_path / "a"))
    b = run_adsorption_full({"element": "Pt", "adsorbate": "CO"},
                            workdir=str(tmp_path / "b"))
    assert a["adsorption_energy_eV"] == b["adsorption_energy_eV"]


def test_adsorption_full_bad_adsorbate_fails_cleanly(mock_mode, tmp_path):
    with pytest.raises(RuntimeError):
        run_adsorption_full({"adsorbate": "Zz9NotAnElement"}, workdir=str(tmp_path))


def test_batch_two_combos_csv(mock_mode, tmp_path):
    combos = [
        {"element": "Pt", "adsorbate": "CO", "site": "top"},
        {"element": "Pt", "adsorbate": "OH", "site": "fcc"},
    ]
    out = tmp_path / "batch.csv"
    summary = run_batch(combos, out_csv=str(out), max_workers=2)
    assert summary["ok"] == 2 and summary["failed"] == 0
    assert len(summary["results"]) == 2
    rows = summary["results"]
    # fcc OH 与 top CO 是不同体系，能量应不同（合成物理亦需可区分）
    assert rows[0]["adsorption_energy_eV"] != rows[1]["adsorption_energy_eV"]
    assert out.exists()
    header = out.read_text(encoding="utf-8").splitlines()[0]
    for col in BATCH_CSV_COLUMNS:
        assert col in header
