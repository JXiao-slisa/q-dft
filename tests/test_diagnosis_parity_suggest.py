"""v0.3.0: failure diagnosis, engine parity QA, RSI P6-lite suggestions."""

import json

import pytest

from slisadft.utils.failure_diagnosis import diagnose, summarize_diagnosis
from slisadft.engines.parity import parity_report
from slisadft.utils.suggest import suggest_next


# ---------------------------------------------------------------------------
# Failure intelligence
# ---------------------------------------------------------------------------

def test_diagnose_scf_not_converged():
    out = diagnose("VASP run stopped: NELM reached, SCF not converged")
    assert out and out[0]["code"] == "SCF_NOT_CONVERGED"
    assert "NELM" in out[0]["advice"]
    assert out[0]["source"]  # traceable


def test_diagnose_potcar_and_priority_order():
    text = "ERROR: POTCAR missing for Pt\nMPI_ABORT was invoked"
    out = diagnose(text)
    codes = [d["code"] for d in out]
    assert "POTCAR_MISSING" in codes and "MPI_ABORT" in codes
    assert out[0]["severity"] == "high"  # high severity sorted first


def test_diagnose_freq_nsw_knowledge_rule():
    out = diagnose("INCAR: IBRION = 5 with NSW = 0")
    assert any(d["code"] == "NO_FREQ_DISPLACEMENT" for d in out)


def test_diagnose_empty_and_unknown():
    assert diagnose("") == []
    assert diagnose("all good, job finished") == []


def test_summarize_diagnosis():
    out = diagnose("ZBRENT: fatal error in bracketing")
    summary = summarize_diagnosis(out)
    assert "ZBRENT_FATAL" in summary
    assert summarize_diagnosis([]) == ""


# ---------------------------------------------------------------------------
# Parity QA
# ---------------------------------------------------------------------------

def test_parity_report_fields_and_budget(mock_mode, co_pt111_poscar):
    r = parity_report(str(co_pt111_poscar), calc_a="emt", calc_b="lj",
                      budget_ev=50.0, budget_force=50.0)
    assert set(r["points"]) == {"emt", "lj"}
    for p in r["points"].values():
        assert isinstance(p["energy_eV"], float)
        assert p["n_atoms"] == 5
    assert r["delta_energy_eV"] >= 0
    assert r["within_budget"] is True  # loose budget passes
    assert r["synthetic"] is True


def test_parity_budget_rejects_large_deviation(mock_mode, co_pt111_poscar):
    r = parity_report(str(co_pt111_poscar), "emt", "lj", budget_ev=1e-6,
                      budget_force=1e-6)
    assert r["within_budget"] is False
    assert r["verdict"].startswith("REVIEW")


def test_parity_unknown_calculator(mock_mode, co_pt111_poscar):
    with pytest.raises(ValueError):
        parity_report(str(co_pt111_poscar), "emt", "quantum-magic")


# ---------------------------------------------------------------------------
# RSI P6-lite suggestions
# ---------------------------------------------------------------------------

def test_suggest_strong_adsorption_site_scan(mock_mode):
    out = suggest_next({"inputs": {"element": "Pt", "site": "top"},
                        "adsorption_energy_eV": -1.8, "synthetic": False})
    actions = [s["action"] for s in out]
    assert any("site_scan" in a and "fcc" in a for a in actions)
    assert all(s["rationale"] and s["source"] for s in out)


def test_suggest_weak_adsorption(mock_mode):
    out = suggest_next({"inputs": {"element": "Au", "site": "top"},
                        "adsorption_energy_eV": -0.1, "synthetic": False})
    assert any("change_exploration" in s["action"] for s in out)


def test_suggest_synthetic_warning_and_pending(knowledge_dir, mock_mode):
    p = knowledge_dir / "auto" / "pending_rules" / "r.yaml"
    p.write_text("title: t\nbody: b\n", encoding="utf-8")
    out = suggest_next({"inputs": {"element": "Pt"}, "adsorption_energy_eV": -1.0,
                        "synthetic": True}, knowledge_dir=str(knowledge_dir))
    actions = [s["action"] for s in out]
    assert any("switch_to_real_engine" in a for a in actions)
    assert any("review_pending_rules" in a for a in actions)


def test_suggest_knowledge_hint_dpa4(knowledge_dir, mock_mode):
    out = suggest_next(None, knowledge_dir=str(knowledge_dir))
    assert any("DPA4-Mini" in s["action"] for s in out)


def test_suggest_cold_start(mock_mode, tmp_path, monkeypatch):
    monkeypatch.setenv("SLISADFT_KNOWLEDGE_DIR", str(tmp_path / "empty"))
    out = suggest_next(None)
    assert out and "start" in out[0]["action"]


def test_suggest_unconverged_retry(mock_mode):
    out = suggest_next({
        "inputs": {"element": "Pt"},
        "adsorption_energy_eV": -1.0,
        "system_details": {"slab": {"dft_converged": False}},
    })
    assert any("retry_converged" in s["action"] for s in out)
