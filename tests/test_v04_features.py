"""v0.4.0 features: init / info / runs / cite / remediation."""

import json

import pytest

from conftest import requires_crewai
from slisadft.utils.inspect import citation, inspect_structure, list_runs
from slisadft.utils.remediation import remediate_incar, remediation_note
from slisadft.main import main


# ---------------------------------------------------------------------------
# qdft init — project scaffolding
# ---------------------------------------------------------------------------

def test_init_creates_project(tmp_path):
    proj = tmp_path / "myproj"
    assert main(["init", str(proj)]) == 0
    assert (proj / ".env").is_file()
    assert (proj / "combos.json").is_file()
    assert (proj / "runs").is_dir()
    combos = json.loads((proj / "combos.json").read_text(encoding="utf-8"))
    assert isinstance(combos, list) and combos[0]["element"]


def test_init_refuses_non_empty_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "x.txt").write_text("data", encoding="utf-8")
    assert main(["init", str(proj)]) == 2


# ---------------------------------------------------------------------------
# qdft info — structure inspection
# ---------------------------------------------------------------------------

def test_info_structure_payload(co_pt111_poscar):
    info = inspect_structure(str(co_pt111_poscar))
    assert info["n_atoms"] == 5
    assert info["composition"]["Pt"] == 3
    assert info["composition"]["C"] == 1
    assert info["composition"]["O"] == 1
    assert info["volume_angstrom3"] > 0
    assert info["feasibility"]["dft"] in ("comfortable", "expensive")
    assert set(info["cell_lengths"]) != {0}


def test_info_missing_file():
    with pytest.raises(FileNotFoundError):
        inspect_structure("no/such/file.vasp")


def test_info_cli(tmp_path, capsys):
    from pathlib import Path as _P
    import json as _json
    struct = _P(str(tmp_path)) / "s.vasp"
    struct.write_text(
        "Po\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0\nPt\n1\nCartesian\n0 0 0\n",
        encoding="utf-8")
    rc = main(["info", str(struct)])
    assert rc == 0
    out = capsys.readouterr().out
    payload = _json.loads(out)
    assert payload["composition"]["Pt"] == 1


# ---------------------------------------------------------------------------
# qdft runs — history browser
# ---------------------------------------------------------------------------

def test_list_runs_from_runs_dir(tmp_path, monkeypatch, mock_mode):
    from slisadft.utils.run_manifest import create_run_manifest
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    create_run_manifest("run", {"element": "Pt", "adsorbate": "CO"},
                        workdir=tmp_path / "runs" / "r1")
    (tmp_path / "runs" / "r1" / "run_manifest.json").write_text(
        json.dumps({"workflow": "run", "created_at": "2026-09-10T00:00:00Z",
                    "engine_mode": "mock", "status": "done",
                    "inputs": {"element": "Pt", "adsorbate": "CO"},
                    "result_summary": {"adsorption_energy_eV": -0.3574}}),
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rows = list_runs()
    assert len(rows) == 1
    assert rows[0]["workflow"] == "run"
    assert rows[0]["e_ad_eV"] == pytest.approx(-0.3574)


def test_runs_cli_empty_is_ok(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["runs"]) == 0


# ---------------------------------------------------------------------------
# qdft cite
# ---------------------------------------------------------------------------

def test_citation_content():
    c = citation()
    assert "q-dft" in c["text"]
    assert "@software{qdft_agent" in c["bibtex"]


def test_cite_cli(capsys):
    assert main(["cite"]) == 0
    assert "BibTeX" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Self-healing remediation
# ---------------------------------------------------------------------------

def test_remediate_scf():
    params, applied, notes = remediate_incar({}, ["SCF_NOT_CONVERGED"])
    assert applied == ["SCF_NOT_CONVERGED"]
    assert params["NELM"] == 200 and params["ALGO"] == "All"
    assert notes and "SCF" in notes[0]


def test_remediate_multiple_and_unknown():
    params, applied, _ = remediate_incar(
        {"ENCUT": 500}, ["ZBRENT_FATAL", "SOMETHING_ELSE"])
    assert applied == ["ZBRENT_FATAL"]
    assert params["POTIM"] == 0.15
    assert params["ENCUT"] == 500  # untouched


def test_remediation_note():
    assert remediation_note([], []) is None
    n = remediation_note(["SCF_NOT_CONVERGED"], ["SCF_NOT_CONVERGED: x"])
    assert "auto-remediated" in n


@requires_crewai
def test_workflow_reports_empty_remediation_in_mock(mock_mode, tmp_path):
    from slisadft.workflows import run_adsorption_full
    r = run_adsorption_full({"element": "Pt", "adsorbate": "CO"},
                            workdir=str(tmp_path))
    assert r["remediation"] == []  # nothing failed → no intervention recorded
