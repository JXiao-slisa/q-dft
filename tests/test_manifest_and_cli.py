"""Run manifest provenance + CLI smoke tests."""

import json

import pytest

from conftest import requires_crewai
from slisadft.utils.run_manifest import create_run_manifest, finalize_run_manifest


def test_manifest_records_engine_mode_and_inputs(tmp_path, mock_mode):
    manifest = create_run_manifest("run", {"element": "Pt"}, workdir=tmp_path)
    assert manifest["engine_mode"] == "mock"
    assert manifest["workflow"] == "run"
    assert manifest["inputs"]["element"] == "Pt"
    assert manifest["manifest_path"] == str(tmp_path / "run_manifest.json")
    on_disk = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["schema"] == "slisadft.run_manifest/1"


def test_manifest_hashes_input_files(tmp_path, co_pt111_poscar):
    manifest = create_run_manifest("run", {"structure": str(co_pt111_poscar)})
    assert str(co_pt111_poscar) in manifest["input_file_hashes"]
    digest = manifest["input_file_hashes"][str(co_pt111_poscar)]
    assert len(digest) == 64  # sha256 hex


def test_finalize_records_status(tmp_path, mock_mode):
    create_run_manifest("adsorption", {}, workdir=tmp_path)
    out = finalize_run_manifest(tmp_path / "run_manifest.json", status="done",
                                result={"energy": -1.0})
    assert out is not None
    data = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "done"
    assert data["result_summary"] == {"energy": -1.0}
    assert "finished_at" in data


def test_finalize_missing_manifest_is_noop(tmp_path):
    assert finalize_run_manifest(tmp_path / "ghost.json", "done") is None


def test_cli_version():
    from slisadft.main import main
    assert main(["version"]) == 0


def test_cli_invalid_inputs_json():
    from slisadft.main import main
    assert main(["run", "--inputs", "{not json"]) == 2


@requires_crewai
def test_prepare_inputs_fills_defaults_and_knowledge(knowledge_dir):
    from slisadft.crew import prepare_inputs
    prepared = prepare_inputs({"element": "Cu"})
    assert prepared["miller"] == "(111)"
    assert prepared["adsorbate"] == "CO"
    assert "knowledge_context" in prepared
    assert prepared["knowledge_context"]  # non-empty from temp KB


@requires_crewai
def test_prepare_inputs_preserves_existing_knowledge():
    from slisadft.crew import prepare_inputs
    assert prepare_inputs({"knowledge_context": "custom"})["knowledge_context"] == "custom"
