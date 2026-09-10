"""Shared pytest fixtures: temp knowledge dir, mock engine mode, sample structures."""

import os
import sys
from pathlib import Path

import pytest

# Make src/ importable without installing (works for pytest runs from repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for p in (str(SRC), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def mock_mode(monkeypatch):
    """Force the mock engine backend for the duration of a test."""
    monkeypatch.setenv("SLISADFT_ENGINE_MODE", "mock")
    return "mock"


@pytest.fixture
def real_mode(monkeypatch):
    monkeypatch.setenv("SLISADFT_ENGINE_MODE", "real")
    return "real"


@pytest.fixture
def knowledge_dir(tmp_path, monkeypatch):
    """Redirect the knowledge base to a temp dir pre-populated with rules."""
    kb = tmp_path / "knowledge"
    (kb / "auto" / "pending_rules").mkdir(parents=True)
    (kb / "computational_rules.md").write_text(
        "# slisaDFT Knowledge Base\n"
        "## MLIP 规则\n"
        "- 3d 金属体系避免使用 DPA4-Mini（误差大）\n"
        "## DFT 规则\n"
        "- VASP 频率计算必须 NSW>=1，否则无位移步\n",
        encoding="utf-8")
    monkeypatch.setenv("SLISADFT_KNOWLEDGE_DIR", str(kb))
    return kb


@pytest.fixture
def co_pt111_poscar(tmp_path):
    """A small Pt3+CO slab model with realistic Pt-Pt distances (fast mock runs)."""
    from ase import Atoms
    from ase.io import write

    slab = Atoms("Pt3CO",
                 positions=[[0.0, 0.0, 0.0],
                            [2.77, 0.0, 0.0],
                            [1.385, 2.40, 0.0],
                            [1.385, 1.20, 1.85],
                            [1.385, 1.20, 2.95]],
                 cell=[9.0, 9.0, 16.0])
    slab.pbc = (True, True, True)
    path = tmp_path / "Pt3CO.vasp"
    write(path, slab, format="vasp")
    return path


def _has_module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# Tests that drive the real crew pipeline need crewai installed; the rest of
# the suite (engines, parsers, API plumbing) runs without it.
requires_crewai = pytest.mark.skipif(
    not _has_module("crewai"), reason="crewai not installed")

# Full pipeline kickoff additionally calls a real LLM endpoint — these tests
# are opt-in: SLISA_E2E_LLM=1 with valid LLM_* config in .env.
requires_llm = pytest.mark.skipif(
    os.environ.get("SLISA_E2E_LLM") != "1",
    reason="set SLISA_E2E_LLM=1 (with valid LLM key) to run LLM end-to-end tests")

requires_crewai_and_llm = pytest.mark.skipif(
    not _has_module("crewai") or os.environ.get("SLISA_E2E_LLM") != "1",
    reason="needs crewai installed and SLISA_E2E_LLM=1")
