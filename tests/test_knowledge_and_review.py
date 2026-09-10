"""RSI P4: knowledge digest injection; RSI P5: pending-rule review flow."""

from slisadft.utils.knowledge_context import build_knowledge_context
from slisadft.utils import rsi_review


def test_knowledge_context_builds_from_reviewed_rules(knowledge_dir):
    ctx = build_knowledge_context()
    assert "DPA4-Mini" in ctx
    assert "NSW>=1" in ctx


def test_knowledge_context_missing_kb_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SLISADFT_KNOWLEDGE_DIR", str(tmp_path / "nope"))
    assert build_knowledge_context() == ""


def test_knowledge_context_respects_max_chars(knowledge_dir):
    ctx = build_knowledge_context(max_chars=60)
    assert len(ctx) <= 60


# ---------------------------------------------------------------------------
# RSI P5 review flow
# ---------------------------------------------------------------------------

def _write_pending(knowledge_dir, name, title="Test Rule"):
    p = knowledge_dir / "auto" / "pending_rules" / name
    p.write_text(f"title: \"{title}\"\nbody: use more k-points\n", encoding="utf-8")
    return p


def test_list_pending(knowledge_dir):
    _write_pending(knowledge_dir, "a.yaml", "Rule A")
    items = rsi_review.list_pending()
    assert [i["name"] for i in items] == ["a.yaml"]
    assert items[0]["title"] == "Rule A"


def test_approve_merges_into_knowledge_base(knowledge_dir):
    _write_pending(knowledge_dir, "a.yaml", "Rule A")
    rf = knowledge_dir / "computational_rules.md"
    size_before = rf.stat().st_size

    result = rsi_review.approve_rule("a.yaml", reviewer="alice")
    assert result["status"] == "approved"
    text = rf.read_text(encoding="utf-8")
    assert "Rule A" in text and "alice" in text
    assert rf.stat().st_size > size_before
    assert not (knowledge_dir / "auto" / "pending_rules" / "a.yaml").exists()
    assert (knowledge_dir / "auto" / "approved" / "a.yaml").exists()


def test_reject_archives_only(knowledge_dir):
    _write_pending(knowledge_dir, "b.yaml", "Rule B")
    rf = knowledge_dir / "computational_rules.md"
    size_before = rf.stat().st_size

    result = rsi_review.reject_rule("b.yaml", note="too vague")
    assert result["status"] == "rejected"
    assert rf.stat().st_size == size_before  # untouched
    assert (knowledge_dir / "auto" / "rejected" / "b.yaml").exists()


def test_path_traversal_rejected(knowledge_dir):
    for bad in ("../a.yaml", "sub/dir.yaml", "..\\a.yaml", "plain.txt", ""):
        try:
            rsi_review.read_pending(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {bad!r}"


def test_approve_missing_rule_raises(knowledge_dir):
    import pytest
    with pytest.raises(FileNotFoundError):
        rsi_review.approve_rule("ghost.yaml")
