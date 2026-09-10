"""RSI P5 — 规则审核：pending_rules 的批准 / 驳回流程。

自动抽取的候选规则落在 ``knowledge/auto/pending_rules/``。专家（或管理员）
通过 Web 界面审核：

* **approve** —— 规则正文以 markdown 章节追加进 ``knowledge/computational_rules.md``
  （该文件由 RSI P4 注入智能体上下文），原文件移入 ``knowledge/auto/approved/``；
* **reject**  —— 原文件移入 ``knowledge/auto/rejected/``，不进入知识库。

所有动作都有文件级审计痕迹（来源文件保留，仅移动目录）。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RULE_TITLE_RE = re.compile(r"^\s*title:\s*(.+)$", re.MULTILINE)


def _knowledge_root() -> Path:
    import os
    override = os.environ.get("SLISADFT_KNOWLEDGE_DIR")
    return Path(override) if override else PROJECT_ROOT / "knowledge"


def pending_dir() -> Path:
    return _knowledge_root() / "auto" / "pending_rules"


def approved_dir() -> Path:
    d = _knowledge_root() / "auto" / "approved"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rejected_dir() -> Path:
    d = _knowledge_root() / "auto" / "rejected"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rules_file() -> Path:
    return _knowledge_root() / "computational_rules.md"


def _safe_name(name: str) -> str:
    """Reject path traversal — only plain YAML filenames are allowed."""
    if not name or "/" in name or "\\" in name or ".." in name or not name.endswith(".yaml"):
        raise ValueError(f"invalid rule file name: {name!r}")
    return name


def list_pending() -> List[Dict[str, str]]:
    """List pending rule files with their extracted titles."""
    items: List[Dict[str, str]] = []
    pd = pending_dir()
    if not pd.is_dir():
        return items
    for f in sorted(pd.glob("*.yaml")):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = RULE_TITLE_RE.search(text)
        items.append({
            "name": f.name,
            "title": m.group(1).strip().strip('"\'') if m else f.stem,
            "size": str(f.stat().st_size),
        })
    return items


def read_pending(name: str) -> str:
    """Return the raw YAML content of a pending rule."""
    name = _safe_name(name)
    path = pending_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"pending rule not found: {name}")
    return path.read_text(encoding="utf-8", errors="ignore")


def approve_rule(name: str, reviewer: str = "", note: str = "") -> Dict[str, str]:
    """Merge a pending rule into the knowledge base and archive the source."""
    name = _safe_name(name)
    src = pending_dir() / name
    if not src.is_file():
        raise FileNotFoundError(f"pending rule not found: {name}")

    body = src.read_text(encoding="utf-8", errors="ignore")
    title_match = RULE_TITLE_RE.search(body)
    title = title_match.group(1).strip().strip('"\'') if title_match else src.stem

    section = (
        f"\n## Auto Rule (approved): {title}\n\n"
        f"- 来源: `knowledge/auto/pending_rules/{name}`\n"
        f"- 审核人: {reviewer or 'web-ui'}\n"
        + (f"- 备注: {note}\n" if note else "")
        + f"\n```yaml\n{body.strip()}\n```\n"
    )

    rf = rules_file()
    if rf.exists():
        with open(rf, "a", encoding="utf-8") as f:
            f.write(section)
    else:  # knowledge base missing — create it with a standard header
        rf.parent.mkdir(parents=True, exist_ok=True)
        rf.write_text("# slisaDFT Knowledge Base\n" + section, encoding="utf-8")

    dest = approved_dir() / name
    shutil.move(str(src), str(dest))
    return {"name": name, "title": title, "status": "approved",
            "merged_into": str(rf), "archived_to": str(dest)}


def reject_rule(name: str, reviewer: str = "", note: str = "") -> Dict[str, str]:
    """Reject a pending rule (archive only, knowledge base untouched)."""
    name = _safe_name(name)
    src = pending_dir() / name
    if not src.is_file():
        raise FileNotFoundError(f"pending rule not found: {name}")

    dest = rejected_dir() / name
    shutil.move(str(src), str(dest))
    if note:
        (rejected_dir() / (name + ".review")).write_text(
            f"reviewer: {reviewer or 'web-ui'}\nnote: {note}\n", encoding="utf-8")
    return {"name": name, "status": "rejected", "archived_to": str(dest)}
