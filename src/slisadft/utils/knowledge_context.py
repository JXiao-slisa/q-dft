"""RSI P4 — 知识反哺：把已审核的知识规则注入智能体上下文。

在每次 kickoff 前调用 :func:`build_knowledge_context`，从
``knowledge/computational_rules.md``（人工审核过的规则，含 pending 审核通过
后合并的自动经验）提取一个紧凑的规则摘要，作为 ``{knowledge_context}``
输入占位符注入相关任务的提示词中——研究经验越多，智能体的参数选择越准。

设计约束：
* 只注入"已审核"内容（computational_rules.md）；``pending_rules/`` 中
  未审核的自动规则绝不直接注入（Rule Zero / 审计要求）。
* 上下文长度有上限（默认 3000 字符），避免提示词膨胀。
* 知识库缺失或为空时返回空字符串，任务提示词中的占位符行自然留空。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

# 项目根 = src/slisadft/utils 的上两级
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RULES_FILE = PROJECT_ROOT / "knowledge" / "computational_rules.md"

_HEADING_RE = re.compile(r"^#{2,4}\s+(.*)$")


def _rules_path() -> Path:
    import os
    override = os.environ.get("SLISADFT_KNOWLEDGE_DIR")
    if override:
        return Path(override) / "computational_rules.md"
    return RULES_FILE


def _digest_rules_text(text: str, max_rules: int) -> List[str]:
    """Extract concise rule bullets: subsection headings + their bullets."""
    rules: List[str] = []
    current_section = ""
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            current_section = m.group(1).strip()
            continue
        stripped = line.strip()
        if stripped.startswith(("-", "*")) and len(stripped) > 4:
            bullet = stripped.lstrip("-*").strip()
            rules.append(f"[{current_section}] {bullet}" if current_section else bullet)
        if len(rules) >= max_rules:
            break
    return rules


def build_knowledge_context(max_rules: int = 24, max_chars: int = 3000) -> str:
    """Build the compact knowledge digest injected into agent prompts.

    Returns an empty string when the knowledge base is missing/empty so
    callers can pass the value straight into crew inputs.
    """
    path = _rules_path()
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    rules = _digest_rules_text(text, max_rules=max_rules)
    if not rules:
        return ""
    parts: List[str] = []
    total = 0
    for rule in rules:
        piece = f"- {rule}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece) + 1
    return "\n".join(parts)
