"""Reviewer Agent — RSI 审稿人机制

将"审问 + 联网检索 + 分析重做/修正或推进"纳入 RSI 流程。

流程：
  1. 审稿人 Agent 审阅当前批次结果（LLM）
  2. 对审稿人提出的质疑进行联网检索（web search）
  3. 生成审阅报告（问题清单 + 质疑 + 检索结论）
  4. 决策：重做/修正 或 推进下一批次
  5. 决策与台账记录 → 知识库
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

LEDGER_PATH = Path(__file__).resolve().parents[2] / "results" / "<project>" / "rsi_ledger.jsonl"


def llm_chat(prompt: str, system: str = "你是资深的计算催化领域学术审稿人。",
             model: str = "sensenova-6.8-flash-lite") -> str:
    """Call SenseNova LLM."""
    api_key = os.getenv("SENSENOVA_API_KEY", "")
    resp = requests.post(
        "https://token.sensenova.cn/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [
                  {"role": "system", "content": system},
                  {"role": "user", "content": prompt},
              ],
              "stream": False},
        timeout=120,
    )
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    raise RuntimeError(f"LLM API error: {json.dumps(data, ensure_ascii=False)[:300]}")


def web_search(query: str, queries: Optional[List[str]] = None) -> str:
    """Perform web search via the tool (no-op shell; the agent calls web_search
    separately). This helper documents the queries to run."""
    qs = queries or [query]
    return "\n".join(f"SEARCH: {q}" for q in qs)


def review_batch(batch_id: str, report_md: str, analysis_json: dict,
                 external_suggestions: Optional[List[str]] = None) -> Dict:
    """Run a full review cycle on a batch.

    Returns a dict with: review_report, search_queries, decisions, ledger_entry.
    """
    external = "\n".join(f"- {s}" for s in (external_suggestions or []))

    review_prompt = f"""请审阅以下 {batch_id} 计算批次的结果报告。

## 批次报告
{report_md}

## 结构化分析数据
{json.dumps(analysis_json, ensure_ascii=False, indent=2)[:3000]}

## 外部审阅建议
{external or "无"}

请以学术审稿人身份：
1. 列出该批次的主要问题/质疑（编号）
2. 对每个问题给出严重程度（高/中/低）
3. 提出需要联网检索确认的科学点（给检索关键词）
4. 给出结论：该批次应【重做】/【修正后推进】/【直接推进下一批次】？
5. 对下一批次（batch_{int(batch_id.split('-')[1])+1 if '-' in batch_id else 2}）的设计建议"""

    review = llm_chat(review_prompt, system="你是资深的计算催化领域学术审稿人，审稿严格、证据导向。")

    # Decide next step based on review keywords (priority: redo > revise > advance)
    next_step = "advance"
    if "重做" in review and "修正后推进" not in review and "不需要完全重做" not in review:
        next_step = "redo"
    elif "修正" in review or "revise" in review.lower() or "修正后推进" in review:
        next_step = "revise-then-advance"

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "batch_id": batch_id,
        "stage": "review",
        "next_step": next_step,
        "review_length": len(review),
        "external_suggestions_count": len(external_suggestions or []),
    }
    _append_ledger(ledger_entry)

    return {
        "review": review,
        "next_step": next_step,
        "ledger_entry": ledger_entry,
    }


def _append_ledger(entry: Dict) -> None:
    """Append an entry to the RSI ledger (JSONL, append-only, traceable)."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_ledger() -> List[Dict]:
    """Read all ledger entries (for traceability)."""
    if not LEDGER_PATH.exists():
        return []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def decide(mode: str, batch_id: str, next_step: str, reason: str) -> Dict:
    """Record a mode/batch decision in the ledger."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,           # quick / normal / advanced
        "batch_id": batch_id,
        "decision": next_step,  # redo / revise / advance / new-mode
        "reason": reason,
    }
    _append_ledger(entry)
    return entry


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

    # Example: review batch-1
    report_path = Path(__file__).resolve().parents[2] / "results/<project>/report.md"
    analysis_path = Path(__file__).resolve().parents[2] / "results/<project>/analysis.json"
    if report_path.exists() and analysis_path.exists():
        report = report_path.read_text()
        analysis = json.loads(analysis_path.read_text())
        ext = [
            "表面太小需扩胞",
            "未做活性火山图",
            "报告无可读性配图",
            "需四类报告（实验/学术/检索/分析）",
            "增加C2路径",
            "内部RSI与外部决策需台账",
            "缺中间体优化日志与前后结构对比",
            "检索学术发表物反思批次1、优化批次2",
            "文件夹标识快速模式batch1",
            "三层迭代（batch内/batch间/模式间）",
        ]
        r = review_batch("batch-1", report, analysis, external_suggestions=ext)
        print(f"Next step: {r['next_step']}")
        print(r["review"][:1500])
        print("\n--- ledger entries ---")
        for e in read_ledger()[-3:]:
            print(e)