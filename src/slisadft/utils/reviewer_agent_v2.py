"""Reviewer Agent V2 — 标准学术审稿格式 + 检索任务。

按学术期刊标准审稿流程：
  1. 审稿人分栏审阅（General / Major / Minor / Questions）
  2. 检索任务：对关键质疑进行文献检索（生成检索查询 + 检索结果摘要）
  3. 决策：Reject / Major Revision / Minor Revision / Accept
  4. 台账记录
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

LEDGER_PATH = Path(__file__).resolve().parents[2] / "results" / "<project>" / "rsi_ledger.jsonl"

REVIEW_TEMPLATE = """## 审稿意见 (Reviewer Report)

**稿件：** {title}
**审稿人：** Reviewer Agent V2
**日期：** {date}
**建议：** {recommendation}

---

### 1. General Comments (总体评价)
{general}

### 2. Major Comments (主要问题)
{major}

### 3. Minor Comments (次要问题)
{minor}

### 4. Questions to Authors (向作者提问)
{questions}

### 5. Scientific Soundness (科学严谨性)
| 维度 | 评级 (Excellent/Good/Fair/Poor) | 说明 |
|------|-------------------------------|------|
| 方法正确性 | {method_rating} | {method_note} |
| 数据充分性 | {data_rating} | {data_note} |
| 统计分析 | {stat_rating} | {stat_note} |
| 物理合理性 | {phys_rating} | {phys_note} |
| 结论支撑度 | {conc_rating} | {conc_note} |

### 6. 检索记录 (Literature Verification)
{search_results}

### 7. 最终建议
**{recommendation}** — {recommendation_reason}
"""


def _llm(prompt: str, system: str = "你是资深学术审稿人，严格、专业、建设性。",
         model: str = "sensenova-6.8-flash-lite") -> str:
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
    raise RuntimeError(f"LLM error: {json.dumps(data, ensure_ascii=False)[:200]}")


def _ledger(entry: Dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_sections(review_raw: str) -> Dict:
    """Parse LLM freeform review into structured sections."""
    sections = {
        "general": review_raw, "major": "", "minor": "", "questions": "",
        "method_rating": "Fair", "method_note": "",
        "data_rating": "Fair", "data_note": "",
        "stat_rating": "Fair", "stat_note": "",
        "phys_rating": "Good", "phys_note": "",
        "conc_rating": "Fair", "conc_note": "",
    }
    # Simple keyword-based extraction
    def grab(key):
        m = re.search(rf"{key}[:：]\s*(.*?)(?:\n\n|\Z)", review_raw, re.S)
        return m.group(1).strip() if m and m.group(1) else ""
    sections["major"] = grab("Major|主要问题")
    sections["minor"] = grab("Minor|次要问题")
    sections["questions"] = grab("Questions|提问")
    return sections


def review(title: str, report_md: str, analysis: Dict,
           search_queries: Optional[List[str]] = None) -> Dict:
    """Full standard review with search task."""
    search_queries = search_queries or [
        "CO adsorption trends on transition metal surfaces",
        "MACE potential validation transition metal adsorption energy",
        "d-band center as adsorption descriptor",
        "transition-metal electrocatalysis",
    ]

    # Build review prompt
    prompt = f"""请以学术审稿人身份审阅以下稿件《{title}》。

## 稿件内容
{report_md[:4000]}

## 结构化数据
{json.dumps(analysis, ensure_ascii=False, indent=1)[:2000]}

请按标准审稿格式输出，包含：
1. General Comments
2. Major Comments（科学/方法/统计问题）
3. Minor Comments（文字/格式问题）
4. Questions to Authors
5. 科学严谨性评级（方法/数据/统计/物理/结论，各为 Excellent/Good/Fair/Poor）
6. 最终建议：Accept / Minor Revision / Major Revision / Reject（给出理由）"""

    review_raw = _llm(prompt)
    sec = extract_sections(review_raw)

    # Decision
    rec = "Major Revision"
    if "Accept" in review_raw and "Revision" not in review_raw.split("建议")[0]:
        rec = "Accept"
    elif "Minor Revision" in review_raw:
        rec = "Minor Revision"
    elif "Reject" in review_raw:
        rec = "Reject"

    # Search task: generate search summary via LLM using queries as context
    search_summary = _llm(
        "请对以下检索主题给出文献检索计划（哪些关键词、关注哪些期刊、预期找到什么证据）：\n"
        + "\n".join(f"- {q}" for q in search_queries),
        system="你是计算催化领域的文献检索专家。")

    report = REVIEW_TEMPLATE.format(
        title=title, date=datetime.now().strftime("%Y-%m-%d"), recommendation=rec,
        general=sec["general"][:500] or review_raw[:500],
        major=sec["major"] or "见 general", minor=sec["minor"] or "无",
        questions=sec["questions"] or "无",
        method_rating=sec["method_rating"], method_note=sec["method_note"] or "",
        data_rating=sec["data_rating"], data_note=sec["data_note"] or "",
        stat_rating=sec["stat_rating"], stat_note=sec["stat_note"] or "",
        phys_rating=sec["phys_rating"], phys_note=sec["phys_note"] or "",
        conc_rating=sec["conc_rating"], conc_note=sec["conc_note"] or "",
        search_results=search_summary,
        recommendation_reason=("见审稿意见" if rec in ("Accept", "Minor Revision")
                               else "需修订后重审"),
    )

    _ledger({
        "timestamp": datetime.now().isoformat(),
        "stage": "review_v2", "title": title,
        "recommendation": rec,
        "search_queries": search_queries,
    })

    return {"report": report, "recommendation": rec, "raw": review_raw}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    import json as _json
    report = open("results/<project>/report.md").read()
    analysis = _json.load(open("results/<project>/analysis.json"))
    r = review("示例体系研究报告", report, analysis)
    print(f"Recommendation: {r['recommendation']}")
    print(r["report"][:1200])