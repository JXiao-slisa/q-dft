"""Reviewer Agent V3 — 点对点 (point-by-point) 标准审稿。

对报告中每个具体问题逐条给出意见，不再笼统。
结构：
  - 逐条 Major/Minor 意见（编号，对应报告具体位置）
  - 每条附：严重度、位置/证据、修改要求
  - 科学严谨性逐维评级
  - 最终建议
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests

LEDGER_PATH = Path(__file__).resolve().parents[2] / "results" / "<project>" / "RSI" / "logs" / "rsi_ledger.jsonl"


def _llm(prompt: str, system: str, model: str = "sensenova-6.8-flash-lite") -> str:
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


def review_point_by_point(title: str, report_md: str, analysis: Dict,
                          figures: List[str]) -> Dict:
    """Point-by-point review with structured per-item comments."""
    fig_list = "\n".join(f"- {f}" for f in figures)

    prompt = f"""请对学术稿件《{title}》进行点对点(Point-by-Point)学术审稿。

## 稿件
{report_md[:5000]}

## 结构化数据
{json.dumps(analysis, ensure_ascii=False, indent=1)[:2500]}

## 报告图片
{fig_list}

请严格按以下格式逐条输出意见（每条都必须具体、可操作、指向明确位置）：

MAJOR-1: [意见内容] | 严重度(高/中/低) | 位置: [报告哪部分/哪个图] | 修改要求: [具体怎么做]
MAJOR-2: ...
MINOR-1: ...
MINOR-2: ...

然后输出：
METHOD_RATING: Excellent/Good/Fair/Poor (理由)
DATA_RATING: ...
STAT_RATING: ...
PHYS_RATING: ...
CONCLUSION_RATING: ...

FINAL: Accept/Minor Revision/Major Revision/Reject (理由)"""

    raw = _llm(prompt, "你是资深学术审稿人，严格、点对点、建设性。")

    # Parse
    major = re.findall(r"MAJOR-\d+:\s*(.*?)(?=(?:MAJOR-\d+:|MINOR-\d+:|METHOD_RATING:))", raw, re.S)
    minor = re.findall(r"MINOR-\d+:\s*(.*?)(?=(?:MAJOR-\d+:|MINOR-\d+:|METHOD_RATING:))", raw, re.S)
    rating = {}
    for key in ["METHOD_RATING","DATA_RATING","STAT_RATING","PHYS_RATING","CONCLUSION_RATING","FINAL"]:
        m = re.search(rf"{key}:\s*(.+)", raw)
        rating[key] = m.group(1).strip() if m else ""

    rec = "Major Revision"
    if "Accept" in rating.get("FINAL",""):
        rec = "Accept"
    elif "Minor Revision" in rating.get("FINAL",""):
        rec = "Minor Revision"
    elif "Reject" in rating.get("FINAL",""):
        rec = "Reject"

    # Build report
    major_txt = "\n".join(f"{i+1}. {m.strip()}" for i, m in enumerate(major) if m.strip()) or "无"
    minor_txt = "\n".join(f"{i+1}. {m.strip()}" for i, m in enumerate(minor) if m.strip()) or "无"

    report = f"""# 审稿意见（点对点）

**稿件：** {title}
**审稿人：** Reviewer Agent V3 | **日期：** {datetime.now().strftime('%Y-%m-%d')}
**建议：** {rec}

---

## 一、Major Comments（主要问题，逐条）
{major_txt}

## 二、Minor Comments（次要问题，逐条）
{minor_txt}

## 三、科学严谨性逐维评级
| 维度 | 评级 | 审稿说明 |
|------|------|----------|
| 方法 | {rating.get('METHOD_RATING','')} | |
| 数据 | {rating.get('DATA_RATING','')} | |
| 统计 | {rating.get('STAT_RATING','')} | |
| 物理 | {rating.get('PHYS_RATING','')} | |
| 结论 | {rating.get('CONCLUSION_RATING','')} | |

## 四、最终建议
**{rec}** — {rating.get('FINAL','')}

## 五、审稿原始意见（全文）
{raw}
"""
    _ledger({
        "timestamp": datetime.now().isoformat(),
        "stage": "review_v3_point_by_point",
        "title": title, "recommendation": rec,
        "n_major": len([m for m in major if m.strip()]),
        "n_minor": len([m for m in minor if m.strip()]),
    })
    return {"report": report, "recommendation": rec, "raw": raw,
            "major": [m.strip() for m in major if m.strip()],
            "minor": [m.strip() for m in minor if m.strip()]}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    import json as _json
    report = open("results/<project>/reports/report.md").read()
    analysis = _json.load(open("results/<project>/analysis.json"))
    figs = ["fig1_surface.png","fig2_ads.png","fig3_ads.png","free_energy.png"]
    r = review_point_by_point("示例 HEA 体系", report, analysis, figs)
    print(f"Recommendation: {r['recommendation']}, major: {len(r['major'])}, minor: {len(r['minor'])}")
    print("\n".join(r['major'][:5]))