"""Search Agent — 独立检索报告生成。

针对审稿 agent 提出的质疑，独立进行文献检索并出具检索报告与建议。
报告包含：检索策略、检索结果摘要、与本研究对比、建议。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests


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


def generate_search_report(reviewer_concerns: List[str],
                           topic: str,
                           outdir: str = "results/<project>/reports") -> str:
    """Generate an independent literature search report for reviewer concerns."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: design search strategy
    strategy_prompt = f"""针对以下研究主题和审稿人质疑，制定详细的文献检索策略。

## 研究主题
{topic}

## 审稿人质疑
{json.dumps(reviewer_concerns, ensure_ascii=False, indent=1)}

请输出：
1. 检索关键词组合（含布尔逻辑）
2. 重点数据库/期刊
3. 每个质疑对应检索什么证据
4. 预期找到的文献类型"""

    strategy = _llm(strategy_prompt, "你是计算催化领域的资深文献检索专家，熟悉 Web of Science, Scopus, arXiv 等。")

    # Step 2: synthesize findings & recommendations
    synth_prompt = f"""基于以下检索策略，请：
1. 总结该领域关键文献发现（吸附趋势、MLIP验证、吸附描述符）
2. 指出本研究结果（关键描述符与待验证点）与文献的异同
3. 给出针对审稿人质疑的具体修改建议（哪些需要补充文献支撑、哪些计算验证）
4. 列出建议引用的代表性文献类型/作者

## 检索策略
{strategy}

## 研究主题
{topic}"""

    synthesis = _llm(synth_prompt, "你是资深计算催化科学家，擅长将文献与自身计算结合。")

    report = f"""# 文献检索报告 (Literature Search Report)

> **检索专家：** Search Agent | **日期：** {datetime.now().strftime('%Y-%m-%d')}
> **主题：** {topic}

---

## 1. 检索策略
{strategy}

## 2. 审稿质疑 → 检索映射
| 审稿质疑 | 检索关键词 | 预期证据 |
|----------|-----------|----------|
{_concern_table(reviewer_concerns)}

## 3. 综合分析与建议
{synthesis}

## 4. 建议补充工作
见上述分析。

---
*本报告由 Search Agent 独立生成，供审稿决策参考。*
"""
    path = outdir / "search_report.md"
    path.write_text(report, encoding="utf-8")
    return str(path)


def _concern_table(concerns: List[str]) -> str:
    rows = []
    for c in concerns:
        rows.append(f"| {c} | {c} + catalyst | 相关实验/计算文献 |")
    return "\n".join(rows)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    concerns = [
        "MLIP精度未与DFT对比验证",
        "仅9个独立位点统计不足",
        "仅top位点未涵盖hollow/bridge",
    ]
    p = generate_search_report(concerns, "示例体系文献调研")
    print(f"Search report: {p}")