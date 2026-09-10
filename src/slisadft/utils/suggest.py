"""RSI P6-lite：假设建议引擎（自动提出下一个最有信息量的计算）。

规则驱动的"假设闭环"第一步：综合三类证据生成结构化建议——
  1. 最近一次工作流结果（能量、收敛、体系）；
  2. 知识库中的已审核经验规则（knowledge/auto/*.yaml + computational_rules.md）；
  3. 待审规则积压状态（pending_rules/）。

每条建议都携带 rationale 与 source（引用触发它的结果字段或规则文件），
保证可审计。后续版本可将建议自动转化为 crew 假设任务（P6 完整版）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from .knowledge_context import _rules_path

_SITE_CYCLE = {"top": ["fcc", "bridge"], "fcc": ["top", "hcp"],
               "bridge": ["top", "fcc"], "hcp": ["top", "fcc"]}


def _pending_count(knowledge_dir: Optional[str]) -> int:
    import os
    root = Path(knowledge_dir) if knowledge_dir else (
        Path(os.environ.get("SLISADFT_KNOWLEDGE_DIR",
                            Path(__file__).resolve().parents[3] / "knowledge")))
    pending = root / "auto" / "pending_rules"
    return len(list(pending.glob("*.yaml"))) if pending.is_dir() else 0


def _knowledge_hints(knowledge_dir: Optional[str]) -> List[str]:
    """Extract short actionable hints from reviewed rules (keyword gated)."""
    hints: List[str] = []
    path = _rules_path() if not knowledge_dir else Path(knowledge_dir) / "computational_rules.md"
    if not path.is_file():
        return hints
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return hints
    if re.search(r"DPA4-?Mini.{0,40}(不准|inaccurate|误差)", text, re.IGNORECASE):
        hints.append("知识库记载 DPA4-Mini 对该类体系误差较大：改用 MACE 或以 DFT 校验。")
    if re.search(r"(频率|frequency).{0,60}(NSW|IBRION)", text, re.IGNORECASE):
        hints.append("自由能精度不足时：对吸附态做 IBRION=5 频率计算（slisadft 已内置安全默认）。")
    if re.search(r"(scaling|标度|关系)", text, re.IGNORECASE):
        hints.append("可利用知识库中的标度关系类规则，由易算描述符预估难算能垒。")
    return hints


def suggest_next(last_result: Optional[dict] = None,
                 knowledge_dir: Optional[str] = None,
                 max_suggestions: int = 6) -> List[Dict[str, object]]:
    """Generate ranked next-step hypotheses for the research loop.

    Args:
        last_result: the dict returned by run_adsorption_full (optional).
        knowledge_dir: override knowledge base location (tests).
        max_suggestions: cap the list length.

    Returns:
        List of {"action", "rationale", "priority", "source"} dicts,
        sorted by priority ("high" first).
    """
    suggestions: List[Dict[str, object]] = []

    # ── 证据 1：最近一次结果 ─────────────────────────────────────────
    if last_result:
        inputs = last_result.get("inputs", {}) or {}
        e_ad = last_result.get("adsorption_energy_eV")
        element = inputs.get("element", "Pt")
        site = inputs.get("site", "top")

        if e_ad is not None:
            e_ad = float(e_ad)
            if e_ad <= -1.5:
                suggestions.append({
                    "action": f"site_scan: 在 {element} 表面扫描相邻位点 "
                              f"（建议位点：{', '.join(_SITE_CYCLE.get(str(site), ['fcc', 'bridge']))}）",
                    "rationale": f"当前 {site} 位点 E_ad={e_ad:.2f} eV 属强吸附，"
                                 "位点敏感性可能显著，需确认最稳定位点。",
                    "priority": "high",
                    "source": "result: adsorption_energy_eV",
                })
            elif e_ad >= -0.3:
                suggestions.append({
                    "action": f"change_exploration: 当前结合较弱（E_ad={e_ad:.2f} eV），"
                              "建议换更活泼表面（如添加过渡金属掺杂）或提高覆盖度",
                    "rationale": "弱吸附体系对催化剂活性贡献有限，探索更强结合环境。",
                    "priority": "medium",
                    "source": "result: adsorption_energy_eV",
                })
            else:
                suggestions.append({
                    "action": f"composition_scan: 在相近结合强度（E_ad={e_ad:.2f} eV）下"
                              "扫描相邻元素/合金组分以寻找火山图顶点",
                    "rationale": "中等吸附强度处于火山图活跃区间，组分扫描收益最大。",
                    "priority": "high",
                    "source": "result: adsorption_energy_eV",
                })

        if last_result.get("synthetic"):
            suggestions.append({
                "action": "switch_to_real_engine: 当前结果为 SYNTHETIC 合成数据，"
                          "正式研究必须在配置好引擎的集群上以 real 模式重跑",
                "rationale": "Rule Zero：mock 数据不得作为科研结论。",
                "priority": "high",
                "source": "result: engine_mode/synthetic",
            })

        if last_result.get("gibbs_correction", {}).get("adsorbate_zpe_eV", 0.0):
            suggestions.append({
                "action": "vibrational_analysis: 对吸附态做频率计算以获得真实 ZPE/熵"
                          "（替代经验校正）",
                "rationale": "经验校正不确定度 ~0.1 eV 量级，频率计算可消除该项。",
                "priority": "medium",
                "source": "result: gibbs_correction.note",
            })

        details = last_result.get("system_details", {}) or {}
        unconverged = [k for k, v in details.items()
                       if isinstance(v, dict) and v.get("dft_converged") is False]
        if unconverged:
            suggestions.append({
                "action": f"retry_converged: 体系 {', '.join(unconverged)} DFT 未收敛，"
                          "收紧 EDIFF/NELM 后重算",
                "rationale": "未收敛能量不可用于 E_ad（Rule Zero）。",
                "priority": "high",
                "source": "result: system_details.*.dft_converged",
            })

    # ── 证据 2：知识库经验 ───────────────────────────────────────────
    for i, hint in enumerate(_knowledge_hints(knowledge_dir)):
        suggestions.append({
            "action": hint,
            "rationale": "来自已审核知识库的匹配规则。",
            "priority": "medium" if i else "high",
            "source": "knowledge: computational_rules.md",
        })

    # ── 证据 3：待审规则积压 ─────────────────────────────────────────
    n_pending = _pending_count(knowledge_dir)
    if n_pending:
        suggestions.append({
            "action": f"review_pending_rules: 有 {n_pending} 条自动抽取规则待审核"
                      "（商业版 Web 面板 / rsi_review API）",
            "rationale": "及时审核可让经验进入知识库反哺后续计算（RSI P4）。",
            "priority": "low" if n_pending < 5 else "medium",
            "source": "knowledge: auto/pending_rules/",
        })

    if not suggestions:
        suggestions.append({
            "action": "start: 无历史结果与知识线索，建议先运行 "
                      "slisadft adsorption-full --inputs '{\"element\":\"Pt\",\"adsorbate\":\"CO\"}'",
            "rationale": "冷启动建议。",
            "priority": "medium",
            "source": "cold-start",
        })

    suggestions.sort(key=lambda s: {"high": 0, "medium": 1, "low": 2}[s["priority"]])
    return suggestions[:max_suggestions]
