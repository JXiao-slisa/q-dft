"""Academic Report Agent — 生成带图文、发表级科学报告。

职责：
  1. 收集计算数据（吸附能矩阵、scaling law、BEP、火山图）
  2. 生成代表性结构图片（优化前后对比、吸附构型）
  3. 调用 LLM 撰写科学论述
  4. 组装带图文的发表级报告（含图注、表格、讨论）

输出：
  - 完整学术报告（markdown + 嵌入图片路径）
  - 四类子报告（实验/学术/检索/分析）
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def render_structure_png(vasp_path: str, output_png: str,
                         title: str = "") -> str:
    """Render a ball-and-stick structure image for reports."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from ase.io import read

    atoms = read(vasp_path)
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    colors = {
        "Pt": "#80808a", "Fe": "#b05a5a", "Co": "#3c64a0",
        "Ni": "#5a8a5a", "Cu": "#c88440", "C": "#505050",
        "O": "#e02020", "H": "#ffffff", "N": "#3c3cdc",
    }
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for s, p in zip(syms, pos):
        ax.scatter(*p, color=colors.get(s, "#888888"), s=180,
                   edgecolors="k", linewidths=0.4, depthshade=True)
    # bonds
    from itertools import combinations
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            d = np.linalg.norm(pos[i]-pos[j])
            if d < 1.8:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        [pos[i][2], pos[j][2]], color="gray", lw=0.5)
    ax.set_axis_off()
    ax.set_title(title or atoms.get_chemical_formula(), fontsize=12)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_png


def build_scaling_bep_plot(analysis: Dict, output_png: str) -> str:
    """Generate publication-quality scaling law + BEP plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scaling = analysis.get("scaling", {})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) descriptor A vs descriptor B
    ax = axes[0]
    pts = []
    for k, v in analysis.get("matrix", {}).items():
        if "CO" in v and "COOH" in v:
            pts.append((v["CO"], v["COOH"]))
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=70, c="tab:blue", edgecolors="k", zorder=3)
        if "COOH_vs_CO" in scaling:
            f = scaling["COOH_vs_CO"]
            xr = np.linspace(min(xs)-0.1, max(xs)+0.1, 20)
            ax.plot(xr, f["slope"]*xr+f["intercept"], "r--", lw=1.5,
                    label=f"slope={f['slope']}, R²={f['r2']}, n={f['n']}")
    ax.set_xlabel("E$_{ads}$(*CO) [eV]")
    ax.set_ylabel("E$_{ads}$(*COOH) [eV]")
    ax.set_title("Scaling: *COOH vs *CO (BEP)")
    ax.legend(); ax.grid(alpha=0.3)

    # (b) CHO vs CO
    ax = axes[1]
    pts = []
    for k, v in analysis.get("matrix", {}).items():
        if "CO" in v and "CHO" in v:
            pts.append((v["CO"], v["CHO"]))
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=70, c="tab:green", edgecolors="k", zorder=3)
        if "CHO_vs_CO" in scaling:
            f = scaling["CHO_vs_CO"]
            xr = np.linspace(min(xs)-0.1, max(xs)+0.1, 20)
            ax.plot(xr, f["slope"]*xr+f["intercept"], "r--", lw=1.5,
                    label=f"slope={f['slope']}, R²={f['r2']}, n={f['n']}")
    ax.set_xlabel("E$_{ads}$(*CO) [eV]")
    ax.set_ylabel("E$_{ads}$(*CHO) [eV]")
    ax.set_title("Scaling: *CHO vs *CO")
    ax.legend(); ax.grid(alpha=0.3)

    # (c) BEP: H vs CO
    ax = axes[2]
    pts = []
    for k, v in analysis.get("matrix", {}).items():
        if "CO" in v and "H" in v:
            pts.append((v["CO"], v["H"]))
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=70, c="tab:orange", edgecolors="k", zorder=3)
        if "H_vs_CO" in scaling:
            f = scaling["H_vs_CO"]
            xr = np.linspace(min(xs)-0.1, max(xs)+0.1, 20)
            ax.plot(xr, f["slope"]*xr+f["intercept"], "r--", lw=1.5,
                    label=f"slope={f['slope']}, R²={f['r2']}, n={f['n']}")
    ax.set_xlabel("E$_{ads}$(*CO) [eV]")
    ax.set_ylabel("E$_{ads}$(*H) [eV]")
    ax.set_title("BEP: *H vs *CO (HER correlation)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def llm_write(prompt: str, system: str = "你是资深计算催化领域科学家，撰写发表级学术论文。",
              model: str = "sensenova-6.8-flash-lite") -> str:
    """Call SenseNova LLM for report writing."""
    import requests
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


def generate_report(workdir: str, analysis: Dict, structures_dir: str,
                    title: str = "CO2RR 高熵电催化剂研究") -> Dict:
    """Generate a publication-quality academic report with figures."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Structure figures (representative)
    fig_structs = []
    if Path(structures_dir).exists():
        # pick clean surface + key adsorbate opt
        for pat in ["*clean.vasp", "*CO*_opt.vasp", "*COOH*_opt.vasp"]:
            for f in sorted(Path(structures_dir).glob(pat))[:1]:
                out = workdir / f"fig_struct_{f.stem}.png"
                try:
                    render_structure_png(str(f), str(out), title=f.stem)
                    fig_structs.append(out.name)
                except Exception:
                    pass

    # 2. Scaling/BEP plot
    fig_scaling = workdir / "fig_scaling_bep.png"
    build_scaling_bep_plot(analysis, str(fig_scaling))

    # 3. Compose report via LLM
    scaling_summary = json.dumps(analysis.get("scaling", {}), ensure_ascii=False, indent=1)
    volcano_summary = json.dumps(analysis.get("volcano", {}), ensure_ascii=False, indent=1)[:2000]

    prompt = f"""请撰写一篇发表级的中文学术报告，主题：{title}。

## 数据
### Scaling Laws
{scaling_summary}

### 火山型/活性数据
{volcano_summary}

## 报告结构要求
1. 摘要
2. 引言（高熵合金CO2RR背景）
3. 计算方法（MLIP/MACE/DPA-4）
4. 结果与讨论
   - Scaling law 分析（引用图fig_scaling_bep.png）
   - 代表性结构（引用结构图）
   - 活性位点分析
5. 结论
请引用文中图片：![结构图](fig_struct_xxx.png) 形式。文字专业、数据准确、讨论深入。"""

    report_text = llm_write(prompt)
    report_path = workdir / "report_academic.md"
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "report": str(report_path),
        "scaling_fig": str(fig_scaling),
        "structure_figs": fig_structs,
        "report_length": len(report_text),
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    import json as _json
    a = _json.load(open("results/<project>/analysis.json"))
    r = generate_report("results/<project>/reports",
                         a, "results/<project>/structures")
    print("Report:", r["report"])
    print("Figures:", r["structure_figs"])