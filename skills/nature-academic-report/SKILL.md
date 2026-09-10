---
name: nature-academic-report
description: "生成 Nature 级别发表级学术报告：调用学术报告 agent 生成带图文的科学报告、代表性结构图、Scaling Law/BEP 图，并按学术期刊标准输出。适用于催化/材料/DFT 计算结果的高质量报告撰写。"
---

# Nature Academic Report Skill

## 用途

将 DFT/MLIP 计算结果转化为发表级学术报告，包含：
- 代表性结构图（3D 球棍模型）
- Scaling Law / BEP 关系图
- 学术语言论述（LLM 生成）
- 标准学术报告结构（摘要/引言/方法/结果/讨论/结论）

## 调用方式

使用 `academic_report_agent.py`：
```python
from slisadft.utils.academic_report_agent import (
    generate_report, render_structure_png,
    build_scaling_bep_plot, llm_write,
)
```

### 1. 生成代表性结构图
```python
render_structure_png('results/.../hea_0_clean.vasp',
                     'reports/fig_hea_surface.png', 'HEA surface')
```

### 2. 生成 Scaling Law / BEP 图
```python
build_scaling_bep_plot(analysis, 'reports/fig_scaling_bep.png')
```

### 3. LLM 撰写学术论述
```python
text = llm_write(prompt, system='你是资深计算催化领域科学家...')
```

### 4. 完整报告生成
```python
result = generate_report(workdir, analysis, structures_dir, title)
```

## 报告结构模板

```markdown
# 标题
## 摘要
## 1. 引言
## 2. 计算方法
## 3. 结果与讨论
   - 3.1 结构与吸附构型（图1）
   - 3.2 Scaling Law 与 BEP（图2）
   - 3.3 活性位点分析
## 4. 结论
```

## 图片引用规范

报告中用 Markdown 图片语法引用：
```markdown
![HEA表面](fig_hea_surface.png)
![CO吸附](fig_CO_ads.png)
![Scaling/BEP](fig_scaling_bep.png)
```

## 质量要求

1. 每张图必须有图注（Figure caption）
2. 文字必须数据准确、物理深入、讨论充分
3. 引用图的位置与文字讨论对应
4. 结论必须由数据支撑