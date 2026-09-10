# slisaDFT RSI — Recursive Self-Improvement 方案（最终版）

## 概述

RSI (Recursive Self-Improvement) 让系统在每次计算后自动学习，将经验沉淀为知识库，并反馈到 Agent 行为和插件行为中。形成**科学假设→计算→分析→科学见解→假设验证→新假设**的闭环。

---

## 闭环流程

```
① 科学假设 (Hypothesis)
   │ 用户/Agent 提出："Pt(111) 上 CO 吸附的 d带中心与吸附能的关系"
   ▼
② 计算设计 (Design)
   │ Agent 查询知识库，选择最佳参数/引擎
   ▼
③ 计算执行 (Calculate)
   │ VASP / CP2K / ABACUS / MACE / DPA-4
   ▼
④ 分析提取 (Analyze)
   │ 知识提取器自动提取能量/力/d带/错误
   ▼
⑤ 科学见解 (Insight)
   │ LLM 结合数据 + 知识库，给出物理/化学解释
   ▼
⑥ 假设验证 (Validate)
   │ 新数据与已有知识比较 → 确认/修正/拒绝假设
   │ 如果是新知识 → 写入知识库
   ▼
⑦ 新假设 (New Hypothesis)
   │ 基于见解提出下一个问题
   └────→ 回到 ①
```

---

## 知识层次

### 经验 (Experience) — 参考和推荐，自动更新

```yaml
type: experience
system: "Pt_CO"
engine: "vasp"
calculation:
  encut: 400
  kpoints: [3, 3, 1]
  converged: true
  n_ionic_steps: 57
  final_energy: -83.37
result:
  adsorption_energy: -0.0044
  d_band_center: -2.222
issues: ["PBE 泛函低估 CO/Pt(111) 吸附能"]
tags: ["vasp", "Pt", "CO", "adsorption"]
```

**自动更新规则**：每次计算完成后，知识提取器自动生成 YAML 文件写入 `knowledge/auto/`。

### 规则 (Rules) — 底层约束，需专家+用户确认

```yaml
type: rule
category: "POTCAR 合并"
description: "VASP POTCAR 必须用 cat 命令合并"
rule: "使用 subprocess.run(['cat'] + parts) 合并 POTCAR 文件"
severity: constraint  # constraint = 必须遵守 / recommendation = 建议
source: "Phase 1 测试验证"
status: pending  # pending | approved | rejected
```

**更新流程**：
1. 知识提取器发现新规则 → 写入 `knowledge/auto/pending_rules/`
2. DFT 专家 Agent 审查 → 加上 `approved` 或 `rejected` 标签
3. 用户确认 → 写入 `knowledge/computational_rules.md`
4. 版本迭代 → 旧规则保留在 YAML 历史中

---

## 对 Agent 行为的影响

### 参数自动调优
```python
# Agent 在决策前查询知识库
def suggest_parameters(system, engine):
    records = get_summary(system)
    if records.n_records > 0:
        # 使用历史最佳参数
        return records.best_parameters
    else:
        # 使用默认参数
        return DEFAULT_PARAMETERS
```

### 失败路径避免
```python
# Engine 选择时检查已知问题
def select_engine(system, elements):
    issues = get_known_issues(system)
    if "PBE 低估" in issues:
        return "RPBE"  # 自动选择更准确的泛函
```

### 插件行为控制
```python
# DSH 插件通过 RSI 接口获取计算状态
# 插件可以显示"基于历史数据，建议使用 RPBE 泛函"
```

---

## 实施阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| **P1** | 知识库静态文件 (computational_rules.md) | ✅ |
| **P2** | 自动知识提取器 + YAML 入库 | ✅ 已实现 |
| **P3** | 经验/规则区分 + pending_rules 审查流程 | ✅ 已实现 |
| **P4** | 知识库 → Agent 行为反馈（参数调优） | 待实现 |
| **P5** | 知识库 → 插件行为反馈（DSH 建议面板） | 待实现 |
| **P6** | 科学假设闭环（自动假设生成 + 验证） | 待实现 |

---

## 文件结构

```
knowledge/
├── computational_rules.md      # 静态规则（P1，需专家确认）
├── slisadft-computation/
│   └── SKILL.md                # 可调用 Skill
├── auto/                       # 自动知识库（P2，YAML 格式）
│   ├── CO/                     # 按体系组织
│   │   └── 20260825_152622_000.yaml
│   ├── Pt12/
│   │   └── 20260825_152622_000.yaml
│   ├── COPt12/
│   │   └── 20260825_152623_000.yaml
│   ├── pending_rules/          # 待审查规则
│   │   └── POTCAR_Concatenation.yaml
│   └── summary/                # 体系汇总
└── user_preference.txt         # 用户偏好
```

---

## 迭代规则版本控制

1. 每条规则有 `status` 字段（pending/approved/rejected/archived）
2. 已批准的规则写入 `computational_rules.md` 并标注版本号
3. 旧版本规则保留在 YAML 历史中，不删除
4. DFT 专家 Agent 定期审查 `pending_rules/`，给出建议
5. 用户最终确认后，规则正式生效