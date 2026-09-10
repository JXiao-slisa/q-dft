---
name: slisadft-computation
description: "slisaDFT 计算平台的专业知识：DFT 计算参数选择、MLIP 优化、VASP/CP2K/ABACUS 引擎配置、后处理分析、常见错误排除。调用此 Skill 后 agent 可获得完整的 DFT 计算决策支持。"
---

# slisaDFT Computation Knowledge

## 概览

此 Skill 包含了 slisaDFT 计算平台在开发/测试过程中积累的全部经验知识，涵盖：

- 模型构建的最佳实践（超胞选择、吸附位点、真空层）
- MLIP 优化（MACE、DPA-4）的配置参数和模型路径
- DFT 引擎（VASP、CP2K、ABACUS）的输入文件模板和参数选择
- 后处理分析（吸附能、自由能、d带中心）的公式和校正
- 常见错误的诊断和修复方法

## 使用方式

Agent 遇到 DFT 计算相关问题时，按以下优先级查找：

1. 计算参数选择 → 查阅 `computational_rules.md` 中的参数表
2. 错误排除 → 查阅 `computational_rules.md` 第 5 节
3. 具体引擎配置 → 查阅 `tools/` 下的对应工具代码
4. 经验值校正 → 查阅 `utils/thermochemistry.py` 中 `EMPIRICAL_GAS_CORRECTIONS`

## 关键路径

- `knowledge/computational_rules.md` — 核心知识库
- `src/slisadft/utils/` — 工具函数实现
- `src/slisadft/tools/` — CrewAI 工具定义
- `PRINCIPLES.md` — 核心规则（不接受编造数据）
- `test_output/test_report.md` — 验证记录