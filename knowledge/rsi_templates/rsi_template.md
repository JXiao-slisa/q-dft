# slisaDFT RSI 任务模板 (RSI Task Template)

> 此模板是 RSI 研究的标准执行框架，任何新催化剂/材料体系的 RSI 研究据此复用。
> 版本: v1.0 | 状态: active

---

## 模板结构

```yaml
rsi_project:
  id: "EXAMPLE-v1"          # 唯一ID
  system: "Cu-Ni fcc(111)"    # 研究体系
  objective: "示例：筛选表面吸附描述符"
  llm: "sensenova-6.8-flash-lite"
  mlip: ["mace", "dpa-4"]     # 仅MLIP
  dft: "none"                 # 本轮不启用DFT

batches:                      # 批次定义（每批一轮RSI）
  - id: "batch-1"
    name: "基础吸附+标度关系"
    computations:
      - {type: "structure", count: 10, detail: "10构型HEA 2x2x3"}
      - {type: "adsorption", count: 400, detail: "10构型 x 10中间体 x 4位点"}
    analyses:
      - "吸附能矩阵"
      - "scaling law"
      - "B-E-P关系"
      - "活性火山图"
    report: "V0.1"
    rsi_round: 1
    knowledge_out: ["adsorption_matrix", "scaling_params"]

  - id: "batch-2"
    name: "高通量+描述符+ML"
    computations:
      - {type: "high-throughput", count: 2000-3000, detail: "50+构型扩展"}
    analyses:
      - "描述符特征库"
      - "ML模型训练(随机森林/XGBoost)"
      - "特征重要性"
      - "全空间性能预测"
    report: "V0.2"
    rsi_round: 2
    knowledge_out: ["descriptors", "ml_model"]

  - id: "batch-3"
    name: "主动学习+MLIP优化"
    computations:
      - {type: "active-learning", count: 500-2000, detail: "不确定性采样"}
    analyses:
      - "主动学习收敛"
      - "最优催化剂识别"
      - "自由能台阶图"
      - "过电位/选择性"
    report: "V0.3"
    rsi_round: 3
    knowledge_out: ["optimized_composition", "activity_map"]

hypotheses:                   # 科学假设链
  - "H1: d带中心分布宽→吸附能可调"
  - "H2: 吸附强度决定活性趋势"
  - "H3: 描述符→火山型关系"
  - "H4: B-E-P关系成立"

rsi_loop:
  trigger: "每批次完成"
  extract: "吸附能/描述符/模型指标入库"
  feedback: "下批次设计参考前批次知识"
  stop_condition: "活性收敛或达到轮次上限"
```

---

## 执行流水线（每批次）

```
┌──────────────────────────────────────────────────────────┐
│ [输入] 上轮知识库（若无则为空）                            │
│                                                          │
│ Step 1: 模型构建（surface_tools / rsi_loop.build_system） │
│ Step 2: MLIP 计算（mace_optimize / dpa_optimize）        │
│ Step 3: 吸附能提取（thermochemistry）                    │
│ Step 4: 分析（标度关系 / 火山图 / ML训练）                │
│ Step 5: 知识入库（knowledge_extractor）                  │
│ Step 6: 报告生成（report_tools + LLM）                   │
│ Step 7: RSI 反思（LLM 总结→下批设计）                     │
│                                                          │
│ [输出] 批次报告 + 知识库更新 + 下批设计建议                │
└──────────────────────────────────────────────────────────┘
```

---

## 标准分析函数（可复用）

| 分析 | 函数 | 输入 | 输出 |
|------|------|------|------|
| 吸附能 | `adsorption_energy()` | E_slab,E_ads,E_free | 吸附能 |
| Scaling law | `fit_scaling()` | {*CO, *COOH, *CHO...} | 斜率/截距/R² |
| 火山图 | `volcano_plot()` | ΔG 台阶 | 火山型图 |
| B-E-P | `linear_relation()` | E_a vs ΔG | 斜率/R² |
| 描述符 | `feature_matrix()` | 组成/结构 | 特征表 |
| ML 训练 | `train_ml()` | X,y | 模型+指标 |

---

## 使用说明

1. 复制此模板到 `knowledge/rsi_templates/<project>.yaml`
2. 修改 `system`, `objective`, `batches` 参数
3. 按批次执行，每批完成后触发 RSI 反思并更新 `<project>.yaml`
4. 模板本身持续迭代（accumulated knowledge improves future templates）