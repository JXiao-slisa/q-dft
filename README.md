<div align="center">

# ⚛️ q-dft agent

### 🚀 自主进化的 AI 多智能体 · 第一性原理计算催化研究系统

**Quantum · Autonomous · Trustworthy**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![crewAI](https://img.shields.io/badge/multi--agent-crewAI-FF4F5B)](https://github.com/crewAIInc/crewAI)
[![Tests](https://img.shields.io/badge/tests-60%20passing-3FB950)](tests)
[![Mock Mode](https://img.shields.io/badge/offline-mock%20engine-8B5CF6)](#-30-秒极速体验)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-ff69b4)](CONTRIBUTING.md)

*让 LLM 智能体替你跑 DFT，让每一张图都可溯源。*

`建模` → `MLIP 预优化` → `DFT 计算` → `后处理分析` → `研究报告`，全自动。

</div>

---

## ⚡ 这是什么

**q-dft agent** 是一个开源（Apache-2.0）的 **AI 多智能体第一性原理计算平台**。
五个领域专家智能体组成数字科研团队，自主完成从晶体表面建模到学术论文级
研究报告的完整 DFT 计算催化流程 —— 而 **Rule Zero 机制** 保证它永远不会
编造一个数字。

```text
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
  │ 🔬 Surface  │──▶│ ⚡ MLIP     │──▶│ 🧮 DFT      │──▶│ 📊 Postproc  │──▶│ 📝 Report   │
  │ Scientist   │   │ Optimizer   │   │ Engineer    │   │ Analyst      │   │ Writer      │
  │ 建模/吸附位  │   │ MACE / DPA  │   │VASP/CP2K/   │   │E_ad/自由能/  │   │ 中英文报告   │
  │             │   │ 预优化      │   │ABACUS+SLURM │   │DOS/d带/火山图│   │             │
  └─────────────┘   └─────────────┘   └─────────────┘   └──────────────┘   └─────────────┘
        ▲                                                                        │
        └────────────── 🧠 RSI 递归自我改进：经验 → 规则 → 反哺 ◀────────────────┘
```

## 🔥 核心黑科技

| | 能力 | 说明 |
|---|---|---|
| 🛡️ | **Rule Zero 反造假** | 每个能量强制溯源到真实计算文件；Mock 数据强制 `SYNTHETIC` 水印；失败绝不伪装成功 |
| 🧠 | **RSI 递归自我改进** | 每次计算自动抽取经验 → 人工审核沉淀为规则 → 自动注入智能体上下文 —— **越用越聪明** |
| 🎯 | **确定性协议层** | 三点吸附能协议固化为纯代码，不依赖 LLM 编排 —— 数值完备性不赌运气 |
| 🔌 | **Real / Mock 双模引擎** | VASP · CP2K · ABACUS + MACE · DPA-4；无引擎机器一键 Mock 模式，全流程离线跑通 |
| 🧬 | **失败智能诊断** | 10 类 DFT 失败模式自动识别，秒级给出修复建议（SCF 不收敛？POTCAR 缺失？） |
| 🔭 | **假设建议引擎** | 根据结果 + 知识库自动提出下一个最有信息量的计算 —— 科研闭环的第一块拼图 |
| 🌐 | **任意 LLM 后端** | DeepSeek / Qwen / OpenAI / SenseNova / vLLM —— 一个环境变量切换 |

## ⚙️ 安装

```bash
git clone https://github.com/JXiao-slisa/q-dft.git
cd q-dft
pip install -e .            # 核心
pip install -e ".[mlip]"    # 真实计算模式（MACE / DPA-4，建议先装 PyTorch）
pip install -e ".[dev]"     # 开发测试
```

要求：Python ≥ 3.10, < 3.13（推荐 3.11）。

```bash
cp .env.example .env        # 填入 LLM_API_KEY（任意 OpenAI 兼容端点）
```

## ⚡ 30 秒极速体验

无需 VASP、无需 GPU、无需集群 —— Mock 引擎全流程离线演示：

```bash
qdft check-env                                       # 引擎自检
qdft adsorption-full --inputs '{"element": "Pt", "adsorbate": "CO"}'
# ✅ 三点吸附能协议：build → MLIP → DFT ×3 → E_ad（12 秒，SYNTHETIC 水印）

qdft batch --inputs-file combos.json --out results.csv   # 高通量筛选
qdft suggest                                         # AI 提出下一个实验假设
```

> ⚠️ **Rule Zero**：Mock 模式输出全部为合成数据（强制水印），仅用于流程演示与
> 开发测试，严禁作为科研结论。正式研究请配置真实引擎（[部署指南](docs/DEPLOYMENT.md)）。

## 🖥️ 真实计算（集群模式）

```ini
# .env
SLISADFT_ENGINE_MODE=real
VASP_EXE=/opt/vasp/vasp.6.5.1/bin/vasp_std
VASP_POTCAR_DIR=/opt/vasp/potpaw_PBE
SLURM_PARTITION=CPU
MPI_RUN=srun
```

```bash
qdft run --inputs '{"element":"Pt","miller":"(111)","adsorbate":"CO","site":"top","dft_calculator":"vasp"}'
```

完整 SLURM/开源引擎接入见 **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**。

## 🧠 RSI：让智能体随你的研究进化

```text
计算完成 ──▶ 自动抽取经验(YAML) ──▶ pending_rules/ ──▶ 人工审核
                        ▲                                   │
                        └── 注入智能体上下文 ◀── knowledge/ ◀─┘
```

- 经验可审计：每条规则保留来源计算与审核记录
- 反哺可配置：`{knowledge_context}` 自动注入建模/DFT 智能体
- API 可编程：`slisadft.utils.rsi_review.approve_rule(...)` 脚本化审核

## 🧪 质量保障

```bash
SLISADFT_ENGINE_MODE=mock pytest    # 60 项测试全绿，全程离线
ruff check src/slisadft             # 代码规范
```

测试矩阵覆盖：VASP 输入/解析、热化学（ZPE/CHE）、Mock 引擎确定性、
三点协议一致性、批量 CSV、失败诊断、一致性 QA、知识注入、规则审核（含
路径穿越防护）、端到端工作流。

## 🗺️ 路线图

- [x] 多智能体流水线 + 三条工作流
- [x] Real/Mock 双模引擎 + SYNTHETIC 水印
- [x] RSI 经验沉淀 / 审核流 / 上下文反哺 / 假设建议
- [x] 确定性三点吸附能协议 · 批量筛选 · 失败智能诊断 · 一致性 QA
- [ ] P6 完整假设闭环（建议 → 自动执行 → 自动验收）
- [ ] 主动学习驱动的 CO2RR 高通量扫描
- [ ] 商业版：Web 界面 + REST API + 任务运行器（[联系获取](#-license)）

## 🤝 贡献

欢迎 PR！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。新功能请附带 mock
模式可运行的测试 —— 守护 Rule Zero，人人有责。

## 📄 License

[Apache-2.0](LICENSE)。VASP 为第三方商业软件，需用户自行持有许可证，
本项目不捆绑不分发（见 [LICENSES/THIRD_PARTY.md](LICENSES/THIRD_PARTY.md)）。

---

<div align="center">

**q-dft agent** · *Trustworthy autonomous DFT research, one commit at a time.*

⚡ 如果这个项目对你有帮助，请点一个 Star ⭐

</div>
