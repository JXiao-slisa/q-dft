# Changelog

## v0.3.0 — 2026-09-10

第二版本：面向"自主规划与迭代"的能力补全（对标 DREAMS/MatClaw 等自主 DFT Agent 的闭环能力），并修复 v0.2 验收中暴露的吸附能工作流完整性缺陷。

### 新增
* **确定性吸附能工作流 `adsorption-full`**：三点能量协议（slab / 吸附物 / slab+ads 各自"建模 → MLIP 预优化 → DFT 单点"）固化为确定性 Python 编排，`E_ad` 不再依赖 LLM 补全上下文；**无需 crewAI 与 LLM Key 即可运行**，可完全离线验收；自动输出 markdown 报告与溯源清单。同时修复 v0.2 中 `adsorption` crew 工作流 e_slab/e_adsorbate 缺失的缺陷。
* **批量筛选 `slisadft batch`**：多组合（元素×吸附物×位点）并行扫描，输出 CSV 汇总表；`POST /api/batch` 每组合生成一个确定性任务。
* **失败智能诊断**：10 类常见 DFT 失败模式（SCF 不收敛、POTCAR 缺失、ZBRENT、OOM、MPI 中止、段错误、频率无位移步…）的可执行诊断规则库（含双语修复建议与知识库溯源）；自动附加到工具错误返回与失败任务的 `diagnosis` 字段（CLI/API/Web UI 可见）。
* **跨引擎一致性校验 `engines/parity.py`**：同一结构双计算器单点对比 + 精度预算（accuracy budget）判定；把知识库中"DPA4-Mini 对某类体系误差大"类经验固化为例行 QA。
* **RSI P6-lite 假设建议**：`slisadft suggest` / `POST /api/suggest` —— 综合最近结果（强/弱吸附、合成数据、未收敛）+ 已审核知识规则 + 待审规则积压，生成带 rationale 与 source 的排序建议列表。
* 新增 pytest：工作流/批量/诊断/一致性/建议 + adsorption-full 的 API 离线端到端。

### 变更
* API `/api/system` 返回 `edition` 字段（community / professional）。
* CLI 新增 `adsorption-full` / `batch` / `suggest` 子命令。

## v0.2.0 — 2026-09-09

首个面向上线的工程化版本。核心流水线与 RSI 机制自 v0.1 起未变，重点为可靠性、可测试性与产品化交付。

### 新增
* **Web 产品形态**：FastAPI 服务（`slisadft-server`）+ 免构建静态 Web UI
  （任务提交/状态/日志/产物/3D 结构查看器、RSI P5 待审规则审核、系统状态）。
* **任务运行器**：JSONL 持久化 JobStore 状态机 + 并发子进程 JobRunner
  （沿用 DSH 插件已验证设计），支持取消与崩溃恢复。
* **Mock 引擎层**：`SLISADFT_ENGINE_MODE=mock` 在无引擎机器端到端跑通流水线；
  产物强制 SYNTHETIC 标记（Rule Zero）。
* **运行溯源清单** `run_manifest.json`：输入、引擎模式、LLM、文件哈希。
* **RSI P4**：已审核知识规则摘要注入智能体上下文（`{knowledge_context}`）。
* **RSI P5**：pending_rules 审核 API + Web 面板（approve 合并进知识库 / reject 归档）。
* **振动自由能工具** `vibrational_thermochemistry_tool`：OUTCAR 频率 → ZPE/G(T)，
  无频率数据时自动回退经验校正并显式标注来源。
* **测试套件**：pytest 覆盖 VASP 输入/解析、热化学、Mock 引擎、知识注入、
  规则审核、JobStore、REST API（无 crewai 环境自动跳过相关用例）。
* Dockerfile / docker-compose（Mock 演示开箱即用）与部署文档 `docs/DEPLOYMENT.md`。

### 修复
* **VASP 频率计算"无位移步"**：新增 `write_incar(freq=True)`，强制 IBRION=5 的
  安全默认（NSW=1、NFREE=2、POTIM=0.015、ISYM=0、EDIFF=1e-7）——此前
  单点基底 NSW=0 会让 VASP 不进入离子循环。
* LLM 端点硬编码 SenseNova → 全量环境变量化（`LLM_BASE_URL/LLM_MODEL/LLM_API_KEY`，
  默认 DeepSeek，兼容任意 OpenAI 兼容服务）。
* `uv.lock` 与 pyproject 失同步（MLIP 依赖缺失）→ 移除陈旧锁文件，
  依赖重组为核心 + `mlip`/`server`/`dev` 可选组。
* 打包元数据（crewAI 模板残留）、`.env` 泄露 Key（移出仓库，提供 `.env.example`）。

### 兼容性
* DSH 聊天插件协议不变（`run_with_trigger` 保持）；CLI 新增 argparse 子命令
  （`run/adsorption/volcano/check-env/version`），旧 `--inputs` 用法保留。
* Python 3.10–3.12；crewAI 1.14.7。

## v0.1.0 — 2026-08-30

* 核心多智能体流水线（crewAI）：建模 → MLIP(MACE/DPA-4) → DFT(VASP/CP2K/ABACUS)
  → 后处理 → 报告；三条工作流。
* RSI P1–P3：知识沉淀、自动经验抽取、pending_rules 流程。
* DSH 聊天插件 v0.0.1（24/24 验收通过）。
* 内部电催化高通量筛选研究案例（数据与结论随论文另行发表）。
