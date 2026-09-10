# slisaDFT 部署指南

slisaDFT 支持三种部署形态，按基础设施从少到多排列：

| 形态 | 适用场景 | 引擎要求 |
|---|---|---|
| Mock 演示（Docker/本机） | 体验流程、开发、演示 | 无（合成数据，SYNTHETIC 标记） |
| 单机 real 模式 | 单台 Linux 服务器 + 开源引擎 | CP2K / ABACUS + MPI |
| 集群 real 模式 | HPC + SLURM | VASP（商业授权）/ CP2K / ABACUS |

> **Rule Zero 提醒**：Mock 模式的所有输出为合成数据（结果带 `SYNTHETIC`
> 标记、`engine_mode="mock"`），只能用于演示与测试，严禁写入任何研究报告。

---

## 0. 环境准备（所有形态通用）

Python ≥ 3.10, < 3.13（推荐 3.11）。

```bash
git clone <your-repo-url> slisa && cd slisa
pip install -e ".[server]"       # Web 服务
# 真实计算模式追加：
pip install -e ".[mlip]"         # MACE / DPA-4（建议先装 PyTorch）
cp .env.example .env             # 填写 LLM Key 等配置
```

`.env` 关键项：

```ini
LLM_BASE_URL=https://api.deepseek.com/v1   # 任意 OpenAI 兼容端点
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-xxx
SLISADFT_ENGINE_MODE=real                  # 或 mock
SLISA_SERVER_TOKEN=<长随机串>               # Web 服务鉴权（生产必须设置）
SLISA_MAX_CONCURRENT_JOBS=2
```

## 1. Mock 演示（Docker）

```bash
docker compose up --build
# 打开 http://localhost:8600
```

无需任何 DFT 引擎即可提交任务、查看结果与规则审核界面。

## 2. 单机 real 模式（开源引擎）

前提：CP2K 或 ABACUS 可执行文件 + MPI（可选 MACE CPU/GPU）。

```bash
# .env 示例
SLISADFT_ENGINE_MODE=real
CP2K_EXE=/opt/cp2k/exe/Linux-x86-64-gfortran/cp2k.popt
# 或 ABACUS_EXE=/opt/abacus/bin/abacus
MPI_RUN=mpirun
```

启动服务：

```bash
slisadft-server                    # 或 python -m slisadft.api.app
```

自检：

```bash
qdft check-env                 # 报告各引擎可用性
qdft run --inputs '{"element":"Pt","adsorbate":"CO"}'
```

MLIP 模型：

* MACE：首次调用自动下载 MACE-MP-0（需外网），或 `MACE_MODEL=/path/to/model`；
* DPA-4：下载 DPA4 权重放入 `models/mlip/` 或设 `DPA4_MODEL=/path/to/dpa-4.pth`
  （仓库不分发权重文件）。

## 3. SLURM 集群 real 模式（VASP）

前提：登录/计算节点装有 SLURM、MPI、VASP（商业软件，自行持有许可证）与 POTCAR 库。

```ini
# .env
SLISADFT_ENGINE_MODE=real
VASP_EXE=/opt/vasp/vasp.6.5.1/bin/vasp_std
VASP_POTCAR_DIR=/opt/vasp/potpaw_PBE
SLURM_PARTITION=CPU
SLURM_NTASKS=16
SLURM_MEM_MB=6000
SLURM_TIME=24:00:00
MPI_RUN=srun                      # 或 mpirun
MPI_ENV_SCRIPT=/opt/intel/oneapi/2024.1/oneapi-vars.sh
MPI_EXTRA_PATH=/opt/vasp/vasp.6.5.1/bin
```

工作方式：任务通过 `sbatch` 提交（无 SLURM 时自动退回本地子进程），
`utils/slurm.py` 轮询作业状态直至完成，再解析 OUTCAR/OSZICAR。

### Web 服务与集群的部署拓扑

* **同机部署（推荐起步）**：`slisadft-server` 跑在集群登录节点（`SLISA_MAX_CONCURRENT_JOBS`
  控制并发，勿超过分区配额）。
* **分离部署**：Web 服务器与计算节点分离时，把任务队列目录放共享存储
  （`SLISA_JOBS_DIR=/shared/jobs`），或在计算节点上以 CLI 方式消费任务。
  REST 转发可由 Nginx 完成：

```nginx
location / {
    proxy_pass http://127.0.0.1:8600;
    proxy_set_header Authorization $http_authorization;
    client_max_body_size 64m;
}
```

## 4. 运维要点

* **鉴权**：生产必须设置 `SLISA_SERVER_TOKEN`；所有 `/api` 路由要求
  `Authorization: Bearer <token>`（产物下载链接也支持 `?token=` 查询参数）。
* **数据目录**：`SLISA_JOBS_DIR`（默认 `./jobs`）保存 JSONL 台账、日志、
  产物与 `run_manifest.json`（溯源清单）——定期备份。
* **知识库**：`knowledge/` 目录即 RSI 数据库；`pending_rules/` 的审核动作
  会改写 `computational_rules.md`，建议纳入 git 管理。
* **LLM**：更换供应商只需改 `LLM_BASE_URL/LLM_MODEL/LLM_API_KEY` 三个变量。
* **升级**：`git pull && pip install -e . && systemctl restart qdft`。

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| 提交任务报 "VASP not found" | `VASP_EXE` 路径不对或登录节点无权限，用 `qdft check-env` 诊断 |
| 任务卡在 running | 看任务详情的运行日志；SLURM 队列可用 `squeue` 核对 |
| 报告里数据标了 SYNTHETIC | 引擎处于 mock 模式，正式研究必须 `SLISADFT_ENGINE_MODE=real` |
| POTCAR generation failed | `VASP_POTCAR_DIR` 未指向包含 `<元素>/POTCAR` 结构的库 |
| MACE 下载失败 | 离线环境手动下载模型后设置 `MACE_MODEL` |
