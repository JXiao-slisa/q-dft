# q-dft Knowledge Base

## 计算经验规则 / Empirical Rules

### 1. 模型构建

#### 1.1 表面超胞选择
- 2×2 超胞（12-16 原子）适合快速测试
- 3×3 超胞（27-36 原子）适合低覆盖度吸附
- 真空层 ≥ 12 Å（Z 方向周期像相互作用 < 0.01 eV）
- 至少固定底部 1-2 层原子（弛豫时保持体相结构）

#### 1.2 吸附位点
- `top`: 表面原子上方，吸附高度 1.8-2.2 Å
- `fcc`: 3 个表面原子构成的空心位（fcc 金属）
- `hcp`: 3 个表面原子构成的空心位（第二层不同）
- `bridge`: 两个表面原子之间的桥位

#### 1.3 分子模型
- 孤立分子需添加 12-15 Å 真空盒子
- 线性分子（CO, N2, O2）沿 Z 轴放置简化分析

### 2. MLIP 优化

#### 2.1 MACE 最佳实践
- 模型路径：`MACE_MODEL` 环境变量或自动发现 `.cache/mace/`
- 默认用 `medium` 模型（MACE-MP-0），覆盖元素周期表大部分元素
- CPU 上使用 `float64` 精度（更准确，稍慢）
- 固定底层原子加速收敛

#### 2.2 DPA-4 最佳实践
- 模型文件：`DPA4_MODEL` 环境变量或 `models/mlip/DPA4-*.pt`
- 需要 `vesin-torch` 包（SeZM 邻域构建）
- 默认 PyTorch 后端（CPU 可用）
- 与 MACE 能量差通常在 0.1-0.2 eV 以内

### 3. DFT 计算

#### 3.1 VASP
- INCAR 默认参数适用于大多数表面吸附体系
- 需要设置 `VASP_POTCAR_DIR` 指向 POTCAR 库
- 常见收敛问题：
  - `BRMIX: did not converge` → 降低 `AMIX`/`BMIX` 或增加 `NELM`
  - 自旋极化（`ISPIN=2`）适用于磁性体系（Fe, Co, Ni, etc.）
  - 表面计算建议 `ISIF=2`（固定晶格体积）

#### 3.2 CP2K
- 基组：`DZVP-MOLOPT-SR-GTH` 双精度，适合大多数体系
- 赝势：`GTH-PBE` 与 PBE 泛函匹配
- 内存设置：`MAX_SCF 100` 适当增加

#### 3.3 ABACUS
- 赝势：ONCV PBE 格式（从 vaspkit 示例或官网下载）
- LCAO 基组需要对应的 `.orb` 文件
- 平面波基组（`basis_type pw`）只需要 `.upf` 赝势
- OpenBLAS 与 OpenMP 冲突时设置 `OMP_NUM_THREADS=4`

### 4. 后处理

#### 4.1 吸附能
- 公式：E_ad = E(slab+ads) - E(slab) - E(ads, free)
- 负值 = 放热吸附（稳定）
- PBE 泛函低估 CO/Pt(111) 吸附能（"CO/Pt puzzle"）

#### 4.2 自由能校正
- 经验值（298.15K）：
  - CO: ZPE=0.13 eV, T×S=0.61 eV
  - H₂: ZPE=0.27 eV, T×S=0.40 eV
  - H₂O: ZPE=0.56 eV, T×S=0.58 eV
  - CO₂: ZPE=0.31 eV, T×S=0.66 eV
- 精确值需频率计算（VASP IBRION=5/6）

#### 4.3 d带中心
- 越接近费米能级 → 吸附越强（Nørskov d带中心理论）
- 过渡金属 d带中心范围：-1.0 ~ -3.0 eV（相对 E_F）

### 5. 错误排除

| 错误 | 原因 | 解决 |
|------|------|------|
| VASP POTCAR I/O error | POTCAR 格式错误 | 用 `cat` 合并 POTCAR，不要用 Python 读写 |
| deepmd mpich not found | CIBUILDWHEEL=1 | 修改 `run_config.ini` 中 CIBUILDWHEEL=0 |
| MACE 模型下载失败 | 网络不可达 | 设置代理或手动下载到 `.cache/mace/` |
| CP2K "Attempt to close unit 5" | 基组文件路径错误 | 设置 `BASIS_SET_FILE_NAME` 为实际路径 |
| Conda 创建环境失败 | ~/.conda 只读 | 用 `conda create --prefix <可写路径>` |
| OpenBLAS/OpenMP hang | 线程冲突 | 设置 `OMP_NUM_THREADS=4` |

### 6. 性能参考

| 体系 | 原子数 | 引擎 | 时间 | 核数 |
|------|--------|------|------|------|
| CO 分子 (单点) | 2 | VASP | ~30s | 16 |
| CO 分子 (单点) | 2 | CP2K | ~15s | 1 |
| Pt(111) 2×2×3 | 12 | VASP | ~30min | 16 |
| Pt(111) 2×2×3 | 12 | MACE | ~5min | 1 |
| 金刚石 8 原子 | 8 | ABACUS | ~15s | 1 |
---

## 7. 表面参数收敛性测试 Rule

**R-SURF-1**: 当出现以下任一情况时，必须对表面参数（超胞大小、原子层数、真空层、k-points）做**收敛性测试**：
  - 新项目/新体系
  - 新空间群的晶体切表面（不同空间群层数/终止面不同）
  - 新晶胞参数
  - 新中间体：原子数或直径超过已有中间体数量级，或直径大于晶胞参数的 1/2

**R-SURF-2**: 收敛判据：相邻两档参数（超胞大小/层数）的吸附能差异 **< 0.1 eV** 视为收敛。2×2 适用于小吸附物（CO, H）；COOH/CHO/CH3OH 等大吸附物需测试 2×2 vs 3×3。

## 8. 数据归档与追踪 Rule

**R-ARCH-1**: 每次涉及能量/优化的计算必须有**独立文件夹**，含：
  - `init.vasp`（初始结构）
  - `opt.vasp`（优化后结构）
  - `opt.traj`（优化轨迹）
  - `optimization.log`（能量/力/步数日志）
  数据 JSON 中每条记录必须含 `dir` 字段指向对应文件夹，可回溯到结构+能量+轨迹。

**R-ARCH-2**: 每次迭代必须有**迭代日志**（`iteration_log.md`），记录：本次迭代的变更、决策、台账引用、未解决问题。文件夹命名标识模式（如 `batch1_v3_quick/`）。

## 9. MLIP 精度选择 Rule

**R-MLIP-1**: DPA4-Mini 对 3d 金属（Cu/Fe）表面吸附误差大（+8~11 eV），不推荐用于含 3d 金属的 HEA 表面。MACE-MP-0建议同时对比多个 MLIP 的 DFT 校验误差后再选型。
**R-MLIP-2**: MLIP 精度必须用 VASP/DFT 单点验证（MAE 报告），且用**相对吸附能**（ΔE_ads - ΔE_clean）而非绝对能量对比。
**R-MLIP-3**: 频率计算用 MLIP 有限差分（DFT-free），ZPE/TS 为结构敏感参数，需对每个结构分别计算。
