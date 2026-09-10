"""失败智能诊断（failure intelligence）。

将 `knowledge/computational_rules.md` 错误排除表中的经验固化为可执行的
正则规则库：当 DFT/MLIP 计算失败（工具 error、run.log、OUTCAR 报错）时，
自动匹配失败模式并给出可操作建议。诊断结果附加到工具返回、任务记录与
API 错误信息中，缩短"失败 -> 修复"的闭环。

规则均可追溯到知识库条目（source 字段），符合 Rule Zero 的可审计要求。
"""

from __future__ import annotations

import re
from typing import Dict, List

# (code, compiled regex, severity, advice_zh, source)
_DIAGNOSIS_RULES = [
    (
        "SCF_NOT_CONVERGED",
        re.compile(r"(NELM|SCF not converged|SCF run NOT converged|"
                   r"convergence NOT achieved)", re.IGNORECASE),
        "high",
        "电子步不收敛：增大 NELM/MAX_SCF，改 ALGO=All/Damped，调整 ISMEAR/SIGMA "
        "（小分子体系用 Sigma=0.05）或检查是否有真空层电荷翻转。",
        "knowledge: 5. 错误排除 / SCF",
    ),
    (
        "POTCAR_MISSING",
        re.compile(r"(POTCAR|pseudopotential.{0,20}not found|no POTCAR)", re.IGNORECASE),
        "high",
        "POTCAR 缺失或不匹配：检查 VASP_POTCAR_DIR 是否指向 <元素>/POTCAR 结构的库，"
        "并确认元素顺序与 POSCAR 一致。",
        "knowledge: 5. 错误排除 / POTCAR",
    ),
    (
        "ZBRENT_FATAL",
        re.compile(r"ZBRENT: fatal", re.IGNORECASE),
        "high",
        "离子步结构灾变（能量/力突变）：减小 POTIM（如 0.1-0.2）、收紧 EDIFF，"
        "或从更合理的初始结构重新开始。",
        "vasp: ZBRENT",
    ),
    (
        "NO_FREQ_DISPLACEMENT",
        re.compile(r"(IBRION\s*[=:]\s*[56])", re.IGNORECASE),
        "medium",
        "频率计算：IBRION=5/6 必须 NSW>=1（本项目 write_incar(freq=True) 已内置），"
        "且需 ISYM=0 与足够紧的 EDIFF，否则不产生位移步。",
        "v0.2 修复记录：freq INCAR defaults",
    ),
    (
        "OUT_OF_MEMORY",
        re.compile(r"(out of memory|OOM|malloc failed|std::bad_alloc|"
                   r"OUT_OF_MEMORY|MEMORY)", re.IGNORECASE),
        "high",
        "内存不足：减小体系/k点、降低 NCORE/NPAR、增加节点内存或改用分块对角化。",
        "knowledge: 5. 错误排除 / 内存",
    ),
    (
        "MPI_ABORT",
        re.compile(r"(MPI_ABORT|MPI_ABORT was invoked|mpirun noticed that)", re.IGNORECASE),
        "medium",
        "MPI 运行时中止：先看上方的具体错误行；常见为二进制/MPI 不匹配或库缺失，"
        "检查 MPI_ENV_SCRIPT 是否已 source。",
        "knowledge: 5. 错误排除 / MPI",
    ),
    (
        "SEGFAULT",
        re.compile(r"(segmentation fault|SIGSEGV|forrtl: severe)", re.IGNORECASE),
        "high",
        "段错误：多为二进制与 MPI/库版本不兼容或栈溢出，尝试 ulimit -s unlimited、"
        "更换 MPI 或重编译；排除结构重叠（距离过近原子）。",
        "knowledge: 5. 错误排除 / 崩溃",
    ),
    (
        "TETRAHEDRON_FAIL",
        re.compile(r"Tetrahedron method fails", re.IGNORECASE),
        "medium",
        "四面体 k 点方法失败：k 点数过少或体系维度不足，改用 Gamma 中心网格或 ISMEAR=0。",
        "vasp: tetrahedron",
    ),
    (
        "ENGINE_NOT_FOUND",
        re.compile(r"(not found \(set |binary not found|No such file or directory.*"
                   r"(vasp|cp2k|abacus))", re.IGNORECASE),
        "high",
        "引擎路径未配置：在 .env 设置对应 *_EXE 路径并运行 `slisadft check-env` 验证；"
        "无引擎机器可临时使用 SLISADFT_ENGINE_MODE=mock。",
        "slisadft: engine layer",
    ),
    (
        "KPINS incompatible",
        re.compile(r"(KPINS|Number of KPOINTS|kpoints.{0,30}incompatible)", re.IGNORECASE),
        "medium",
        "k 点与并行配置不兼容：调整 KPOINTS 数使其能被 NCORE/NPAR 分组整除。",
        "vasp: parallel",
    ),
]


def diagnose(text: str) -> List[Dict[str, object]]:
    """Match failure patterns in a log/error text.

    Returns a list of {"code", "severity", "advice", "source", "match"} —
    empty when nothing matches (i.e. no known failure pattern found).
    """
    if not text:
        return []
    out: List[Dict[str, object]] = []
    seen = set()
    for code, pattern, severity, advice, source in _DIAGNOSIS_RULES:
        if code in seen:
            continue
        m = pattern.search(text)
        if m:
            seen.add(code)
            out.append({
                "code": code,
                "severity": severity,
                "advice": advice,
                "source": source,
                "match": m.group(0)[:120],
            })
    # high severity first
    out.sort(key=lambda d: 0 if d["severity"] == "high" else 1)
    return out


def summarize_diagnosis(diagnoses: List[Dict[str, object]], max_items: int = 3) -> str:
    """One-line summary for job records / API errors."""
    if not diagnoses:
        return ""
    parts = [f"{d['code']}: {str(d['advice'])[:80]}…" for d in diagnoses[:max_items]]
    return " | ".join(parts)
