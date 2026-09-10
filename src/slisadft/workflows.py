"""确定性工作流层（v0.3.0）：不依赖 LLM 编排的完整科研工作流。

背景：v0.2 端到端验收发现，`adsorption` crew 工作流只计算吸附体系总能量，
E_slab 与 E_adsorbate 依赖智能体"从上下文补全"，导致 E_ad 不可靠（LLM 在
报告中如实指出 e_slab=e_adsorbate=0）。本模块将"三点能量协议"固化为
确定性 Python 编排：

    E_ad = E(slab+ads) - E(slab) - E(adsorbate, free)

流程（对 slab / molecule / slab+ads 三个体系一致执行）：
    1. ASE 构建模型（surface_tools）
    2. MLIP 预优化（MACE/DPA-4，mock 模式为 EMT/LJ）
    3. DFT 单点精修（VASP/CP2K/ABACUS，mock 模式为合成输出）
    4. 吸附能 + 吉布斯自由能校正 + 溯源清单 + markdown 报告

确定性层无需 crewAI 与 LLM Key 即可运行（LLM 仅在 crew 工作流中参与
编排与科学解读），因此可在任何机器上离线验收。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .engines import engine_mode
from .utils.run_manifest import create_run_manifest, finalize_run_manifest

DEFAULT_INPUTS = {
    "element": "Pt",
    "miller": "(111)",
    "layers": 3,
    "adsorbate": "CO",
    "site": "top",
    "dft_calculator": "vasp",
    "size": "2,2",
    "vacuum": 10.0,
    "fmax": 0.05,
    "mlip_steps": 200,
    "temperature": 298.15,
}

REPORT_NAME = "adsorption_full_report.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scratch_dir() -> str:
    """Scratch dir for intermediate build files (never pollutes the repo)."""
    import tempfile
    d = Path(tempfile.gettempdir()) / "qdft_scratch"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _parse_json_tool_output(raw: str, what: str) -> Dict[str, object]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{what}: tool returned non-JSON output: {raw[:200]}") from exc
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{what}: {data['error']}")
    return data


def run_adsorption_full(inputs: Optional[dict] = None,
                        workdir: Optional[str] = None) -> Dict[str, object]:
    """Execute the deterministic three-point adsorption-energy workflow.

    Args:
        inputs: element / miller / layers / adsorbate / site / dft_calculator /
                size / vacuum / fmax / mlip_steps / temperature.
        workdir: directory for the report (default: cwd).

    Returns:
        Structured result dict with the three single-point energies,
        adsorption energy, Gibbs correction and artifact paths.
    """
    from .tools.surface_tools import (
        build_surface_tool,
        build_adsorption_tool,
        build_molecule_tool,
    )
    from .tools.mlip_tools import mace_optimize_tool
    from .tools.dft_tools import dft_optimize_tool
    from .tools.postprocess_tools import adsorption_energy_tool
    from .utils.thermochemistry import gibbs_free_energy, EMPIRICAL_GAS_CORRECTIONS

    cfg = dict(DEFAULT_INPUTS)
    cfg.update(inputs or {})
    element = str(cfg["element"])
    adsorbate = str(cfg["adsorbate"])
    calculator = str(cfg["dft_calculator"])
    started = time.time()

    manifest = create_run_manifest("adsorption-full", cfg)

    # -- 三体系建模 ------------------------------------------------------
    surfaces = {}
    slab_raw = build_surface_tool.func(
        element, str(cfg["miller"]), int(cfg["layers"]),
        str(cfg["size"]), float(cfg["vacuum"]))
    if str(slab_raw).startswith("Error"):
        raise RuntimeError(f"slab build failed: {slab_raw}")
    surfaces["slab"] = str(slab_raw)

    mol_raw = build_molecule_tool.func(
        adsorbate,
        output_file=str(Path(_scratch_dir()) / f"molecule_{adsorbate}.vasp"))
    if str(mol_raw).startswith("Error"):
        raise RuntimeError(f"molecule build failed: {mol_raw}")
    surfaces["adsorbate"] = str(mol_raw)

    ads_raw = build_adsorption_tool.func(surfaces["slab"], adsorbate,
                                         str(cfg["site"]))
    if str(ads_raw).startswith("Error"):
        raise RuntimeError(f"adsorption build failed: {ads_raw}")
    surfaces["slab_ads"] = str(ads_raw)

    # -- 三体系一致执行：MLIP 预优化 -> DFT 单点 --------------------------
    energies: Dict[str, float] = {}
    details: Dict[str, Dict[str, object]] = {}
    for name, structure in surfaces.items():
        mlip = _parse_json_tool_output(
            mace_optimize_tool.func(
                structure,
                fmax=float(cfg["fmax"]),
                steps=int(cfg["mlip_steps"]),
                output_file=str(Path(structure).with_name(
                    Path(structure).stem + f"_{name}_mlip_opt.vasp")),
            ),
            f"{name}: MLIP optimization")
        dft = _parse_json_tool_output(
            dft_optimize_tool.func(
                str(mlip["output_file"]), calculator=calculator, relax=False),
            f"{name}: DFT single point")
        e = dft.get("final_energy_eV")
        if e is None:
            raise RuntimeError(f"{name}: DFT returned no energy ({dft.get('error')})")
        energies[name] = float(e)
        details[name] = {
            "structure_built": structure,
            "mlip_optimized": mlip.get("output_file"),
            "mlip_energy_eV": mlip.get("energy_eV"),
            "mlip_converged": mlip.get("converged"),
            "dft_energy_eV": e,
            "dft_converged": dft.get("converged"),
            "dft_workdir": dft.get("workdir"),
        }

    e_ad = energies["slab_ads"] - energies["slab"] - energies["adsorbate"]

    # 吉布斯校正：吸附物用经验表（吸附态振动需频率计算，见 vibrational 工具）
    zpe, s_emp = EMPIRICAL_GAS_CORRECTIONS.get(adsorbate.upper(), (0.0, 0.0))
    temperature = float(cfg["temperature"])
    g_adsorbate = gibbs_free_energy(energies["adsorbate"], T=temperature,
                                    formula=adsorbate.upper() or None)
    e_ad_gibbs_approx = energies["slab_ads"] - energies["slab"] - g_adsorbate

    synthetic = engine_mode() == "mock"
    result: Dict[str, object] = {
        "workflow": "adsorption-full",
        "protocol": "build -> MLIP relax -> DFT single point (three systems)",
        "inputs": cfg,
        "energies_eV": energies,
        "adsorption_energy_eV": round(e_ad, 4),
        "gibbs_correction": {
            "adsorbate_zpe_eV": zpe,
            "adsorbate_ts_eV_at_T": round(
                (zpe + (g_adsorbate - energies["adsorbate"])) - zpe, 4),
            "g_adsorbate_eV": round(g_adsorbate, 4),
            "adsorption_energy_gibbs_approx_eV": round(e_ad_gibbs_approx, 4),
            "note": "approximate: adsorbate empirical corrections; for the "
                    "adsorbed state use vibrational_thermochemistry_tool",
        },
        "system_details": details,
        "engine_mode": engine_mode(),
        "synthetic": synthetic,
        "elapsed_s": round(time.time() - started, 1),
        "manifest": manifest.get("manifest_path"),
        "finished_at": _now(),
    }
    if synthetic:
        result["synthetic_notice"] = (
            "SYNTHETIC-DATA: mock engine output (EMT/LJ), not real DFT — "
            "do not use as research data (Rule Zero).")

    # -- 报告 + 溯源 ------------------------------------------------------
    workdir_path = Path(workdir) if workdir else Path.cwd()
    try:
        report = _render_report(result)
        report_path = workdir_path / REPORT_NAME
        report_path.write_text(report, encoding="utf-8")
        result["report"] = str(report_path)
    except OSError:
        pass
    manifest_path = manifest.get("manifest_path")
    if manifest_path:
        finalize_run_manifest(Path(manifest_path), status="done", result={
            "adsorption_energy_eV": result["adsorption_energy_eV"]})
    return result


def _render_report(r: Dict[str, object]) -> str:
    energies = r["energies_eV"]
    gc = r["gibbs_correction"]
    lines = [
        "# Adsorption-energy workflow report (adsorption-full)",
        "",
        f"- generated: {r['finished_at']}",
        f"- engine mode: **{r['engine_mode']}**"
        + ("  ⚠ SYNTHETIC mock data (Rule Zero)" if r["synthetic"] else ""),
        f"- protocol: {r['protocol']}",
        f"- elapsed: {r['elapsed_s']} s",
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(r["inputs"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Single-point energies (eV)",
        "",
        "| system | DFT energy |",
        "|---|---|",
    ]
    for name, e in energies.items():
        lines.append(f"| {name} | {e:.6f} |")
    lines += [
        "",
        f"**E_ad = E(slab+ads) - E(slab) - E(adsorbate) = "
        f"{r['adsorption_energy_eV']} eV**",
        "",
        "## Gibbs correction (adsorbate, empirical)",
        "",
        f"- G(adsorbate) = {gc['g_adsorbate_eV']} eV",
        f"- E_ad(Gibbs approx.) = {gc['adsorption_energy_gibbs_approx_eV']} eV",
        f"- note: {gc['note']}",
        "",
        "## Per-system details",
        "",
    ]
    for name, d in r["system_details"].items():
        lines += [
            f"### {name}",
            f"- structure: `{d['structure_built']}`",
            f"- MLIP optimized: `{d['mlip_optimized']}` (E={d['mlip_energy_eV']}, "
            f"converged={d['mlip_converged']})",
            f"- DFT workdir: `{d['dft_workdir']}` (converged={d['dft_converged']})",
            "",
        ]
    lines.append("*All energies traceable to the calculation files in the "
                 "referenced workdirs (Rule Zero).*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 批量筛选：多组合确定性扫描，输出 CSV
# ---------------------------------------------------------------------------

BATCH_CSV_COLUMNS = [
    "element", "miller", "adsorbate", "site", "calculator",
    "e_slab_ads_eV", "e_slab_eV", "e_adsorbate_eV",
    "adsorption_energy_eV", "elapsed_s", "engine_mode", "synthetic",
    "status", "error",
]


def run_batch(combos: List[dict], out_csv: Optional[str] = None,
              max_workers: int = 2, workdir: Optional[str] = None
              ) -> Dict[str, object]:
    """Run the adsorption-full workflow over many input combinations.

    Args:
        combos: list of input dicts (same schema as run_adsorption_full).
        out_csv: write a summary CSV to this path.
        max_workers: parallel threads (each combo is CPU/engine bound).
        workdir: directory for the CSV (default: cwd).

    Returns:
        {"results": [per-combo rows], "csv": path, "ok": n_ok, "failed": n_failed}
    """
    from concurrent.futures import ThreadPoolExecutor

    rows: List[Dict[str, object]] = []

    def _one(idx: int, combo: dict) -> Dict[str, object]:
        base = dict(DEFAULT_INPUTS)
        base.update(combo or {})
        row = {
            "element": base["element"], "miller": base["miller"],
            "adsorbate": base["adsorbate"], "site": base["site"],
            "calculator": base["dft_calculator"],
        }
        try:
            r = run_adsorption_full(combo, workdir=workdir)
            en = r["energies_eV"]
            row.update({
                "e_slab_ads_eV": en["slab_ads"], "e_slab_eV": en["slab"],
                "e_adsorbate_eV": en["adsorbate"],
                "adsorption_energy_eV": r["adsorption_energy_eV"],
                "elapsed_s": r["elapsed_s"], "engine_mode": r["engine_mode"],
                "synthetic": r["synthetic"], "status": "done", "error": "",
            })
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc),
                        "engine_mode": engine_mode()})
        row["_idx"] = idx
        return row

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        rows = list(pool.map(lambda t: _one(*t), enumerate(combos)))

    rows.sort(key=lambda x: x["_idx"])
    for r in rows:
        r.pop("_idx", None)

    csv_path = None
    if out_csv:
        import csv as _csv
        csv_path = str(Path(workdir) / out_csv) if workdir else str(out_csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=BATCH_CSV_COLUMNS,
                                     extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "done")
    return {"results": rows, "csv": csv_path, "ok": n_ok,
            "failed": len(rows) - n_ok, "engine_mode": engine_mode()}
