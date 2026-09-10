#!/usr/bin/env python
"""slisaDFT command-line interface.

Usage:
    slisadft [run]        [--inputs '<json>'] [--json-result PATH]  full pipeline
    slisadft adsorption   [--inputs '<json>'] [--json-result PATH]  adsorption workflow
    slisadft volcano      [--inputs '<json>'] [--json-result PATH]  free-energy diagram
    slisadft check-env                                              engine availability
    slisadft version                                                print version
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def _get_crew_class():
    from slisadft.crew import Slisadft
    return Slisadft


# ---------------------------------------------------------------------------
# Workflow runners (shared by CLI and the REST-API job runner)
# ---------------------------------------------------------------------------

WORKFLOWS = ("run", "adsorption", "volcano", "adsorption-full")


def run_workflow(workflow: str, inputs: dict, manifest_dir: str = "") -> dict:
    """Execute a workflow and return a machine-readable result dict.

    Deterministic workflows (``adsorption-full``) run without crewAI/LLM;
    crew workflows (``run``/``adsorption``/``volcano``) drive the agents.

    This is the single entry point used by both the CLI and the REST API
    job runner.  ``manifest_dir`` (default: cwd) receives the run manifest.
    """
    from slisadft.utils.run_manifest import create_run_manifest, finalize_run_manifest

    # v0.4: CLI/API runs always persist a manifest under runs/ so history is
    # browsable via `qdft runs` (callers may override manifest_dir).
    if not manifest_dir:
        from datetime import datetime as _dt
        manifest_dir = str(Path("runs") /
                           f"{_dt.now().strftime('%Y%m%d-%H%M%S')}-{workflow}")

    # ── deterministic workflows (no LLM required) ────────────────────
    if workflow == "adsorption-full":
        from slisadft.workflows import run_adsorption_full
        manifest = create_run_manifest(workflow, inputs, Path(manifest_dir))
        result = run_adsorption_full(inputs, workdir=manifest_dir)
        manifest_path = manifest.get("manifest_path")
        if manifest_path:
            finalize_run_manifest(Path(manifest_path), status="done", result={
                "adsorption_energy_eV": result.get("adsorption_energy_eV")})
        return {
            "workflow": workflow,
            "inputs": result.get("inputs", inputs),
            "engine_mode": result.get("engine_mode"),
            "synthetic": result.get("synthetic"),
            "adsorption_energy_eV": result.get("adsorption_energy_eV"),
            "report": result.get("report"),
            "final_answer": json.dumps(
                {k: result.get(k) for k in
                 ("workflow", "energies_eV", "adsorption_energy_eV",
                  "gibbs_correction", "engine_mode", "synthetic")},
                ensure_ascii=False),
        }

    from slisadft.crew import prepare_inputs
    inputs = prepare_inputs(inputs)
    manifest = create_run_manifest(workflow, inputs, Path(manifest_dir))

    crew_cls = _get_crew_class()
    if workflow == "adsorption":
        crew = crew_cls().adsorption_crew()
    elif workflow == "volcano":
        crew = crew_cls().volcano_crew()
    elif workflow == "run":
        crew = crew_cls().crew()
    else:
        raise ValueError(f"unknown workflow: {workflow}")

    try:
        result = crew.kickoff(inputs=inputs)
    except Exception as exc:
        manifest_path = manifest.get("manifest_path")
        finalize_run_manifest(Path(manifest_path) if manifest_path else Path("run_manifest.json"),
                              status="failed", result={"error": str(exc)})
        raise

    return {
        "workflow": workflow,
        "inputs": inputs,
        "engine_mode": manifest.get("engine_mode"),
        "final_answer": str(result),
    }


# ---------------------------------------------------------------------------
# crewAI standard hooks (kept for `crewai run/train/test` compatibility)
# ---------------------------------------------------------------------------

def run():
    """Run the default DFT pipeline (full workflow)."""
    try:
        result = run_workflow("run", {})
        print("✅ Full pipeline finished.")
        print("Final result:", result["final_answer"])
    except Exception as e:
        print(f"❌ Crew execution failed: {e}")
        sys.exit(1)


def train():
    """Train the crew for a given number of iterations."""
    from slisadft.crew import prepare_inputs
    try:
        _get_crew_class()().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=prepare_inputs({}))
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        _get_crew_class()().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and returns the results."""
    from slisadft.crew import prepare_inputs
    try:
        _get_crew_class()().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=prepare_inputs({}))
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """Run the crew with trigger payload (DSH plugin runner protocol)."""
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")
    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    return run_workflow("run", {
        "element": trigger_payload.get("element", "Pt"),
        "miller": trigger_payload.get("miller", "(111)"),
        "layers": trigger_payload.get("layers", 3),
        "adsorbate": trigger_payload.get("adsorbate", "CO"),
        "site": trigger_payload.get("site", "top"),
        "dft_calculator": trigger_payload.get("calculator", "vasp"),
    })


# ---------------------------------------------------------------------------
# argparse CLI
# ---------------------------------------------------------------------------

def _cmd_check_env(_args) -> int:
    from slisadft.tools.dft_tools import check_dft_environment_tool
    from slisadft.engines import engine_mode
    print(f"engine mode: {engine_mode()}")
    print(check_dft_environment_tool.func())
    return 0


def _cmd_suggest(args) -> int:
    import json as _json
    from slisadft.utils.suggest import suggest_next
    last = None
    if args.inputs:
        try:
            last = _json.loads(args.inputs)
        except json.JSONDecodeError as exc:
            print(f"❌ --inputs is not valid JSON: {exc}")
            return 2
    suggestions = suggest_next(last_result=last)
    print(_json.dumps(suggestions, ensure_ascii=False, indent=2))
    if args.json_result:
        _write_json_result(args.json_result,
                           {"status": "done", "suggestions": suggestions})
    return 0


_ENV_TEMPLATE = """# q-dft agent project configuration
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-your-key-here
SLISADFT_ENGINE_MODE=mock

# real-computation mode (configure engines, then set SLISADFT_ENGINE_MODE=real)
VASP_EXE=
VASP_POTCAR_DIR=
CP2K_EXE=
ABACUS_EXE=
SLURM_PARTITION=CPU
MPI_RUN=mpirun
REPORT_LANGUAGE=zh
"""

_COMBOS_TEMPLATE = [
    {"element": "Pt", "miller": "(111)", "adsorbate": "CO", "site": "top",
     "dft_calculator": "vasp"},
    {"element": "Cu", "miller": "(111)", "adsorbate": "OH", "site": "fcc",
     "dft_calculator": "vasp"},
]


def _cmd_init(args) -> int:
    """Scaffold a research project directory (env + combos + runs/)."""
    import json as _json
    project = Path(args.target) if args.target else Path("qdft-project")
    if project.exists() and any(project.iterdir()):
        print(f"❌ directory not empty: {project}")
        return 2
    project.mkdir(parents=True, exist_ok=True)
    (project / ".env").write_text(_ENV_TEMPLATE, encoding="utf-8")
    (project / "combos.json").write_text(
        _json.dumps(_COMBOS_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (project / "runs").mkdir(exist_ok=True)
    print(f"✅ project created: {project.resolve()}")
    print("next steps:")
    print(f"  1. edit {project / '.env'} (LLM key; engines for real mode)")
    print(f"  2. cd {project}")
    print("  3. qdft adsorption-full --inputs '{\"element\":\"Pt\",\"adsorbate\":\"CO\"}'")
    print("  4. qdft batch --inputs-file combos.json --out results.csv")
    print("  5. qdft runs")
    if args.json_result:
        _write_json_result(args.json_result, {"status": "done",
                                              "project": str(project.resolve())})
    return 0


def _cmd_info(args) -> int:
    import json as _json
    from slisadft.utils.inspect import inspect_structure
    if not args.target:
        print("❌ info requires a structure file path")
        return 2
    try:
        info = inspect_structure(args.target)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1
    except Exception as exc:  # unparsable structure
        print(f"❌ cannot parse structure: {exc}")
        return 1
    print(_json.dumps(info, ensure_ascii=False, indent=2))
    if args.json_result:
        _write_json_result(args.json_result, {"status": "done", "info": info})
    return 0


def _cmd_runs(args) -> int:
    import json as _json
    from slisadft.utils.inspect import list_runs
    rows = list_runs(limit=args.limit)
    if not rows:
        print("no runs recorded yet — manifests are written to runs/ on every run")
        return 0
    header = f"{'created_at':<20} {'workflow':<16} {'engine':<6} {'status':<8} {'system':<16} {'E_ad(eV)':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        system = f"{r['element']}{''}/{r['adsorbate']}" if r["element"] else "-"
        e = f"{r['e_ad_eV']:.4f}" if isinstance(r["e_ad_eV"], (int, float)) else "-"
        print(f"{r['created_at']:<20} {r['workflow']:<16} {r['engine_mode']:<6} "
              f"{r['status']:<8} {system:<16} {e:>10}")
    if args.json_result:
        _write_json_result(args.json_result, {"status": "done", "runs": rows})
    return 0


def _cmd_cite(args) -> int:
    import json as _json
    from slisadft.utils.inspect import citation
    c = citation()
    print(c["text"])
    print("\nBibTeX:\n" + c["bibtex"])
    if args.json_result:
        _write_json_result(args.json_result, {"status": "done", "citation": c})
    return 0


def _cmd_batch(args) -> int:
    import json as _json
    from slisadft.workflows import run_batch
    if not args.inputs_file:
        print("❌ batch requires --inputs-file (JSON list of input combos)")
        return 2
    try:
        data = _json.loads(Path(args.inputs_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"❌ cannot read --inputs-file: {exc}")
        return 2
    combos = data.get("combos") if isinstance(data, dict) else data
    if not isinstance(combos, list) or not combos:
        print("❌ --inputs-file must contain a JSON list of combos "
              "(or {'combos': [...]})")
        return 2

    summary = run_batch(combos, out_csv=args.out, max_workers=args.max_workers)
    print(f"✅ batch finished: {summary['ok']} ok, {summary['failed']} failed, "
          f"engine_mode={summary['engine_mode']}")
    if summary["csv"]:
        print(f"CSV: {summary['csv']}")
    for row in summary["results"]:
        e = row.get("adsorption_energy_eV")
        mark = "OK " if row["status"] == "done" else "ERR"
        print(f"  [{mark}] {row['element']}{row['miller']}/{row['adsorbate']}"
              f"@{row['site']} ({row['calculator']}): "
              f"E_ad={e if e is not None else row['error']}")
    if args.json_result:
        _write_json_result(args.json_result, {"status": "done", **summary})
    return 0 if summary["failed"] == 0 else 1


def _cmd_version(_args) -> int:
    from slisadft import __edition__, __version__
    print(f"slisadft {__version__} ({__edition__} edition)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="slisadft",
        description="slisaDFT — AI multi-agent DFT computational catalysis platform")
    parser.add_argument(
        "workflow", nargs="?", default="run",
        choices=["run", "adsorption", "volcano", "adsorption-full",
                 "batch", "suggest", "init", "info", "runs", "cite",
                 "check-env", "version"],
        help="workflow to execute (default: run; adsorption-full/batch/suggest/"
             "init/info/runs/cite are deterministic and need no LLM)")
    parser.add_argument(
        "target", nargs="?", default="",
        help="init: project directory to create; info: structure file to inspect")
    parser.add_argument("--limit", type=int, default=20,
                        help="runs: how many recent runs to show")
    parser.add_argument("--inputs", default="",
                        help='workflow inputs as JSON, e.g. \'{"element": "Cu", "adsorbate": "OH"}\'')
    parser.add_argument("--json-result", default="",
                        help="write the final result as JSON to this file (used by the API runner)")
    parser.add_argument("--inputs-file", default="",
                        help="batch: JSON file with a list of input combos "
                             "(or {'combos': [...]})")
    parser.add_argument("--out", default="batch_results.csv",
                        help="batch: output CSV path")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="batch: parallel combos")
    args = parser.parse_args(argv)

    if args.workflow == "version":
        return _cmd_version(args)
    if args.workflow == "check-env":
        return _cmd_check_env(args)
    if args.workflow == "suggest":
        return _cmd_suggest(args)
    if args.workflow == "batch":
        return _cmd_batch(args)
    if args.workflow == "init":
        return _cmd_init(args)
    if args.workflow == "info":
        return _cmd_info(args)
    if args.workflow == "runs":
        return _cmd_runs(args)
    if args.workflow == "cite":
        return _cmd_cite(args)

    inputs = {}
    if args.inputs:
        try:
            inputs = json.loads(args.inputs)
        except json.JSONDecodeError as exc:
            print(f"❌ --inputs is not valid JSON: {exc}")
            return 2

    try:
        result = run_workflow(args.workflow, inputs)
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        if args.json_result:
            _write_json_result(args.json_result, {"status": "failed", "error": str(e)})
        return 1

    print("✅ Workflow finished.")
    print("Final result:", result["final_answer"])
    if args.json_result:
        _write_json_result(args.json_result, {"status": "done", **result})
    return 0


def _write_json_result(path: str, payload: dict) -> None:
    try:
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
    except OSError as exc:
        print(f"⚠️ could not write JSON result to {path}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
