#!/usr/bin/env python
"""q-dft agent command-line interface (package: slisadft).

Usage:
    qdft [run]              [--inputs '<json>'] [--json-result PATH]  full pipeline
    qdft adsorption         [--inputs '<json>'] [--json-result PATH]  adsorption workflow
    qdft volcano            [--inputs '<json>'] [--json-result PATH]  free-energy diagram
    qdft adsorption-full    [--inputs '<json>'] [--json-result PATH]  deterministic three-point E_ad
    qdft batch              --inputs-file FILE [--out CSV]            batch screening
    qdft suggest            [--inputs '<json>']                       next-step hypotheses
    qdft check-env                                                    engine availability
    qdft version                                                      print version
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

    # ── deterministic workflows (no LLM required) ────────────────────
    if workflow == "adsorption-full":
        from slisadft.workflows import run_adsorption_full
        manifest = create_run_manifest(workflow, inputs,
                                       Path(manifest_dir) if manifest_dir else None)
        result = run_adsorption_full(inputs, workdir=manifest_dir or None)
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
    manifest = create_run_manifest(
        workflow, inputs, Path(manifest_dir) if manifest_dir else None)

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
        prog="qdft",
        description="q-dft agent — AI multi-agent DFT computational catalysis platform")
    parser.add_argument(
        "workflow", nargs="?", default="run",
        choices=["run", "adsorption", "volcano", "adsorption-full",
                 "batch", "suggest", "check-env", "version"],
        help="workflow to execute (default: run; adsorption-full/batch/suggest "
             "are deterministic and need no LLM)")
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
