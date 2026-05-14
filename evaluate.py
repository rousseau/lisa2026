#!/usr/bin/env python
"""
Unified evaluation entry point for LISA 2026.

Routes to the correct evaluation script based on the run ID. For Task 1a runs,
a second pass through compute_metrics.py is triggered automatically after
predictions are generated.

Usage:
    python evaluate.py --run 0001
    python evaluate.py --run 0002
    python evaluate.py --run 0003
    python evaluate.py --run 0003_EXP_C
    python evaluate.py --run 0004
    python evaluate.py --run 0001 --smoke-test

Run IDs accept an optional "RUN_" / "run_" prefix:
    python evaluate.py --run RUN_0003_EXP_C   # same as --run 0003_EXP_C
"""

import argparse
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Evaluation registry
# ---------------------------------------------------------------------------
# mode:
#   "task1a_single_model"  – evaluate_task1a.py  then compute_metrics.py
#   "task1a_multilabel"    – evaluate_task1a_multilabel.py  then compute_metrics.py
#   "task2"                – evaluate_task2_dynunet.py  (single pass)
#   "task1b"               – evaluate_task1b.py  (single pass)
#
# supports_smoke_test:
#   False when the underlying eval script does not expose --smoke_test.

EVAL_REGISTRY = {
    "0001": {
        "task": "1a",
        "eval_script": "evaluate_task1a.py",
        "metrics_script": "compute_metrics.py",
        "config": "configs/run_0001_baseline.yaml",
        "mode": "task1a_single_model",
        "supports_smoke_test": False,  # evaluate_task1a.py does not expose --smoke_test
    },
    "0002": {
        "task": "1a",
        "eval_script": "evaluate_task1a_multilabel.py",
        "metrics_script": "compute_metrics.py",
        "config": "configs/run_0002_upf.yaml",
        "mode": "task1a_multilabel",
        "supports_smoke_test": True,
    },
    "0003": {
        "task": "2",
        "eval_script": "evaluate_task2_dynunet.py",
        "config": "configs/run_0003_task2_dynunet.yaml",
        "mode": "task2",
        "supports_smoke_test": True,
    },
    "0003_COLLAPSED": {
        "task": "2",
        "eval_script": "evaluate_task2_dynunet.py",
        "config": "configs/run_0003_task2_dynunet_collapsed.yaml",
        "mode": "task2",
        "supports_smoke_test": True,
    },
    "0003_EXP_C": {
        "task": "2",
        "eval_script": "evaluate_task2_dynunet.py",
        "config": "configs/run_0003_task2_dynunet_expC.yaml",
        "mode": "task2",
        "supports_smoke_test": True,
    },
    "0004": {
        "task": "1b",
        "eval_script": "evaluate_task1b.py",
        "config": "configs/run_0004_task1b_unet.yaml",
        "mode": "task1b",
        "supports_smoke_test": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalise_run_id(raw: str) -> str:
    """Strip optional 'RUN_' / 'run_' prefix.

    Examples
    --------
    "RUN_0001"           -> "0001"
    "run_0003_EXP_C_TTA" -> "0003_EXP_C_TTA"
    "0003_EXP_C"         -> "0003_EXP_C"
    """
    if raw.strip().upper().startswith("RUN_"):
        return raw.strip()[4:]
    return raw.strip()


def _load_config(config_path: str) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    try:
        with open(config_path, "r") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)


def _run_subprocess(cmd: list[str]) -> None:
    """Run *cmd* and exit on failure."""
    print(f"  $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Command failed with return code {exc.returncode}:")
        print(f"  {' '.join(cmd)}")
        sys.exit(1)


def _smoke_args(entry: dict, smoke_test: bool) -> list[str]:
    """Return ['--smoke_test'] when applicable, else []."""
    if not smoke_test:
        return []
    if entry.get("supports_smoke_test", True):
        return ["--smoke_test"]
    print(
        f"  [WARNING] '{entry['eval_script']}' does not support --smoke_test; "
        "running without it."
    )
    return []


# ---------------------------------------------------------------------------
# Evaluation dispatch
# ---------------------------------------------------------------------------


def evaluate_task1a(entry: dict, smoke_test: bool) -> None:
    """Two-step Task 1a evaluation.

    Step 1 – generate predictions CSV via the eval script.
    Step 2 – compute challenge metrics via compute_metrics.py.
    """
    config_path = entry["config"]
    config = _load_config(config_path)
    smoke_extra = _smoke_args(entry, smoke_test)

    # ── Step 1 : generate predictions ──────────────────────────────────────
    print(f"\n[1/2] Generating predictions")
    eval_cmd = [sys.executable, entry["eval_script"], "--config", config_path]
    eval_cmd += smoke_extra
    _run_subprocess(eval_cmd)

    # ── Step 2 : compute metrics ────────────────────────────────────────────
    print(f"\n[2/2] Computing metrics")

    predictions_file = config["output"]["predictions_file"]
    ground_truth_csv = config["data"]["csv_path"]
    metrics_file = config["output"]["metrics_file"]
    run_id = str(config.get("run_id", "unknown"))

    metrics_cmd = [
        sys.executable,
        entry["metrics_script"],
        "--predictions",
        predictions_file,
        "--ground-truth",
        ground_truth_csv,
        "--output",
        metrics_file,
        "--run-id",
        run_id,
    ]
    _run_subprocess(metrics_cmd)

    print(f"\n[OK] Evaluation complete.")
    print(f"  Predictions : {predictions_file}")
    print(f"  Metrics     : {metrics_file}")


def evaluate_task2(entry: dict, smoke_test: bool) -> None:
    """Single-step Task 2 evaluation."""
    config_path = entry["config"]
    smoke_extra = _smoke_args(entry, smoke_test)

    print(f"\n[1/1] Running Task 2 evaluation")
    cmd = [sys.executable, entry["eval_script"], "--config", config_path]
    cmd += smoke_extra
    _run_subprocess(cmd)

    print(f"\n[OK] Evaluation complete.")


def evaluate_task1b(entry: dict, smoke_test: bool) -> None:
    """Single-step Task 1b evaluation."""
    config_path = entry["config"]
    smoke_extra = _smoke_args(entry, smoke_test)

    print(f"\n[1/1] Running Task 1b evaluation")
    cmd = [sys.executable, entry["eval_script"], "--config", config_path]
    cmd += smoke_extra
    _run_subprocess(cmd)

    print(f"\n[OK] Evaluation complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified evaluation entry point for LISA 2026.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        metavar="RUN_ID",
        help="Run identifier, e.g. 0001, 0003_EXP_C_TTA, RUN_0002.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        dest="smoke_test",
        help="Quick sanity check (limited batches).",
    )
    args = parser.parse_args()

    run_id = normalise_run_id(args.run)

    if run_id not in EVAL_REGISTRY:
        available = "\n  ".join(sorted(EVAL_REGISTRY.keys()))
        print(f"\n[ERROR] Run '{run_id}' is not registered in the evaluation registry.")
        print(f"\nAvailable runs:\n  {available}")
        print("\nTip: prefix 'RUN_' is automatically stripped (e.g. RUN_0001 -> 0001).")
        sys.exit(1)

    entry = EVAL_REGISTRY[run_id]

    print(f"\n[LISA 2026] Starting evaluation")
    print(f"  Run ID      : {run_id}")
    print(f"  Task        : {entry['task']}")
    print(f"  Eval script : {entry['eval_script']}")
    print(f"  Config      : {entry['config']}")
    if args.smoke_test:
        print(f"  Mode        : SMOKE TEST")

    mode = entry["mode"]

    if mode in ("task1a_single_model", "task1a_multilabel"):
        evaluate_task1a(entry, smoke_test=args.smoke_test)
    elif mode == "task2":
        evaluate_task2(entry, smoke_test=args.smoke_test)
    elif mode == "task1b":
        evaluate_task1b(entry, smoke_test=args.smoke_test)
    else:
        print(f"[ERROR] Unknown evaluation mode '{mode}' for run {run_id}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
