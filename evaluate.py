#!/usr/bin/env python
"""
Unified evaluation entry point for LISA 2026.

All evaluation scripts live in src/ and are invoked as Python modules so that
their `from src.xxx import ...` imports always resolve from the project root.

Usage
-----
    python evaluate.py --run 0001
    python evaluate.py --run 0002
    python evaluate.py --run 0003
    python evaluate.py --run 0004
    python evaluate.py --run 0005
    python evaluate.py --run 0001 --smoke-test

Run IDs accept an optional "RUN_" prefix:
    python evaluate.py --run RUN_0003   # same as --run 0003
"""

import argparse
import sys

import yaml

from src.utils import normalise_run_id, run_cmd, smoke_args as _smoke_args_util

# ---------------------------------------------------------------------------
# Evaluation registry
# ---------------------------------------------------------------------------
# mode:
#   "task1a_single_model"  – src.evaluate_task1a  then src.compute_metrics
#   "task1a_multilabel"    – src.evaluate_task1a_multilabel  then src.compute_metrics
#   "task2"                – src.evaluate_task2_dynunet  (single pass)
#   "task1b"               – src.evaluate_task1b  (single pass)

EVAL_REGISTRY = {
    "0001": {
        "task": "1a",
        "eval_module": "src.evaluate_task1a",
        "metrics_module": "src.compute_metrics",
        "config": "configs/run_0001_baseline.yaml",
        "mode": "task1a_single_model",
        "supports_smoke_test": False,
    },
    "0002": {
        "task": "1a",
        "eval_module": "src.evaluate_task1a_multilabel",
        "metrics_module": "src.compute_metrics",
        "config": "configs/run_0002_upf.yaml",
        "mode": "task1a_multilabel",
        "supports_smoke_test": True,
    },
    "0003": {
        "task": "2",
        "eval_module": "src.evaluate_task2_dynunet",
        "config": "configs/run_0003_task2_dynunet.yaml",
        "mode": "task2",
        "supports_smoke_test": True,
    },
    "0004": {
        "task": "1a+1b+2",
        "eval_module": "src.evaluate_multitask",
        "config": "configs/run_0004_multitask.yaml",
        "mode": "multitask",
        "supports_smoke_test": True,
    },
    "0005": {
        "task": "1b",
        "eval_module": "src.evaluate_task1b",
        "config": "configs/run_0005_task1b_unet.yaml",
        "mode": "task1b",
        "supports_smoke_test": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers — thin wrappers around src.utils to preserve local names
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> None:
    run_cmd(cmd)


def _smoke_args(entry: dict, smoke_test: bool) -> list[str]:
    return _smoke_args_util(entry, smoke_test)


def _load_config(path: str) -> dict:
    try:
        with open(path) as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"[ERROR] Config not found: {path}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Evaluation dispatch
# ---------------------------------------------------------------------------


def evaluate_task1a(entry: dict, smoke_test: bool) -> None:
    """Two-step: generate predictions, then compute metrics."""
    config_path = entry["config"]
    cfg = _load_config(config_path)
    smoke_extra = _smoke_args(entry, smoke_test)

    print(f"\n[1/2] Generating predictions")
    _run(
        [sys.executable, "-m", entry["eval_module"], "--config", config_path]
        + smoke_extra
    )

    print(f"\n[2/2] Computing metrics")
    _run(
        [
            sys.executable,
            "-m",
            entry["metrics_module"],
            "--predictions",
            cfg["output"]["predictions_file"],
            "--ground-truth",
            cfg["data"]["csv_path"],
            "--output",
            cfg["output"]["metrics_file"],
            "--run-id",
            str(cfg.get("run_id", "unknown")),
        ]
    )
    print(f"\n[OK] Evaluation complete.")


def evaluate_task2(entry: dict, smoke_test: bool) -> None:
    smoke_extra = _smoke_args(entry, smoke_test)
    print(f"\n[1/1] Running Task 2 evaluation")
    _run(
        [sys.executable, "-m", entry["eval_module"], "--config", entry["config"]]
        + smoke_extra
    )
    print(f"\n[OK] Evaluation complete.")


def evaluate_multitask(entry: dict, smoke_test: bool) -> None:
    smoke_extra = _smoke_args(entry, smoke_test)
    print(f"\n[1/1] Running multi-task evaluation (Tasks 1a, 1b, 2)")
    _run(
        [sys.executable, "-m", entry["eval_module"], "--config", entry["config"]]
        + smoke_extra
    )
    print(f"\n[OK] Evaluation complete.")


def evaluate_task1b(entry: dict, smoke_test: bool) -> None:
    smoke_extra = _smoke_args(entry, smoke_test)
    print(f"\n[1/1] Running Task 1b evaluation")
    _run(
        [sys.executable, "-m", entry["eval_module"], "--config", entry["config"]]
        + smoke_extra
    )
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
        required=True,
        metavar="RUN_ID",
        help="Run ID, e.g. 0001, 0003, RUN_0002.",
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
        print(f"\n[ERROR] Run '{run_id}' not registered.\nAvailable:\n  {available}")
        sys.exit(1)

    entry = EVAL_REGISTRY[run_id]
    print(f"\n[LISA 2026] Evaluation")
    print(f"  Run    : {run_id}")
    print(f"  Task   : {entry['task']}")
    print(f"  Module : {entry['eval_module']}")
    print(f"  Config : {entry['config']}")
    if args.smoke_test:
        print(f"  Mode   : SMOKE TEST")

    mode = entry["mode"]
    if mode in ("task1a_single_model", "task1a_multilabel"):
        evaluate_task1a(entry, smoke_test=args.smoke_test)
    elif mode == "task2":
        evaluate_task2(entry, smoke_test=args.smoke_test)
    elif mode == "multitask":
        evaluate_multitask(entry, smoke_test=args.smoke_test)
    elif mode == "task1b":
        evaluate_task1b(entry, smoke_test=args.smoke_test)
    else:
        print(f"[ERROR] Unknown mode '{mode}'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
