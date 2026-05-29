#!/usr/bin/env python
"""
Unified evaluation entry point for LISA 2026.

All evaluation scripts live in src/ and are invoked as Python modules so that
their ``from src.xxx import ...`` imports always resolve from the project root.

Usage
-----
    python evaluate.py --run 0001
    python evaluate.py --run 0002
    python evaluate.py --run 0003
    python evaluate.py --run 0004
    python evaluate.py --run 0001 --smoke-test

Run IDs accept an optional "RUN_" prefix:
    python evaluate.py --run RUN_0003   # same as --run 0003
"""

import argparse
import sys

from src.utils import normalise_run_id, run_cmd, smoke_args as _smoke_args_util

# ---------------------------------------------------------------------------
# Evaluation registry
# ---------------------------------------------------------------------------

EVAL_REGISTRY = {
    "0001": {
        "task": "1a",
        "eval_module": "src.evaluate_task1a",
        "config": "configs/run_0001_baseline.yaml",
        "supports_smoke_test": True,
    },
    "0002": {
        "task": "1a",
        "eval_module": "src.evaluate_task1a_multilabel",
        "config": "configs/run_0002_upf.yaml",
        "supports_smoke_test": True,
    },
    "0003": {
        "task": "2",
        "eval_module": "src.evaluate_task2_dynunet",
        "config": "configs/run_0003_task2_dynunet.yaml",
        "supports_smoke_test": True,
    },
    "0004": {
        "task": "1a+1b+2",
        "eval_module": "src.evaluate_multitask",
        "config": "configs/run_0004_multitask.yaml",
        "supports_smoke_test": True,
    },
}


def _run(cmd: list[str]) -> None:
    run_cmd(cmd)


def _smoke_args(entry: dict, smoke_test: bool) -> list[str]:
    return _smoke_args_util(entry, smoke_test)


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

    smoke_extra = _smoke_args(entry, args.smoke_test)
    _run(
        [sys.executable, "-m", entry["eval_module"], "--config", entry["config"]]
        + smoke_extra
    )
    print(f"\n[OK] Evaluation completed.")


if __name__ == "__main__":
    main()
