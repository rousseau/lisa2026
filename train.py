#!/usr/bin/env python
"""
Unified training entry point for LISA 2026.

All training scripts live in src/ and are invoked as Python modules so that
their `from src.xxx import ...` imports always resolve from the project root.

Usage
-----
    python train.py --run 0001
    python train.py --run 0002
    python train.py --run 0003
    python train.py --run 0004
    python train.py --run 0005
    python train.py --run 0001 --smoke-test

Run IDs accept an optional "RUN_" prefix:
    python train.py --run RUN_0003   # same as --run 0003
"""

import argparse
import sys

from src.utils import normalise_run_id, run_cmd, smoke_args as _smoke_args_util

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

RUN_REGISTRY = {
    "0001": {
        "task": "1a",
        "module": "src.train_task1a",
        "config": "configs/run_0001_baseline.yaml",
        "mode": "per_task",
        "tasks": [
            "Noise",
            "Zipper",
            "Positioning",
            "Banding",
            "Motion",
            "Contrast",
            "Distortion",
        ],
        "supports_smoke_test": False,
    },
    "0002": {
        "task": "1a",
        "module": "src.train_task1a_multilabel",
        "config": "configs/run_0002_upf.yaml",
        "mode": "single",
        "supports_smoke_test": True,
    },
    "0003": {
        "task": "2",
        "module": "src.train_task2_dynunet",
        "config": "configs/run_0003_task2_dynunet.yaml",
        "mode": "single",
        "supports_smoke_test": True,
    },
    "0004": {
        "task": "1a+1b+2",
        "module": "src.train_multitask",
        "config": "configs/run_0004_multitask.yaml",
        "mode": "single",
        "supports_smoke_test": True,
    },
    "0005": {
        "task": "1b",
        "module": "src.train_task1b",
        "config": "configs/run_0005_task1b_unet.yaml",
        "mode": "single",
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


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run_per_task(entry: dict, smoke_test: bool) -> None:
    tasks = entry["tasks"]
    smoke_extra = _smoke_args(entry, smoke_test)
    print(f"\n[INFO] mode=per_task – {len(tasks)} independent models to train")
    for idx, task_name in enumerate(tasks, start=1):
        print(f"\n{'=' * 60}\n  [{idx}/{len(tasks)}] Task: {task_name}\n{'=' * 60}")
        _run(
            [
                sys.executable,
                "-m",
                entry["module"],
                "--config",
                entry["config"],
                "--task",
                task_name,
            ]
            + smoke_extra
        )
    print(f"\n[OK] All {len(tasks)} models trained.")


def run_single(entry: dict, smoke_test: bool) -> None:
    smoke_extra = _smoke_args(entry, smoke_test)
    print(
        f"\n{'=' * 60}\n  Module : {entry['module']}\n  Config : {entry['config']}\n{'=' * 60}\n"
    )
    _run(
        [sys.executable, "-m", entry["module"], "--config", entry["config"]]
        + smoke_extra
    )
    print(f"\n[OK] Training completed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified training entry point for LISA 2026.",
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
        help="Quick sanity check (limited epochs/batches).",
    )
    args = parser.parse_args()

    run_id = normalise_run_id(args.run)
    if run_id not in RUN_REGISTRY:
        available = "\n  ".join(sorted(RUN_REGISTRY.keys()))
        print(f"\n[ERROR] Run '{run_id}' not registered.\nAvailable:\n  {available}")
        sys.exit(1)

    entry = RUN_REGISTRY[run_id]
    print(f"\n[LISA 2026] Training")
    print(f"  Run    : {run_id}")
    print(f"  Task   : {entry['task']}")
    print(f"  Module : {entry['module']}")
    print(f"  Config : {entry['config']}")
    if args.smoke_test:
        print(f"  Mode   : SMOKE TEST")

    if entry.get("mode", "single") == "per_task":
        run_per_task(entry, smoke_test=args.smoke_test)
    else:
        run_single(entry, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
