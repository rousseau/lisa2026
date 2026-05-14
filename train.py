#!/usr/bin/env python
"""
Unified training entry point for LISA 2026.

Routes to the correct training script based on the run ID. All scripts are
launched via subprocess so that their environment and working directory are
preserved.

Usage:
    python train.py --run 0001
    python train.py --run 0002
    python train.py --run 0003
    python train.py --run 0003_COLLAPSED
    python train.py --run 0003_EXP_C
    python train.py --run 0004
    python train.py --run 0001 --smoke-test

Run IDs are case-insensitive and accept optional "RUN_" / "run_" prefix:
    python train.py --run RUN_0003_EXP_C   # same as --run 0003_EXP_C
"""

import argparse
import subprocess
import sys

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------
# Each entry describes which script and config to use, and how to call it.
#
# mode:
#   "per_task"  – script is called once per task name in the "tasks" list;
#                 each invocation receives --task TASK_NAME (no --smoke_test,
#                 because train_task1a.py does not expose that flag).
#   "single"    – script is called once with --config CONFIG.
#
# supports_smoke_test:
#   Set to False when the underlying script does not parse --smoke_test.
#   A warning is printed and the run proceeds without the flag.

RUN_REGISTRY = {
    "0001": {
        "task": "1a",
        "script": "train_task1a.py",
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
        "supports_smoke_test": False,  # train_task1a.py does not expose --smoke_test
    },
    "0002": {
        "task": "1a",
        "script": "train_task1a_multilabel.py",
        "config": "configs/run_0002_upf.yaml",
        "mode": "single",
        "supports_smoke_test": True,
    },
    "0003": {
        "task": "2",
        "script": "train_task2_dynunet.py",
        "config": "configs/run_0003_task2_dynunet.yaml",
        "mode": "single",
        "supports_smoke_test": True,
    },
    "0003_COLLAPSED": {
        "task": "2",
        "script": "train_task2_dynunet.py",
        "config": "configs/run_0003_task2_dynunet_collapsed.yaml",
        "mode": "single",
        "supports_smoke_test": True,
    },
    "0003_EXP_C": {
        "task": "2",
        "script": "train_task2_dynunet.py",
        "config": "configs/run_0003_task2_dynunet_expC.yaml",
        "mode": "single",
        "supports_smoke_test": True,
    },
    "0004": {
        "task": "1b",
        "script": "train_task1b.py",
        "config": "configs/run_0004_task1b_unet.yaml",
        "mode": "single",
        "supports_smoke_test": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalise_run_id(raw: str) -> str:
    """Strip optional 'RUN_' / 'run_' prefix and upper-case variant prefixes.

    Examples
    --------
    "RUN_0001"       -> "0001"
    "run_0003_EXP_C" -> "0003_EXP_C"
    "0003_EXP_C"     -> "0003_EXP_C"
    """
    upper = raw.strip().upper()
    if upper.startswith("RUN_"):
        return raw.strip()[4:]  # preserve original case of the suffix
    return raw.strip()


def _run_subprocess(cmd: list[str]) -> None:
    """Run *cmd* with error handling. Exits the process on failure."""
    print(f"  $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Command failed with return code {exc.returncode}:")
        print(f"  {' '.join(cmd)}")
        sys.exit(1)


def _smoke_args(entry: dict, smoke_test: bool) -> list[str]:
    """Return ['--smoke_test'] if applicable, else []."""
    if not smoke_test:
        return []
    if entry.get("supports_smoke_test", True):
        return ["--smoke_test"]
    print(
        f"  [WARNING] '{entry['script']}' does not support --smoke_test; "
        "running without it."
    )
    return []


# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------


def run_per_task(entry: dict, smoke_test: bool) -> None:
    """Call the script once per task name (Task 1a per-model mode)."""
    script = entry["script"]
    config = entry["config"]
    tasks = entry["tasks"]
    smoke_extra = _smoke_args(entry, smoke_test)

    print(f"\n[INFO] mode=per_task – {len(tasks)} independent models to train")
    for idx, task_name in enumerate(tasks, start=1):
        print(f"\n{'=' * 60}")
        print(f"  [{idx}/{len(tasks)}] Training task: {task_name}")
        print(f"{'=' * 60}")
        cmd = [sys.executable, script, "--config", config, "--task", task_name]
        cmd += smoke_extra
        _run_subprocess(cmd)

    print(f"\n[OK] All {len(tasks)} task models trained successfully.")


def run_single(entry: dict, smoke_test: bool) -> None:
    """Call the script once with --config (single-model mode)."""
    script = entry["script"]
    config = entry["config"]
    smoke_extra = _smoke_args(entry, smoke_test)

    print(f"\n{'=' * 60}")
    print(f"  Training  script : {script}")
    print(f"  Config           : {config}")
    print(f"{'=' * 60}\n")

    cmd = [sys.executable, script, "--config", config] + smoke_extra
    _run_subprocess(cmd)

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
        type=str,
        required=True,
        metavar="RUN_ID",
        help="Run identifier, e.g. 0001, 0003_EXP_C, RUN_0002.",
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
        print(f"\n[ERROR] Run '{run_id}' is not registered.")
        print(f"\nAvailable runs:\n  {available}")
        print("\nTip: prefix 'RUN_' is automatically stripped (e.g. RUN_0001 -> 0001).")
        sys.exit(1)

    entry = RUN_REGISTRY[run_id]

    print(f"\n[LISA 2026] Starting training")
    print(f"  Run ID : {run_id}")
    print(f"  Task   : {entry['task']}")
    print(f"  Script : {entry['script']}")
    print(f"  Config : {entry['config']}")
    if args.smoke_test:
        print(f"  Mode   : SMOKE TEST")

    mode = entry.get("mode", "single")
    if mode == "per_task":
        run_per_task(entry, smoke_test=args.smoke_test)
    elif mode == "single":
        run_single(entry, smoke_test=args.smoke_test)
    else:
        print(f"[ERROR] Unknown mode '{mode}' for run {run_id}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
