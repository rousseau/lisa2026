"""Common CLI helpers shared by train.py and evaluate.py entry points."""

import argparse
import sys
from typing import Callable

from src.utils import normalise_run_id, run_cmd, smoke_args as _smoke_args_util
from src.cli.registry import discover_runs, list_runs


def _smoke_args(entry: dict, smoke_test: bool) -> list[str]:
    """Return smoke-test CLI flags if supported by the run entry."""
    return _smoke_args_util(entry, smoke_test)


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build an ArgumentParser with the standard LISA 2026 CLI flags."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        metavar="RUN_ID",
        help="Run ID, e.g. 0001, 0003, RUN_0002.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        dest="smoke_test",
        help="Quick sanity check (limited epochs/batches).",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all discovered runs and exit.",
    )
    return parser


def resolve_run(args, registry: dict) -> dict:
    """Normalise run_id, validate existence, and return the registry entry."""
    if args.list_runs:
        list_runs()
        sys.exit(0)

    if not args.run:
        print("[ERROR] --run is required (use --list-runs to see available runs).")
        sys.exit(1)

    run_id = normalise_run_id(args.run)
    if run_id not in registry:
        available = "\n  ".join(sorted(registry.keys()))
        print(f"\n[ERROR] Run '{run_id}' not registered.\nAvailable:\n  {available}")
        sys.exit(1)

    return registry[run_id]


def dispatch_single(module: str, config_path: str, smoke_test: bool, entry: dict) -> None:
    """Dispatch a single training or evaluation command via subprocess."""
    smoke_extra = _smoke_args(entry, smoke_test)
    print(
        f"\n{'=' * 60}\n  Module : {module}\n  Config : {config_path}\n{'=' * 60}\n"
    )
    run_cmd([sys.executable, "-m", module, "--config", config_path] + smoke_extra)
    print("\n[OK] Completed.")


def dispatch_per_task(entry: dict, smoke_test: bool, *, eval_mode: bool = False) -> None:
    """Dispatch training/evaluation in per-task mode.

    In training mode each task gets an independent subprocess with --task.
    In evaluation mode we currently evaluate each task sequentially in a single
    subprocess (the downstream evaluator decides how to split tasks).
    """
    tasks = entry.get("tasks", [])
    if not tasks:
        print("[ERROR] per_task mode requested but no tasks listed in config.")
        sys.exit(1)

    smoke_extra = _smoke_args(entry, smoke_test)
    module = entry.get("eval_module" if eval_mode else "module")
    config_path = entry["config"]

    print(f"\n[INFO] mode=per_task – {len(tasks)} task(s) to process")

    for idx, task_name in enumerate(tasks, start=1):
        print(f"\n{'=' * 60}\n  [{idx}/{len(tasks)}] Task: {task_name}\n{'=' * 60}")
        cmd = [sys.executable, "-m", module, "--config", config_path, "--task", task_name]
        run_cmd(cmd + smoke_extra)

    print(f"\n[OK] All {len(tasks)} tasks processed.")
