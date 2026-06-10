#!/usr/bin/env python
"""Unified evaluation entry point for LISA 2026.

Usage
-----
    python evaluate.py --run 0001
    python evaluate.py --run 0002 --smoke-test
    python evaluate.py --list-runs
"""

import sys

from src.cli.registry import discover_runs
from src.cli.common import build_parser, resolve_run, dispatch_single, dispatch_per_task


def main() -> None:
    parser = build_parser("Unified evaluation entry point for LISA 2026.")
    args = parser.parse_args()

    registry = discover_runs()
    entry = resolve_run(args, registry)

    run_id = entry["run_id"]
    print(f"\n[LISA 2026] Evaluation")
    print(f"  Run    : {run_id}")
    print(f"  Task   : {entry['task']}")
    print(f"  Module : {entry['eval_module']}")
    print(f"  Config : {entry['config']}")
    if args.smoke_test:
        print("  Mode   : SMOKE TEST")

    if entry.get("mode", "single") == "per_task":
        dispatch_per_task(entry, smoke_test=args.smoke_test, eval_mode=True)
    else:
        dispatch_single(entry["eval_module"], entry["config"], args.smoke_test, entry)


if __name__ == "__main__":
    main()
