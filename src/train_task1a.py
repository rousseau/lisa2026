#!/usr/bin/env python
"""Train Task 1a ordinal classifier — one model per artifact type (RUN_0001)."""

import argparse

import yaml

from src.training import Task1aOrdinalTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 1a per-task model (RUN_0001)")
    parser.add_argument("--config", type=str, default="configs/run_0001_baseline.yaml")
    parser.add_argument("--task", type=str, required=True,
                        help="Artifact task name, e.g. Noise, Zipper, Motion …")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Task1aOrdinalTrainer(config=config, task_name=args.task, smoke_test=args.smoke_test).train()


if __name__ == "__main__":
    main()
