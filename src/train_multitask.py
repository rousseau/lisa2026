#!/usr/bin/env python
"""Multi-task training: Tasks 1a, 1b and 2 with shared DynUNet encoder (RUN_0004)."""

import argparse

import yaml

from src.training import MultiTaskTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-task model (RUN_0004)")
    parser.add_argument("--config", type=str, default="configs/run_0004_multitask.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    MultiTaskTrainer(config=config, smoke_test=args.smoke_test).train()


if __name__ == "__main__":
    main()
