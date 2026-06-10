#!/usr/bin/env python
"""Train DynUNet baseline for Task 2 (RUN_0003)."""

import argparse

import yaml

from src.training import Task2Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 2 segmentation model")
    parser.add_argument("--config", type=str, default="configs/run_0003_task2_dynunet.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Task2Trainer(config=config, smoke_test=args.smoke_test).train()


if __name__ == "__main__":
    main()
