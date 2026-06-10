#!/usr/bin/env python
"""Train MedSAM2 + DynUNet for Task 2 (RUN_0003b)."""

import argparse

import yaml

from src.training.task2_medsam2 import Task2MedSAM2Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 2 MedSAM2 segmentation model")
    parser.add_argument("--config", type=str, default="configs/run_0003b_task2_medsam2.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Task2MedSAM2Trainer(config=config, smoke_test=args.smoke_test).train()


if __name__ == "__main__":
    main()
