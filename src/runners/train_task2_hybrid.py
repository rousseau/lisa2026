#!/usr/bin/env python
"""Train Hybrid nnU-Net + MedSAM2 for Task 2 (RUN_0003c)."""

import argparse

import yaml

from src.training.task2_hybrid import Task2HybridTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 2 Hybrid segmentation model")
    parser.add_argument("--config", type=str, default="configs/run_0003c_task2_hybrid.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Task2HybridTrainer(config=config, smoke_test=args.smoke_test).train()


if __name__ == "__main__":
    main()
