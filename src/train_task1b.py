#!/usr/bin/env python
"""Train 3D BasicUNet self-supervised denoising for Task 1b (RUN_0005)."""

import argparse

import yaml

from src.training import Task1bTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 1b denoising model")
    parser.add_argument("--config", type=str, default="configs/run_0005_task1b_unet.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Task1bTrainer(config=config, smoke_test=args.smoke_test).train()


if __name__ == "__main__":
    main()
