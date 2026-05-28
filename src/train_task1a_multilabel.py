#!/usr/bin/env python
"""Train Task 1a multi-label classifier — single model, 7 heads (RUN_0002)."""

import argparse

import yaml

from src.training import Task1aMultiLabelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 1a multi-label model (RUN_0002)")
    parser.add_argument("--config", type=str, default="configs/run_0002_upf.yaml")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick sanity check: 2 epochs, 2 batches each")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Task1aMultiLabelTrainer(config=config, smoke_test=args.smoke_test).train()


if __name__ == "__main__":
    main()
