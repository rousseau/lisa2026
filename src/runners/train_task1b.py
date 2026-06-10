"""Training entry point for RUN_0002 — CycleGAN Task 1b."""

import argparse

import yaml

from src.training.task1b import CycleGANTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CycleGAN for Task 1b (RUN_0002).")
    parser.add_argument("--config", default="configs/run_0002_cyclegan_task1b.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--smoke_test", action="store_true", dest="smoke_test",
                        help="Quick sanity check (2 epochs, 2 batches).")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    trainer = CycleGANTrainer(config=config, smoke_test=args.smoke_test)
    trainer.train()


if __name__ == "__main__":
    main()
