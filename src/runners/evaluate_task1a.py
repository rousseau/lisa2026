#!/usr/bin/env python
"""RUN_0001 — Task 1a ordinal evaluation (inline metrics).

Loads the 7 independent ordinal classifiers, infers on the validation set,
computes per-artifact and global metrics directly, and writes ``metrics.json``.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.datasets import Task1aDataset
from src.evaluation import load_config, get_device, evaluate_task1a_ordinal
from src.models import Task1aOrdinalModel


TASK_ORDER = [
    "Noise", "Zipper", "Positioning",
    "Banding", "Motion", "Contrast", "Distortion",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/run_0001_baseline.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke_test:
        print("⚡ SMOKE TEST MODE")

    device, _ = get_device(config)
    ckpt_dir = config["output"]["checkpoint_dir"]
    csv_path = config["data"]["csv_path"]
    bids_root = config["data"]["bids_root"]
    split_pkl = config["data"]["split_pkl"]

    def _make_loader(task: str):
        ds = Task1aDataset(
            csv_path=csv_path,
            bids_root=bids_root,
            split_pkl=split_pkl,
            fold="val",
            task_name=task,
            stage="val",
        )
        return DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    results = evaluate_task1a_ordinal(
        model_factory=Task1aOrdinalModel,
        task_order=TASK_ORDER,
        val_loader_factory=_make_loader,
        ckpt_dir=ckpt_dir,
        device=device,
        smoke_test=args.smoke_test,
    )

    from src.evaluation.metrics_io import build_payload, write_metrics

    payload = build_payload(
        run_id=config.get("run_id", "0001"),
        task="task1a",
        model="ordinal",
        global_metrics=results["global"],
        per_class=results["per_task"],
        extra={"mode": "ordinal"},
    )
    metrics_path = write_metrics(payload, config["output"]["results_dir"])

    print(f"\n✓ Metrics saved to {metrics_path}")
    g = results["global"]
    print(f"  Aggregate : {g['aggregate']:.4f}")
    print(f"  Accuracy  : {g['accuracy']:.4f}")
    print(f"  F1 macro  : {g['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
