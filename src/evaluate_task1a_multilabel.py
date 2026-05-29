#!/usr/bin/env python
"""RUN_0002 — Multi-label Task 1a evaluation (inline metrics).

Loads the single multi-head model, infers on the validation set, computes
per-artifact and global metrics directly, and writes ``metrics.json``.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.datasets import Task1aMultiLabelDataset, TASK_NAMES
from src.evaluation import load_config, get_device, evaluate_task1a_multilabel
from src.models import Task1aMultiLabelModel


def main():
    parser = argparse.ArgumentParser(description="RUN_0002 – Multi-label Task 1a Evaluation")
    parser.add_argument("--config", type=str, default="configs/run_0002_upf.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke_test:
        print("⚡ SMOKE TEST MODE")

    device, use_amp = get_device(config)

    # Load model
    model = Task1aMultiLabelModel().to(device)
    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "multilabel_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    print(f"✓ Loaded checkpoint: {ckpt_path}")

    # Data
    dataset = Task1aMultiLabelDataset(
        csv_path=config["data"]["csv_path"],
        bids_root=config["data"]["bids_root"],
        split_pkl=config["data"]["split_pkl"],
        fold=args.split,
        stage="val",
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    print(f"  {len(dataset)} samples in '{args.split}' split")

    # Evaluate
    results = evaluate_task1a_multilabel(
        model, loader, device, smoke_test=args.smoke_test
    )

    # Save metrics.json
    payload = {
        "run_id": config.get("run_id", "0002"),
        "task": "1a",
        "mode": "multilabel",
        **results,
    }
    out_dir = config["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n✓ Metrics saved to {metrics_path}")
    print(f"  Aggregate : {results['global']['aggregate']:.4f}")
    print(f"  Accuracy  : {results['global']['accuracy']:.4f}")
    print(f"  F1 macro  : {results['global']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
