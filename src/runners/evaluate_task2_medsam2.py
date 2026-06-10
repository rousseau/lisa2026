#!/usr/bin/env python
"""Evaluate MedSAM2 model for LISA 2026 Task 2 (RUN_0003b)."""

import argparse
import json
import os

import torch

from src.datasets import get_task2_seg_dataloaders
from src.evaluation import load_config, get_device, evaluate_task2
from src.models import Task2MedSAM2Model


def to_3tuple(values):
    return tuple(int(v) for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/run_0003b_task2_medsam2.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device, use_amp = get_device(config)

    data_root = os.getenv("LISA_DATA_ROOT", config["data"]["data_root"])
    split_pkl = os.getenv("LISA_TASK2_SPLIT_PKL", config["data"]["split_pkl"])

    _, val_loader, _, _ = get_task2_seg_dataloaders(
        data_root=data_root,
        split_pkl=split_pkl,
        batch_size=1,
        num_workers=int(config["training"]["num_workers"]),
        image_suffix=config["data"].get("image_suffix", "_ciso.nii.gz"),
        label_suffix=config["data"].get("label_suffix", "_LF_seg.nii.gz"),
        patch_size=to_3tuple(config["data"]["patch_size"]),
        num_samples_per_volume=1,
        collapse_labels=bool(config["data"].get("collapse_labels", False)),
    )

    model_cfg = config["model"]
    num_classes = int(model_cfg["out_channels"])
    model = Task2MedSAM2Model(
        medsam2_checkpoint=model_cfg["medsam2_checkpoint"],
        medsam2_config=model_cfg.get("medsam2_config", "configs/sam2.1_hiera_t512.yaml"),
        num_classes=num_classes,
        decoder_filters=tuple(int(x) for x in model_cfg.get("decoder_filters", [32, 64, 128, 256, 320])),
        device=device,
    ).to(device)

    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "task2_medsam2_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    results = evaluate_task2(model, val_loader, config, device, smoke_test=args.smoke_test)

    from src.evaluation.metrics_io import build_payload, write_metrics

    payload = build_payload(
        run_id=config.get("run_id", "0003b"),
        task="task2",
        model="medsam2",
        global_metrics=results["global"],
        per_class=results["per_class"],
    )
    metrics_path = write_metrics(payload, config["output"]["results_dir"])

    print(f"\n✓ Metrics saved to {metrics_path}")
    g = results["global"]
    print(f"  mean DSC  : {g['mean_dsc']:.4f}")
    print(f"  mean HD95 : {g['mean_hd95']:.2f}")
    print(f"  mean ASSD : {g['mean_assd']:.2f}")


if __name__ == "__main__":
    main()
