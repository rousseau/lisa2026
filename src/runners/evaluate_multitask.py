#!/usr/bin/env python
"""Evaluate multi-task model for LISA 2026 Tasks 1a, 1b and 2 (RUN_0004)."""

import argparse
import json
import os

import torch
import yaml

from src.datasets import TASK_NAMES, get_multitask_dataloaders
from src.evaluation import (
    evaluate_task1a_multilabel,
    evaluate_task1b,
    evaluate_task2,
)
from src.models import DynUNetMultiHeadModel


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-task model (RUN_0004): Tasks 1a, 1b and 2."
    )
    parser.add_argument(
        "--config", type=str, default="configs/run_0004_multitask.yaml"
    )
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["data"]["data_root"] = os.getenv(
        "LISA_DATA_ROOT", config["data"]["data_root"]
    )

    model_cfg = config["model"]
    model = DynUNetMultiHeadModel(
        in_channels=int(model_cfg.get("in_channels", 1)),
        filters=tuple(int(x) for x in model_cfg["filters"]),
        num_seg_classes=int(model_cfg["num_seg_classes"]),
        num_artifact_tasks=int(model_cfg["num_artifact_tasks"]),
        num_artifact_classes=int(model_cfg["num_artifact_classes"]),
    ).to(device)

    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "multitask_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.eval()
    print(
        f"[INFO] Loaded checkpoint from {ckpt_path} "
        f"(epoch {state.get('epoch', '?')}, "
        f"val_dice_2={state.get('val_dice_2', '?'):.4f})"
    )

    loaders = get_multitask_dataloaders(config)
    _, val_loader_1a, _, n_val_1a = loaders["1a"]
    _, val_loader_1b, _, n_val_1b = loaders["1b"]
    _, val_loader_2, _, n_val_2 = loaders["2"]
    print(f"Val sizes — 1a: {n_val_1a}, 1b: {n_val_1b}, 2: {n_val_2}")

    print("\n── Task 1a — Artifact classification ──")
    results_1a = evaluate_task1a_multilabel(
        model, val_loader_1a, device, smoke_test=args.smoke_test
    )

    print("\n── Task 1b — Image reconstruction ──")
    fid_slices = int(config.get("inference", {}).get("fid_num_slices_per_volume", 10))
    results_1b = evaluate_task1b(
        model, val_loader_1b, device,
        smoke_test=args.smoke_test,
        fid_num_slices_per_volume=fid_slices,
    )

    print("\n── Task 2 — Multi-structure segmentation ──")
    results_2 = evaluate_task2(
        model, val_loader_2, config, device,
        smoke_test=args.smoke_test,
        model_fn=model.forward_task2,
    )

    from src.evaluation.metrics_io import build_payload, write_metrics

    payload = build_payload(
        run_id=config.get("run_id", "0004"),
        task="multitask",
        model="dynunet_multihead",
        global_metrics={
            "task1a_aggregate": results_1a["global"]["aggregate"],
            "task1a_accuracy": results_1a["global"]["accuracy"],
            "task1b_psnr": results_1b.get("psnr", float("nan")),
            "task1b_lpips": results_1b.get("lpips", float("nan")),
            "task2_mean_dsc": results_2["global"]["mean_dsc"],
            "task2_mean_hd95": results_2["global"]["mean_hd95"],
        },
        extra={
            "task1a": results_1a,
            "task1b": results_1b,
            "task2": results_2,
        },
    )
    metrics_path = write_metrics(payload, config["output"]["results_dir"])

    print(f"\n[INFO] Metrics saved to {metrics_path}")
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Task 1a — aggregate       : {results_1a['global']['aggregate']:.4f}")
    print(f"  Task 1a — accuracy (global): {results_1a['global']['accuracy']:.4f}")
    print(f"  Task 1b — PSNR            : {results_1b['psnr']:.2f} dB")
    print(f"  Task 1b — LPIPS           : {results_1b['lpips']:.4f}")
    print(f"  Task 1b — FID             : {results_1b['fid']:.4f}")
    print(f"  Task 1b — L1              : {results_1b['l1']:.4f}")
    print(f"  Task 1b — n_subjects      : {results_1b['n_subjects']}")
    print(f"  Task 2  — mean DSC        : {results_2['global']['mean_dsc']:.4f}")
    print(f"  Task 2  — mean HD95       : {results_2['global']['mean_hd95']:.2f}")
    print(f"  Task 2  — mean ASSD       : {results_2['global']['mean_assd']:.2f}")
    print(f"  Task 2  — n_subjects      : {results_2['global']['n_subjects']}")


if __name__ == "__main__":
    main()
