#!/usr/bin/env python
"""Evaluate DynUNet baseline for LISA 2026 Task 2 (RUN_0003)."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
from tqdm import tqdm

from src.datasets import get_task2_seg_dataloaders
from src.models import Task2DynUNetModel


EPS = 1e-8


def to_3tuple(values):
    return tuple(int(v) for v in values)


def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    inter = torch.logical_and(pred, target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return float((2.0 * inter + EPS) / (den + EPS))


def compute_rve(pred: torch.Tensor, target: torch.Tensor):
    pred_vol = float(pred.sum().item())
    true_vol = float(target.sum().item())
    if true_vol <= 0.0 and pred_vol <= 0.0:
        return 0.0
    if true_vol <= 0.0 and pred_vol > 0.0:
        return np.nan
    return float(abs(pred_vol - true_vol) / (true_vol + EPS))


@torch.no_grad()
def evaluate(config: dict, smoke_test: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(config["environment"].get("mixed_precision", True)) and device == "cuda"

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
    )

    model_cfg = config["model"]
    num_classes = int(model_cfg["out_channels"])

    model = Task2DynUNetModel(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=num_classes,
        kernel_size=tuple(tuple(int(x) for x in ks) for ks in model_cfg["kernel_size"]),
        strides=tuple(tuple(int(x) for x in st) for st in model_cfg["strides"]),
        upsample_kernel_size=tuple(tuple(int(x) for x in st) for st in model_cfg["upsample_kernel_size"]),
        filters=tuple(int(x) for x in model_cfg["filters"]),
        norm_name=model_cfg.get("norm_name", "instance"),
        deep_supervision=bool(model_cfg.get("deep_supervision", False)),
    ).to(device)

    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "task2_dynunet_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    roi_size = to_3tuple(config["data"]["val_roi_size"])
    overlap = float(config["inference"]["overlap"])
    sw_batch_size = int(config["inference"]["sw_batch_size"])

    results_rows = []

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Eval")):
        image = batch["img"].to(device)
        label = batch["label"].to(device).long()
        subject = batch["subject"][0]

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )

        pred = torch.argmax(logits, dim=1).cpu()
        target = label.squeeze(1).cpu()

        for class_id in range(1, num_classes):
            pred_bin = (pred == class_id)
            target_bin = (target == class_id)

            pred_has = bool(pred_bin.any())
            target_has = bool(target_bin.any())

            if not pred_has and not target_has:
                dsc = 1.0
                hd95 = 0.0
                hd = 0.0
                assd = 0.0
                rve = 0.0
            elif pred_has and target_has:
                dsc = dice_binary(pred_bin, target_bin)

                pred_metric = pred_bin.unsqueeze(1).float()
                target_metric = target_bin.unsqueeze(1).float()

                hd95_t = compute_hausdorff_distance(
                    y_pred=pred_metric,
                    y=target_metric,
                    include_background=True,
                    percentile=95,
                )
                hd_t = compute_hausdorff_distance(
                    y_pred=pred_metric,
                    y=target_metric,
                    include_background=True,
                    percentile=None,
                )
                assd_t = compute_average_surface_distance(
                    y_pred=pred_metric,
                    y=target_metric,
                    include_background=True,
                    symmetric=True,
                )

                hd95 = float(torch.nan_to_num(hd95_t, nan=0.0, posinf=1e6, neginf=0.0).item())
                hd = float(torch.nan_to_num(hd_t, nan=0.0, posinf=1e6, neginf=0.0).item())
                assd = float(torch.nan_to_num(assd_t, nan=0.0, posinf=1e6, neginf=0.0).item())
                rve = compute_rve(pred_bin, target_bin)
            else:
                dsc = 0.0
                hd95 = np.nan
                hd = np.nan
                assd = np.nan
                rve = compute_rve(pred_bin, target_bin)

            results_rows.append(
                {
                    "subject": subject,
                    "class_id": class_id,
                    "dsc": dsc,
                    "hd95": hd95,
                    "hd": hd,
                    "rve": rve,
                    "assd": assd,
                }
            )

        if smoke_test:
            break

    df = pd.DataFrame(results_rows)

    class_summary = (
        df.groupby("class_id")[["dsc", "hd95", "hd", "rve", "assd"]]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )

    global_summary = {
        "mean_dsc": float(np.nanmean(df["dsc"].values)),
        "mean_hd95": float(np.nanmean(df["hd95"].values)),
        "mean_hd": float(np.nanmean(df["hd"].values)),
        "mean_rve": float(np.nanmean(df["rve"].values)),
        "mean_assd": float(np.nanmean(df["assd"].values)),
        "n_subjects": int(df["subject"].nunique()),
        "n_classes_eval": int(df["class_id"].nunique()),
    }

    output_dir = config["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    predictions_file = os.path.join(output_dir, "predictions_val_task2.csv")
    metrics_file = os.path.join(output_dir, "metrics.json")

    df.to_csv(predictions_file, index=False)

    payload = {
        "run_id": config.get("run_id", "0003"),
        "task": "task2",
        "model": "dynunet",
        "global": global_summary,
        "per_class": class_summary,
    }

    with open(metrics_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Evaluation complete. Metrics saved to {metrics_file}")
    print(json.dumps(global_summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/run_0003_task2_dynunet.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config=config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
