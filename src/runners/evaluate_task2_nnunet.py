#!/usr/bin/env python
"""Evaluate nnU-Net v2 predictions for Task 2 (RUN_0003a).

Reads NIfTI predictions from disk and computes DSC, HD95, HD, RVE, ASSD
using the same metric logic as the shared ``src.evaluation.task2_eval``.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance

from src.evaluation.common import load_config, should_break
from src.evaluation.metrics_io import build_payload, write_metrics
from src.utils.metrics.segmentation import dice_binary, compute_rve, keep_largest_connected_per_class


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib
    return nib.load(path).get_fdata()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, help="nnU-Net predictions directory")
    parser.add_argument("--gt-dir", required=True, help="Ground truth labels directory")
    parser.add_argument("--config", default="configs/run_0003a_task2_nnunet.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"[WARN] No prediction files found in {pred_dir}")
        return

    results_rows = []
    num_classes = int(config["model"].get("out_channels", 12))

    for idx, pred_path in enumerate(pred_files):
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            print(f"[WARN] Ground truth not found for {case_id}")
            continue

        pred = _load_nifti(pred_path).astype(np.int16)
        target = _load_nifti(gt_path).astype(np.int16)

        if bool(config.get("inference", {}).get("keep_largest_component", False)):
            pred = keep_largest_connected_per_class(pred, num_classes=num_classes)

        for class_id in range(1, num_classes):
            pred_bin = pred == class_id
            target_bin = target == class_id
            pred_has = bool(pred_bin.any())
            target_has = bool(target_bin.any())

            if not pred_has and not target_has:
                dsc, hd95, hd, assd, rve = 1.0, 0.0, 0.0, 0.0, 0.0
            elif pred_has and target_has:
                dsc = dice_binary(torch.from_numpy(pred_bin), torch.from_numpy(target_bin))
                pred_m = torch.from_numpy(pred_bin).unsqueeze(0).unsqueeze(0).float()
                target_m = torch.from_numpy(target_bin).unsqueeze(0).unsqueeze(0).float()

                hd95_t = compute_hausdorff_distance(
                    y_pred=pred_m, y=target_m,
                    include_background=True, percentile=95,
                )
                hd_t = compute_hausdorff_distance(
                    y_pred=pred_m, y=target_m,
                    include_background=True, percentile=None,
                )
                assd_t = compute_average_surface_distance(
                    y_pred=pred_m, y=target_m,
                    include_background=True, symmetric=True,
                )
                hd95 = float(torch.nan_to_num(hd95_t, nan=0.0, posinf=1e6, neginf=0.0).item())
                hd = float(torch.nan_to_num(hd_t, nan=0.0, posinf=1e6, neginf=0.0).item())
                assd = float(torch.nan_to_num(assd_t, nan=0.0, posinf=1e6, neginf=0.0).item())
                rve = compute_rve(torch.from_numpy(pred_bin), torch.from_numpy(target_bin))
            else:
                dsc, hd95, hd, assd = 0.0, float("nan"), float("nan"), float("nan")
                rve = compute_rve(torch.from_numpy(pred_bin), torch.from_numpy(target_bin))

            results_rows.append({
                "subject": case_id,
                "class_id": class_id,
                "dsc": dsc,
                "hd95": hd95,
                "hd": hd,
                "rve": rve,
                "assd": assd,
            })

        if should_break(idx, args.smoke_test, limit=1):
            break

    df = pd.DataFrame(results_rows)
    per_class = (
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

    payload = build_payload(
        run_id=config.get("run_id", "0003a"),
        task="task2",
        model="nnunetv2",
        global_metrics=global_summary,
        per_class=per_class,
    )
    metrics_path = write_metrics(payload, config["output"]["results_dir"])

    print(f"\n[OK] Metrics saved to {metrics_path}")
    g = global_summary
    print(f"  mean DSC  : {g['mean_dsc']:.4f}")
    print(f"  mean HD95 : {g['mean_hd95']:.2f}")
    print(f"  mean ASSD : {g['mean_assd']:.2f}")


if __name__ == "__main__":
    main()
