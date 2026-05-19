#!/usr/bin/env python
"""Evaluate multi-task model for LISA 2026 Tasks 1a, 1b and 2 (RUN_0004)."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from monai.inferers import sliding_window_inference
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
from scipy import ndimage
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.datasets import TASK_NAMES, get_multitask_dataloaders
from src.models import SharedEncoderMultiTaskModel

EPS = 1e-8


# ─── Utility functions (mirrored from evaluate_task2_dynunet.py) ────────────────


def to_3tuple(values):
    return tuple(int(v) for v in values)


def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Dice similarity coefficient for a binary pair of tensors."""
    inter = torch.logical_and(pred, target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return float((2.0 * inter + EPS) / (den + EPS))


def compute_rve(pred: torch.Tensor, target: torch.Tensor):
    """Relative Volume Error — returns NaN when prediction is present but GT is empty."""
    pred_vol = float(pred.sum().item())
    true_vol = float(target.sum().item())
    if true_vol <= 0.0 and pred_vol <= 0.0:
        return 0.0
    if true_vol <= 0.0 and pred_vol > 0.0:
        return np.nan
    return float(abs(pred_vol - true_vol) / (true_vol + EPS))


def keep_largest_connected_per_class(pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Retain only the largest connected component for each foreground class."""
    out = np.zeros_like(pred)
    for class_id in range(1, num_classes):
        mask = pred == class_id
        if not np.any(mask):
            continue
        labeled, n_comp = ndimage.label(mask)
        if n_comp <= 1:
            out[mask] = class_id
            continue
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_comp + 1))
        keep = int(np.argmax(sizes)) + 1
        out[labeled == keep] = class_id
    return out


def infer_logits_tta(
    model_fn, image, roi_size, overlap, sw_batch_size, use_amp, tta_axes=None
):
    """Sliding-window inference with optional test-time augmentation (axis flips)."""
    if not tta_axes:
        with torch.amp.autocast("cuda", enabled=use_amp):
            return sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model_fn,
                overlap=overlap,
            )

    logits_sum = None
    variants = [None] + [tuple(ax) for ax in tta_axes]

    for axes in variants:
        inp = torch.flip(image, dims=axes) if axes is not None else image
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = sliding_window_inference(
                inp,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model_fn,
                overlap=overlap,
            )
        if axes is not None:
            logits = torch.flip(logits, dims=axes)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    return logits_sum / float(len(variants))


# ─── Per-task evaluation functions ──────────────────────────────────────────────


@torch.no_grad()
def evaluate_task1a(model, val_loader_1a, device, smoke_test: bool = False) -> dict:
    """Evaluate Task 1a artifact-classification metrics.

    Computes per-artifact and global accuracy, F1 (macro), F2 (macro),
    precision (macro) and recall (macro), following the same methodology as
    RUN_0001 / RUN_0002 evaluations.

    Args:
        model:         Loaded ``SharedEncoderMultiTaskModel`` in eval mode.
        val_loader_1a: Validation DataLoader for Task 1a.
                       Batches: ``{"img": [B,1,H,W,D], "labels": [B,7], ...}``.
        device:        Torch device string.
        smoke_test:    If True, stop after 2 batches.

    Returns:
        ``{"per_task": {task_name: {accuracy, f1_macro, ...}},
           "global":   {accuracy, f1_macro, f2_macro, precision_macro,
                        recall_macro, aggregate}}``
    """
    model.eval()
    use_amp = device == "cuda"

    all_preds = [[] for _ in TASK_NAMES]
    all_labels = [[] for _ in TASK_NAMES]

    for batch_idx, batch in enumerate(tqdm(val_loader_1a, desc="Eval-Task1a")):
        images = batch["img"].to(device)
        labels = batch["labels"].to(device)  # [B, 7]

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model.forward_task1a(images)  # [B, 7, 3]

        preds = torch.argmax(logits, dim=-1).cpu().numpy()  # [B, 7]
        lbl_np = labels.cpu().numpy()  # [B, 7]

        for t in range(len(TASK_NAMES)):
            all_preds[t].extend(preds[:, t].tolist())
            all_labels[t].extend(lbl_np[:, t].tolist())

        if smoke_test and batch_idx >= 1:  # 2 batches max
            break

    per_task = {}
    for t, task_name in enumerate(TASK_NAMES):
        y_true = all_labels[t]
        y_pred = all_preds[t]
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f2 = float(
            fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
        )
        pre = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        agg = float(np.mean([acc, f1, f2, pre, rec]))
        per_task[task_name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "f2_macro": f2,
            "precision_macro": pre,
            "recall_macro": rec,
            "aggregate": agg,
        }

    # Global = mean of each metric across the 7 artifact tasks
    global_acc = float(np.mean([per_task[t]["accuracy"] for t in TASK_NAMES]))
    global_f1 = float(np.mean([per_task[t]["f1_macro"] for t in TASK_NAMES]))
    global_f2 = float(np.mean([per_task[t]["f2_macro"] for t in TASK_NAMES]))
    global_pre = float(np.mean([per_task[t]["precision_macro"] for t in TASK_NAMES]))
    global_rec = float(np.mean([per_task[t]["recall_macro"] for t in TASK_NAMES]))
    global_agg = float(
        np.mean([global_acc, global_f1, global_f2, global_pre, global_rec])
    )

    return {
        "per_task": per_task,
        "global": {
            "accuracy": global_acc,
            "f1_macro": global_f1,
            "f2_macro": global_f2,
            "precision_macro": global_pre,
            "recall_macro": global_rec,
            "aggregate": global_agg,
        },
    }


@torch.no_grad()
def evaluate_task1b(model, val_loader_1b, device, smoke_test: bool = False) -> dict:
    """Evaluate Task 1b reconstruction metrics: PSNR and L1.

    LPIPS and FID are skipped here because the small validation set gives
    unstable estimates; they can be added in post-processing with dedicated
    scripts if needed.

    PSNR is computed per volume on values clamped to [0, 1]:
        PSNR = 10 * log10(1.0 / (MSE + 1e-8))

    Args:
        model:         Loaded ``SharedEncoderMultiTaskModel`` in eval mode.
        val_loader_1b: Validation DataLoader for Task 1b.
                       Batches: ``{"img": [B,1,H,W,D], "subject": [...]}``.
        device:        Torch device string.
        smoke_test:    If True, stop after 3 volumes.

    Returns:
        ``{"psnr": float, "l1": float, "n_subjects": int}``
    """
    model.eval()
    use_amp = device == "cuda"

    psnr_values = []
    l1_values = []
    n_subjects = 0

    for batch_idx, batch in enumerate(tqdm(val_loader_1b, desc="Eval-Task1b")):
        images = batch["img"].to(device)  # [B, 1, H, W, D]

        with torch.amp.autocast("cuda", enabled=use_amp):
            recon = model.forward_task1b(images)  # [B, 1, H, W, D]

        # Clamp to [0, 1] for PSNR computation
        recon_c = recon.clamp(0.0, 1.0)
        target_c = images.clamp(0.0, 1.0)

        # PSNR per volume (val loader uses batch_size=1 but handle >1 gracefully)
        for i in range(images.shape[0]):
            mse = F.mse_loss(recon_c[i : i + 1], target_c[i : i + 1]).item()
            psnr = 10.0 * np.log10(1.0 / (mse + 1e-8))
            psnr_values.append(float(psnr))

        # L1 on unclamped values for consistency with training loss
        l1_values.append(float(F.l1_loss(recon, images).item()))
        n_subjects += int(images.shape[0])

        if smoke_test and batch_idx >= 2:  # 3 volumes max
            break

    return {
        "psnr": float(np.mean(psnr_values)) if psnr_values else float("nan"),
        "l1": float(np.mean(l1_values)) if l1_values else float("nan"),
        "n_subjects": n_subjects,
    }


@torch.no_grad()
def evaluate_task2(
    model, val_loader_2, config, device, smoke_test: bool = False
) -> dict:
    """Evaluate Task 2 segmentation: DSC, HD95, HD, RVE, ASSD per class + global.

    Mirrors the evaluation logic of ``evaluate_task2_dynunet.py`` (RUN_0003)
    for direct comparability.  Uses ``model.forward_task2`` as the sliding-window
    predictor so the shared encoder is exercised with the segmentation decoder.

    Args:
        model:         Loaded ``SharedEncoderMultiTaskModel`` in eval mode.
        val_loader_2:  Validation DataLoader for Task 2.
                       Batches: ``{"img": [B,1,H,W,D], "label": [B,1,H,W,D],
                                  "subject": [...]}``.
        config:        Full config dict (used for inference params and model dims).
        device:        Torch device string.
        smoke_test:    If True, stop after 2 volumes.

    Returns:
        ``{"global": {mean_dsc, mean_hd95, mean_hd, mean_rve, mean_assd,
                      n_subjects, n_classes_eval},
           "per_class": [{class_id, dsc, hd95, hd, rve, assd}, ...]}``
    """
    model.eval()
    use_amp = device == "cuda"

    roi_size = to_3tuple(config["data"]["val_roi_size"])
    overlap = float(config["inference"]["overlap"])
    sw_batch_size = int(config["inference"]["sw_batch_size"])
    tta_axes = config["inference"].get("tta_flip_axes", [])
    keep_largest = bool(config["inference"].get("keep_largest_component", False))
    num_classes = int(config["model"]["num_seg_classes"])

    results_rows = []

    for batch_idx, batch in enumerate(tqdm(val_loader_2, desc="Eval-Task2")):
        image = batch["img"].to(device)
        label = batch["label"].to(device).long()
        subject = batch["subject"][0]

        logits = infer_logits_tta(
            model_fn=model.forward_task2,
            image=image,
            roi_size=roi_size,
            overlap=overlap,
            sw_batch_size=sw_batch_size,
            use_amp=use_amp,
            tta_axes=tta_axes,
        )

        pred = torch.argmax(logits, dim=1).cpu()

        if keep_largest:
            pred_np = pred.squeeze(0).numpy().astype(np.int16)
            pred_np = keep_largest_connected_per_class(pred_np, num_classes=num_classes)
            pred = torch.from_numpy(pred_np[None])

        target = label.squeeze(1).cpu()

        for class_id in range(1, num_classes):
            pred_bin = pred == class_id
            target_bin = target == class_id

            pred_has = bool(pred_bin.any())
            target_has = bool(target_bin.any())

            if not pred_has and not target_has:
                dsc, hd95, hd, assd, rve = 1.0, 0.0, 0.0, 0.0, 0.0
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

                hd95 = float(
                    torch.nan_to_num(hd95_t, nan=0.0, posinf=1e6, neginf=0.0).item()
                )
                hd = float(
                    torch.nan_to_num(hd_t, nan=0.0, posinf=1e6, neginf=0.0).item()
                )
                assd = float(
                    torch.nan_to_num(assd_t, nan=0.0, posinf=1e6, neginf=0.0).item()
                )
                rve = compute_rve(pred_bin, target_bin)
            else:
                dsc, hd95, hd, assd = 0.0, np.nan, np.nan, np.nan
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

        if smoke_test and batch_idx >= 1:  # 2 volumes max
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

    return {"global": global_summary, "per_class": per_class}


# ─── Main ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-task model (RUN_0004): Tasks 1a, 1b and 2."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run_0004_multitask.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help=(
            "Quick sanity check: 2 batches for Task 1a, "
            "3 volumes for Task 1b, 2 volumes for Task 2."
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Allow environment variable override for data root
    config["data"]["data_root"] = os.getenv(
        "LISA_DATA_ROOT", config["data"]["data_root"]
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model_cfg = config["model"]
    model = SharedEncoderMultiTaskModel(
        in_channels=int(model_cfg["in_channels"]),
        filters=tuple(int(x) for x in model_cfg["filters"]),
        num_seg_classes=int(model_cfg["num_seg_classes"]),
        num_artifact_tasks=int(model_cfg["num_artifact_tasks"]),
        num_artifact_classes=int(model_cfg["num_artifact_classes"]),
    ).to(device)

    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "multitask_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train_multitask.py first to produce a checkpoint."
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(
        f"[INFO] Loaded checkpoint from {ckpt_path} "
        f"(epoch {state.get('epoch', '?')}, "
        f"val_dice_2={state.get('val_dice_2', '?'):.4f})"
    )

    # ── Dataloaders (val splits only) ─────────────────────────────────────────
    loaders = get_multitask_dataloaders(config)
    _, val_loader_1a, _, n_val_1a = loaders["1a"]
    _, val_loader_1b, _, n_val_1b = loaders["1b"]
    _, val_loader_2, _, n_val_2 = loaders["2"]

    print(f"Val sizes — 1a: {n_val_1a}, 1b: {n_val_1b}, 2: {n_val_2}")

    # ── Run evaluations ───────────────────────────────────────────────────────
    print("\n── Task 1a — Artifact classification ──")
    results_1a = evaluate_task1a(
        model, val_loader_1a, device, smoke_test=args.smoke_test
    )

    print("\n── Task 1b — Image reconstruction ──")
    results_1b = evaluate_task1b(
        model, val_loader_1b, device, smoke_test=args.smoke_test
    )

    print("\n── Task 2 — Multi-structure segmentation ──")
    results_2 = evaluate_task2(
        model, val_loader_2, config, device, smoke_test=args.smoke_test
    )

    # ── Assemble and save metrics.json ────────────────────────────────────────
    payload = {
        "run_id": config.get("run_id", "0004"),
        "task1a": results_1a,
        "task1b": results_1b,
        "task2": results_2,
    }

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")

    with open(metrics_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n[INFO] Metrics saved to {metrics_path}")
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Task 1a — aggregate       : {results_1a['global']['aggregate']:.4f}")
    print(f"  Task 1a — accuracy (global): {results_1a['global']['accuracy']:.4f}")
    print(f"  Task 1b — PSNR            : {results_1b['psnr']:.2f} dB")
    print(f"  Task 1b — L1              : {results_1b['l1']:.4f}")
    print(f"  Task 1b — n_subjects      : {results_1b['n_subjects']}")
    print(f"  Task 2  — mean DSC        : {results_2['global']['mean_dsc']:.4f}")
    print(f"  Task 2  — mean HD95       : {results_2['global']['mean_hd95']:.2f}")
    print(f"  Task 2  — mean ASSD       : {results_2['global']['mean_assd']:.2f}")
    print(f"  Task 2  — n_subjects      : {results_2['global']['n_subjects']}")


if __name__ == "__main__":
    main()
