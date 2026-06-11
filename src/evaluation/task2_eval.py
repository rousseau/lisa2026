"""Task 2 evaluation — DSC, HD95, HD, RVE, ASSD per class + global.

Shared between standalone Task 2 evaluator and ``evaluate_multitask.py``.
"""

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
from tqdm import tqdm

from src.utils.metrics.segmentation import (
    dice_binary,
    compute_rve,
    keep_largest_connected_per_class,
    infer_logits_tta,
)


_EVAL_CLASSES_CACHE = {}


def to_3tuple(values):
    return tuple(int(v) for v in values)


def _get_num_segmentation_classes(config: dict) -> int:
    """Resolve segmentation class count from model config.

    Supports both single-task and multitask config conventions.
    """
    model_cfg = config.get("model", {})
    for key in ("num_seg_classes", "out_channels", "num_classes"):
        if key in model_cfg:
            return int(model_cfg[key])

    available = ", ".join(sorted(model_cfg.keys())) if model_cfg else "<none>"
    raise KeyError(
        "Missing segmentation class count in config['model']. "
        "Expected one of: num_seg_classes, out_channels, num_classes. "
        f"Available keys: {available}"
    )


def evaluate_task2(
    model,
    val_loader,
    config: dict,
    device: str,
    smoke_test: bool = False,
    model_fn=None,
) -> dict:
    """Evaluate Task 2 segmentation metrics inline.

    Parameters
    ----------
    model : nn.Module
        The segmentation model.
    val_loader : DataLoader
        Validation loader yielding ``{img, label, subject}``.
    config : dict
        Full config (needs ``data.val_roi_size``, ``inference.*``, and one of
        ``model.num_seg_classes``, ``model.out_channels`` or ``model.num_classes``).
    device : str
        Torch device.
    smoke_test : bool
    model_fn : callable, optional
        If provided, called as ``model_fn(image)`` instead of ``model(image)``.
        Used by multitask evaluation to route through ``model.forward_task2``.

    Returns
    -------
    dict with keys ``global`` and ``per_class``.
    """
    model.eval()
    use_amp = device == "cuda"
    roi_size = to_3tuple(config["data"]["val_roi_size"])
    overlap = float(config["inference"]["overlap"])
    sw_batch_size = int(config["inference"]["sw_batch_size"])
    tta_axes = config["inference"].get("tta_flip_axes", [])
    stitch_device = config["inference"].get("stitch_device", "cpu")
    keep_largest = bool(config["inference"].get("keep_largest_component", False))
    num_classes = _get_num_segmentation_classes(config)

    _predict_fn = model_fn if model_fn is not None else model

    results_rows = []

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Eval-Task2")):
        image = batch["img"].to(device)
        label = batch["label"].long()
        subject = batch["subject"][0]

        with torch.inference_mode():
            logits = infer_logits_tta(
                model_fn=_predict_fn,
                image=image,
                roi_size=roi_size,
                overlap=overlap,
                sw_batch_size=sw_batch_size,
                use_amp=use_amp,
                tta_axes=tta_axes,
                stitch_device=stitch_device,
            )
            pred = torch.argmax(logits, dim=1).cpu()
        del logits

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
                pred_m = pred_bin.unsqueeze(1).float()
                target_m = target_bin.unsqueeze(1).float()
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
                hd   = float(torch.nan_to_num(hd_t,   nan=0.0, posinf=1e6, neginf=0.0).item())
                assd = float(torch.nan_to_num(assd_t, nan=0.0, posinf=1e6, neginf=0.0).item())
                rve = compute_rve(pred_bin, target_bin)
            else:
                dsc, hd95, hd, assd = 0.0, float("nan"), float("nan"), float("nan")
                rve = compute_rve(pred_bin, target_bin)

            results_rows.append({
                "subject": subject,
                "class_id": class_id,
                "dsc": dsc,
                "hd95": hd95,
                "hd": hd,
                "rve": rve,
                "assd": assd,
            })

        if smoke_test and batch_idx >= 1:
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
