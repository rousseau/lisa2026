"""Task 2 segmentation metrics and inference helpers."""

from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage

EPS = 1e-8


def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Dice similarity coefficient for a binary tensor pair.

    Args:
        pred:   Boolean or 0/1 tensor (any shape).
        target: Boolean or 0/1 tensor (same shape as pred).

    Returns:
        DSC ∈ [0, 1].
    """
    inter = torch.logical_and(pred, target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return float((2.0 * inter + EPS) / (den + EPS))


def compute_rve(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative Volume Error.

    Returns:
        RVE ≥ 0 when GT is non-empty, 0.0 when both are empty,
        NaN when GT is empty but prediction is non-empty.
    """
    pred_vol = float(pred.sum().item())
    true_vol = float(target.sum().item())
    if true_vol <= 0.0 and pred_vol <= 0.0:
        return 0.0
    if true_vol <= 0.0 and pred_vol > 0.0:
        return np.nan
    return float(abs(pred_vol - true_vol) / (true_vol + EPS))


def keep_largest_connected_per_class(
    pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Retain only the largest connected component for each foreground class.

    Args:
        pred:        Integer label map (H, W, D).
        num_classes: Total number of classes (background = 0).

    Returns:
        Label map with only the largest component per class.
    """
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
    model_fn: Callable,
    image: torch.Tensor,
    roi_size: Sequence[int],
    overlap: float,
    sw_batch_size: int,
    use_amp: bool,
    tta_axes: Optional[List] = None,
) -> torch.Tensor:
    """Sliding-window inference with optional test-time augmentation (axis flips).

    Args:
        model_fn:      Callable that maps a batch tensor to logits.
        image:         Input image tensor [B, C, H, W, D].
        roi_size:      Sliding window patch size (3-tuple).
        overlap:       Sliding window overlap fraction.
        sw_batch_size: Number of windows per forward pass.
        use_amp:       Whether to use automatic mixed precision.
        tta_axes:      List of axis tuples to flip for TTA, e.g. [[2], [3], [4]].
                       Pass None or [] to disable TTA.

    Returns:
        Averaged logits tensor [B, num_classes, H, W, D].
    """
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
