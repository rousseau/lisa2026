#!/usr/bin/env python
"""Evaluate Task 1b denoising model: PSNR, LPIPS, FID (RUN_0004).

Evaluation protocol
--------------------
  * A fixed mid-range Gaussian noise (mean of noise_std_min and noise_std_max)
    is applied to each clean validation volume to produce the degraded input.
  * The model predicts the denoised output.
  * Per-volume PSNR is computed in 3D on intensities normalised to [0, 1].
  * Per-slice LPIPS is computed on axial slices converted to 3-channel [−1, 1].
  * FID is computed globally on InceptionV3 pool features extracted from all
    axial slices (real = clean, fake = predicted).

Usage
-----
  python evaluate_task1b.py --config configs/run_0004_task1b_unet.yaml
  python evaluate_task1b.py --config configs/run_0004_task1b_unet.yaml --smoke_test
"""

import argparse
import json
import os
from typing import Any

import lpips
import numpy as np
import torch
import yaml
from scipy.linalg import sqrtm
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

from src.datasets import get_task1b_dataloaders
from src.models import Task1bUNetModel

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def add_synthetic_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Add Gaussian noise at a fixed std level for deterministic evaluation."""
    noise = torch.randn_like(x) * noise_std
    return (x + noise).clamp(x.min(), x.max())


def compute_psnr(
    pred: np.ndarray, target: np.ndarray, data_range: float = 1.0
) -> float:
    """Compute Peak Signal-to-Noise Ratio (volumetric, 3D).

    Args:
        pred:       Predicted array, same shape as *target*, values in [0, 1].
        target:     Ground-truth array, values in [0, 1].
        data_range: Maximum possible value (1.0 for normalised inputs).

    Returns:
        PSNR in dB, or ``float('inf')`` for a perfect reconstruction.
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range**2 / mse)


def slice_to_rgb(volume: np.ndarray, z: int) -> np.ndarray:
    """Extract an axial slice from a 3-D volume and duplicate to 3 channels.

    Args:
        volume: Array of shape ``[H, W, D]`` with values in [0, 1].
        z:      Axial index along the last dimension.

    Returns:
        RGB array of shape ``[3, H, W]`` in [0, 1].
    """
    slc = volume[:, :, z]
    vmin, vmax = float(slc.min()), float(slc.max())
    if vmax > vmin:
        slc = (slc - vmin) / (vmax - vmin)
    else:
        slc = np.zeros_like(slc)
    return np.stack([slc, slc, slc], axis=0).astype(np.float32)


def extract_inception_features(
    slices_rgb: np.ndarray,
    model: torch.nn.Module,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract 2048-dim InceptionV3 pool features from a set of 2D RGB slices.

    Args:
        slices_rgb: Array of shape ``[N, 3, H, W]`` with values in [0, 1].
        model:      InceptionV3 model with ``fc`` replaced by ``Identity``.
        device:     Torch device string.
        batch_size: Inference mini-batch size.

    Returns:
        Features array of shape ``[N, 2048]``.
    """
    model.eval()
    features = []
    n = len(slices_rgb)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(slices_rgb[start:end], dtype=torch.float32).to(device)
        # InceptionV3 expects 299×299 inputs.
        batch = torch.nn.functional.interpolate(
            batch, size=(299, 299), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            feat = model(batch)  # [B, 2048]
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute Fréchet Inception Distance between two sets of feature vectors.

    Args:
        real_features: ``[N, D]`` features from real (clean) slices.
        fake_features: ``[N, D]`` features from predicted (denoised) slices.

    Returns:
        Scalar FID score.
    """
    mu1 = real_features.mean(0)
    mu2 = fake_features.mean(0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)
    diff = mu1 - mu2
    covmean_raw = sqrtm(sigma1.dot(sigma2))
    # np.real handles both complex and real arrays, avoiding ndarray.real property issues.
    covmean: np.ndarray = np.real(covmean_raw)  # type: ignore[arg-type]
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(config: "dict[str, Any]", smoke_test: bool = False) -> None:
    """Run the full Task 1b evaluation pipeline and write metrics.json."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (
        bool(config["environment"].get("mixed_precision", True)) and device == "cuda"
    )

    # ── data ───────────────────────────────────────────────────────────────
    data_root = os.getenv("LISA_DATA_ROOT", config["data"]["data_root"])
    split_pkl = config["data"]["split_pkl"]
    spatial_size = tuple(int(s) for s in config["data"]["spatial_size"])
    num_workers = int(config["training"]["num_workers"])
    # Use fixed mid-range noise for reproducible evaluation.
    noise_std = (
        float(config["data"].get("noise_std_min", 0.05))
        + float(config["data"].get("noise_std_max", 0.20))
    ) / 2.0
    num_slices_per_volume = int(
        config["inference"].get("fid_num_slices_per_volume", 10)
    )

    _, val_loader, _, n_val = get_task1b_dataloaders(
        data_root=data_root,
        split_pkl=split_pkl,
        batch_size=1,
        num_workers=num_workers,
        image_suffix=config["data"].get("image_suffix", "_ciso.nii.gz"),
        spatial_size=spatial_size,
    )
    print(f"Validation volumes: {n_val}")

    # ── model ──────────────────────────────────────────────────────────────
    model_cfg = config["model"]
    model = Task1bUNetModel(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        features=tuple(int(f) for f in model_cfg["features"]),
    ).to(device)

    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "task1b_unet_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train_task1b.py first or check the checkpoint_dir in the config."
        )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── LPIPS ──────────────────────────────────────────────────────────────
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    # ── InceptionV3 for FID (pool features, no classification head) ────────
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception_model.aux_logits = False
    # Replace FC with Identity to get the 2048-dim pool layer output.
    # setattr used to bypass static type-checker (fc is typed as Linear).
    setattr(inception_model, "fc", torch.nn.Identity())
    inception_model = inception_model.to(device)
    inception_model.eval()

    # ── accumulate metrics ─────────────────────────────────────────────────
    psnr_values: list[float] = []
    lpips_values: list[float] = []
    real_slices: list[np.ndarray] = []
    fake_slices: list[np.ndarray] = []

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Eval")):
        clean = batch["img"].to(device)  # [1, 1, H, W, D]
        degraded = add_synthetic_noise(clean, noise_std=noise_std)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(degraded)

        # ── PSNR (3-D volumetric) ──────────────────────────────────────────
        pred_np = pred.squeeze().cpu().float().numpy()  # [H, W, D]
        clean_np = clean.squeeze().cpu().float().numpy()  # [H, W, D]

        c_min, c_max = float(clean_np.min()), float(clean_np.max())
        if c_max > c_min:
            pred_norm = np.clip((pred_np - c_min) / (c_max - c_min), 0.0, 1.0)
            clean_norm = np.clip((clean_np - c_min) / (c_max - c_min), 0.0, 1.0)
        else:
            pred_norm = np.zeros_like(pred_np)
            clean_norm = np.zeros_like(clean_np)

        psnr_values.append(compute_psnr(pred_norm, clean_norm, data_range=1.0))

        # ── slice selection (axial, central 50 %) ─────────────────────────
        depth = clean_norm.shape[-1]
        step = max(1, (depth // 2) // num_slices_per_volume)
        z_indices = list(range(depth // 4, 3 * depth // 4, step))[
            :num_slices_per_volume
        ]

        for z in z_indices:
            # LPIPS – expects [B, 3, H, W] in [−1, 1]
            pred_slc = torch.from_numpy(pred_norm[:, :, z]).float()
            clean_slc = torch.from_numpy(clean_norm[:, :, z]).float()

            pred_3ch = pred_slc[None, None].expand(1, 3, -1, -1).to(device)
            clean_3ch = clean_slc[None, None].expand(1, 3, -1, -1).to(device)
            pred_3ch = pred_3ch * 2.0 - 1.0
            clean_3ch = clean_3ch * 2.0 - 1.0

            with torch.no_grad():
                lpips_val = float(lpips_fn(pred_3ch, clean_3ch).item())
            lpips_values.append(lpips_val)

            # FID – [3, H, W] in [0, 1]
            fake_slices.append(slice_to_rgb(pred_norm, z))
            real_slices.append(slice_to_rgb(clean_norm, z))

        if smoke_test and batch_idx >= 1:
            break

    # ── aggregate PSNR & LPIPS ─────────────────────────────────────────────
    mean_psnr = float(np.mean(psnr_values)) if psnr_values else float("nan")
    mean_lpips = float(np.mean(lpips_values)) if lpips_values else float("nan")

    # ── FID ────────────────────────────────────────────────────────────────
    print(f"Computing FID on {len(real_slices)} slice pairs …")
    if len(real_slices) >= 2:
        real_arr = np.stack(real_slices, axis=0)  # [N, 3, H, W]
        fake_arr = np.stack(fake_slices, axis=0)  # [N, 3, H, W]
        real_features = extract_inception_features(real_arr, inception_model, device)
        fake_features = extract_inception_features(fake_arr, inception_model, device)
        fid_score = compute_fid(real_features, fake_features)
    else:
        print("  ! Not enough slices to compute FID (need >= 2). Setting FID = nan.")
        fid_score = float("nan")

    # ── report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 44}")
    print("  Task 1b Evaluation Results (RUN_0004)")
    print(f"  PSNR:  {mean_psnr:.4f} dB")
    print(f"  LPIPS: {mean_lpips:.4f}")
    print(f"  FID:   {fid_score:.4f}")
    print(f"{'=' * 44}")

    # ── write metrics.json ─────────────────────────────────────────────────
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, "metrics.json")

    payload = {
        "run_id": config.get("run_id", "0004"),
        "task": "task1b",
        "model": "basicunet_3d",
        "noise_std_eval": noise_std,
        "n_subjects_eval": len(psnr_values),
        "global": {
            "fid": fid_score,
            "psnr": mean_psnr,
            "lpips": mean_lpips,
        },
    }

    with open(metrics_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Metrics written to {metrics_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Task 1b denoising model (RUN_0004)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run_0004_task1b_unet.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Evaluate on 2 validation volumes only (fast sanity check).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config=config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
