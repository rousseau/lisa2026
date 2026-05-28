"""Task 1b reconstruction metrics: PSNR, LPIPS, FID."""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_psnr(recon: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (dB) for a single volume pair.

    Both tensors are clamped to [0, 1] before computing MSE.
    PSNR = 10 * log10(1 / (MSE + 1e-8))

    Args:
        recon:  Reconstructed volume [1, C, H, W, D] or [C, H, W, D].
        target: Reference volume (same shape as recon).

    Returns:
        PSNR in dB (float).
    """
    recon_c = recon.clamp(0.0, 1.0)
    target_c = target.clamp(0.0, 1.0)
    mse = F.mse_loss(recon_c, target_c).item()
    return float(10.0 * np.log10(1.0 / (mse + 1e-8)))


def compute_lpips_batch(
    recon_list: List[np.ndarray],
    target_list: List[np.ndarray],
    device: str = "cpu",
) -> float:
    """Mean LPIPS over a list of volume pairs.

    Each volume is processed slice-by-slice along the last axis (axial).
    LPIPS is computed on 3-channel (repeated) 2D slices as required by the
    lpips library, then averaged over slices and volumes.

    Requires ``lpips`` package (``pip install lpips``).

    Args:
        recon_list:  List of numpy arrays [H, W, D] in [0, 1].
        target_list: List of numpy arrays [H, W, D] in [0, 1] (same shapes).
        device:      Torch device string.

    Returns:
        Mean LPIPS (float), or NaN if lpips is unavailable.
    """
    try:
        import lpips as _lpips
    except ImportError:
        return float("nan")

    loss_fn = _lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    scores = []
    with torch.no_grad():
        for recon_np, target_np in zip(recon_list, target_list):
            # recon_np: [H, W, D] → iterate over D slices
            d = recon_np.shape[-1]
            for z in range(d):
                r_slice = torch.from_numpy(recon_np[:, :, z]).float().to(device)
                t_slice = torch.from_numpy(target_np[:, :, z]).float().to(device)
                # LPIPS expects [B, 3, H, W] in [-1, 1]
                r_slice = r_slice.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
                t_slice = t_slice.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
                scores.append(float(loss_fn(r_slice, t_slice).item()))

    return float(np.mean(scores)) if scores else float("nan")


def compute_fid_from_features(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> float:
    """Fréchet Inception Distance from pre-extracted feature matrices.

    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2 * sqrt(Σ_r Σ_f))

    Args:
        real_features: [N, D] feature matrix for real samples.
        fake_features: [N, D] feature matrix for generated/reconstructed samples.

    Returns:
        FID score (float).
    """
    from scipy.linalg import sqrtm

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean))
    return fid
