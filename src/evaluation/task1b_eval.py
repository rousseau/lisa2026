"""Task 1b evaluation — PSNR, LPIPS and FID (inline).

Extracted and shared between the standalone Task 1b evaluator and
``evaluate_multitask.py``.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.metrics.reconstruction import (
    build_inception_model,
    compute_fid,
    compute_psnr,
    compute_lpips_batch,
    extract_inception_features,
    slice_to_rgb,
)


def evaluate_task1b(
    model,
    val_loader,
    device: str,
    smoke_test: bool = False,
    fid_num_slices_per_volume: int = 10,
    task_name: str | None = None,
) -> dict:
    """Evaluate Task 1b reconstruction metrics inline.

    Parameters
    ----------
    task_name :
        If the model requires a task dispatch argument (e.g. ``"1b"`` for
        ``DynUNetMultiHeadModel``), it will be passed as ``model(images,
        task=task_name)``.  When ``None`` the model is called directly.

    Returns
    -------
    dict with keys ``psnr``, ``lpips``, ``fid``, ``l1``, ``n_subjects``.
    """
    model.eval()
    use_amp = device == "cuda"

    psnr_values = []
    l1_values = []
    recon_vols = []
    target_vols = []
    real_slices = []
    fake_slices = []
    n_subjects = 0

    inception_model = build_inception_model(device)

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Eval-Task1b")):
        images = batch["img_B"].to(device) if "img_B" in batch else batch["img"].to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                if task_name is not None:
                    recon = model(images, task=task_name)
                else:
                    recon = model(images)
        recon_c = recon.clamp(0.0, 1.0)
        target_c = images.clamp(0.0, 1.0)

        for i in range(images.shape[0]):
            psnr_values.append(compute_psnr(recon_c[i : i + 1], target_c[i : i + 1]))
            pred_np = recon_c[i, 0].cpu().numpy()
            clean_np = target_c[i, 0].cpu().numpy()
            recon_vols.append(pred_np)
            target_vols.append(clean_np)

            d = pred_np.shape[-1]
            indices = np.linspace(0, d - 1, fid_num_slices_per_volume, dtype=int)
            for z in indices:
                fake_slices.append(slice_to_rgb(pred_np, z))
                real_slices.append(slice_to_rgb(clean_np, z))

        l1_values.append(float(F.l1_loss(recon, images).item()))
        n_subjects += int(images.shape[0])
        if smoke_test and batch_idx >= 2:
            break

    lpips_score = compute_lpips_batch(recon_vols, target_vols, device=device)

    if len(real_slices) >= 2:
        real_arr = np.stack(real_slices, axis=0)
        fake_arr = np.stack(fake_slices, axis=0)
        real_features = extract_inception_features(real_arr, inception_model, device)
        fake_features = extract_inception_features(fake_arr, inception_model, device)
        fid_score = compute_fid(real_features, fake_features)
    else:
        fid_score = float("nan")

    return {
        "psnr": float(np.mean(psnr_values)) if psnr_values else float("nan"),
        "lpips": lpips_score,
        "fid": fid_score,
        "l1": float(np.mean(l1_values)) if l1_values else float("nan"),
        "n_subjects": n_subjects,
    }
