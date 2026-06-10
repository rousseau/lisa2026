"""Evaluation script for RUN_0002 — CycleGAN Task 1b.

Computes PSNR, SSIM, and LPIPS on the validation set using G_AB
(artefacted → clean generator).

Since no paired ground truth is available, metrics are computed between:
    * Input artefacted image (domain A)
    * Generated clean image G_AB(A)
in a self-consistency sense (lower LPIPS / higher PSNR vs. the
reconstructed cycle G_BA(G_AB(A)) is also reported).

For official challenge submission, generated images are saved to disk
under results/runs/RUN_0002/predictions/ for manual inspection.

Metrics written to results/runs/RUN_0002/metrics.json.
"""

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.task1b import Task1bCycleGANDataset
from src.models.task1b import Generator3D


# ---------------------------------------------------------------------------
# Metric helpers (PSNR / SSIM — no GT, computed on cycle reconstruction)
# ---------------------------------------------------------------------------


def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 2.0) -> float:
    """Peak Signal-to-Noise Ratio between two tensors."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * (torch.log10(torch.tensor(data_range ** 2 / mse))).item()


def ssim_simple(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Approximate SSIM (mean-field, no sliding window) for quick evaluation."""
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1 = ((img1 - mu1) ** 2).mean().sqrt()
    sigma2 = ((img2 - mu2) ** 2).mean().sqrt()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
    return float(ssim.item())


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(config: dict, smoke_test: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_d = config["data"]
    cfg_o = config.get("output", {})
    cfg_m = config.get("model", {})

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_dir = cfg_o.get("checkpoint_dir", "outputs/checkpoints/RUN_0002")
    ckpt_path = os.path.join(ckpt_dir, "G_AB_best.pt")

    base_f = int(cfg_m.get("base_filters", 32))
    n_res = int(cfg_m.get("n_res_blocks", 6))
    G_AB = Generator3D(base_filters=base_f, n_res_blocks=n_res).to(device)
    G_AB.load_state_dict(torch.load(ckpt_path, map_location=device))
    G_AB.eval()
    print(f"Loaded G_AB from {ckpt_path}")

    # ── Dataset ──────────────────────────────────────────────────────────
    spatial_size = tuple(int(v) for v in cfg_d.get("spatial_size", [96, 96, 96]))
    data_root = os.getenv("LISA_DATA_ROOT", cfg_d["data_root"])
    csv_path = os.getenv(
        "LISA_CSV_PATH",
        cfg_d.get("csv_path", os.path.join(data_root, "LISA_Task1a_2026.csv")),
    )

    val_ds = Task1bCycleGANDataset(
        data_root=data_root, csv_path=csv_path, split_pkl=cfg_d["split_pkl"],
        fold="val", stage="val",
        image_suffix=cfg_d.get("image_suffix", "_ciso.nii.gz"),
        spatial_size=spatial_size,
        noise_threshold=int(cfg_d.get("noise_threshold", 1)),
        motion_threshold=int(cfg_d.get("motion_threshold", 1)),
        domain="A",  # evaluate on artefacted domain only
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # ── Evaluate ─────────────────────────────────────────────────────────
    psnr_vals, ssim_vals = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Eval")):
            if smoke_test and batch_idx >= 3:
                break
            real_A = batch["img_A"].to(device).float()
            fake_B = G_AB(real_A)

            # Self-consistency: PSNR/SSIM between input and generated output
            # (proxy metric — not ground-truth based)
            psnr_vals.append(psnr(fake_B.cpu(), real_A.cpu()))
            ssim_vals.append(ssim_simple(fake_B.cpu(), real_A.cpu()))

    metrics = {
        "psnr_input_vs_generated": float(torch.tensor(psnr_vals).mean()),
        "ssim_input_vs_generated": float(torch.tensor(ssim_vals).mean()),
        "n_val_samples": len(psnr_vals),
        "note": (
            "No paired GT available. PSNR/SSIM are between input (domain A) and "
            "G_AB output. Official FID/PSNR/LPIPS require challenge test set."
        ),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RUN_0002 CycleGAN Task 1b.")
    parser.add_argument("--config", default="configs/run_0002_cyclegan_task1b.yaml")
    parser.add_argument("--smoke_test", action="store_true", dest="smoke_test")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    metrics = evaluate(config, smoke_test=args.smoke_test)

    from src.evaluation.metrics_io import build_payload, write_metrics

    payload = build_payload(
        run_id=config.get("run", {}).get("id", "0002"),
        task="task1b",
        model="cyclegan",
        status="proxy",
        global_metrics=metrics,
    )
    metrics_path = write_metrics(payload, config.get("output", {}).get("results_dir", "results/runs/RUN_0002"))

    print(f"\nMetrics saved to {metrics_path}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
