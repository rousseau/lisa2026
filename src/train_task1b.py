#!/usr/bin/env python
"""Train 3D BasicUNet self-supervised denoising baseline for LISA 2026 Task 1b (RUN_0004).

Training strategy
-----------------
  * For each clean volume loaded from disk, Gaussian noise is added on-the-fly
    to create a (degraded, clean) pair.
  * The model learns to reconstruct the original image – no paired acquisitions
    are required.
  * Loss: weighted combination of pixel-wise L1 and 3D SSIM.
  * Optimiser: AdamW with cosine-annealing LR schedule.
  * Early stopping monitored on validation reconstruction loss.

Usage
-----
  python train_task1b.py --config configs/run_0004_task1b_unet.yaml
  python train_task1b.py --config configs/run_0004_task1b_unet.yaml --smoke_test
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from monai.losses import SSIMLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import get_task1b_dataloaders
from src.models import Task1bUNetModel
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_synthetic_noise(
    x: torch.Tensor,
    noise_std_range: tuple = (0.05, 0.20),
) -> torch.Tensor:
    """Add Gaussian noise for self-supervised denoising training.

    The noise standard deviation is sampled uniformly from *noise_std_range*
    once per batch so that the model sees a variety of degradation levels
    during training.

    Args:
        x:               Clean input tensor (any shape).
        noise_std_range: (min_std, max_std) for the uniform noise draw.

    Returns:
        Degraded tensor clipped to the original value range.
    """
    std = torch.empty(1).uniform_(*noise_std_range).item()
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(x.min(), x.max())


class CombinedL1SSIMLoss(nn.Module):
    """Weighted sum of pixel-wise L1 loss and 3D SSIM loss.

    Args:
        l1_weight:   Scalar weight for the L1 term (default 1.0).
        ssim_weight: Scalar weight for the SSIM term (default 1.0).
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()
        # SSIMLoss from MONAI returns (1 - SSIM) so minimising it maximises SSIM.
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Task1bTrainer:
    """Self-supervised denoising trainer for Task 1b (RUN_0004).

    Instantiate with a parsed YAML config dict and call :py:meth:`train`.
    """

    def __init__(self, config: dict, smoke_test: bool = False):
        self.config = config
        self.smoke_test = smoke_test

        # ── reproducibility ────────────────────────────────────────────────
        seed = int(config["environment"]["seed"])
        set_seed(seed)
        if config["environment"].get("deterministic", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ── device / AMP ───────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (
            bool(config["environment"].get("mixed_precision", True))
            and self.device == "cuda"
        )

        # ── data ───────────────────────────────────────────────────────────
        data_root = os.getenv("LISA_DATA_ROOT", config["data"]["data_root"])
        split_pkl = config["data"]["split_pkl"]
        spatial_size = tuple(int(s) for s in config["data"]["spatial_size"])

        batch_size = int(config["training"]["batch_size"])
        num_workers = int(config["training"]["num_workers"])
        if smoke_test:
            batch_size = 1

        self.train_loader, self.val_loader, self.n_train, self.n_val = (
            get_task1b_dataloaders(
                data_root=data_root,
                split_pkl=split_pkl,
                batch_size=batch_size,
                num_workers=num_workers,
                image_suffix=config["data"].get("image_suffix", "_ciso.nii.gz"),
                spatial_size=spatial_size,
            )
        )

        # ── model ──────────────────────────────────────────────────────────
        model_cfg = config["model"]
        self.model = Task1bUNetModel(
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            features=tuple(int(f) for f in model_cfg["features"]),
        ).to(self.device)

        # ── loss ───────────────────────────────────────────────────────────
        loss_cfg = config["training"].get("loss", {})
        self.loss_fn = CombinedL1SSIMLoss(
            l1_weight=float(loss_cfg.get("l1_weight", 1.0)),
            ssim_weight=float(loss_cfg.get("ssim_weight", 1.0)),
        )

        self.noise_std_range = (
            float(config["data"].get("noise_std_min", 0.05)),
            float(config["data"].get("noise_std_max", 0.20)),
        )

        # ── optimiser & scheduler ──────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        num_epochs_total = int(config["training"]["num_epochs"])
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, num_epochs_total),
            eta_min=float(config["training"].get("min_learning_rate", 1.0e-6)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── output paths ───────────────────────────────────────────────────
        out_dir = config["output"]
        self.ckpt_dir = out_dir["checkpoint_dir"]
        self.log_dir = out_dir["log_dir"]
        self.results_dir = out_dir["results_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.ckpt_path = os.path.join(self.ckpt_dir, "task1b_unet_best.pt")
        self.history_path = os.path.join(self.results_dir, "training_history.json")

        # ── early stopping ─────────────────────────────────────────────────
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.num_epochs = num_epochs_total
        self.patience = int(config["early_stopping"]["patience"])

        if smoke_test:
            self.num_epochs = 1

        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"Task1b train volumes: {self.n_train}, val volumes: {self.n_val}")
        print(f"Noise std range: {self.noise_std_range}")
        print(
            f"Loss weights – L1: {self.loss_fn.l1_weight}, SSIM: {self.loss_fn.ssim_weight}"
        )

    # -----------------------------------------------------------------------

    def train_one_epoch(self) -> float:
        """Run one full training epoch and return the mean loss."""
        self.model.train()
        total_loss = 0.0
        last_idx = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train")):
            clean = batch["img"].to(self.device)
            degraded = add_synthetic_noise(clean, noise_std_range=self.noise_std_range)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(degraded)
                loss = self.loss_fn(pred, clean)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            last_idx = batch_idx

            if self.smoke_test and batch_idx >= 1:
                break

        return total_loss / max(1, last_idx + 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Evaluate on the validation set with fixed mid-range noise.

        Using a fixed noise std during validation ensures the metric is
        reproducible across epochs and comparable between runs.
        """
        self.model.eval()
        total_loss = 0.0
        last_idx = 0

        mid_std = (self.noise_std_range[0] + self.noise_std_range[1]) / 2.0

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Val  ")):
            clean = batch["img"].to(self.device)
            noise = torch.randn_like(clean) * mid_std
            degraded = (clean + noise).clamp(clean.min(), clean.max())

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(degraded)
                loss = self.loss_fn(pred, clean)

            total_loss += float(loss.item())
            last_idx = batch_idx

            if self.smoke_test and batch_idx >= 0:
                break

        return total_loss / max(1, last_idx + 1)

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Persist model + optimiser state to disk."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            },
            self.ckpt_path,
        )

    def train(self) -> None:
        """Full training loop with early stopping."""
        history = []

        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )

            print(
                f"Epoch {epoch + 1:03d}/{self.num_epochs:03d} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch=epoch + 1, val_loss=val_loss)
                print(f"  -> New best checkpoint saved ({self.ckpt_path})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"  -> Early stopping at epoch {epoch + 1}")
                    break

            self.scheduler.step()

            if self.smoke_test:
                break

        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"History saved to {self.history_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Task 1b denoising model (RUN_0004)"
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
        help="Run 1 epoch / 2 batches to verify the pipeline end-to-end.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Task1bTrainer(config=config, smoke_test=args.smoke_test)
    trainer.train()


if __name__ == "__main__":
    main()
