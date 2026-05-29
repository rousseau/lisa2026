"""Task 1b trainer — self-supervised denoising (RUN_0005)."""

import json
import os

import torch
import torch.nn as nn
from monai.losses import SSIMLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import get_task1b_dataloaders
from src.models import Task1bUNetModel
from src.training.base import BaseTrainer


# ─── Loss ────────────────────────────────────────────────────────────────────


class CombinedL1SSIMLoss(nn.Module):
    """Weighted sum of pixel-wise L1 loss and 3D SSIM loss.

    Args:
        l1_weight:   Weight for the L1 term.
        ssim_weight: Weight for the SSIM term.
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1_weight * self.l1_loss(pred, target) + self.ssim_weight * self.ssim_loss(pred, target)


# ─── Noise helper ────────────────────────────────────────────────────────────


def add_synthetic_noise(
    x: torch.Tensor,
    noise_std_range: tuple = (0.05, 0.20),
) -> torch.Tensor:
    """Add random-std Gaussian noise for self-supervised training."""
    std = torch.empty(1).uniform_(*noise_std_range).item()
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(x.min(), x.max())


# ─── Trainer ─────────────────────────────────────────────────────────────────


class Task1bTrainer(BaseTrainer):
    """Self-supervised 3D denoising trainer for Task 1b (RUN_0005).

    The model is trained to reconstruct clean volumes from synthetically-degraded
    inputs. No paired acquisitions are required.

    Loss: ``l1_weight * L1 + ssim_weight * (1 - SSIM)``.
    Early stopping monitors validation reconstruction loss (lower = better).
    """

    val_metric_key = "val_loss"
    val_metric_direction = "min"

    def __init__(self, config: dict, smoke_test: bool = False):
        super().__init__(config, smoke_test)
        self._build_dataloaders()
        self._build_model()
        self._build_optimizer()

        # Noise range
        self.noise_std_range = (
            float(config["data"].get("noise_std_min", 0.05)),
            float(config["data"].get("noise_std_max", 0.20)),
        )

        # Loss function
        loss_cfg = config["training"].get("loss", {})
        self.loss_fn = CombinedL1SSIMLoss(
            l1_weight=float(loss_cfg.get("l1_weight", 1.0)),
            ssim_weight=float(loss_cfg.get("ssim_weight", 1.0)),
        )

        # Training length
        self.num_epochs = 1 if smoke_test else int(config["training"]["num_epochs"])
        self.ckpt_path = os.path.join(self.ckpt_dir, "task1b_unet_best.pt")

        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"Task1b train: {self.n_train}  val: {self.n_val}")
        print(f"Noise std range: {self.noise_std_range}")
        print(f"Loss — L1 w={self.loss_fn.l1_weight}, SSIM w={self.loss_fn.ssim_weight}")

    # ── BaseTrainer interface ────────────────────────────────────────────────

    def _build_model(self) -> None:
        cfg = self.config["model"]
        self.model = Task1bUNetModel(
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg["out_channels"]),
            features=tuple(int(f) for f in cfg["features"]),
        ).to(self.device)

    def _build_dataloaders(self) -> None:
        cfg = self.config
        data_root = os.getenv("LISA_DATA_ROOT", cfg["data"]["data_root"])
        batch_size = 1 if self.smoke_test else int(cfg["training"]["batch_size"])
        (
            self.train_loader,
            self.val_loader,
            self.n_train,
            self.n_val,
        ) = get_task1b_dataloaders(
            data_root=data_root,
            split_pkl=cfg["data"]["split_pkl"],
            batch_size=batch_size,
            num_workers=int(cfg["training"]["num_workers"]),
            image_suffix=cfg["data"].get("image_suffix", "_ciso.nii.gz"),
            spatial_size=tuple(int(s) for s in cfg["data"]["spatial_size"]),
        )

    def _build_optimizer(self) -> None:
        cfg = self.config["training"]
        num_epochs = int(cfg["num_epochs"])
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, num_epochs),
            eta_min=float(cfg.get("min_learning_rate", 1e-6)),
        )

    # ── Epoch methods ────────────────────────────────────────────────────────

    def train_one_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0

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
            if self.smoke_test and batch_idx >= 1:
                break

        n = min(batch_idx + 1, 2) if self.smoke_test else batch_idx + 1
        return {"train_loss": total_loss / max(1, n)}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        mid_std = (self.noise_std_range[0] + self.noise_std_range[1]) / 2.0

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Val  ")):
            clean = batch["img"].to(self.device)
            noise = torch.randn_like(clean) * mid_std
            degraded = (clean + noise).clamp(clean.min(), clean.max())
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(degraded)
                loss = self.loss_fn(pred, clean)
            total_loss += float(loss.item())
            if self.smoke_test and batch_idx >= 0:
                break

        val_loss = total_loss / max(1, batch_idx + 1)
        return {self.val_metric_key: val_loss}
