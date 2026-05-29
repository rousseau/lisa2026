"""CycleGAN trainer for Task 1b unpaired artifact removal (RUN_0002).

Training procedure
------------------
Two generators (G_AB: artefacted→clean, G_BA: clean→artefacted) and two
PatchGAN discriminators (D_A, D_B) are trained adversarially.

Loss components:
    * Adversarial (LSGAN / MSE):    forces G_AB(A) to fool D_B, and G_BA(B) to
                                    fool D_A.
    * Cycle-consistency (L1):       G_BA(G_AB(A)) ≈ A, G_AB(G_BA(B)) ≈ B.
                                    Weight: lambda_cycle (default 10).
    * Identity (L1):                G_AB(B) ≈ B, G_BA(A) ≈ A.
                                    Weight: lambda_identity (default 5).

Optimisation:
    * Separate Adam optimisers for generators and discriminators.
    * LR: linear decay from lr_initial to 0 over the second half of training.
    * Image buffer of size 50 for discriminator updates (reduces oscillation).

Checkpointing:
    * Best checkpoint selected by minimum val cycle-consistency loss.
    * Saves G_AB (the inference generator) as ``G_AB_best.pt``.
    * Full state (both generators + discriminators) saved as
      ``cyclegan_full_best.pt``.
"""

import os
import random
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.task1b import Task1bCycleGANDataset
from src.models.task1b import Discriminator3D, Generator3D


# ---------------------------------------------------------------------------
# Image replay buffer
# ---------------------------------------------------------------------------


class ImageBuffer:
    """Discriminator image buffer (size 50) to stabilise training."""

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self.buffer: List[torch.Tensor] = []

    @torch.no_grad()
    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                out = self.buffer[idx].clone()
                self.buffer[idx] = img
                result.append(out)
            else:
                result.append(img)
        return torch.cat(result, dim=0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CycleGANTrainer:
    """Full CycleGAN training loop for Task 1b (RUN_0002).

    Args:
        config:     Loaded YAML config dict.
        smoke_test: If True, limit epochs and batches for quick validation.
    """

    def __init__(self, config: dict, smoke_test: bool = False) -> None:
        self.config = config
        self.smoke_test = smoke_test

        cfg_t = config["training"]
        cfg_d = config["data"]
        cfg_m = config.get("model", {})
        cfg_o = config.get("output", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # ── Hyper-parameters ──────────────────────────────────────────────
        self.num_epochs = 2 if smoke_test else int(cfg_t["num_epochs"])
        self.decay_epoch = int(cfg_t.get("decay_epoch", self.num_epochs // 2))
        self.lr = float(cfg_t["learning_rate"])
        self.lambda_cycle = float(cfg_t.get("lambda_cycle", 10.0))
        self.lambda_identity = float(cfg_t.get("lambda_identity", 5.0))
        self.batch_size = int(cfg_t["batch_size"])
        self.num_workers = int(cfg_t.get("num_workers", 2))

        # ── Output paths ──────────────────────────────────────────────────
        self.ckpt_dir = cfg_o.get("checkpoint_dir", "outputs/checkpoints/RUN_0002")
        self.results_dir = cfg_o.get("results_dir", "results/runs/RUN_0002")
        self.log_dir = os.path.join("outputs", "logs", "RUN_0002")
        for d in (self.ckpt_dir, self.results_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)

        self._log_fh = open(os.path.join(self.log_dir, "train.log"), "w")

        # ── Data ─────────────────────────────────────────────────────────
        spatial_size = tuple(int(v) for v in cfg_d.get("spatial_size", [96, 96, 96]))
        data_root = os.getenv("LISA_DATA_ROOT", cfg_d["data_root"])
        csv_path = os.getenv(
            "LISA_CSV_PATH",
            cfg_d.get("csv_path", os.path.join(data_root, "LISA_Task1a_2026.csv")),
        )
        split_pkl = cfg_d["split_pkl"]
        image_suffix = cfg_d.get("image_suffix", "_ciso.nii.gz")
        noise_thr = int(cfg_d.get("noise_threshold", 1))
        motion_thr = int(cfg_d.get("motion_threshold", 1))

        train_ds = Task1bCycleGANDataset(
            data_root=data_root, csv_path=csv_path, split_pkl=split_pkl,
            fold="train", stage="train", image_suffix=image_suffix,
            spatial_size=spatial_size, noise_threshold=noise_thr,
            motion_threshold=motion_thr, domain="both",
        )
        val_ds = Task1bCycleGANDataset(
            data_root=data_root, csv_path=csv_path, split_pkl=split_pkl,
            fold="val", stage="val", image_suffix=image_suffix,
            spatial_size=spatial_size, noise_threshold=noise_thr,
            motion_threshold=motion_thr, domain="both",
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=max(1, self.num_workers // 2), pin_memory=True,
        )
        self.n_train = len(train_ds)
        self.n_val = len(val_ds)

        # ── Models ────────────────────────────────────────────────────────
        base_f = int(cfg_m.get("base_filters", 32))
        n_res = int(cfg_m.get("n_res_blocks", 6))
        d_base = int(cfg_m.get("disc_base_filters", 64))
        d_layers = int(cfg_m.get("disc_n_layers", 4))

        self.G_AB = Generator3D(base_filters=base_f, n_res_blocks=n_res).to(self.device)
        self.G_BA = Generator3D(base_filters=base_f, n_res_blocks=n_res).to(self.device)
        self.D_A = Discriminator3D(base_filters=d_base, n_layers=d_layers).to(self.device)
        self.D_B = Discriminator3D(base_filters=d_base, n_layers=d_layers).to(self.device)

        # Wrap with DataParallel when multiple GPUs are available
        if self.n_gpus > 1:
            self.G_AB = nn.DataParallel(self.G_AB)
            self.G_BA = nn.DataParallel(self.G_BA)
            self.D_A  = nn.DataParallel(self.D_A)
            self.D_B  = nn.DataParallel(self.D_B)

        # ── Optimisers ────────────────────────────────────────────────────
        self.opt_G = Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=self.lr, betas=(0.5, 0.999),
        )
        self.opt_D = Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=self.lr, betas=(0.5, 0.999),
        )

        # LR: constant for decay_epoch epochs, then linear decay to 0
        def _lr_lambda(epoch: int) -> float:
            if epoch < self.decay_epoch:
                return 1.0
            return max(0.0, 1.0 - (epoch - self.decay_epoch) /
                       max(1, self.num_epochs - self.decay_epoch))

        self.sched_G = LambdaLR(self.opt_G, lr_lambda=_lr_lambda)
        self.sched_D = LambdaLR(self.opt_D, lr_lambda=_lr_lambda)

        # ── Losses ────────────────────────────────────────────────────────
        self.criterion_adv = nn.MSELoss()   # LSGAN
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # ── Buffers ───────────────────────────────────────────────────────
        self.buffer_A = ImageBuffer()
        self.buffer_B = ImageBuffer()

        # ── Tracking ──────────────────────────────────────────────────────
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        n_G = sum(p.numel() for p in self.G_AB.parameters() if p.requires_grad)
        n_D = sum(p.numel() for p in self.D_A.parameters() if p.requires_grad)
        self._log(
            f"CycleGAN RUN_0002 | device={self.device} | n_gpus={self.n_gpus} | "
            f"train={self.n_train} val={self.n_val} | "
            f"G params={2*n_G:,} | D params={2*n_D:,}"
        )

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        print(msg)
        self._log_fh.write(msg + "\n")
        self._log_fh.flush()

    # ── Training utilities ────────────────────────────────────────────────────

    @staticmethod
    def _real_label(tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tensor)

    @staticmethod
    def _fake_label(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor)

    def _train_generators(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> dict:
        self.opt_G.zero_grad()

        # Identity loss
        idt_B = self.G_AB(real_B)
        loss_idt_AB = self.criterion_identity(idt_B, real_B) * self.lambda_identity
        idt_A = self.G_BA(real_A)
        loss_idt_BA = self.criterion_identity(idt_A, real_A) * self.lambda_identity

        # Adversarial loss
        fake_B = self.G_AB(real_A)
        pred_fake_B = self.D_B(fake_B)
        loss_adv_AB = self.criterion_adv(pred_fake_B, self._real_label(pred_fake_B))

        fake_A = self.G_BA(real_B)
        pred_fake_A = self.D_A(fake_A)
        loss_adv_BA = self.criterion_adv(pred_fake_A, self._real_label(pred_fake_A))

        # Cycle-consistency loss
        rec_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle
        rec_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle

        loss_G = (loss_adv_AB + loss_adv_BA +
                  loss_cycle_A + loss_cycle_B +
                  loss_idt_AB + loss_idt_BA)
        loss_G.backward()
        self.opt_G.step()

        return {
            "loss_G": loss_G.item(),
            "loss_adv": (loss_adv_AB + loss_adv_BA).item(),
            "loss_cycle": (loss_cycle_A + loss_cycle_B).item(),
            "loss_identity": (loss_idt_AB + loss_idt_BA).item(),
            "fake_A": fake_A.detach(),
            "fake_B": fake_B.detach(),
        }

    def _train_discriminator(
        self,
        D: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        buffer: ImageBuffer,
    ) -> float:
        fake_buf = buffer.push_and_pop(fake)
        pred_real = D(real)
        pred_fake = D(fake_buf.detach())
        loss = 0.5 * (
            self.criterion_adv(pred_real, self._real_label(pred_real)) +
            self.criterion_adv(pred_fake, self._fake_label(pred_fake))
        )
        return loss

    # ── Epoch methods ─────────────────────────────────────────────────────────

    def _train_epoch(self) -> dict:
        self.G_AB.train(); self.G_BA.train()
        self.D_A.train(); self.D_B.train()

        metrics = {"loss_G": 0.0, "loss_adv": 0.0, "loss_cycle": 0.0,
                   "loss_identity": 0.0, "loss_D": 0.0}
        n = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train")):
            if self.smoke_test and batch_idx >= 2:
                break
            real_A = batch["img_A"].to(self.device).float()
            real_B = batch["img_B"].to(self.device).float()

            # Train generators
            g_out = self._train_generators(real_A, real_B)
            fake_A, fake_B = g_out.pop("fake_A"), g_out.pop("fake_B")

            # Train discriminators
            self.opt_D.zero_grad()
            loss_D_A = self._train_discriminator(self.D_A, real_A, fake_A, self.buffer_A)
            loss_D_B = self._train_discriminator(self.D_B, real_B, fake_B, self.buffer_B)
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            self.opt_D.step()

            for k, v in g_out.items():
                metrics[k] += v
            metrics["loss_D"] += loss_D.item()
            n += 1

        n = max(n, 1)
        return {k: v / n for k, v in metrics.items()}

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.G_AB.eval(); self.G_BA.eval()

        cycle_loss = 0.0
        n = 0
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Val")):
            if self.smoke_test and batch_idx >= 2:
                break
            real_A = batch["img_A"].to(self.device).float()
            real_B = batch["img_B"].to(self.device).float()

            fake_B = self.G_AB(real_A)
            rec_A = self.G_BA(fake_B)
            fake_A = self.G_BA(real_B)
            rec_B = self.G_AB(fake_A)

            cycle_loss += (
                self.criterion_cycle(rec_A, real_A) +
                self.criterion_cycle(rec_B, real_B)
            ).item()
            n += 1

        return {"val_cycle_loss": cycle_loss / max(n, 1)}

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        self._log(f"\n{'='*60}\nStarting CycleGAN training — {self.num_epochs} epochs\n{'='*60}")

        for epoch in range(1, self.num_epochs + 1):
            train_m = self._train_epoch()
            val_m = self._val_epoch()

            self.sched_G.step()
            self.sched_D.step()

            val_loss = val_m["val_cycle_loss"]
            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                # Unwrap DataParallel for portable checkpoints
                _sd = lambda m: m.module.state_dict() if isinstance(m, nn.DataParallel) else m.state_dict()  # noqa: E731
                torch.save(_sd(self.G_AB),
                           os.path.join(self.ckpt_dir, "G_AB_best.pt"))
                torch.save({
                    "G_AB": _sd(self.G_AB),
                    "G_BA": _sd(self.G_BA),
                    "D_A":  _sd(self.D_A),
                    "D_B":  _sd(self.D_B),
                    "epoch": epoch,
                }, os.path.join(self.ckpt_dir, "cyclegan_full_best.pt"))

            marker = " ✓" if improved else ""
            self._log(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"G={train_m['loss_G']:.4f} "
                f"adv={train_m['loss_adv']:.4f} "
                f"cyc={train_m['loss_cycle']:.4f} "
                f"idt={train_m['loss_identity']:.4f} "
                f"D={train_m['loss_D']:.4f} | "
                f"val_cyc={val_loss:.4f}{marker}"
            )

        self._log(
            f"\nTraining complete. Best val_cycle={self.best_val_loss:.4f} "
            f"at epoch {self.best_epoch}."
        )
        self._log_fh.close()
