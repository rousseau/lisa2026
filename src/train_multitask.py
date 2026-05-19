#!/usr/bin/env python
"""Multi-task training: Tasks 1a, 1b and 2 with shared encoder (RUN_0004)."""

import argparse
import itertools
import json
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, SSIMLoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import get_multitask_dataloaders
from src.models import DynUNetMultiHeadModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiTaskTrainer:
    def __init__(self, config: dict, smoke_test: bool = False):
        self.config = config
        self.smoke_test = smoke_test

        seed = int(config["environment"]["seed"])
        set_seed(seed)

        if config["environment"].get("deterministic", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (
            bool(config["environment"].get("mixed_precision", True))
            and self.device == "cuda"
        )

        # Allow environment variable override for data root
        config["data"]["data_root"] = os.getenv(
            "LISA_DATA_ROOT", config["data"]["data_root"]
        )

        # Force batch_size=1 for smoke test (all 3 task loaders)
        if smoke_test:
            config["data"]["batch_size"] = 1

        # ── Dataloaders ───────────────────────────────────────────────────────
        loaders = get_multitask_dataloaders(config)
        self.train_loader_1a, self.val_loader_1a, n_train_1a, n_val_1a = loaders["1a"]
        self.train_loader_1b, self.val_loader_1b, n_train_1b, n_val_1b = loaders["1b"]
        self.train_loader_2, self.val_loader_2, n_train_2, n_val_2 = loaders["2"]

        print(f"Task 1a : train={n_train_1a}, val={n_val_1a}")
        print(f"Task 1b : train={n_train_1b}, val={n_val_1b}")
        print(f"Task 2  : train={n_train_2},  val={n_val_2}")

        # ── Model ─────────────────────────────────────────────────────────────
        model_cfg = config["model"]
        self.model = DynUNetMultiHeadModel(
            in_channels=int(model_cfg.get("in_channels", 1)),
            filters=tuple(int(x) for x in model_cfg["filters"]),
            num_seg_classes=int(model_cfg["num_seg_classes"]),
            num_artifact_tasks=int(model_cfg["num_artifact_tasks"]),
            num_artifact_classes=int(model_cfg["num_artifact_classes"]),
        ).to(self.device)

        # ── Losses ────────────────────────────────────────────────────────────
        loss_cfg = config["training"].get("loss", {})
        self.l1_weight = float(loss_cfg.get("l1_weight", 1.0))
        self.ssim_weight = float(loss_cfg.get("ssim_weight", 1.0))
        # SSIMLoss(spatial_dims=3, data_range=1.0) returns 1-SSIM (to minimise).
        # Inputs must be in [0, 1] — clamp before calling (see _loss_1b).
        self.ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0)
        self.seg_loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )

        # Task loss weights (λ)
        self.lambda_1a = float(config["training"].get("lambda_1a", 1.0))
        self.lambda_1b = float(config["training"].get("lambda_1b", 1.0))
        self.lambda_2 = float(config["training"].get("lambda_2", 1.0))

        # ── Optimiser & scheduler ─────────────────────────────────────────────
        self.num_warmup_epochs = int(config["training"].get("num_warmup_epochs", 10))
        self.num_epochs = int(config["training"].get("num_epochs", 80))

        if smoke_test:
            self.num_warmup_epochs = 1
            self.num_epochs = 1

        total_epochs = self.num_warmup_epochs + self.num_epochs

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_epochs),
            eta_min=float(config["training"].get("min_learning_rate", 1.0e-6)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── Validation inference config ───────────────────────────────────────
        self.val_roi_size = tuple(int(x) for x in config["data"]["val_roi_size"])
        self.sw_batch_size = int(config["inference"]["sw_batch_size"])
        self.overlap = float(config["inference"]["overlap"])

        # ── Output paths ──────────────────────────────────────────────────────
        out_cfg = config["output"]
        self.ckpt_dir = out_cfg["checkpoint_dir"]
        self.log_dir = out_cfg["log_dir"]
        self.results_dir = out_cfg["results_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.ckpt_path = os.path.join(self.ckpt_dir, "multitask_best.pt")
        self.history_path = os.path.join(self.results_dir, "training_history.json")

        # ── Pretrained encoder warm-start (partial loading from RUN_0003) ─────
        pretrained = config["training"].get("pretrained_checkpoint", "")
        self.pretrained_loaded = False
        if pretrained and os.path.exists(pretrained):
            try:
                ckpt_state = torch.load(pretrained, map_location=self.device)
                src_sd = ckpt_state.get("model_state_dict", ckpt_state)
                tgt_sd = self.model.state_dict()
                loaded_keys = []
                for k, v in src_sd.items():
                    if k in tgt_sd and v.shape == tgt_sd[k].shape:
                        tgt_sd[k] = v
                        loaded_keys.append(k)
                self.model.load_state_dict(tgt_sd)
                self.pretrained_loaded = True
                print(
                    f"[INFO] Partial encoder warm-start from {pretrained}: "
                    f"{len(loaded_keys)}/{len(tgt_sd)} keys matched."
                )
            except Exception as exc:
                warnings.warn(
                    f"[WARNING] Failed to load pretrained checkpoint '{pretrained}': {exc}. "
                    "Training from scratch (random initialisation).",
                    stacklevel=2,
                )
        elif pretrained and not os.path.exists(pretrained):
            warnings.warn(
                f"[WARNING] Pretrained checkpoint path not found: {pretrained}. "
                "Training from scratch.",
                stacklevel=2,
            )

        # ── Early stopping state ──────────────────────────────────────────────
        self.patience = int(config["early_stopping"]["patience"])
        self.best_val_dice = -1.0
        self.patience_counter = 0

        # ── Warmup early-exit target ──────────────────────────────────────────
        self.dice_warmup_target = float(
            config["training"].get("dice_warmup_target", 0.15)
        )

        # ── Loss normalization scales (calibrated before joint phase) ─────────
        self.loss_scale_1a = 1.0
        self.loss_scale_1b = 1.0
        self.loss_scale_2 = 1.0

        print(f"Device : {self.device} | AMP : {self.use_amp}")
        print(
            f"Warm-up: {self.num_warmup_epochs} epochs "
            f"(early exit if DSC≥{self.dice_warmup_target:.2f}) | "
            f"Joint: {self.num_epochs} epochs | "
            f"λ=(1a={self.lambda_1a}, 1b={self.lambda_1b}, 2={self.lambda_2}) "
            f"[loss-normalized]"
        )

    # ─── Loss helpers ──────────────────────────────────────────────────────────

    def _loss_1a(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sum of cross-entropy losses over the 7 artifact classification heads.

        Args:
            logits: [B, 7, 3]  — per-artifact logits.
            labels: [B, 7]     — per-artifact integer class labels.

        Returns:
            Scalar tensor (sum over 7 heads).
        """
        loss = torch.tensor(0.0, device=self.device)
        for t in range(logits.shape[1]):
            loss = loss + F.cross_entropy(logits[:, t, :], labels[:, t])
        return loss

    def _loss_1b(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined L1 + SSIM reconstruction loss.

        Both operands are clamped to [0, 1] before SSIM because SSIMLoss
        (MONAI, data_range=1.0) requires non-negative inputs.  L1 is computed
        on the unclamped tensors to preserve the full intensity gradient.

        Args:
            recon:  [B, 1, H, W, D] — model reconstruction output.
            target: [B, 1, H, W, D] — clean input volume (identity target).

        Returns:
            Scalar tensor (l1_weight * L1 + ssim_weight * (1 - SSIM)).
        """
        l1 = F.l1_loss(recon, target)
        recon_c = torch.clamp(recon, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        ssim = self.ssim_loss_fn(recon_c, target_c)
        return self.l1_weight * l1 + self.ssim_weight * ssim

    @torch.no_grad()
    def _calibrate_losses(self) -> tuple:
        """Measure initial loss magnitudes (no grad) to normalise joint training.

        Runs one forward pass on one batch from each task loader and records the
        raw loss values *before* any joint-phase update.  These values are used
        as divisors in ``train_one_epoch_joint`` so that every task contributes
        equally to the total gradient at the start of joint training.

        Returns:
            Tuple (scale_1a, scale_1b, scale_2) — initial loss values.
            Each is clamped to ≥ 1e-6 to prevent division by zero.
        """
        self.model.eval()
        MIN_SCALE = 1e-6

        b1a = next(iter(self.train_loader_1a))
        b1b = next(iter(self.train_loader_1b))
        b2 = next(iter(self.train_loader_2))

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            l_1a = self._loss_1a(
                self.model.forward_task1a(b1a["img"].to(self.device)),
                b1a["labels"].to(self.device),
            )
            l_1b = self._loss_1b(
                self.model.forward_task1b(b1b["img"].to(self.device)),
                b1b["img"].to(self.device),
            )
            l_2 = self.seg_loss_fn(
                self.model.forward_task2(b2["img"].to(self.device)),
                b2["label"].to(self.device),
            )

        self.model.train()

        s1a = max(float(l_1a.item()), MIN_SCALE)
        s1b = max(float(l_1b.item()), MIN_SCALE)
        s2 = max(float(l_2.item()), MIN_SCALE)

        print(f"[Calibration] Initial losses — 1a={s1a:.4f}  1b={s1b:.4f}  2={s2:.4f}")
        print(
            f"[Calibration] Effective weights after normalisation — "
            f"λ_1a/L0={self.lambda_1a / s1a:.4f}  "
            f"λ_1b/L0={self.lambda_1b / s1b:.4f}  "
            f"λ_2/L0={self.lambda_2 / s2:.4f}"
        )
        return s1a, s1b, s2

    # ─── Training phases ───────────────────────────────────────────────────────

    def train_one_epoch_warmup(self) -> dict:
        """Warm-up: only the Task 2 segmentation head is active.

        Returns:
            Dict with train_loss_total, train_loss_1a (0), train_loss_1b (0),
            train_loss_2.
        """
        self.model.train()
        total_loss_2 = 0.0

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader_2, desc="Warmup-Train")
        ):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                seg_logits = self.model.forward_task2(images)
                loss = self.seg_loss_fn(seg_logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_2 += float(loss.item())

            if self.smoke_test and batch_idx >= 1:  # 2 steps max
                break

        denom = max(1, batch_idx + 1)
        avg = total_loss_2 / denom
        return {
            "train_loss_total": avg,
            "train_loss_1a": 0.0,
            "train_loss_1b": 0.0,
            "train_loss_2": avg,
        }

    def train_one_epoch_joint(self) -> dict:
        """Joint training: all 3 task heads are active in every step.

        Iterates until the longest loader is exhausted; shorter loaders are
        cycled with ``itertools.cycle``.  Each step performs 3 forward passes
        (one per task) and a single backward + optimiser update.

        Returns:
            Dict with train_loss_total, train_loss_1a, train_loss_1b,
            train_loss_2.
        """
        self.model.train()

        n1a = len(self.train_loader_1a)
        n1b = len(self.train_loader_1b)
        n2 = len(self.train_loader_2)
        n_steps = max(n1a, n1b, n2)
        steps = min(n_steps, 2) if self.smoke_test else n_steps

        iter_1a = itertools.cycle(self.train_loader_1a)
        iter_1b = itertools.cycle(self.train_loader_1b)
        iter_2 = itertools.cycle(self.train_loader_2)

        total_loss = 0.0
        total_l_1a = 0.0
        total_l_1b = 0.0
        total_l_2 = 0.0

        for _step in tqdm(range(steps), desc="Joint-Train"):
            batch_1a = next(iter_1a)
            batch_1b = next(iter_1b)
            batch_2 = next(iter_2)

            img_1a = batch_1a["img"].to(self.device)
            labels_1a = batch_1a["labels"].to(self.device)  # [B, 7]
            img_1b = batch_1b["img"].to(self.device)  # target = img_1b
            img_2 = batch_2["img"].to(self.device)
            labels_2 = batch_2["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Task 1a — artifact classification (7 heads × 3 classes)
                logits_1a = self.model.forward_task1a(img_1a)  # [B, 7, 3]
                l_1a = self._loss_1a(logits_1a, labels_1a)

                # Task 1b — reconstruction autoencoder (target = input)
                recon_1b = self.model.forward_task1b(img_1b)  # [B, 1, H, W, D]
                l_1b = self._loss_1b(recon_1b, img_1b)

                # Task 2 — multi-structure segmentation
                seg_logits = self.model.forward_task2(img_2)  # [B, 12, H, W, D]
                l_2 = self.seg_loss_fn(seg_logits, labels_2)

                loss = (
                    self.lambda_1a * l_1a / self.loss_scale_1a
                    + self.lambda_1b * l_1b / self.loss_scale_1b
                    + self.lambda_2 * l_2 / self.loss_scale_2
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_l_1a += float(l_1a.item())
            total_l_1b += float(l_1b.item())
            total_l_2 += float(l_2.item())

        denom = max(1, steps)
        return {
            "train_loss_total": total_loss / denom,
            "train_loss_1a": total_l_1a / denom,
            "train_loss_1b": total_l_1b / denom,
            "train_loss_2": total_l_2 / denom,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validation pass for all 3 tasks.

        - Task 2  : sliding-window inference + MONAI DiceMetric.
        - Task 1a : average cross-entropy over the val loader.
        - Task 1b : average L1 loss over the val loader.

        Returns:
            Dict with val_dice_2, val_loss_1a, val_loss_1b.
        """
        self.model.eval()
        num_classes = int(self.config["model"]["num_seg_classes"])

        # ── Task 2: sliding window + Dice ─────────────────────────────────────
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        for batch_idx, batch in enumerate(tqdm(self.val_loader_2, desc="Val-Task2")):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = sliding_window_inference(
                    images,
                    roi_size=self.val_roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model.forward_task2,
                    overlap=self.overlap,
                )

            pred = torch.argmax(logits, dim=1, keepdim=True)
            pred_onehot = (
                F.one_hot(pred.squeeze(1), num_classes=num_classes)
                .permute(0, 4, 1, 2, 3)
                .float()
            )
            lbl_onehot = (
                F.one_hot(labels.squeeze(1).long(), num_classes=num_classes)
                .permute(0, 4, 1, 2, 3)
                .float()
            )
            dice_metric(y_pred=pred_onehot, y=lbl_onehot)

            if self.smoke_test and batch_idx >= 0:  # 1 subject max in smoke test
                break

        val_dice_2 = float(dice_metric.aggregate().item())
        dice_metric.reset()

        # ── Task 1a: average CE ────────────────────────────────────────────────
        val_loss_1a = 0.0
        for batch_idx, batch in enumerate(tqdm(self.val_loader_1a, desc="Val-Task1a")):
            images = batch["img"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits_1a = self.model.forward_task1a(images)
                loss_1a = self._loss_1a(logits_1a, labels)

            val_loss_1a += float(loss_1a.item())

            if self.smoke_test and batch_idx >= 1:
                break

        val_loss_1a /= max(1, batch_idx + 1)

        # ── Task 1b: average L1 ───────────────────────────────────────────────
        val_loss_1b = 0.0
        for batch_idx, batch in enumerate(tqdm(self.val_loader_1b, desc="Val-Task1b")):
            images = batch["img"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                recon = self.model.forward_task1b(images)
                loss_1b = F.l1_loss(recon, images)

            val_loss_1b += float(loss_1b.item())

            if self.smoke_test and batch_idx >= 1:
                break

        val_loss_1b /= max(1, batch_idx + 1)

        return {
            "val_dice_2": val_dice_2,
            "val_loss_1a": val_loss_1a,
            "val_loss_1b": val_loss_1b,
        }

    def save_checkpoint(self, epoch: int, val_dice: float):
        """Save model + optimiser state when val_dice_2 improves."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_dice_2": val_dice,
                "config": self.config,
            },
            self.ckpt_path,
        )

    def train(self):
        """Full training loop: warm-up phase then joint phase."""
        run_meta = {
            "run_id": self.config.get("run_id", "0004"),
            "model": "DynUNetMultiHeadModel",
            "pretrained_checkpoint": self.config["training"].get(
                "pretrained_checkpoint", ""
            ),
            "pretrained_loaded": self.pretrained_loaded,
            "device": self.device,
            "num_warmup_epochs": self.num_warmup_epochs,
            "dice_warmup_target": self.dice_warmup_target,
            "num_epochs": self.num_epochs,
            "num_seg_classes": int(self.config["model"]["num_seg_classes"]),
            "filters": list(self.config["model"].get("filters", [])),
            "loss_normalization": True,
            "loss_scale_1a": None,  # filled after calibration
            "loss_scale_1b": None,
            "loss_scale_2": None,
        }
        history = [run_meta]

        # ════════════════════════════════════════════════════════════════════
        # Phase 1 — Warm-up (Task 2 only)
        # ════════════════════════════════════════════════════════════════════
        print(
            f"\n{'=' * 60}\n"
            f"Warm-up phase ({self.num_warmup_epochs} epochs) — Task 2 only\n"
            f"{'=' * 60}"
        )
        for epoch in range(self.num_warmup_epochs):
            train_metrics = self.train_one_epoch_warmup()
            val_metrics = self.validate()

            row = {
                "epoch": epoch + 1,
                "phase": "warmup",
                **train_metrics,
                **val_metrics,
            }
            history.append(row)

            print(
                f"[Warmup] Epoch {epoch + 1:03d}/{self.num_warmup_epochs:03d} | "
                f"train_loss={train_metrics['train_loss_total']:.4f} | "
                f"val_dice_2={val_metrics['val_dice_2']:.4f} | "
                f"val_loss_1a={val_metrics['val_loss_1a']:.4f} | "
                f"val_loss_1b={val_metrics['val_loss_1b']:.4f}"
            )

            if val_metrics["val_dice_2"] > self.best_val_dice:
                self.best_val_dice = val_metrics["val_dice_2"]
                self.save_checkpoint(epoch=epoch + 1, val_dice=self.best_val_dice)
                print(
                    f"  -> New best checkpoint saved "
                    f"(warmup, val_dice_2={self.best_val_dice:.4f})"
                )

            self.scheduler.step()

            if val_metrics["val_dice_2"] >= self.dice_warmup_target:
                print(
                    f"  -> Warmup early exit at epoch {epoch + 1}: "
                    f"val_dice_2={val_metrics['val_dice_2']:.4f} "
                    f">= target {self.dice_warmup_target:.2f}"
                )
                break

            if self.smoke_test:
                break

        # ════════════════════════════════════════════════════════════════════
        # Phase 2 — Joint training (all 3 tasks)
        # ════════════════════════════════════════════════════════════════════
        # ── Calibrate loss scales before joint phase ──────────────────────────
        self.loss_scale_1a, self.loss_scale_1b, self.loss_scale_2 = (
            self._calibrate_losses()
        )
        # Back-fill scales in run_meta for history logging
        history[0]["loss_scale_1a"] = self.loss_scale_1a
        history[0]["loss_scale_1b"] = self.loss_scale_1b
        history[0]["loss_scale_2"] = self.loss_scale_2

        print(
            f"\n{'=' * 60}\n"
            f"Joint phase ({self.num_epochs} epochs) — all 3 tasks\n"
            f"{'=' * 60}"
        )
        for epoch in range(self.num_epochs):
            train_metrics = self.train_one_epoch_joint()
            val_metrics = self.validate()

            global_epoch = self.num_warmup_epochs + epoch + 1
            row = {
                "epoch": global_epoch,
                "phase": "joint",
                **train_metrics,
                **val_metrics,
            }
            history.append(row)

            print(
                f"[Joint] Epoch {epoch + 1:03d}/{self.num_epochs:03d} "
                f"(global {global_epoch:03d}) | "
                f"total={train_metrics['train_loss_total']:.4f} | "
                f"1a={train_metrics['train_loss_1a']:.4f} | "
                f"1b={train_metrics['train_loss_1b']:.4f} | "
                f"2={train_metrics['train_loss_2']:.4f} | "
                f"val_dice_2={val_metrics['val_dice_2']:.4f}"
            )

            if val_metrics["val_dice_2"] > self.best_val_dice:
                self.best_val_dice = val_metrics["val_dice_2"]
                self.patience_counter = 0
                self.save_checkpoint(epoch=global_epoch, val_dice=self.best_val_dice)
                print(
                    f"  -> New best checkpoint saved "
                    f"(val_dice_2={self.best_val_dice:.4f})"
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(
                        f"  -> Early stopping triggered at epoch {global_epoch} "
                        f"(patience={self.patience})"
                    )
                    break

            self.scheduler.step()

            if self.smoke_test:
                break

        # ── Save full training history ────────────────────────────────────────
        with open(self.history_path, "w") as fh:
            json.dump(history, fh, indent=2)

        print(
            f"\nTraining complete. "
            f"Best val Dice (Task 2): {self.best_val_dice:.4f}\n"
            f"Checkpoint : {self.ckpt_path}\n"
            f"History    : {self.history_path}"
        )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task training: Tasks 1a, 1b and 2 (RUN_0004)."
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
            "Quick sanity check: 1 warmup epoch + 1 joint epoch, "
            "2 steps max per epoch, batch_size forced to 1."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = MultiTaskTrainer(config=config, smoke_test=args.smoke_test)
    trainer.train()


if __name__ == "__main__":
    main()
