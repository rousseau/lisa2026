#!/usr/bin/env python
"""Multi-task training: Tasks 1a, 1b and 2 with shared encoder (RUN_0004)."""

import argparse
import itertools
import json
import math
import os
import pickle
import warnings

import numpy as np
import pandas as pd
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
from src.utils.seed import set_seed


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
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.focal_weight = float(loss_cfg.get("focal_weight", 0.5))
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

        # ── Warmup early-exit target ────────────────────────────────────────
        self.dice_warmup_target = float(
            config["training"].get("dice_warmup_target", 0.10)
        )

        # ── Loss normalization scales (calibrated before joint phase) ─────────
        self.loss_scale_1a = 1.0
        self.loss_scale_1b = 1.0
        self.loss_scale_2 = 1.0

        # ── Gradient clipping ──────────────────────────────────────────────
        self.max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))

        # ── Joint phase LR (lower than warmup to stabilise NaN-prone heads) ───
        self.joint_lr = float(config["training"].get("joint_learning_rate", 1.0e-5))

        # ── Head warm-up LR (use joint_lr by default — much safer than warmup LR
        #    which at epoch 43/130 is still ~75% of max LR and causes fp16 overflow)
        self.head_warmup_lr = float(
            config["training"].get("head_warmup_lr", self.joint_lr)
        )

        # ── Head warm-up epochs (encoder frozen, 1a + 1b seulement) ────────
        self.num_head_warmup_epochs = int(
            config["training"].get("num_head_warmup_epochs", 0)
        )

        # ── Class weights for Task 1a (inverse-frequency balancing) ───────────
        # Severe imbalance: Banding has 96% class-0, weights up to 18× for cls 1.
        # Computed from training split at init-time (reproducible, split-aware).
        self.class_weights_1a = self._compute_1a_class_weights()

        print(f"Device : {self.device} | AMP : {self.use_amp}")
        if self.class_weights_1a:
            w_str = ", ".join(
                f"{n}: [{'/'.join(f'{w:.1f}' for w in ws.tolist())}]"
                for n, ws in zip(
                    ["Noise", "Zipper", "Pos", "Band", "Motion", "Contr", "Dist"],
                    self.class_weights_1a,
                )
            )
            print(f"Task-1a class weights — {w_str}")
        print(
            f"Warm-up: {self.num_warmup_epochs} epochs "
            f"(early exit if DSC≥{self.dice_warmup_target:.2f}) | "
            f"Head warm-up: {self.num_head_warmup_epochs} epochs "
            + (
                f"@ lr={self.head_warmup_lr:.1e} | "
                if self.num_head_warmup_epochs
                else "(disabled) | "
            )
            + f"Joint: {self.num_epochs} epochs @ lr={self.joint_lr:.1e} | "
            f"λ=(1a={self.lambda_1a}, 1b={self.lambda_1b}, 2={self.lambda_2}) "
            f"[loss-normalized, grad_clip={self.max_grad_norm}]"
        )

    # ─── Loss helpers + class-weight computation ─────────────────────────────────────

    def _compute_1a_class_weights(self) -> list:
        """Compute per-artifact inverse-frequency class weights for Task 1a.

        Loads the CSV and the training split to get the actual class distribution
        seen during training (split-aware, reproducible).

        Returns:
            List of 7 float32 tensors, each of shape [num_artifact_classes].
            Returns empty list if CSV or split file is not found (graceful fallback).
        """
        csv_path = self.config["data"].get("csv_path", "")
        split_pkl = self.config["data"].get("split_pkl_1a", "")
        n_classes = int(self.config["model"].get("num_artifact_classes", 3))
        task_names = [
            "Noise",
            "Zipper",
            "Positioning",
            "Banding",
            "Motion",
            "Contrast",
            "Distortion",
        ]

        if not (os.path.exists(csv_path) and os.path.exists(split_pkl)):
            warnings.warn(
                f"[Task-1a weights] CSV or split file not found — "
                "using uniform class weights.",
                stacklevel=2,
            )
            return []

        df = pd.read_csv(csv_path)
        with open(split_pkl, "rb") as fh:
            split = pickle.load(fh)
        train_df = df.iloc[split.get("train_indices", [])]
        n_total = len(train_df)

        weights = []
        for col in task_names:
            if col not in train_df.columns:
                weights.append(None)
                continue
            w = torch.ones(n_classes, dtype=torch.float32)
            for k in range(n_classes):
                n_k = int((train_df[col] == k).sum())
                # w_k = N / (n_classes * N_k) — standard inverse-frequency
                w[k] = n_total / (n_classes * max(n_k, 1))
            # Cap weights to prevent loss explosion (max ratio 3× is sufficient)
            w = torch.clamp(w, max=3.0)
            weights.append(w.to(self.device))
        return weights

    def _loss_1a(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Class-balanced CE + focal loss summed over the 7 artifact heads.

        Uses per-artifact inverse-frequency class weights (computed at init from
        the training split) to correct for severe class imbalance — e.g. Banding
        has 96 % class-0 samples, weight ratio ~18× for minority classes.

        Args:
            logits: [B, 7, 3]  — per-artifact logits.
            labels: [B, 7]     — per-artifact integer class labels.

        Returns:
            Scalar tensor (weighted sum over 7 heads).
        """
        loss = torch.tensor(0.0, device=self.device)
        for t in range(logits.shape[1]):
            w = (
                self.class_weights_1a[t]
                if self.class_weights_1a
                and t < len(self.class_weights_1a)
                and self.class_weights_1a[t] is not None
                else None
            )
            ce = F.cross_entropy(logits[:, t, :], labels[:, t], weight=w)
            logp = F.log_softmax(logits[:, t, :], dim=1)
            pt = torch.exp(
                logp.gather(1, labels[:, t].unsqueeze(1)).squeeze(1)
            ).clamp(1e-6, 1.0)
            focal = -((1.0 - pt) ** self.focal_gamma) * torch.log(pt)
            if w is not None:
                focal = focal * w[labels[:, t]]
            focal = focal.mean()
            loss = loss + (1.0 - self.focal_weight) * ce + self.focal_weight * focal
        return loss

    def _loss_1b(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple L1 reconstruction loss.

        Args:
            recon:  [B, 1, H, W, D] — model reconstruction output.
            target: [B, 1, H, W, D] — clean input volume (identity target).

        Returns:
            Scalar tensor (L1).
        """
        return F.l1_loss(recon, target)

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

        # Use math.isfinite to avoid the Python max(nan, x)==nan pitfall
        # (max(nan, 1e-6) returns NaN because nan>1e-6 is False and Python's
        # max() keeps the first element unchanged when no other is larger).
        raw_1a = float(l_1a.item())
        raw_1b = float(l_1b.item())
        raw_2 = float(l_2.item())
        s1a = raw_1a if (math.isfinite(raw_1a) and raw_1a > 0) else MIN_SCALE
        s1b = raw_1b if (math.isfinite(raw_1b) and raw_1b > 0) else MIN_SCALE
        s2 = raw_2 if (math.isfinite(raw_2) and raw_2 > 0) else MIN_SCALE

        print(f"[Calibration] Initial losses — 1a={s1a:.4f}  1b={s1b:.4f}  2={s2:.4f}")
        print(
            f"[Calibration] Effective weights after normalisation — "
            f"λ_1a/L0={self.lambda_1a / s1a:.4f}  "
            f"λ_1b/L0={self.lambda_1b / s1b:.4f}  "
            f"λ_2/L0={self.lambda_2 / s2:.4f}"
        )
        return s1a, s1b, s2

    def train_one_epoch_head_warmup(self) -> dict:
        """Head warm-up : entraîne seulement les têtes 1a et 1b, encodeur gelé.

        L'encodeur DynUNet (model.model.*) est gelé.  Seuls cls_gap, cls_mlp,
        recon_up*, recon_out sont mis à jour.  Ceci permet aux têtes
        aléatoires de converger vers des sorties raisonnables avant le joint
        training, prévenant l'explosion de gradients liée aux logits initiaux.

        Returns:
            Dict avec train_loss_1a, train_loss_1b.
        """
        # Geler l'encodeur backbone
        for param in self.model.model.parameters():
            param.requires_grad = False
        self.model.train()

        n1a = len(self.train_loader_1a)
        n1b = len(self.train_loader_1b)
        steps = min(max(n1a, n1b), 2) if self.smoke_test else max(n1a, n1b)
        iter_1a = itertools.cycle(self.train_loader_1a)
        iter_1b = itertools.cycle(self.train_loader_1b)

        total_l_1a = 0.0
        total_l_1b = 0.0
        nan_steps = 0

        for _step in tqdm(range(steps), desc="HeadWarmup-Train"):
            b1a = next(iter_1a)
            b1b = next(iter_1b)
            img_1a = b1a["img"].to(self.device)
            labels_1a = b1a["labels"].to(self.device)
            img_1b = b1b["img"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                l_1a = self._loss_1a(self.model.forward_task1a(img_1a), labels_1a)
                l_1b = self._loss_1b(self.model.forward_task1b(img_1b), img_1b)
                loss = self.lambda_1a * l_1a + self.lambda_1b * l_1b

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                nan_steps += 1
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_l_1a += float(l_1a.item())
            total_l_1b += float(l_1b.item())

            if self.smoke_test and _step >= 1:
                break

        # Dégeler l'encodeur pour la suite
        for param in self.model.model.parameters():
            param.requires_grad = True

        valid = max(1, steps - nan_steps)
        if nan_steps > 0 and (nan_steps / max(1, steps)) > 0.25:
            raise RuntimeError(
                f"Head warm-up unstable: {nan_steps}/{steps} NaN/Inf steps."
            )
        return {
            "train_loss_1a": total_l_1a / valid,
            "train_loss_1b": total_l_1b / valid,
            "nan_steps": nan_steps,
        }

    # ─── Training phases ────────────────────────────────────────────────────────────────

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
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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
        nan_steps = 0

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

            # ── NaN / Inf guard ────────────────────────────────────────────────
            if not torch.isfinite(loss):
                # Skip this step : do NOT call backward (NaN gradients would
                # corrupt model weights permanently).
                self.optimizer.zero_grad(set_to_none=True)
                nan_steps += 1
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_l_1a += float(l_1a.item())
            total_l_1b += float(l_1b.item())
            total_l_2 += float(l_2.item())

        valid_steps = max(1, steps - nan_steps)
        if nan_steps > 0:
            print(
                f"  [WARNING] {nan_steps}/{steps} steps had NaN/Inf loss "
                f"and were skipped (grad clip={self.max_grad_norm})."
            )
        if nan_steps > 0 and (nan_steps / max(1, steps)) > 0.25:
            raise RuntimeError(
                f"Joint phase unstable: {nan_steps}/{steps} NaN/Inf steps."
            )
        return {
            "train_loss_total": total_loss / valid_steps,
            "train_loss_1a": total_l_1a / valid_steps,
            "train_loss_1b": total_l_1b / valid_steps,
            "train_loss_2": total_l_2 / valid_steps,
            "nan_steps": nan_steps,
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
            "num_head_warmup_epochs": self.num_head_warmup_epochs,
            "num_epochs": self.num_epochs,
            "num_seg_classes": int(self.config["model"]["num_seg_classes"]),
            "filters": list(self.config["model"].get("filters", [])),
            "loss_normalization": True,
            "max_grad_norm": self.max_grad_norm,
            "joint_learning_rate": self.joint_lr,
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

        # ════════════════════════════════════════════════════════════════
        # Phase 2 — Head warm-up (têtes 1a + 1b, encodeur gelé)
        # ════════════════════════════════════════════════════════════════
        if self.num_head_warmup_epochs > 0 and not self.smoke_test:
            print(
                f"\n{'=' * 60}\n"
                f"Head warm-up phase ({self.num_head_warmup_epochs} epochs) "
                f"— têtes 1a + 1b, encodeur gelé\n"
                f"{'=' * 60}"
            )

            # FIX: reset LR to head_warmup_lr before head warmup.
            # At epoch 43/130 the cosine scheduler LR is still ~75% of max
            # (≈7.5e-5). Running fresh random heads at that LR with AMP fp16
            # causes fp16 overflow after ~870 steps (epochs 3-5 → 100% NaN).
            # head_warmup_lr defaults to joint_lr (1e-5), 7.5× smaller.
            prev_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.head_warmup_lr
            # FIX: reset GradScaler — after many clean warmup epochs the
            # internal scale factor may be very large; a fresh scaler starts
            # at the safe default (65536) preventing fp16 overflow on new heads.
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            print(
                f"[HeadWarmup setup] LR reset from {prev_lrs[0]:.2e} "
                f"to {self.head_warmup_lr:.2e} | GradScaler reset."
            )

            for epoch in range(self.num_head_warmup_epochs):
                hw_metrics = self.train_one_epoch_head_warmup()
                row = {
                    "epoch": self.num_warmup_epochs + epoch + 1,
                    "phase": "head_warmup",
                    **hw_metrics,
                }
                history.append(row)
                print(
                    f"[HeadWarmup] Epoch {epoch + 1:03d}/{self.num_head_warmup_epochs:03d} "
                    f"| loss_1a={hw_metrics['train_loss_1a']:.4f} "
                    f"| loss_1b={hw_metrics['train_loss_1b']:.4f} "
                    f"| nan_steps={hw_metrics['nan_steps']}"
                )

            # FIX: after head warmup, detect NaN weights (corruption).
            # If any parameter contains NaN (can happen even at low LR with
            # badly-conditioned fp16 gradients), reload the best warmup
            # checkpoint to restore clean encoder + decoder weights.
            has_nan = any(
                p.data.isnan().any().item()
                for p in self.model.parameters()
                if p.data.is_floating_point()
            )
            if has_nan:
                warnings.warn(
                    "[WARNING] Head warm-up produced NaN weights. "
                    "Reloading best warmup checkpoint before joint phase.",
                    stacklevel=2,
                )
                if os.path.exists(self.ckpt_path):
                    ckpt = torch.load(self.ckpt_path, map_location=self.device)
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    print(
                        f"[INFO] Reloaded checkpoint (epoch={ckpt.get('epoch')}, "
                        f"val_dice_2={ckpt.get('val_dice_2', 'N/A'):.4f})."
                    )
                else:
                    print(
                        "[WARNING] No checkpoint found to reload after NaN head warmup."
                    )
            else:
                print("[HeadWarmup] Weights are clean (no NaN detected).")

        # ════════════════════════════════════════════════════════════════
        # Phase 3 — Joint training (all 3 tasks)
        # ════════════════════════════════════════════════════════════════
        # ── Reset LR for joint phase (lower to prevent NaN on fresh heads) ───
        prev_lr = self.optimizer.param_groups[0]["lr"]
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.joint_lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.num_epochs),
            eta_min=float(self.config["training"].get("min_learning_rate", 1.0e-6)),
        )
        # FIX: reset GradScaler before joint phase to clear any inflated scale
        # factor accumulated during warmup / head warmup, avoiding fp16 overflow
        # on the first joint backward pass.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        print(
            f"[Joint setup] LR reset to {self.joint_lr:.2e} (was {prev_lr:.2e}). "
            f"GradScaler reset."
        )

        # ── Calibrate loss scales before joint phase ─────────────────────────────
        self.loss_scale_1a, self.loss_scale_1b, self.loss_scale_2 = (
            self._calibrate_losses()
        )
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

            global_epoch = (
                self.num_warmup_epochs + self.num_head_warmup_epochs + epoch + 1
            )
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
