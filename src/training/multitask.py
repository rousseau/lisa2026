"""Multi-task trainer — shared DynUNet encoder, 3 heads (RUN_0004)."""

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
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, SSIMLoss
from monai.metrics import DiceMetric
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import get_multitask_dataloaders
from src.models import DynUNetMultiHeadModel, PlainConvMultiHeadModel
from src.training.base import BaseTrainer


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning-rate decay: lr = initial_lr * (1 - epoch/max_epoch)^power."""

    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / max(1, self.max_epochs)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class MultiTaskTrainer(BaseTrainer):
    """Multi-task trainer for Tasks 1a + 1b + 2 with shared DynUNet encoder (RUN_0004).

    Training phases
    ---------------
    1. **Warm-up** (Task 2 only) — establishes a stable segmentation backbone
       before introducing the other heads.
    2. **Head warm-up** (optional) — encoder frozen, trains 1a + 1b heads only.
    3. **Joint** — all three tasks trained simultaneously with loss normalisation
       and gradient clipping.

    Early stopping monitors ``val_dice_2`` (higher = better).
    """

    val_metric_key = "val_dice_2"
    val_metric_direction = "max"

    def __init__(self, config: dict, smoke_test: bool = False):
        super().__init__(config, smoke_test)

        config["data"]["data_root"] = os.getenv("LISA_DATA_ROOT", config["data"]["data_root"])
        if smoke_test:
            config["data"]["batch_size"] = 1

        self._build_dataloaders()
        self._build_model()
        self._build_optimizer()
        self._load_pretrained()

        # ── Loss functions ────────────────────────────────────────────────────
        loss_cfg = config["training"].get("loss", {})
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.focal_weight = float(loss_cfg.get("focal_weight", 0.5))
        self.ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0)
        self.seg_loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)

        # Task loss weights λ
        self.lambda_1a = float(config["training"].get("lambda_1a", 1.0))
        self.lambda_1b = float(config["training"].get("lambda_1b", 1.0))
        self.lambda_2  = float(config["training"].get("lambda_2",  1.0))

        # ── Phase lengths ────────────────────────────────────────────────────
        self.num_warmup_epochs     = 1 if smoke_test else int(config["training"].get("num_warmup_epochs", 10))
        self.num_head_warmup_epochs= 0 if smoke_test else int(config["training"].get("num_head_warmup_epochs", 0))
        self.num_epochs            = 1 if smoke_test else int(config["training"].get("num_epochs", 80))

        # ── Inference config ─────────────────────────────────────────────────
        self.val_roi_size   = tuple(int(x) for x in config["data"]["val_roi_size"])
        self.sw_batch_size  = int(config["inference"]["sw_batch_size"])
        self.overlap        = float(config["inference"]["overlap"])

        # ── Paths ────────────────────────────────────────────────────────────
        self.ckpt_path    = os.path.join(self.ckpt_dir, "multitask_best.pt")
        self.history_path = os.path.join(self.results_dir, "training_history.json")

        # ── Additional config ────────────────────────────────────────────────
        self.dice_warmup_target   = float(config["training"].get("dice_warmup_target", 0.10))
        self.max_grad_norm        = float(config["training"].get("max_grad_norm", 1.0))
        self.joint_lr             = float(config["training"].get("joint_learning_rate", 1e-5))
        self.head_warmup_lr       = float(config["training"].get("head_warmup_lr", self.joint_lr))
        self.loss_scale_1a = self.loss_scale_1b = self.loss_scale_2 = 1.0

        # ── Class weights Task 1a ────────────────────────────────────────────
        self.class_weights_1a = self._compute_1a_class_weights()

        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(
            f"Phases — warmup={self.num_warmup_epochs}ep  "
            f"head_warmup={self.num_head_warmup_epochs}ep  "
            f"joint={self.num_epochs}ep | "
            f"λ=(1a={self.lambda_1a}, 1b={self.lambda_1b}, 2={self.lambda_2}) "
            f"grad_clip={self.max_grad_norm}"
        )

    # ── BaseTrainer interface ────────────────────────────────────────────────

    def _build_model(self) -> None:
        cfg = self.config["model"]
        model_type = cfg.get("type", "dynunet").lower()
        if model_type in ("plainconv", "nnunet", "plainconv_multihead"):
            self.model = PlainConvMultiHeadModel(
                input_channels=int(cfg.get("in_channels", 1)),
                n_stages=int(cfg.get("n_stages", 6)),
                features_per_stage=tuple(int(x) for x in cfg.get("features_per_stage", (32, 64, 128, 256, 320, 320))),
                strides=tuple(tuple(int(s) for s in st) for st in cfg.get("strides", ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)))),
                num_seg_classes=int(cfg["num_seg_classes"]),
                num_artifact_tasks=int(cfg["num_artifact_tasks"]),
                num_artifact_classes=int(cfg["num_artifact_classes"]),
            ).to(self.device)
        else:
            self.model = DynUNetMultiHeadModel(
                in_channels=int(cfg.get("in_channels", 1)),
                filters=tuple(int(x) for x in cfg["filters"]),
                num_seg_classes=int(cfg["num_seg_classes"]),
                num_artifact_tasks=int(cfg["num_artifact_tasks"]),
                num_artifact_classes=int(cfg["num_artifact_classes"]),
            ).to(self.device)

    def _build_dataloaders(self) -> None:
        loaders = get_multitask_dataloaders(self.config)
        self.train_loader_1a, self.val_loader_1a, n_tr_1a, n_val_1a = loaders["1a"]
        self.train_loader_1b, self.val_loader_1b, n_tr_1b, n_val_1b = loaders["1b"]
        self.train_loader_2,  self.val_loader_2,  n_tr_2,  n_val_2  = loaders["2"]
        # Satisfy BaseTrainer expectations
        self.train_loader = self.train_loader_2
        self.val_loader   = self.val_loader_2
        print(f"1a: train={n_tr_1a}, val={n_val_1a}")
        print(f"1b: train={n_tr_1b}, val={n_val_1b}")
        print(f"2:  train={n_tr_2},  val={n_val_2}")

    def _build_optimizer(self) -> None:
        cfg = self.config["training"]
        total_epochs = int(cfg.get("num_warmup_epochs", 10)) + int(cfg.get("num_epochs", 80))
        lr = float(cfg["learning_rate"])
        wd = float(cfg.get("weight_decay", 3e-5))
        opt_type = cfg.get("optimizer", "adamw").lower()

        if opt_type == "sgd":
            momentum = float(cfg.get("momentum", 0.99))
            nesterov = bool(cfg.get("nesterov", True))
            self.optimizer = SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=wd,
                nesterov=nesterov,
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
            )

        sched_type = cfg.get("scheduler", "cosine").lower()
        if sched_type == "poly":
            power = float(cfg.get("poly_power", 0.9))
            self.scheduler = PolyLR(self.optimizer, max_epochs=max(1, total_epochs), power=power)
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_epochs),
                eta_min=float(cfg.get("min_learning_rate", 1e-6)),
            )

        # Deep supervision config
        self.use_deep_supervision = bool(self.config.get("model", {}).get("deep_supervision", False))
        self.ds_weights = cfg.get("deep_supervision_weights", [0.5, 0.25, 0.125, 0.0625, 0.03125])

    # ── Pretrained ──────────────────────────────────────────────────────────

    def _load_pretrained(self) -> None:
        pretrained = self.config["training"].get("pretrained_checkpoint", "")
        self.pretrained_loaded = False
        if not pretrained:
            return
        if not os.path.exists(pretrained):
            warnings.warn(f"[WARNING] Pretrained checkpoint not found: {pretrained}. Training from scratch.", stacklevel=2)
            return
        try:
            # Handle nnU-Net v2 checkpoint format (for PlainConvMultiHeadModel)
            if hasattr(self.model, 'load_pretrained_nnunet'):
                n_loaded = self.model.load_pretrained_nnunet(pretrained, device=self.device)
                if n_loaded > 0:
                    self.pretrained_loaded = True
                    print(f"[INFO] PlainConvMultiHead warm-started: {n_loaded} keys loaded.")
                else:
                    warnings.warn(f"[WARNING] No keys loaded from nnU-Net checkpoint. Training from scratch.", stacklevel=2)
                return

            # Legacy DynUNet path
            src_sd = torch.load(pretrained, map_location=self.device).get("model_state_dict", {})
            tgt_sd = self.model.state_dict()
            load_mode = self.config["training"].get("pretrained_load_mode", "all")
            if load_mode == "encoder_only":
                loaded = [k for k, v in src_sd.items() if k.startswith("model.") and k in tgt_sd and v.shape == tgt_sd[k].shape]
            else:
                loaded = [k for k, v in src_sd.items() if k in tgt_sd and v.shape == tgt_sd[k].shape]
            for k in loaded:
                tgt_sd[k] = src_sd[k]
            self.model.load_state_dict(tgt_sd)
            self.pretrained_loaded = True
            print(f"[INFO] Partial warm-start from {pretrained}: {len(loaded)}/{len(tgt_sd)} keys.")

            if self.config["model"].get("reset_heads", False):
                self.model.reset_heads()
        except Exception as exc:
            warnings.warn(f"[WARNING] Failed to load pretrained checkpoint: {exc}. Training from scratch.", stacklevel=2)

    # ── Class weights ────────────────────────────────────────────────────────

    def _compute_1a_class_weights(self) -> list:
        csv_path  = self.config["data"].get("csv_path", "")
        split_pkl = self.config["data"].get("split_pkl_1a", "")
        n_classes = int(self.config["model"].get("num_artifact_classes", 3))
        task_names = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]

        if not (os.path.exists(csv_path) and os.path.exists(split_pkl)):
            warnings.warn("[Task-1a weights] CSV or split file not found — using uniform weights.", stacklevel=2)
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
                w[k] = n_total / (n_classes * max(n_k, 1))
            w = torch.clamp(w, max=3.0)
            weights.append(w.to(self.device))
        return weights

    # ── Loss helpers ─────────────────────────────────────────────────────────

    def _loss_1a(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for t in range(logits.shape[1]):
            w = (self.class_weights_1a[t] if self.class_weights_1a and t < len(self.class_weights_1a) and self.class_weights_1a[t] is not None else None)
            ce = F.cross_entropy(logits[:, t, :], labels[:, t], weight=w)
            logp = F.log_softmax(logits[:, t, :], dim=1)
            pt = torch.exp(logp.gather(1, labels[:, t].unsqueeze(1)).squeeze(1)).clamp(1e-6, 1.0)
            focal = -((1.0 - pt) ** self.focal_gamma) * torch.log(pt)
            if w is not None:
                focal = focal * w[labels[:, t]]
            loss = loss + (1.0 - self.focal_weight) * ce + self.focal_weight * focal.mean()
        return loss

    @staticmethod
    def _loss_1b(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(recon, target)

    # ── Loss calibration ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _calibrate_losses(self) -> tuple:
        self.model.eval()
        MIN_SCALE = 1e-6
        b1a = next(iter(self.train_loader_1a))
        b1b = next(iter(self.train_loader_1b))
        b2  = next(iter(self.train_loader_2))
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            l1a = self._loss_1a(self.model.forward_task1a(b1a["img"].to(self.device)), b1a["labels"].to(self.device))
            clean = b1b["img_B"].to(self.device)
            l1b = self._loss_1b(self.model.forward_task1b(clean), clean)
            l2  = self._seg_loss_with_ds(self.model.forward_task2(b2["img"].to(self.device)), b2["label"].to(self.device))
        self.model.train()
        def _safe(x):
            v = float(x.item())
            return v if (math.isfinite(v) and v > 0) else MIN_SCALE
        s1a, s1b, s2 = _safe(l1a), _safe(l1b), _safe(l2)
        print(f"[Calibration] 1a={s1a:.4f}  1b={s1b:.4f}  2={s2:.4f}")
        return s1a, s1b, s2

    # ── Deep-supervision helper ──────────────────────────────────────────────

    def _seg_loss_with_ds(self, model_out, labels):
        """Compute segmentation loss, handling deep-supervision 6-D tensor.

        When deep supervision is enabled, MONAI DynUNet returns a tensor of
        shape ``[B, N_levels, C, H, W, D]``.  Level 0 is the main output;
        deeper levels (1..N-1) are progressively downsampled.  We index each
        level, downsample ``labels`` to match, and apply weighted DiceCE.
        """
        if not self.use_deep_supervision or model_out.dim() != 6:
            return self.seg_loss_fn(model_out, labels)

        total = 0.0
        n_levels = model_out.shape[1]
        weights = self.ds_weights
        for i in range(n_levels):
            w = float(weights[i]) if i < len(weights) else 0.0
            if w == 0.0:
                continue
            out = model_out[:, i]  # [B, C, H, W, D]
            # Downsample labels to match this output's spatial size
            if out.shape[2:] != labels.shape[2:]:
                lbl_ds = F.interpolate(
                    labels.float(),
                    size=out.shape[2:],
                    mode="nearest",
                )
            else:
                lbl_ds = labels
            total = total + w * self.seg_loss_fn(out, lbl_ds)
        return total

    # ── Training epoch methods ───────────────────────────────────────────────

    def train_one_epoch(self) -> dict:
        """Alias for joint epoch (used by BaseTrainer.train if called directly)."""
        return self.train_one_epoch_joint()

    def train_one_epoch_warmup(self) -> dict:
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(self.train_loader_2, desc="Warmup-Train")):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model.forward_task2(images)
                loss = self._seg_loss_with_ds(out, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += float(loss.item())
            if self.smoke_test and batch_idx >= 1:
                break
        avg = total_loss / max(1, batch_idx + 1)
        return {"train_loss_total": avg, "train_loss_1a": 0.0, "train_loss_1b": 0.0, "train_loss_2": avg}

    def train_one_epoch_head_warmup(self) -> dict:
        for p in self.model.model.parameters():
            p.requires_grad = False
        self.model.train()
        n1a, n1b = len(self.train_loader_1a), len(self.train_loader_1b)
        steps = min(max(n1a, n1b), 2) if self.smoke_test else max(n1a, n1b)
        iter_1a = itertools.cycle(self.train_loader_1a)
        iter_1b = itertools.cycle(self.train_loader_1b)
        total_1a = total_1b = nan_steps = 0.0
        for step in tqdm(range(steps), desc="HeadWarmup"):
            b1a = next(iter_1a)
            b1b = next(iter_1b)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                l1a = self._loss_1a(self.model.forward_task1a(b1a["img"].to(self.device)), b1a["labels"].to(self.device))
                clean = b1b["img_B"].to(self.device)
                l1b = self._loss_1b(self.model.forward_task1b(clean), clean)
                loss = self.lambda_1a * l1a + self.lambda_1b * l1b
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                nan_steps += 1
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_1a += float(l1a.item())
            total_1b += float(l1b.item())
            if self.smoke_test and step >= 1:
                break
        for p in self.model.model.parameters():
            p.requires_grad = True
        valid = max(1, steps - nan_steps)
        if nan_steps > 0 and (nan_steps / max(1, steps)) > 0.25:
            raise RuntimeError(f"Head warm-up unstable: {int(nan_steps)}/{steps} NaN/Inf steps.")
        return {"train_loss_1a": total_1a / valid, "train_loss_1b": total_1b / valid, "nan_steps": int(nan_steps)}

    def train_one_epoch_joint(self) -> dict:
        self.model.train()
        n1a, n1b, n2 = len(self.train_loader_1a), len(self.train_loader_1b), len(self.train_loader_2)
        n_steps = max(n1a, n1b, n2)
        steps = min(n_steps, 2) if self.smoke_test else n_steps
        iter_1a = itertools.cycle(self.train_loader_1a)
        iter_1b = itertools.cycle(self.train_loader_1b)
        iter_2  = itertools.cycle(self.train_loader_2)
        total = t1a = t1b = t2 = nan_steps = 0.0
        for step in tqdm(range(steps), desc="Joint-Train"):
            b1a = next(iter_1a); b1b = next(iter_1b); b2 = next(iter_2)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                l1a = self._loss_1a(self.model.forward_task1a(b1a["img"].to(self.device)), b1a["labels"].to(self.device))
                clean = b1b["img_B"].to(self.device)
                l1b = self._loss_1b(self.model.forward_task1b(clean), clean)
                l2  = self._seg_loss_with_ds(self.model.forward_task2(b2["img"].to(self.device)), b2["label"].to(self.device))
                loss = (self.lambda_1a * l1a / self.loss_scale_1a
                      + self.lambda_1b * l1b / self.loss_scale_1b
                      + self.lambda_2  * l2  / self.loss_scale_2)
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                nan_steps += 1
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.item()); t1a += float(l1a.item()); t1b += float(l1b.item()); t2 += float(l2.item())
        valid = max(1, steps - nan_steps)
        if nan_steps > 0 and (nan_steps / max(1, steps)) > 0.25:
            raise RuntimeError(f"Joint phase unstable: {int(nan_steps)}/{steps} NaN/Inf steps.")
        if nan_steps > 0:
            print(f"  [WARNING] {int(nan_steps)}/{steps} NaN/Inf steps skipped.")
        return {"train_loss_total": total / valid, "train_loss_1a": t1a / valid, "train_loss_1b": t1b / valid, "train_loss_2": t2 / valid, "nan_steps": int(nan_steps)}

    # ── Validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        num_classes = int(self.config["model"]["num_seg_classes"])
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        for batch_idx, batch in enumerate(tqdm(self.val_loader_2, desc="Val-Task2")):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = sliding_window_inference(images, roi_size=self.val_roi_size, sw_batch_size=self.sw_batch_size, predictor=self.model.forward_task2_main, overlap=self.overlap)
            pred = torch.argmax(logits, dim=1, keepdim=True)
            pred_oh = F.one_hot(pred.squeeze(1), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            lbl_oh  = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            dice_metric(y_pred=pred_oh, y=lbl_oh)
            if self.smoke_test and batch_idx >= 0:
                break
        val_dice_2 = float(dice_metric.aggregate().item())
        dice_metric.reset()

        val_loss_1a = 0.0
        for batch_idx, batch in enumerate(tqdm(self.val_loader_1a, desc="Val-Task1a")):
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                val_loss_1a += float(self._loss_1a(self.model.forward_task1a(batch["img"].to(self.device)), batch["labels"].to(self.device)).item())
            if self.smoke_test and batch_idx >= 1:
                break
        val_loss_1a /= max(1, batch_idx + 1)

        val_loss_1b = 0.0
        for batch_idx, batch in enumerate(tqdm(self.val_loader_1b, desc="Val-Task1b")):
            images = batch["img_B"].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                val_loss_1b += float(F.l1_loss(self.model.forward_task1b(images), images).item())
            if self.smoke_test and batch_idx >= 1:
                break
        val_loss_1b /= max(1, batch_idx + 1)

        return {self.val_metric_key: val_dice_2, "val_loss_1a": val_loss_1a, "val_loss_1b": val_loss_1b}

    # ── Full training loop ───────────────────────────────────────────────────

    def train(self) -> None:
        run_meta = {
            "run_id": self.config.get("run_id", "0004"),
            "pretrained_loaded": self.pretrained_loaded,
            "device": self.device,
            "num_warmup_epochs": self.num_warmup_epochs,
            "num_head_warmup_epochs": self.num_head_warmup_epochs,
            "num_epochs": self.num_epochs,
            "loss_scale_1a": None, "loss_scale_1b": None, "loss_scale_2": None,
        }
        history = [run_meta]

        # ── Phase 1: Warm-up ─────────────────────────────────────────────────
        print(f"\n{'='*60}\nWarm-up ({self.num_warmup_epochs} ep) — Task 2 only\n{'='*60}")
        for epoch in range(self.num_warmup_epochs):
            tm = self.train_one_epoch_warmup()
            vm = self.validate()
            history.append({"epoch": epoch + 1, "phase": "warmup", **tm, **vm})
            print(f"[Warmup] {epoch+1:03d}/{self.num_warmup_epochs:03d} | loss={tm['train_loss_total']:.4f} | val_dice_2={vm['val_dice_2']:.4f}")
            if self.is_improvement(vm[self.val_metric_key]):
                self.update_best(vm[self.val_metric_key])
                self.save_checkpoint(epoch + 1, vm[self.val_metric_key], self.ckpt_path)
                print(f"  -> Checkpoint saved (val_dice_2={self.best_val_metric:.4f})")
            self.scheduler.step()
            if vm["val_dice_2"] >= self.dice_warmup_target:
                print(f"  -> Warmup early exit (dice={vm['val_dice_2']:.4f} >= {self.dice_warmup_target:.2f})")
                break
            if self.smoke_test:
                break

        # ── Phase 2: Head warm-up ────────────────────────────────────────────
        if self.num_head_warmup_epochs > 0 and not self.smoke_test:
            print(f"\n{'='*60}\nHead warm-up ({self.num_head_warmup_epochs} ep) — 1a+1b, encoder frozen\n{'='*60}")
            prev_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.head_warmup_lr
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            print(f"[HeadWarmup] LR {prev_lrs[0]:.2e} -> {self.head_warmup_lr:.2e} | GradScaler reset")
            for epoch in range(self.num_head_warmup_epochs):
                hm = self.train_one_epoch_head_warmup()
                history.append({"epoch": self.num_warmup_epochs + epoch + 1, "phase": "head_warmup", **hm})
                print(f"[HeadWarmup] {epoch+1:03d} | loss_1a={hm['train_loss_1a']:.4f} loss_1b={hm['train_loss_1b']:.4f} nan={hm['nan_steps']}")
            # NaN guard
            if any(p.data.isnan().any().item() for p in self.model.parameters() if p.data.is_floating_point()):
                warnings.warn("[WARNING] NaN weights after head warmup. Reloading best checkpoint.", stacklevel=2)
                if os.path.exists(self.ckpt_path):
                    ckpt = torch.load(self.ckpt_path, map_location=self.device)
                    self.model.load_state_dict(ckpt["model_state_dict"])

        # ── Phase 3: Joint ───────────────────────────────────────────────────
        prev_lr = self.optimizer.param_groups[0]["lr"]
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.joint_lr

        # Rebuild scheduler for joint phase respecting config
        sched_type = self.config["training"].get("scheduler", "cosine").lower()
        if sched_type == "poly":
            power = float(self.config["training"].get("poly_power", 0.9))
            self.scheduler = PolyLR(self.optimizer, max_epochs=max(1, self.num_epochs), power=power)
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.num_epochs),
                eta_min=float(self.config["training"].get("min_learning_rate", 1e-6)),
            )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        print(f"[Joint setup] LR {prev_lr:.2e} -> {self.joint_lr:.2e} | scheduler={sched_type} | GradScaler reset")

        self.loss_scale_1a, self.loss_scale_1b, self.loss_scale_2 = self._calibrate_losses()
        run_meta["loss_scale_1a"] = self.loss_scale_1a
        run_meta["loss_scale_1b"] = self.loss_scale_1b
        run_meta["loss_scale_2"]  = self.loss_scale_2

        print(f"\n{'='*60}\nJoint ({self.num_epochs} ep) — all 3 tasks\n{'='*60}")
        for epoch in range(self.num_epochs):
            tm = self.train_one_epoch_joint()
            vm = self.validate()
            g_epoch = self.num_warmup_epochs + self.num_head_warmup_epochs + epoch + 1
            history.append({"epoch": g_epoch, "phase": "joint", **tm, **vm})
            print(
                f"[Joint] {epoch+1:03d}/{self.num_epochs:03d} (g={g_epoch:03d}) | "
                f"total={tm['train_loss_total']:.4f} 1a={tm['train_loss_1a']:.4f} "
                f"1b={tm['train_loss_1b']:.4f} 2={tm['train_loss_2']:.4f} | "
                f"val_dice_2={vm['val_dice_2']:.4f}"
            )
            if self.is_improvement(vm[self.val_metric_key]):
                self.update_best(vm[self.val_metric_key])
                self.patience_counter = 0
                self.save_checkpoint(g_epoch, vm[self.val_metric_key], self.ckpt_path)
                print(f"  -> New best (val_dice_2={self.best_val_metric:.4f})")
            elif self.increment_patience():
                print(f"  -> Early stopping at epoch {g_epoch}")
                break
            self.scheduler.step()
            if self.smoke_test:
                break

        with open(self.history_path, "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"\nTraining complete. Best val Dice (Task 2): {self.best_val_metric:.4f}")
        print(f"Checkpoint: {self.ckpt_path}")
