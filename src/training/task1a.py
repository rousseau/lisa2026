"""Task 1a trainers for LISA 2026.

Exposes two concrete trainers:
    - ``Task1aOrdinalTrainer`` — single-task per-artifact ordinal model (RUN_0001).
    - ``Task1aMultiLabelTrainer`` — multi-head DenseNet trained on all 7 artifacts
      simultaneously with EMD + Focal loss (RUN_0002).
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import TASK_NAMES, get_dataloaders, get_multilabel_dataloaders
from src.models import Task1aMultiLabelModel, Task1aOrdinalModel
from src.training.base import BaseTrainer


# ─── Loss functions ──────────────────────────────────────────────────────────


def ordinal_emd_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Earth Mover's Distance (Cramér distance) for ordinal classification.

    Args:
        logits:  [B, 7, 3]  raw model output.
        targets: [B, 7]     integer class labels in {0, 1, 2}.

    Returns:
        Scalar mean EMD loss over batch and tasks.
    """
    num_classes = logits.shape[-1]
    probs = torch.softmax(logits, dim=-1)
    targets_oh = F.one_hot(targets, num_classes).float()
    pred_cdf = torch.cumsum(probs, dim=-1)
    target_cdf = torch.cumsum(targets_oh, dim=-1)
    return torch.mean((pred_cdf[..., :-1] - target_cdf[..., :-1]) ** 2)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: tuple = (0.25, 0.5, 1.0),
) -> torch.Tensor:
    """Focal loss to address class imbalance.

    Args:
        logits:  [B, 7, 3]
        targets: [B, 7]
        gamma:   focusing parameter.
        alpha:   per-class weights.

    Returns:
        Scalar mean focal loss.
    """
    B, T, C = logits.shape
    logits_2d = logits.reshape(B * T, C)
    targets_1d = targets.reshape(B * T)
    weight = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
    ce = F.cross_entropy(logits_2d, targets_1d, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return torch.mean((1.0 - pt) ** gamma * ce)


# ─── Task1aOrdinalTrainer (RUN_0001) ─────────────────────────────────────────


class Task1aOrdinalTrainer(BaseTrainer):
    """Single-task ordinal classifier for one artifact type (RUN_0001).

    Args:
        config:    Loaded YAML config dict.
        task_name: Artifact name — one of TASK_NAMES.
        smoke_test: If True, limit epochs/batches.
    """

    val_metric_key = "val_f1"
    val_metric_direction = "max"

    def __init__(self, config: dict, task_name: str, smoke_test: bool = False):
        super().__init__(config, smoke_test)
        self.task_name = task_name
        self._build_dataloaders()
        self._build_model()
        self._build_optimizer()

        self.num_epochs = 2 if smoke_test else int(config["training"]["num_epochs"])
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{task_name}_best.pt")

        log_path = os.path.join(self.log_dir, f"task1a_{task_name}.log")
        self._log_fh = open(log_path, "w")
        self._print(f"Task: {task_name}")
        self._print(f"Train samples: {self.n_train}, Val samples: {self.n_val}")

    # ── BaseTrainer interface ────────────────────────────────────────────────

    def _build_model(self) -> None:
        self.model = Task1aOrdinalModel(num_classes=3).to(self.device)

    def _build_dataloaders(self) -> None:
        cfg = self.config
        self.train_loader, self.val_loader, self.n_train, self.n_val = get_dataloaders(
            csv_path=cfg["data"]["csv_path"],
            bids_root=cfg["data"]["bids_root"],
            split_pkl=cfg["data"]["split_pkl"],
            task_name=self.task_name,
            batch_size=cfg["training"]["batch_size"],
            num_workers=2,
        )

    def _build_optimizer(self) -> None:
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
        )
        self.scheduler = None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _print(self, msg: str) -> None:
        print(msg)
        if hasattr(self, "_log_fh"):
            self._log_fh.write(msg + "\n")
            self._log_fh.flush()

    @staticmethod
    def _compute_metrics(y_true, y_pred) -> dict:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        return {
            "acc": acc,
            "f1": f1,
            "f2": f2,
            "prec": prec,
            "rec": rec,
            "agg": float(np.mean([acc, f1, f2, prec, rec])),
        }

    # ── Epoch methods ────────────────────────────────────────────────────────

    def train_one_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in tqdm(self.train_loader, desc="Train"):
            img = batch["img"].to(self.device).float()
            label = batch["label"].to(self.device)
            logits = self.model(img)
            loss = cross_entropy(logits, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        metrics = self._compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val"):
                img = batch["img"].to(self.device).float()
                label = batch["label"].to(self.device)
                logits = self.model(img)
                loss = cross_entropy(logits, label)
                total_loss += loss.item()
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        metrics = self._compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(self.val_loader)
        metrics[self.val_metric_key] = metrics["f1"]
        return metrics

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            train_m = self.train_one_epoch()
            val_m = self.validate()

            msg = (
                f"Epoch {epoch + 1:3d}/{self.num_epochs} | "
                f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} | "
                f"val_f1={val_m['f1']:.4f} val_agg={val_m['agg']:.4f}"
            )
            self._print(msg)

            if self.is_improvement(val_m[self.val_metric_key]):
                self.update_best(val_m[self.val_metric_key])
                torch.save(self.model.state_dict(), self.ckpt_path)
                self._print(f"  -> Best checkpoint saved (F1={self.best_val_metric:.4f})")
            elif self.increment_patience():
                self._print(f"  -> Early stopping at epoch {epoch + 1}")
                break

        self._print(f"\nTraining completed. Best F1: {self.best_val_metric:.4f}")
        self._log_fh.close()


# ─── Task1aMultiLabelTrainer (RUN_0002) ──────────────────────────────────────


class Task1aMultiLabelTrainer(BaseTrainer):
    """Multi-head DenseNet trainer for Task 1a — all 7 artifacts (RUN_0002).

    Loss = EMD_weight * ordinal_emd_loss + focal_weight * focal_loss.
    """

    val_metric_key = "val_mean_f1"
    val_metric_direction = "max"

    def __init__(self, config: dict, smoke_test: bool = False):
        super().__init__(config, smoke_test)
        self._build_dataloaders()
        self._build_model()
        self._build_optimizer()

        loss_cfg = config["training"].get("loss", {})
        self.emd_weight = float(loss_cfg.get("emd_weight", 1.0))
        self.focal_weight = float(loss_cfg.get("focal_weight", 1.0))
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.focal_alpha = tuple(loss_cfg.get("focal_alpha", [0.25, 0.5, 1.0]))

        self.num_epochs = 2 if smoke_test else int(config["training"]["num_epochs"])
        self.ckpt_path = os.path.join(self.ckpt_dir, "multilabel_best.pt")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Train: {self.n_train}  Val: {self.n_val}  Params: {n_params:,}")
        print(f"Loss: EMD (w={self.emd_weight}) + Focal (w={self.focal_weight})")

    # ── BaseTrainer interface ────────────────────────────────────────────────

    def _build_model(self) -> None:
        self.model = Task1aMultiLabelModel().to(self.device)

    def _build_dataloaders(self) -> None:
        cfg = self.config
        (
            self.train_loader,
            self.val_loader,
            self.n_train,
            self.n_val,
        ) = get_multilabel_dataloaders(
            csv_path=cfg["data"]["csv_path"],
            bids_root=cfg["data"]["bids_root"],
            split_pkl=cfg["data"]["split_pkl"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"].get("num_workers", 2),
        )

    def _build_optimizer(self) -> None:
        cfg = self.config["training"]
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.num_epochs if hasattr(self, "num_epochs") else int(self.config["training"]["num_epochs"])),
            eta_min=1e-6,
        )

    # ── Epoch helpers ────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool) -> dict:
        self.model.train(train)
        context = torch.enable_grad() if train else torch.no_grad()
        total_loss = 0.0
        all_preds = [[] for _ in TASK_NAMES]
        all_labels = [[] for _ in TASK_NAMES]

        with context:
            for batch_idx, batch in enumerate(loader):
                if self.smoke_test and batch_idx >= 2:
                    break
                imgs = batch["img"].to(self.device).float()
                labels = batch["labels"].to(self.device)
                logits = self.model(imgs)
                loss = (
                    self.emd_weight * ordinal_emd_loss(logits, labels)
                    + self.focal_weight * focal_loss(
                        logits, labels,
                        gamma=self.focal_gamma,
                        alpha=self.focal_alpha,
                    )
                )
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                lbl_np = labels.cpu().numpy()
                for t in range(len(TASK_NAMES)):
                    all_preds[t].extend(preds[:, t].tolist())
                    all_labels[t].extend(lbl_np[:, t].tolist())

        n_batches = min(3, len(loader)) if self.smoke_test else len(loader)
        avg_loss = total_loss / max(n_batches, 1)
        per_task_f1 = {
            task: float(f1_score(all_labels[t], all_preds[t], average="macro", zero_division=0))
            for t, task in enumerate(TASK_NAMES)
        }
        mean_f1 = float(np.mean(list(per_task_f1.values())))
        return {"loss": avg_loss, "mean_f1": mean_f1, "per_task_f1": per_task_f1}

    def train_one_epoch(self) -> dict:
        return self._run_epoch(self.train_loader, train=True)

    def validate(self) -> dict:
        result = self._run_epoch(self.val_loader, train=False)
        result[self.val_metric_key] = result["mean_f1"]
        return result

    def train(self) -> None:
        print(f"\n{'=' * 60}\nRUN_0002 – Multi-label Training ({self.num_epochs} epochs)\n{'=' * 60}\n")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_m = self.train_one_epoch()
            val_m = self.validate()
            self.scheduler.step()
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:03d}/{self.num_epochs:03d} | "
                f"train_loss={train_m['loss']:.4f} | "
                f"val_f1={val_m['mean_f1']:.4f} | {elapsed:.0f}s"
            )
            if epoch % 5 == 0 or epoch == self.num_epochs:
                for task, f1 in val_m["per_task_f1"].items():
                    print(f"    {task:<16} F1={f1:.3f}")

            if self.is_improvement(val_m[self.val_metric_key]):
                self.update_best(val_m[self.val_metric_key])
                torch.save(self.model.state_dict(), self.ckpt_path)
                print(f"  -> New best val_f1={self.best_val_metric:.4f} – checkpoint saved")
            elif self.increment_patience() and not self.smoke_test:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\nTraining complete. Best val_f1={self.best_val_metric:.4f}")
        print(f"Checkpoint: {self.ckpt_path}")
