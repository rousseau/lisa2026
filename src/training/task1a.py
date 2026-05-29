"""Task 1a trainers for LISA 2026.

Exposes one concrete trainer:
    - ``Task1aOrdinalTrainer`` — single-task per-artifact ordinal model (RUN_0001).
"""

import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from tqdm import tqdm

from src.datasets import TASK_NAMES, get_dataloaders
from src.models import Task1aOrdinalModel
from src.training.base import BaseTrainer


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

    def _print_epoch(self, epoch: int, train_m: dict, val_m: dict) -> None:
        msg = (
            f"Epoch {epoch:3d}/{self.num_epochs} | "
            f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} | "
            f"val_f1={val_m['f1']:.4f} val_agg={val_m['agg']:.4f}"
        )
        self._print(msg)

    def _on_training_complete(self) -> None:
        self._print(f"\nTraining completed. Best F1: {self.best_val_metric:.4f}")
        self._log_fh.close()

