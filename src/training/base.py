"""Base trainer — shared boilerplate for all LISA 2026 training scripts."""

import json
import os
from abc import ABC, abstractmethod

import torch

from src.utils.config import apply_env_overrides
from src.utils.seed import set_seed


class BaseTrainer(ABC):
    """Abstract base class for all LISA 2026 trainers.

    Handles the common boilerplate:
        - Reproducibility (seed, deterministic mode).
        - Device selection and AMP GradScaler.
        - Environment-variable overrides for data paths.
        - Output directory creation.
        - Checkpoint save / load.
        - Early stopping state.
        - Training-history JSON serialisation.

    Subclasses must implement:
        - ``_build_model()``       → assign ``self.model``
        - ``_build_dataloaders()`` → assign ``self.train_loader``, ``self.val_loader``
        - ``_build_optimizer()``   → assign ``self.optimizer`` (+ ``self.scheduler``)
        - ``train_one_epoch()``    → return metrics dict
        - ``validate()``           → return metrics dict with ``self.val_metric_key``
        - ``train()``              → full training loop (calls helpers below)

    Args:
        config:     Loaded YAML config dict.
        smoke_test: If True, limit epochs and steps for quick sanity checks.
    """

    #: Metric key used for early stopping and checkpoint selection.
    #: Subclasses should override if they use a different primary metric.
    val_metric_key: str = "val_loss"
    #: Direction of improvement: ``"min"`` (lower is better) or ``"max"``.
    val_metric_direction: str = "min"

    def __init__(self, config: dict, smoke_test: bool = False):
        self.config = config
        self.smoke_test = smoke_test

        # ── Reproducibility ──────────────────────────────────────────────────
        seed = int(config.get("environment", {}).get("seed", 42))
        set_seed(seed)

        if config.get("environment", {}).get("deterministic", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ── Device + AMP ─────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (
            bool(config.get("environment", {}).get("mixed_precision", True))
            and self.device == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── Data-path overrides ───────────────────────────────────────────────
        apply_env_overrides(config)

        # ── Early stopping state ─────────────────────────────────────────────
        self.patience = int(
            config.get("early_stopping", {}).get("patience", 20)
        )
        self.patience_counter = 0
        if self.val_metric_direction == "min":
            self.best_val_metric = float("inf")
        else:
            self.best_val_metric = float("-inf")

        # ── Output directories ───────────────────────────────────────────────
        out_cfg = config.get("output", {})
        self.ckpt_dir = out_cfg.get("checkpoint_dir", "outputs/checkpoints")
        self.log_dir = out_cfg.get("log_dir", "outputs/logs")
        self.results_dir = out_cfg.get("results_dir", "results")
        for d in (self.ckpt_dir, self.log_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

        # Subclass contract: set num_epochs, ckpt_path in __init__
        self.num_epochs = 0
        self.ckpt_path = None
        self.history_path = os.path.join(self.results_dir, "training_history.json")

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def _build_model(self) -> None:
        """Instantiate model and assign to ``self.model``."""

    @abstractmethod
    def _build_dataloaders(self) -> None:
        """Build train/val dataloaders."""

    @abstractmethod
    def _build_optimizer(self) -> None:
        """Build optimizer (and optionally scheduler)."""

    @abstractmethod
    def train_one_epoch(self) -> dict:
        """Run one training epoch; return dict of metrics."""

    @abstractmethod
    def validate(self) -> dict:
        """Run validation; return dict containing ``self.val_metric_key``."""

    # NOTE: train() is concrete below — subclasses may override it or rely on
    # the generic loop if their epoch/validate interface is sufficient.

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_metric: float, path: str) -> None:
        """Save model + optimizer + scaler state dict.

        Args:
            epoch:      Current epoch index (1-based).
            val_metric: Value of the primary validation metric.
            path:       Full path to write the checkpoint.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            self.val_metric_key: val_metric,
            "config": self.config,
        }
        if self.use_amp and hasattr(self, "scaler") and self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> dict:
        """Load a checkpoint and restore model, optimizer, and scaler weights.

        Args:
            path:   Path to the checkpoint file.
            strict: Passed to ``model.load_state_dict``.

        Returns:
            The full checkpoint dict (contains epoch, metric, config, …).
        """
        ckpt = torch.load(path, map_location=self.device)
        sd = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(sd, strict=strict)

        if "optimizer_state_dict" in ckpt and hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scaler_state_dict" in ckpt and self.use_amp and hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        return ckpt

    # ── Early stopping helper ────────────────────────────────────────────────

    def is_improvement(self, val_metric: float) -> bool:
        """Return True if *val_metric* is an improvement over the current best."""
        if self.val_metric_direction == "min":
            return val_metric < self.best_val_metric
        return val_metric > self.best_val_metric

    def update_best(self, val_metric: float) -> None:
        """Update best metric and reset patience counter."""
        self.best_val_metric = val_metric
        self.patience_counter = 0

    def increment_patience(self) -> bool:
        """Increment patience counter; return True if early stopping triggered."""
        self.patience_counter += 1
        return self.patience_counter >= self.patience

    # ── History helper ───────────────────────────────────────────────────────

    def save_history(self, history: list) -> None:
        """Serialise training history to JSON.

        Args:
            history: List of per-epoch metric dicts.
        """
        with open(self.history_path, "w") as fh:
            json.dump(history, fh, indent=2)

    # ── Generic training loop ────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop.

        Subclasses can override ``_print_epoch()`` for custom log formatting
        or ``_on_training_complete()`` for post-processing (e.g. closing files).
        """
        history = []
        for epoch in range(self.num_epochs):
            train_m = self.train_one_epoch()
            val_m = self.validate()

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()

            row = {"epoch": epoch + 1, **train_m, **val_m}
            history.append(row)

            self._print_epoch(epoch + 1, train_m, val_m)

            if self.is_improvement(val_m[self.val_metric_key]):
                self.update_best(val_m[self.val_metric_key])
                self.save_checkpoint(
                    epoch + 1, val_m[self.val_metric_key], self.ckpt_path
                )
                print(f"  -> New best checkpoint ({self.ckpt_path})")
            elif self.increment_patience():
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break

            if self.smoke_test:
                break

        self.save_history(history)
        self._on_training_complete()
        print(
            f"Training complete. Best {self.val_metric_key}: "
            f"{self.best_val_metric:.4f}"
        )

    def _print_epoch(self, epoch: int, train_m: dict, val_m: dict) -> None:
        """Print a one-line summary for the current epoch.

        Subclasses may override for custom formatting.
        """
        parts = [f"Epoch {epoch:03d}/{self.num_epochs:03d}"]
        parts.append(", ".join(f"{k}={v:.4f}" for k, v in train_m.items()))
        parts.append(", ".join(f"{k}={v:.4f}" for k, v in val_m.items()))
        print(" | ".join(parts))

    def _on_training_complete(self) -> None:
        """Hook called after training finishes (default: no-op)."""
        pass
