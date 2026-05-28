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

    @abstractmethod
    def train(self) -> None:
        """Full training loop."""

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_metric: float, path: str) -> None:
        """Save model + optimizer state dict.

        Args:
            epoch:      Current epoch index (1-based).
            val_metric: Value of the primary validation metric.
            path:       Full path to write the checkpoint.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                self.val_metric_key: val_metric,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str, strict: bool = True) -> dict:
        """Load a checkpoint and restore model weights.

        Args:
            path:   Path to the checkpoint file.
            strict: Passed to ``model.load_state_dict``.

        Returns:
            The full checkpoint dict (contains epoch, metric, config, …).
        """
        ckpt = torch.load(path, map_location=self.device)
        sd = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(sd, strict=strict)
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
