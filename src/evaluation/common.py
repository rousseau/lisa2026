"""Shared evaluation boilerplate for all LISA 2026 evaluators.

Provides: config loading, device selection, checkpoint loading, and
smoke-test helpers so that every evaluate_*.py script can start with

    from src.evaluation.common import load_config, get_device, load_checkpoint
"""

import os
from typing import Any

import torch
import yaml

from src.utils.config import apply_env_overrides


def load_config(path: str) -> dict:
    """Load YAML config, apply environment overrides for data paths."""
    with open(path, "r") as fh:
        config = yaml.safe_load(fh)
    apply_env_overrides(config)
    return config


def get_device(config: dict | None = None) -> str:
    """Return ``'cuda'`` if a GPU is available, else ``'cpu'``.

    If *config* is provided, mixed-precision flag is read from
    ``config["environment"]["mixed_precision"]``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config is not None:
        use_amp = (
            bool(config.get("environment", {}).get("mixed_precision", True))
            and device == "cuda"
        )
        return device, use_amp
    return device, False


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: str,
    strict: bool = True,
) -> dict[str, Any]:
    """Load model weights and checkpoint metadata.

    Returns the full checkpoint dict (contains ``epoch``,
    ``model_state_dict``, ``val_metric_key``, etc.).
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=strict)
    else:
        # Fallback: raw state-dict (e.g. old Task 1a checkpoints)
        model.load_state_dict(state, strict=strict)
    return state


def should_break(batch_idx: int, smoke_test: bool, limit: int = 1) -> bool:
    """Return ``True`` if smoke-test early-exit should trigger."""
    return smoke_test and batch_idx >= limit
