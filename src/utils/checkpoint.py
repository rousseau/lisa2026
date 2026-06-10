"""Checkpoint utilities shared across training and evaluation."""

import torch


def load_checkpoint_safe(
    path: str,
    model: torch.nn.Module,
    *,
    strict: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: str = "cpu",
) -> dict:
    """Load a checkpoint, restoring model/optimizer/scaler state.

    Args:
        path:       Path to the checkpoint file.
        model:      Model to load weights into.
        strict:     Passed to ``model.load_state_dict``.
        optimizer:  If provided, restore its state dict when present.
        scaler:     If provided, restore its state dict when present.
        device:     Map location for ``torch.load``.

    Returns:
        The full checkpoint dictionary (may contain epoch, metric, config, …).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle both bare state_dict and wrapped dicts
    sd = ckpt.get("model_state_dict", ckpt)
    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint at {path} does not contain a valid state dict.")

    model.load_state_dict(sd, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt
