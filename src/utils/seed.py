"""Reproducibility helpers."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets Python, NumPy and PyTorch (CPU + all CUDA devices) seeds.
    Call this before any dataset construction or model initialisation.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
