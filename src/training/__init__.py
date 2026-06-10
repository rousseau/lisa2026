"""Training sub-package for LISA 2026."""

from .base import BaseTrainer
from .task1a import Task1aOrdinalTrainer
from .task1b import CycleGANTrainer
from .task2 import Task2Trainer
from .task2_hybrid import Task2HybridTrainer
from .task2_medsam2 import Task2MedSAM2Trainer
from .multitask import MultiTaskTrainer

__all__ = [
    "BaseTrainer",
    "Task1aOrdinalTrainer",
    "CycleGANTrainer",
    "Task2Trainer",
    "Task2MedSAM2Trainer",
    "Task2HybridTrainer",
    "MultiTaskTrainer",
]
