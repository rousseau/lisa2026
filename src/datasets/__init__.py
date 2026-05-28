"""Datasets for LISA 2026 — public re-exports.

All original public names are preserved for backward compatibility with
existing train/evaluate scripts.
"""

from .task1a import TASK_NAMES, Task1aDataset
from .task1b import (
    CleanImageDataset,
    Task1bDataset,
    _subjects_from_split,
    build_task1b_records,
)
from .task2 import Task2SegmentationDataset, build_task2_records
from .loaders import (
    get_dataloaders,
    get_multilabel_dataloaders,
    get_multitask_dataloaders,
    get_task1b_dataloaders,
    get_task2_seg_dataloaders,
)

# Legacy aliases kept for scripts that import these names directly
Task1aMultiLabelDataset = Task1aDataset       # multilabel=True (task_name=None)
Task1aMultiTask128Dataset = Task1aDataset     # same class, spatial_size kwarg

__all__ = [
    # Constants
    "TASK_NAMES",
    # Datasets
    "Task1aDataset",
    "Task1aMultiLabelDataset",
    "Task1aMultiTask128Dataset",
    "Task1bDataset",
    "CleanImageDataset",
    "Task2SegmentationDataset",
    # Helpers
    "build_task1b_records",
    "build_task2_records",
    "_subjects_from_split",
    # Loaders
    "get_dataloaders",
    "get_multilabel_dataloaders",
    "get_multitask_dataloaders",
    "get_task1b_dataloaders",
    "get_task2_seg_dataloaders",
]
