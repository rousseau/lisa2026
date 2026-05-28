"""Task 1a quality-assessment dataset — unified parameterised class."""

import pickle
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import build_image_only_transforms

TASK_NAMES = [
    "Noise",
    "Zipper",
    "Positioning",
    "Banding",
    "Motion",
    "Contrast",
    "Distortion",
]


class Task1aDataset(Dataset):
    """Unified Task 1a dataset covering both per-task and multi-label modes.

    Replaces three separate classes from the original codebase:
        * ``Task1aDataset``             (single task, 150³, RUN_0001)
        * ``Task1aMultiLabelDataset``   (multi-label, 150³, RUN_0002)
        * ``Task1aMultiTask128Dataset`` (multi-label, 128³, RUN_0004)

    The behaviour is controlled by ``task_name`` and ``spatial_size``:
        - ``task_name=None``  → multi-label mode (returns ``labels`` [7]).
        - ``task_name="Noise"`` → single-task mode (returns ``label`` scalar).

    Args:
        csv_path:     Path to Task 1a CSV with per-artifact labels.
        bids_root:    BIDS data root directory containing NIfTI files.
        split_pkl:    Path to patient-level split pickle.
        fold:         ``"train"`` or ``"val"``.
        stage:        ``"train"`` (with augmentation) or ``"val"``.
        task_name:    Single artifact name for per-task mode, or ``None`` for
                      multi-label mode.
        spatial_size: 3-tuple target crop/pad size (default (128, 128, 128)).
    """

    def __init__(
        self,
        csv_path: str,
        bids_root: str,
        split_pkl: str,
        fold: str,
        stage: str = "train",
        task_name: Optional[str] = None,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
    ):
        import os

        self.bids_root = bids_root
        self.task_name = task_name
        self.multilabel = task_name is None
        self.spatial_size = tuple(spatial_size)

        self.df = pd.read_csv(csv_path)

        with open(split_pkl, "rb") as fh:
            split = pickle.load(fh)

        indices = (
            split.get("train_indices", [])
            if fold == "train"
            else split.get("val_indices", [])
        )
        self.df = self.df.iloc[indices].reset_index(drop=True)

        # use_to_tensor=True keeps legacy compatibility for per-task mode
        self.transforms = build_image_only_transforms(
            spatial_size=self.spatial_size,
            stage=stage,
            use_to_tensor=True,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        import os

        row = self.df.iloc[idx]
        img_path = os.path.join(self.bids_root, row["filename"])

        data = self.transforms({"img": img_path})

        if self.multilabel:
            labels = torch.tensor(
                [int(row[t]) for t in TASK_NAMES], dtype=torch.long
            )
            return {
                "img": data["img"].float(),
                "labels": labels,
                "filename": row["filename"],
            }
        else:
            label = int(row[self.task_name])
            return {
                "img": data["img"],
                "label": torch.tensor(label, dtype=torch.long),
                "filename": row["filename"],
            }
