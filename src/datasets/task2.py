"""Task 2 multi-structure segmentation dataset."""

import os
import pickle
import re
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import build_segmentation_transforms


def build_task2_records(
    data_root: str,
    image_suffix: str = "_ciso.nii.gz",
    label_suffix: str = "_LF_seg.nii.gz",
) -> list:
    """Build Task 2 image/label pairs from LISA naming convention.

    Returns:
        List of dicts with keys: subject, img, label.
    """
    records = []
    for filename in sorted(os.listdir(data_root)):
        if not filename.endswith(label_suffix):
            continue
        match = re.match(r"(LISA_\d+)", filename)
        if match is None:
            continue
        subject = match.group(1)
        label_path = os.path.join(data_root, filename)
        img_path = os.path.join(data_root, f"{subject}{image_suffix}")
        if os.path.exists(img_path):
            records.append({"subject": subject, "img": img_path, "label": label_path})
    return records


class Task2SegmentationDataset(Dataset):
    """Task 2 segmentation dataset for DynUNet (RUN_0003, RUN_0004).

    Args:
        data_root:            Directory with image + label NIfTI files.
        split_pkl:            Path to subject-level split pickle.
        fold:                 ``"train"`` or ``"val"``.
        stage:                ``"train"`` or ``"val"``.
        image_suffix:         Image filename suffix (default ``"_ciso.nii.gz"``).
        label_suffix:         Label filename suffix (default ``"_LF_seg.nii.gz"``).
        patch_size:           Training patch size (default (128, 128, 128)).
        num_samples_per_volume: Crops per volume in training (default 1).
        num_classes:          Number of segmentation classes (default 12).
        collapse_labels:      If True, remap labels via ``src.collapsed_labels``.
    """

    def __init__(
        self,
        data_root: str,
        split_pkl: str,
        fold: str,
        stage: str = "train",
        image_suffix: str = "_ciso.nii.gz",
        label_suffix: str = "_LF_seg.nii.gz",
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_samples_per_volume: int = 1,
        num_classes: int = 12,
        collapse_labels: bool = False,
    ):
        self.stage = stage
        self.patch_size = tuple(patch_size)
        self.num_classes = int(num_classes)
        self.collapse_labels = bool(collapse_labels)

        records = build_task2_records(
            data_root, image_suffix=image_suffix, label_suffix=label_suffix
        )
        records_df = pd.DataFrame(records)

        with open(split_pkl, "rb") as fh:
            split = pickle.load(fh)

        keep = set(
            split.get("train_subjects", [])
            if fold == "train"
            else split.get("val_subjects", [])
        )
        self.records = records_df[records_df["subject"].isin(keep)].to_dict("records")

        self.transforms = build_segmentation_transforms(
            spatial_size=self.patch_size,
            stage=stage,
            num_classes=self.num_classes,
            num_samples=int(num_samples_per_volume),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        data = self.transforms({"img": row["img"], "label": row["label"]})

        # RandCropByLabelClassesd can return a list when num_samples > 1.
        if isinstance(data, list):
            data = data[0]

        label = data["label"].long()
        if label.shape[0] != 1:
            label = label[:1]

        if self.collapse_labels:
            from src.collapsed_labels import COLLAPSED_MAP

            lookup = torch.tensor(
                [COLLAPSED_MAP.get(i, i) for i in range(12)],
                dtype=label.dtype,
                device=label.device,
            )
            label = lookup[label]

        return {
            "img": data["img"].float(),
            "label": label,
            "subject": row["subject"],
            "img_path": row["img"],
            "label_path": row["label"],
        }
