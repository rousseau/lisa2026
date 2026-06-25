"""Task 2 multi-structure segmentation dataset.

Updated for RUN_0006 iteration: uses 79 patients from Task2/ + Task2Extra/,
with image fallback to Task1b/ for the 25 orphan patients.
Returns both HF seg (scored) and LF_seg (auxiliary supervision).
"""

import os
import pickle
import re
from typing import Tuple, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import build_segmentation_transforms


def build_task2_records(
    task2_dir: str,
    task2extra_dir: str,
    task1b_dir: str,
    image_suffix: str = "_ciso.nii.gz",
    label_suffix: str = "_seg.nii.gz",
    label_suffix_lf: str = "_LF_seg.nii.gz",
) -> list:
    """Build Task 2 image/label pairs from new LISA 2026 directory layout.

    Scans ``task2_dir`` for HF labels (``*_seg.nii.gz``), looks up the image in
    ``task2_dir`` first, then falls back to ``task1b_dir``.  The LF label is
    looked up in ``task2extra_dir``.

    Returns:
        List of dicts with keys: subject, img, label, label_lf (optional).
    """
    if not os.path.isdir(task2_dir):
        raise FileNotFoundError(f"Task2 directory not found: {task2_dir}")

    records = []
    seg_files = sorted([f for f in os.listdir(task2_dir) if f.endswith(label_suffix)])
    for filename in seg_files:
        match = re.match(r"(LISA_\d+)", filename)
        if match is None:
            continue
        subject = match.group(1)
        label_path = os.path.join(task2_dir, filename)

        # Image: Task2 first, then Task1b fallback
        img_path = os.path.join(task2_dir, f"{subject}{image_suffix}")
        if not os.path.exists(img_path):
            img_path = os.path.join(task1b_dir, f"{subject}{image_suffix}")
        if not os.path.exists(img_path):
            continue  # cannot train without image

        record = {"subject": subject, "img": img_path, "label": label_path}

        # LF label (auxiliary) — optional
        lf_label_path = os.path.join(task2extra_dir, f"{subject}{label_suffix_lf}")
        if os.path.exists(lf_label_path):
            record["label_lf"] = lf_label_path

        records.append(record)

    return records


class Task2SegmentationDataset(Dataset):
    """Task 2 segmentation dataset for PlainConvUNet (RUN_0006 iteration).

    Args:
        split_pkl: Path to subject-level split pickle.
        fold: ``"train"`` or ``"val"``.
        stage: ``"train"`` or ``"val"``.
        task2_dir: Directory with HF images and ``*_seg.nii.gz`` labels.
        task2extra_dir: Directory with ``*_LF_seg.nii.gz`` auxiliary labels.
        task1b_dir: Fallback directory for missing ``*_ciso.nii.gz`` images.
        image_suffix: Image filename suffix (default ``"_ciso.nii.gz"``).
        label_suffix: HF label suffix (default ``"_seg.nii.gz"``).
        label_suffix_lf: LF label suffix (default ``"_LF_seg.nii.gz"``).
        patch_size: Training patch size (default (128, 128, 128)).
        num_samples_per_volume: Crops per volume in training (default 1).
        num_classes: Number of segmentation classes (default 12).
        collapse_labels: If True, remap labels via ``src.collapsed_labels``.
        use_nnunet_preprocessing: If True, apply exact nnU-Net preprocessing.
    """

    def __init__(
        self,
        split_pkl: str,
        fold: str,
        stage: str = "train",
        task2_dir: str = "",
        task2extra_dir: str = "",
        task1b_dir: str = "",
        image_suffix: str = "_ciso.nii.gz",
        label_suffix: str = "_seg.nii.gz",
        label_suffix_lf: str = "_LF_seg.nii.gz",
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_samples_per_volume: int = 1,
        num_classes: int = 12,
        collapse_labels: bool = False,
        use_nnunet_preprocessing: bool = False,
    ):
        self.stage = stage
        self.patch_size = tuple(patch_size)
        self.num_classes = int(num_classes)
        self.collapse_labels = bool(collapse_labels)

        records = build_task2_records(
            task2_dir=task2_dir,
            task2extra_dir=task2extra_dir,
            task1b_dir=task1b_dir,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            label_suffix_lf=label_suffix_lf,
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
            use_nnunet_preprocessing=use_nnunet_preprocessing,
        )

        # Pre-compute collapsed label lookup Tensor if needed (outside __getitem__)
        if self.collapse_labels:
            from src.collapsed_labels import COLLAPSED_MAP

            self._collapsed_lookup = torch.tensor(
                [COLLAPSED_MAP.get(i, i) for i in range(self.num_classes)],
                dtype=torch.long,
            )
        else:
            self._collapsed_lookup = None

    def __len__(self) -> int:
        return len(self.records)

    def _process_label(self, label_tensor: torch.Tensor) -> torch.Tensor:
        """Crop channel dim and optionally collapse labels."""
        if label_tensor.shape[0] != 1:
            label_tensor = label_tensor[:1]
        label_tensor = label_tensor.long()
        if self._collapsed_lookup is not None:
            # Make sure lookup is on same device as tensor
            lookup = self._collapsed_lookup.to(label_tensor.device)
            label_tensor = lookup[label_tensor]
        return label_tensor

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        data = self.transforms({"img": row["img"], "label": row["label"]})

        if isinstance(data, list):
            data = data[0]

        label = self._process_label(data["label"])

        result = {
            "img": data["img"].float(),
            "label": label,
            "subject": row["subject"],
            "img_path": row["img"],
            "label_path": row["label"],
        }

        # LF label (auxiliary supervision) — processed separately
        if "label_lf" in row:
            data_lf = self.transforms({"img": row["img"], "label": row["label_lf"]})
            if isinstance(data_lf, list):
                data_lf = data_lf[0]
            label_lf = self._process_label(data_lf["label"])
            result["label_lf"] = label_lf

        return result
