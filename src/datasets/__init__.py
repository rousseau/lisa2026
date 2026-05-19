"""
Task 1a Dataset and DataLoader
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    ToTensord,
)
from torch.utils.data import DataLoader, Dataset


class Task1aDataset(Dataset):
    """Task 1a Quality Assessment Dataset"""

    def __init__(self, csv_path, bids_root, split_pkl, fold, task_name, stage="train"):
        """
        Args:
            csv_path: Path to CSV with labels
            bids_root: Root directory of BIDS data
            split_pkl: Path to pickle file with train/val split
            fold: 'train' or 'val'
            task_name: One of ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
            stage: 'train' or 'val' (for augmentation)
        """
        self.csv_path = csv_path
        self.bids_root = bids_root
        self.fold = fold
        self.task_name = task_name
        self.stage = stage

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Load split
        with open(split_pkl, "rb") as f:
            self.split = pickle.load(f)

        # Filter by fold
        if fold == "train":
            indices = self.split.get("train_indices", [])
        else:
            indices = self.split.get("val_indices", [])

        self.df = self.df.iloc[indices].reset_index(drop=True)

        # Build transforms
        self.transforms = self._build_transforms(stage)

    def _build_transforms(self, stage):
        """Build MONAI transform pipeline"""
        base_transforms = [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            CenterSpatialCropd(keys=["img"], roi_size=(150, 150, 150)),
            SpatialPadd(keys=["img"], spatial_size=(150, 150, 150), mode="symmetric"),
        ]

        if stage == "train":
            augmentations = [
                RandRotated(
                    keys=["img"],
                    prob=0.2,
                    range_x=np.deg2rad(15),
                    range_y=np.deg2rad(15),
                    range_z=np.deg2rad(10),
                    mode="bilinear",
                ),
                RandAffined(
                    keys=["img"],
                    prob=0.2,
                    scale_range=(0.05, 0.05, 0.05),
                    translate_range=(3, 3, 2),
                    mode="bilinear",
                ),
                RandShiftIntensityd(keys=["img"], prob=0.2, offsets=0.1),
                RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.8, 1.2)),
            ]
            base_transforms.extend(augmentations)

        base_transforms.append(ToTensord(keys=["img"]))

        return Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.bids_root, row["filename"])
        label = int(row[self.task_name])

        data = {"img": img_path, "label": label}
        data = self.transforms(data)

        return {
            "img": data["img"],
            "label": torch.tensor(label, dtype=torch.long),
            "filename": row["filename"],
        }


def get_dataloaders(
    csv_path, bids_root, split_pkl, task_name, batch_size=8, num_workers=2
):
    """Get train and val dataloaders"""
    train_dataset = Task1aDataset(
        csv_path, bids_root, split_pkl, "train", task_name, "train"
    )
    val_dataset = Task1aDataset(csv_path, bids_root, split_pkl, "val", task_name, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


# ─── RUN_0002 – Multi-label ──────────────────────────────────────────────────

TASK_NAMES = [
    "Noise",
    "Zipper",
    "Positioning",
    "Banding",
    "Motion",
    "Contrast",
    "Distortion",
]


class Task1aMultiLabelDataset(Dataset):
    """Task 1a – returns all 7 artifact labels in a single sample (RUN_0002+)."""

    def __init__(self, csv_path, bids_root, split_pkl, fold, stage="train"):
        self.bids_root = bids_root
        self.fold = fold
        self.stage = stage

        self.df = pd.read_csv(csv_path)

        with open(split_pkl, "rb") as f:
            split = pickle.load(f)

        indices = (
            split.get("train_indices", [])
            if fold == "train"
            else split.get("val_indices", [])
        )
        self.df = self.df.iloc[indices].reset_index(drop=True)

        self.transforms = self._build_transforms(stage)

    def _build_transforms(self, stage):
        base_transforms = [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            CenterSpatialCropd(keys=["img"], roi_size=(150, 150, 150)),
            SpatialPadd(keys=["img"], spatial_size=(150, 150, 150), mode="symmetric"),
        ]

        if stage == "train":
            base_transforms.extend(
                [
                    RandRotated(
                        keys=["img"],
                        prob=0.2,
                        range_x=np.deg2rad(15),
                        range_y=np.deg2rad(15),
                        range_z=np.deg2rad(10),
                        mode="bilinear",
                    ),
                    RandAffined(
                        keys=["img"],
                        prob=0.2,
                        scale_range=(0.05, 0.05, 0.05),
                        translate_range=(3, 3, 2),
                        mode="bilinear",
                    ),
                    RandShiftIntensityd(keys=["img"], prob=0.2, offsets=0.1),
                    RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.8, 1.2)),
                ]
            )

        base_transforms.append(ToTensord(keys=["img"]))
        return Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.bids_root, row["filename"])
        labels = torch.tensor(
            [int(row[t]) for t in TASK_NAMES], dtype=torch.long
        )  # [7]

        data = self.transforms({"img": img_path})

        return {
            "img": data["img"],
            "labels": labels,
            "filename": row["filename"],
        }


def get_multilabel_dataloaders(
    csv_path, bids_root, split_pkl, batch_size=8, num_workers=2
):
    """Get train and val dataloaders for multi-label Task 1a (RUN_0002+)."""
    train_ds = Task1aMultiLabelDataset(csv_path, bids_root, split_pkl, "train", "train")
    val_ds = Task1aMultiLabelDataset(csv_path, bids_root, split_pkl, "val", "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


# ─── RUN_0003 – Task 2 Multi-structure Segmentation ─────────────────────────


def build_task2_records(
    data_root, image_suffix="_ciso.nii.gz", label_suffix="_LF_seg.nii.gz"
):
    """Build Task 2 image/label pairs from LISA naming convention.

    Returns a list of dict records with keys: subject, img, label.
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
        img_filename = f"{subject}{image_suffix}"
        img_path = os.path.join(data_root, img_filename)

        if os.path.exists(img_path):
            records.append({"subject": subject, "img": img_path, "label": label_path})

    return records


class Task2SegmentationDataset(Dataset):
    """Task 2 segmentation dataset for DynUNet baseline (RUN_0003)."""

    def __init__(
        self,
        data_root,
        split_pkl,
        fold,
        stage="train",
        image_suffix="_ciso.nii.gz",
        label_suffix="_LF_seg.nii.gz",
        patch_size=(128, 128, 128),
        num_samples_per_volume=1,
        num_classes=12,
        collapse_labels=False,
    ):
        self.stage = stage
        self.fold = fold
        self.patch_size = tuple(patch_size)
        self.num_samples_per_volume = int(num_samples_per_volume)
        self.num_classes = int(num_classes)
        self.collapse_labels = bool(collapse_labels)

        records = build_task2_records(
            data_root, image_suffix=image_suffix, label_suffix=label_suffix
        )
        records_df = pd.DataFrame(records)

        with open(split_pkl, "rb") as f:
            split = pickle.load(f)

        if fold == "train":
            keep_subjects = set(split.get("train_subjects", []))
        else:
            keep_subjects = set(split.get("val_subjects", []))

        self.records = records_df[records_df["subject"].isin(keep_subjects)].to_dict(
            "records"
        )
        self.transforms = self._build_transforms(stage)

    def _build_transforms(self, stage):
        base = [
            LoadImaged(keys=["img", "label"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img", "label"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            EnsureTyped(keys=["img", "label"]),
        ]

        if stage == "train":
            base.extend(
                [
                    SpatialPadd(
                        keys=["img", "label"],
                        spatial_size=self.patch_size,
                        mode=("constant", "constant"),
                    ),
                    RandCropByLabelClassesd(
                        keys=["img", "label"],
                        label_key="label",
                        spatial_size=self.patch_size,
                        num_classes=self.num_classes,
                        ratios=[0.25] + [1.0] * (self.num_classes - 1),
                        num_samples=self.num_samples_per_volume,
                    ),
                    RandAffined(
                        keys=["img", "label"],
                        prob=0.25,
                        rotate_range=(np.deg2rad(20), np.deg2rad(20), np.deg2rad(20)),
                        scale_range=(0.15, 0.15, 0.15),
                        translate_range=(8, 8, 8),
                        mode=("bilinear", "nearest"),
                    ),
                    RandFlipd(keys=["img", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["img", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["img", "label"], prob=0.5, spatial_axis=2),
                    RandGaussianNoised(keys=["img"], prob=0.15, mean=0.0, std=0.01),
                    RandScaleIntensityd(keys=["img"], factors=0.1, prob=0.2),
                    RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.7, 1.5)),
                    RandGaussianSmoothd(
                        keys=["img"],
                        prob=0.15,
                        sigma_x=(0.5, 1.0),
                        sigma_y=(0.5, 1.0),
                        sigma_z=(0.5, 1.0),
                    ),
                ]
            )
        else:
            base.extend(
                [
                    SpatialPadd(
                        keys=["img", "label"],
                        spatial_size=self.patch_size,
                        mode=("constant", "constant"),
                    ),
                ]
            )

        return Compose(base)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        data = self.transforms({"img": row["img"], "label": row["label"]})

        # RandCropByPosNegLabeld can return a list when num_samples > 1.
        if isinstance(data, list):
            data = data[0]

        label = data["label"].long()
        if label.shape[0] != 1:
            label = label[:1]

        if self.collapse_labels:
            from src.collapsed_labels import COLLAPSED_MAP

            # Build lookup tensor: index=original_label, value=collapsed_label
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


def get_task2_seg_dataloaders(
    data_root,
    split_pkl,
    batch_size=1,
    num_workers=2,
    image_suffix="_ciso.nii.gz",
    label_suffix="_LF_seg.nii.gz",
    patch_size=(128, 128, 128),
    num_samples_per_volume=1,
    num_classes=12,
    collapse_labels=False,
):
    """Get train/val dataloaders for Task 2 segmentation (RUN_0003)."""
    train_ds = Task2SegmentationDataset(
        data_root=data_root,
        split_pkl=split_pkl,
        fold="train",
        stage="train",
        image_suffix=image_suffix,
        label_suffix=label_suffix,
        patch_size=patch_size,
        num_samples_per_volume=num_samples_per_volume,
        num_classes=num_classes,
        collapse_labels=collapse_labels,
    )
    val_ds = Task2SegmentationDataset(
        data_root=data_root,
        split_pkl=split_pkl,
        fold="val",
        stage="val",
        image_suffix=image_suffix,
        label_suffix=label_suffix,
        patch_size=patch_size,
        num_samples_per_volume=1,
        num_classes=num_classes,
        collapse_labels=collapse_labels,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


# ─── RUN_0004 – Task 1b Self-supervised Denoising ────────────────────────────


def build_task1b_records(data_root: str, image_suffix: str = "_ciso.nii.gz") -> list:
    """Scan *data_root* for volumes matching *image_suffix* and return a record list.

    Returns:
        List of dicts with keys: ``subject`` (str), ``img_path`` (str).
    """
    records = []
    for filename in sorted(os.listdir(data_root)):
        if not filename.endswith(image_suffix):
            continue
        match = re.match(r"(LISA_\d+)", filename)
        if match is None:
            continue
        subject = match.group(1)
        records.append(
            {"subject": subject, "img_path": os.path.join(data_root, filename)}
        )
    return records


def _subjects_from_split(split: dict, all_subjects: list, fold: str) -> set:
    """Extract the subject set for a given fold from any supported split format.

    Supported formats:
      * ``{train_subjects, val_subjects}`` – Task2-style subject-level split.
      * ``{train_indices, val_indices}``   – Task1a-style CSV-index split.  Since
        these indices refer to an external CSV we cannot re-map them here; the
        train/val fraction is used to derive an equivalent subject partition.
    """
    if "train_subjects" in split:
        key = "train_subjects" if fold == "train" else "val_subjects"
        return set(split[key])

    if "train_indices" in split:
        # Approximate split by replicating the train fraction on the sorted
        # subject list so the ratio is preserved across tasks.
        n_total_samples = split.get("n_samples", len(all_subjects))
        n_train_samples = len(split["train_indices"])
        fraction = n_train_samples / max(1, n_total_samples)
        n_train_subjects = max(1, round(fraction * len(all_subjects)))
        if fold == "train":
            return set(all_subjects[:n_train_subjects])
        return set(all_subjects[n_train_subjects:])

    # Unknown format – default to 80/20
    n_train = max(1, int(0.8 * len(all_subjects)))
    if fold == "train":
        return set(all_subjects[:n_train])
    return set(all_subjects[n_train:])


class Task1bDataset(Dataset):
    """Task 1b dataset: self-supervised denoising (RUN_0004).

    For each sample the dataset returns the preprocessed *clean* volume.
    Synthetic degradation is applied externally by the trainer so that noise
    parameters can vary epoch by epoch.

    The train/val split reuses ``task2_fixed.pkl`` (``train_subjects``/
    ``val_subjects``) or ``task1a_fixed.pkl`` (``train_indices``/``val_indices``)
    if the referenced file is available.  Otherwise a deterministic
    ``GroupShuffleSplit`` is created from the filesystem records.

    Args:
        data_root:    Directory containing ``*_ciso.nii.gz`` volumes.
        split_pkl:    Path to the patient-level split pickle.  Both subject-level
                      (Task2 format) and index-level (Task1a format) pickles are
                      supported.
        fold:         ``"train"`` or ``"val"``.
        stage:        ``"train"`` or ``"val"`` (reserved for future augmentations).
        image_suffix: Filename suffix to match (default ``"_ciso.nii.gz"``).
        spatial_size: 3-tuple for centre-crop / spatial-pad target size.
    """

    def __init__(
        self,
        data_root: str,
        split_pkl: str,
        fold: str = "train",
        stage: str = "train",
        image_suffix: str = "_ciso.nii.gz",
        spatial_size: tuple = (96, 96, 96),
    ):
        self.stage = stage
        self.fold = fold
        self.spatial_size = tuple(spatial_size)

        records = build_task1b_records(data_root, image_suffix=image_suffix)
        if not records:
            raise RuntimeError(
                f"No '{image_suffix}' volumes found in {data_root}. "
                "Check data_root and image_suffix in the config."
            )

        records_df = pd.DataFrame(records)
        all_subjects = sorted(records_df["subject"].unique().tolist())

        if os.path.exists(split_pkl):
            with open(split_pkl, "rb") as f:
                split = pickle.load(f)
            keep = _subjects_from_split(split, all_subjects, fold)
        else:
            # No split file – create a deterministic patient-level split on-the-fly.
            from sklearn.model_selection import GroupShuffleSplit as _GSS

            gss = _GSS(n_splits=1, test_size=0.2, random_state=42)
            groups = records_df["subject"].values
            train_idx, val_idx = next(gss.split(records_df, groups=groups))
            if fold == "train":
                keep = set(records_df.iloc[train_idx]["subject"].unique())
            else:
                keep = set(records_df.iloc[val_idx]["subject"].unique())

        self.records = records_df[records_df["subject"].isin(keep)].to_dict("records")

        if not self.records:
            raise RuntimeError(
                f"Task1bDataset ({fold}): no volumes remain after applying split. "
                f"Subjects found: {all_subjects[:5]} ... "
                f"Subjects requested: {sorted(keep)[:5]} ..."
            )

        self.transforms = self._build_transforms(stage)

    def _build_transforms(self, stage: str):
        base = [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            CenterSpatialCropd(keys=["img"], roi_size=self.spatial_size),
            SpatialPadd(keys=["img"], spatial_size=self.spatial_size, mode="symmetric"),
            EnsureTyped(keys=["img"]),
        ]
        # Stage-specific augmentations can be inserted here in future runs.
        return Compose(base)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        data = self.transforms({"img": row["img_path"]})
        return {
            "img": data["img"].float(),
            "subject": row["subject"],
            "img_path": row["img_path"],
        }


def get_task1b_dataloaders(
    data_root: str,
    split_pkl: str,
    batch_size: int = 2,
    num_workers: int = 4,
    image_suffix: str = "_ciso.nii.gz",
    spatial_size: tuple = (96, 96, 96),
):
    """Return (train_loader, val_loader, n_train, n_val) for Task 1b (RUN_0004)."""
    train_ds = Task1bDataset(
        data_root=data_root,
        split_pkl=split_pkl,
        fold="train",
        stage="train",
        image_suffix=image_suffix,
        spatial_size=spatial_size,
    )
    val_ds = Task1bDataset(
        data_root=data_root,
        split_pkl=split_pkl,
        fold="val",
        stage="val",
        image_suffix=image_suffix,
        spatial_size=spatial_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


# ─── RUN_0004 – Multi-task datasets ──────────────────────────────────────────────


def build_clean_subject_set(csv_path: str) -> set:
    """Return subject IDs where ALL 7 artifact scores equal 0.

    Parses the Task 1a CSV and extracts the LISA_XXXX identifier from
    the ``filename`` column using the same regex as Task 2.

    Returns an empty set if the CSV cannot be read or has no clean rows.
    """
    try:
        df = pd.read_csv(csv_path)
        artifact_cols = [c for c in TASK_NAMES if c in df.columns]
        if not artifact_cols:
            return set()
        clean_mask = (df[artifact_cols] == 0).all(axis=1)
        subjects: set = set()
        for fn in df.loc[clean_mask, "filename"]:
            m = re.match(r"(LISA_\d+)", str(fn))
            if m:
                subjects.add(m.group(1))
        return subjects
    except Exception:
        return set()


class Task1aMultiTask128Dataset(Dataset):
    """Task 1a multi-label dataset resized to 128³ for the shared encoder (RUN_0004).

    Identical to Task1aMultiLabelDataset but with configurable spatial_size
    (default 128³ instead of 150³).  Returns ``labels`` as [7] int64 tensor
    plus ``img`` as [1, 128, 128, 128] float32 tensor.
    """

    def __init__(
        self,
        csv_path: str,
        bids_root: str,
        split_pkl: str,
        fold: str,
        stage: str = "train",
        spatial_size: tuple = (128, 128, 128),
    ):
        self.bids_root = bids_root
        self.fold = fold
        self.stage = stage
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
        self.transforms = self._build_transforms(stage)

    def _build_transforms(self, stage: str):
        base = [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            CenterSpatialCropd(keys=["img"], roi_size=self.spatial_size),
            SpatialPadd(keys=["img"], spatial_size=self.spatial_size, mode="symmetric"),
            EnsureTyped(keys=["img"]),
        ]
        if stage == "train":
            base.extend(
                [
                    RandRotated(
                        keys=["img"],
                        prob=0.2,
                        range_x=np.deg2rad(15),
                        range_y=np.deg2rad(15),
                        range_z=np.deg2rad(10),
                        mode="bilinear",
                    ),
                    RandAffined(
                        keys=["img"],
                        prob=0.2,
                        scale_range=(0.05, 0.05, 0.05),
                        translate_range=(3, 3, 2),
                        mode="bilinear",
                    ),
                    RandShiftIntensityd(keys=["img"], prob=0.2, offsets=0.1),
                    RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.8, 1.2)),
                ]
            )
        return Compose(base)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.bids_root, row["filename"])
        labels = torch.tensor([int(row[t]) for t in TASK_NAMES], dtype=torch.long)
        data = self.transforms({"img": img_path})
        return {
            "img": data["img"].float(),
            "labels": labels,
            "filename": row["filename"],
        }


class CleanImageDataset(Dataset):
    """Volumes with all artifact scores = 0, at 128³ (Task 1b autoencoder, RUN_0004).

    Scans *data_root* for ``*_ciso.nii.gz`` files and keeps only subjects
    whose LISA_XXXX ID appears in *clean_subjects*.  If *clean_subjects* is
    empty or None, ALL subjects are used (fallback for datasets without labels).

    Each sample returns:
        ``img``     : [1, 128, 128, 128] float32 – normalised clean volume
        ``subject`` : str
    The reconstruction target is the same as the input (identity mapping).
    """

    def __init__(
        self,
        data_root: str,
        split_pkl: str,
        fold: str,
        stage: str = "train",
        image_suffix: str = "_ciso.nii.gz",
        spatial_size: tuple = (128, 128, 128),
        clean_subjects: set = None,
    ):
        self.spatial_size = tuple(spatial_size)
        self.stage = stage

        records = build_task1b_records(data_root, image_suffix=image_suffix)
        if not records:
            raise RuntimeError(f"No '{image_suffix}' volumes found in {data_root}.")

        records_df = pd.DataFrame(records)
        all_subjects = sorted(records_df["subject"].unique().tolist())

        # Apply task split
        if os.path.exists(split_pkl):
            with open(split_pkl, "rb") as fh:
                split = pickle.load(fh)
            fold_subjects = _subjects_from_split(split, all_subjects, fold)
        else:
            n_train = max(1, int(0.8 * len(all_subjects)))
            fold_subjects = (
                set(all_subjects[:n_train])
                if fold == "train"
                else set(all_subjects[n_train:])
            )

        # Filter to clean subjects (if provided)
        if clean_subjects:
            fold_subjects = fold_subjects & clean_subjects
            if not fold_subjects:
                import warnings

                warnings.warn(
                    f"CleanImageDataset ({fold}): no clean subjects found after "
                    "intersecting split with clean_subjects. Using all split subjects "
                    "as fallback (autoencoder will not be restricted to clean images)."
                )
                fold_subjects = _subjects_from_split(
                    split if os.path.exists(split_pkl) else {}, all_subjects, fold
                )

        self.records = records_df[records_df["subject"].isin(fold_subjects)].to_dict(
            "records"
        )

        if not self.records:
            raise RuntimeError(
                f"CleanImageDataset ({fold}): no volumes remain after filtering."
            )

        self.transforms = self._build_transforms(stage)

    def _build_transforms(self, stage: str):
        return Compose(
            [
                LoadImaged(keys=["img"], reader="nibabelreader"),
                EnsureChannelFirstd(keys=["img"]),
                NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
                CenterSpatialCropd(keys=["img"], roi_size=self.spatial_size),
                SpatialPadd(
                    keys=["img"], spatial_size=self.spatial_size, mode="symmetric"
                ),
                EnsureTyped(keys=["img"]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        data = self.transforms({"img": row["img_path"]})
        return {
            "img": data["img"].float(),
            "subject": row["subject"],
        }


def get_multitask_dataloaders(config: dict) -> dict:
    """Return dataloaders for all three tasks used in RUN_0004.

    Returns a dict::

        {
          "1a": (train_loader, val_loader, n_train, n_val),
          "1b": (train_loader, val_loader, n_train, n_val),
          "2":  (train_loader, val_loader, n_train, n_val),
        }

    Config keys expected under ``data``:
        data_root, csv_path (Task 1a + 1b), bids_root (Task 1a),
        split_pkl_1a, split_pkl_2, image_suffix, label_suffix,
        patch_size, train_num_samples.
    Config keys expected under ``training``:
        batch_size, num_workers.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 2))
    spatial_size = tuple(int(v) for v in data_cfg.get("patch_size", [128, 128, 128]))

    data_root = os.getenv("LISA_DATA_ROOT", data_cfg["data_root"])
    bids_root = os.getenv(
        "LISA_DATA_ROOT", data_cfg.get("bids_root", data_cfg["data_root"])
    )
    csv_path = os.getenv(
        "LISA_CSV_PATH",
        data_cfg.get("csv_path", os.path.join(data_root, "LISA_Task1a_2026.csv")),
    )
    split_pkl_1a = data_cfg.get("split_pkl_1a", "results/splits/task1a_fixed.pkl")
    split_pkl_2 = data_cfg.get("split_pkl_2", "results/splits/task2_fixed.pkl")
    image_suffix = data_cfg.get("image_suffix", "_ciso.nii.gz")
    label_suffix = data_cfg.get("label_suffix", "_LF_seg.nii.gz")
    num_samples = int(data_cfg.get("train_num_samples", 1))

    # ── Task 1a ─────────────────────────────────────────────────────────────────
    train_ds_1a = Task1aMultiTask128Dataset(
        csv_path=csv_path,
        bids_root=bids_root,
        split_pkl=split_pkl_1a,
        fold="train",
        stage="train",
        spatial_size=spatial_size,
    )
    val_ds_1a = Task1aMultiTask128Dataset(
        csv_path=csv_path,
        bids_root=bids_root,
        split_pkl=split_pkl_1a,
        fold="val",
        stage="val",
        spatial_size=spatial_size,
    )

    # ── Task 1b (clean images) ──────────────────────────────────────────────────
    clean_subjects = build_clean_subject_set(csv_path)
    train_ds_1b = CleanImageDataset(
        data_root=data_root,
        split_pkl=split_pkl_2,
        fold="train",
        stage="train",
        image_suffix=image_suffix,
        spatial_size=spatial_size,
        clean_subjects=clean_subjects if clean_subjects else None,
    )
    val_ds_1b = CleanImageDataset(
        data_root=data_root,
        split_pkl=split_pkl_2,
        fold="val",
        stage="val",
        image_suffix=image_suffix,
        spatial_size=spatial_size,
        clean_subjects=clean_subjects if clean_subjects else None,
    )

    # ── Task 2 ───────────────────────────────────────────────────────────────────
    train_ds_2 = Task2SegmentationDataset(
        data_root=data_root,
        split_pkl=split_pkl_2,
        fold="train",
        stage="train",
        image_suffix=image_suffix,
        label_suffix=label_suffix,
        patch_size=spatial_size,
        num_samples_per_volume=num_samples,
        num_classes=12,
    )
    val_ds_2 = Task2SegmentationDataset(
        data_root=data_root,
        split_pkl=split_pkl_2,
        fold="val",
        stage="val",
        image_suffix=image_suffix,
        label_suffix=label_suffix,
        patch_size=spatial_size,
        num_samples_per_volume=1,
        num_classes=12,
    )

    def _make_loader(ds, shuffle, bs=batch_size):
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=True
        )

    return {
        "1a": (
            _make_loader(train_ds_1a, True),
            _make_loader(val_ds_1a, False),
            len(train_ds_1a),
            len(val_ds_1a),
        ),
        "1b": (
            _make_loader(train_ds_1b, True),
            _make_loader(val_ds_1b, False),
            len(train_ds_1b),
            len(val_ds_1b),
        ),
        "2": (
            _make_loader(train_ds_2, True),
            DataLoader(
                val_ds_2,
                batch_size=1,
                shuffle=False,
                num_workers=max(1, num_workers // 2),
                pin_memory=True,
            ),
            len(train_ds_2),
            len(val_ds_2),
        ),
    }
