"""DataLoader factory functions for all LISA 2026 tasks."""

import os
import re

import pandas as pd
from torch.utils.data import DataLoader

from .task1a import TASK_NAMES, Task1aDataset
from .task1b import CleanImageDataset, Task1bDataset, build_task1b_records
from .task2 import Task2SegmentationDataset


# ─── Task 1a ─────────────────────────────────────────────────────────────────


def get_dataloaders(
    csv_path: str,
    bids_root: str,
    split_pkl: str,
    task_name: str,
    batch_size: int = 8,
    num_workers: int = 2,
    spatial_size: tuple = (150, 150, 150),
):
    """Train/val dataloaders for Task 1a per-task mode (RUN_0001).

    Args:
        csv_path:     Path to Task 1a CSV.
        bids_root:    BIDS data root.
        split_pkl:    Patient-level split pickle.
        task_name:    Single artifact name.
        batch_size:   Batch size.
        num_workers:  DataLoader workers.
        spatial_size: Crop/pad target size.

    Returns:
        (train_loader, val_loader, n_train, n_val)
    """
    train_ds = Task1aDataset(
        csv_path, bids_root, split_pkl, "train", "train",
        task_name=task_name, spatial_size=spatial_size,
    )
    val_ds = Task1aDataset(
        csv_path, bids_root, split_pkl, "val", "val",
        task_name=task_name, spatial_size=spatial_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def get_multilabel_dataloaders(
    csv_path: str,
    bids_root: str,
    split_pkl: str,
    batch_size: int = 8,
    num_workers: int = 2,
    spatial_size: tuple = (150, 150, 150),
):
    """Train/val dataloaders for Task 1a multi-label mode (RUN_0002).

    Returns:
        (train_loader, val_loader, n_train, n_val)
    """
    train_ds = Task1aDataset(
        csv_path, bids_root, split_pkl, "train", "train",
        task_name=None, spatial_size=spatial_size,
    )
    val_ds = Task1aDataset(
        csv_path, bids_root, split_pkl, "val", "val",
        task_name=None, spatial_size=spatial_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


# ─── Task 1b ─────────────────────────────────────────────────────────────────


def get_task1b_dataloaders(
    data_root: str,
    split_pkl: str,
    batch_size: int = 2,
    num_workers: int = 4,
    image_suffix: str = "_ciso.nii.gz",
    spatial_size: tuple = (96, 96, 96),
):
    """Train/val dataloaders for Task 1b standalone (RUN_0005).

    Returns:
        (train_loader, val_loader, n_train, n_val)
    """
    train_ds = Task1bDataset(
        data_root=data_root, split_pkl=split_pkl, fold="train", stage="train",
        image_suffix=image_suffix, spatial_size=spatial_size,
    )
    val_ds = Task1bDataset(
        data_root=data_root, split_pkl=split_pkl, fold="val", stage="val",
        image_suffix=image_suffix, spatial_size=spatial_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


# ─── Task 2 ──────────────────────────────────────────────────────────────────


def get_task2_seg_dataloaders(
    data_root: str,
    split_pkl: str,
    batch_size: int = 1,
    num_workers: int = 2,
    image_suffix: str = "_ciso.nii.gz",
    label_suffix: str = "_LF_seg.nii.gz",
    patch_size: tuple = (128, 128, 128),
    num_samples_per_volume: int = 1,
    num_classes: int = 12,
    collapse_labels: bool = False,
):
    """Train/val dataloaders for Task 2 segmentation (RUN_0003).

    Returns:
        (train_loader, val_loader, n_train, n_val)
    """
    train_ds = Task2SegmentationDataset(
        data_root=data_root, split_pkl=split_pkl, fold="train", stage="train",
        image_suffix=image_suffix, label_suffix=label_suffix,
        patch_size=patch_size, num_samples_per_volume=num_samples_per_volume,
        num_classes=num_classes, collapse_labels=collapse_labels,
    )
    val_ds = Task2SegmentationDataset(
        data_root=data_root, split_pkl=split_pkl, fold="val", stage="val",
        image_suffix=image_suffix, label_suffix=label_suffix,
        patch_size=patch_size, num_samples_per_volume=1,
        num_classes=num_classes, collapse_labels=collapse_labels,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


# ─── Multi-task (RUN_0004) ────────────────────────────────────────────────────


def _build_clean_subject_set(csv_path: str) -> set:
    """Return subject IDs where ALL 7 artifact scores equal 0."""
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


def get_multitask_dataloaders(config: dict) -> dict:
    """Return dataloaders for all three tasks used in RUN_0004.

    Returns::

        {
          "1a": (train_loader, val_loader, n_train, n_val),
          "1b": (train_loader, val_loader, n_train, n_val),
          "2":  (train_loader, val_loader, n_train, n_val),
        }
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

    # ── Task 1a ──────────────────────────────────────────────────────────────
    train_ds_1a = Task1aDataset(
        csv_path=csv_path, bids_root=bids_root, split_pkl=split_pkl_1a,
        fold="train", stage="train", task_name=None, spatial_size=spatial_size,
    )
    val_ds_1a = Task1aDataset(
        csv_path=csv_path, bids_root=bids_root, split_pkl=split_pkl_1a,
        fold="val", stage="val", task_name=None, spatial_size=spatial_size,
    )

    # ── Task 1b (clean images only) ───────────────────────────────────────────
    clean_subjects = _build_clean_subject_set(csv_path) or None
    train_ds_1b = CleanImageDataset(
        data_root=data_root, split_pkl=split_pkl_2, fold="train", stage="train",
        image_suffix=image_suffix, spatial_size=spatial_size,
        clean_subjects=clean_subjects,
    )
    val_ds_1b = CleanImageDataset(
        data_root=data_root, split_pkl=split_pkl_2, fold="val", stage="val",
        image_suffix=image_suffix, spatial_size=spatial_size,
        clean_subjects=clean_subjects,
    )

    # ── Task 2 ───────────────────────────────────────────────────────────────
    train_ds_2 = Task2SegmentationDataset(
        data_root=data_root, split_pkl=split_pkl_2, fold="train", stage="train",
        image_suffix=image_suffix, label_suffix=label_suffix,
        patch_size=spatial_size, num_samples_per_volume=num_samples, num_classes=12,
    )
    val_ds_2 = Task2SegmentationDataset(
        data_root=data_root, split_pkl=split_pkl_2, fold="val", stage="val",
        image_suffix=image_suffix, label_suffix=label_suffix,
        patch_size=spatial_size, num_samples_per_volume=1, num_classes=12,
    )

    def _make_loader(ds, shuffle, bs=batch_size):
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
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
                val_ds_2, batch_size=1, shuffle=False,
                num_workers=max(1, num_workers // 2), pin_memory=True,
            ),
            len(train_ds_2),
            len(val_ds_2),
        ),
    }
