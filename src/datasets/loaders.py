"""DataLoader factory functions for all LISA 2026 tasks."""

import os
import re

import pandas as pd
from torch.utils.data import DataLoader

from .task1a import TASK_NAMES, Task1aDataset
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


# ─── Task 2 ──────────────────────────────────────────────────────────────────


def get_task2_seg_dataloaders(
    split_pkl: str,
    batch_size: int = 1,
    num_workers: int = 2,
    task2_dir: str = "",
    task2extra_dir: str = "",
    task1b_dir: str = "",
    image_suffix: str = "_ciso.nii.gz",
    label_suffix: str = "_seg.nii.gz",
    label_suffix_lf: str = "_LF_seg.nii.gz",
    patch_size: tuple = (128, 128, 128),
    num_samples_per_volume: int = 1,
    num_classes: int = 12,
    collapse_labels: bool = False,
):
    """Train/val dataloaders for Task 2 segmentation (RUN_0006 iteration).

    Returns:
        (train_loader, val_loader, n_train, n_val)
    """
    train_ds = Task2SegmentationDataset(
        split_pkl=split_pkl, fold="train", stage="train",
        task2_dir=task2_dir, task2extra_dir=task2extra_dir, task1b_dir=task1b_dir,
        image_suffix=image_suffix, label_suffix=label_suffix, label_suffix_lf=label_suffix_lf,
        patch_size=patch_size, num_samples_per_volume=num_samples_per_volume,
        num_classes=num_classes, collapse_labels=collapse_labels,
        use_nnunet_preprocessing=True,
    )
    val_ds = Task2SegmentationDataset(
        split_pkl=split_pkl, fold="val", stage="val",
        task2_dir=task2_dir, task2extra_dir=task2extra_dir, task1b_dir=task1b_dir,
        image_suffix=image_suffix, label_suffix=label_suffix, label_suffix_lf=label_suffix_lf,
        patch_size=patch_size, num_samples_per_volume=1,
        num_classes=num_classes, collapse_labels=collapse_labels,
        use_nnunet_preprocessing=True,
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
    # New layout: Task1a images live in Task1a/ subdir
    task1a_dir = data_cfg.get("task1a_dir", os.path.join(data_root, "Task1a"))
    if os.path.isdir(task1a_dir):
        bids_root = task1a_dir

    csv_path = os.getenv(
        "LISA_CSV_PATH",
        data_cfg.get("csv_path", os.path.join(data_root, "LISA_Task1a_2026.csv")),
    )
    split_pkl_1a = data_cfg.get("split_pkl_1a", "results/splits/task1a_fixed.pkl")
    split_pkl_2 = data_cfg.get("split_pkl_2", "results/splits/task2_fixed.pkl")
    image_suffix = data_cfg.get("image_suffix", "_ciso.nii.gz")
    label_suffix = data_cfg.get("label_suffix", "_seg.nii.gz")
    label_suffix_lf = data_cfg.get("label_suffix_lf", "_LF_seg.nii.gz")
    num_samples = int(data_cfg.get("train_num_samples", 1))

    # ── Task 2 directories (new layout) ─────────────────────────────────────
    task2_dir = data_cfg.get("task2_dir", os.path.join(data_root, "Task2"))
    task2extra_dir = data_cfg.get("task2extra_dir", os.path.join(data_root, "Task2Extra"))
    task1b_dir = data_cfg.get("task1b_dir", os.path.join(data_root, "Task1b"))
    if not os.path.isdir(task2_dir):
        task2_dir = data_root
    if not os.path.isdir(task2extra_dir):
        task2extra_dir = data_root
    if not os.path.isdir(task1b_dir):
        task1b_dir = data_root

    # Task 1b image search roots (ciso lives in Task2/ + Task1b/)
    task1b_image_roots = [task2_dir]
    if task1b_dir != task2_dir:
        task1b_image_roots.append(task1b_dir)
    # Fallback to data_root if neither exists (legacy flat layout)
    if not os.path.isdir(task2_dir) and not os.path.isdir(task1b_dir):
        task1b_image_roots = [data_root]
    task1b_data_root = task1b_image_roots[0]
    task1b_fallbacks = task1b_image_roots[1:] if len(task1b_image_roots) > 1 else []

    # ── Task 1a ──────────────────────────────────────────────────────────────
    train_ds_1a = Task1aDataset(
        csv_path=csv_path, bids_root=bids_root, split_pkl=split_pkl_1a,
        fold="train", stage="train", task_name=None, spatial_size=spatial_size,
        use_nnunet_preprocessing=True,
    )
    val_ds_1a = Task1aDataset(
        csv_path=csv_path, bids_root=bids_root, split_pkl=split_pkl_1a,
        fold="val", stage="val", task_name=None, spatial_size=spatial_size,
        use_nnunet_preprocessing=True,
    )

    # ── Task 1b (clean images for reconstruction head) ────────────────────────
    clean_subjects = _build_clean_subject_set(csv_path)
    from .task1b import Task1bCycleGANDataset
    train_ds_1b = Task1bCycleGANDataset(
        data_root=task1b_data_root, csv_path=csv_path, split_pkl=split_pkl_2,
        fold="train", stage="train",
        image_suffix=image_suffix, spatial_size=spatial_size,
        domain="both", use_nnunet_preprocessing=True,
        data_root_fallbacks=task1b_fallbacks,
    )
    val_ds_1b = Task1bCycleGANDataset(
        data_root=task1b_data_root, csv_path=csv_path, split_pkl=split_pkl_2,
        fold="val", stage="val",
        image_suffix=image_suffix, spatial_size=spatial_size,
        domain="both", use_nnunet_preprocessing=True,
        data_root_fallbacks=task1b_fallbacks,
    )

    # ── Task 2 ───────────────────────────────────────────────────────────────
    train_ds_2 = Task2SegmentationDataset(
        split_pkl=split_pkl_2, fold="train", stage="train",
        task2_dir=task2_dir, task2extra_dir=task2extra_dir, task1b_dir=task1b_dir,
        image_suffix=image_suffix, label_suffix=label_suffix, label_suffix_lf=label_suffix_lf,
        patch_size=spatial_size, num_samples_per_volume=num_samples, num_classes=12,
        use_nnunet_preprocessing=True,
    )
    val_ds_2 = Task2SegmentationDataset(
        split_pkl=split_pkl_2, fold="val", stage="val",
        task2_dir=task2_dir, task2extra_dir=task2extra_dir, task1b_dir=task1b_dir,
        image_suffix=image_suffix, label_suffix=label_suffix, label_suffix_lf=label_suffix_lf,
        patch_size=spatial_size, num_samples_per_volume=1, num_classes=12,
        use_nnunet_preprocessing=True,
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
