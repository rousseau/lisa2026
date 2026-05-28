"""Task 1b denoising/reconstruction datasets."""

import os
import pickle
import re
import warnings
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import build_image_only_transforms


def build_task1b_records(data_root: str, image_suffix: str = "_ciso.nii.gz") -> list:
    """Scan *data_root* for volumes matching *image_suffix*.

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
    """Extract the subject set for *fold* from any supported split format.

    Supported formats:
      * ``{train_subjects, val_subjects}`` – Task2-style subject-level split.
      * ``{train_indices, val_indices}``   – Task1a-style CSV-index split.
    """
    if "train_subjects" in split:
        key = "train_subjects" if fold == "train" else "val_subjects"
        return set(split[key])

    if "train_indices" in split:
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
    """Task 1b dataset: self-supervised denoising (RUN_0005).

    Returns the preprocessed clean volume; synthetic degradation is applied
    externally by the trainer so noise parameters can vary epoch by epoch.

    Args:
        data_root:    Directory containing ``*_ciso.nii.gz`` volumes.
        split_pkl:    Path to patient-level split pickle.
        fold:         ``"train"`` or ``"val"``.
        stage:        ``"train"`` or ``"val"``.
        image_suffix: Filename suffix to match (default ``"_ciso.nii.gz"``).
        spatial_size: Target crop/pad size (default (96, 96, 96)).
    """

    def __init__(
        self,
        data_root: str,
        split_pkl: str,
        fold: str = "train",
        stage: str = "train",
        image_suffix: str = "_ciso.nii.gz",
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
    ):
        self.stage = stage
        self.spatial_size = tuple(spatial_size)

        records = build_task1b_records(data_root, image_suffix=image_suffix)
        if not records:
            raise RuntimeError(
                f"No '{image_suffix}' volumes found in {data_root}."
            )

        records_df = pd.DataFrame(records)
        all_subjects = sorted(records_df["subject"].unique().tolist())

        if os.path.exists(split_pkl):
            with open(split_pkl, "rb") as fh:
                split = pickle.load(fh)
            keep = _subjects_from_split(split, all_subjects, fold)
        else:
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
                f"Task1bDataset ({fold}): no volumes remain after applying split."
            )

        self.transforms = build_image_only_transforms(
            spatial_size=self.spatial_size, stage=stage, use_to_tensor=False
        )

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


class CleanImageDataset(Dataset):
    """Volumes with all artifact scores = 0 for Task 1b autoencoder (RUN_0004).

    Args:
        data_root:      Directory containing ``*_ciso.nii.gz`` volumes.
        split_pkl:      Path to patient-level split pickle.
        fold:           ``"train"`` or ``"val"``.
        stage:          ``"train"`` or ``"val"``.
        image_suffix:   Filename suffix (default ``"_ciso.nii.gz"``).
        spatial_size:   Target crop/pad size (default (128, 128, 128)).
        clean_subjects: Set of subject IDs with no artifacts.  If None or empty,
                        all subjects in the fold are used as fallback.
    """

    def __init__(
        self,
        data_root: str,
        split_pkl: str,
        fold: str,
        stage: str = "train",
        image_suffix: str = "_ciso.nii.gz",
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        clean_subjects: Optional[set] = None,
    ):
        self.spatial_size = tuple(spatial_size)

        records = build_task1b_records(data_root, image_suffix=image_suffix)
        if not records:
            raise RuntimeError(f"No '{image_suffix}' volumes found in {data_root}.")

        records_df = pd.DataFrame(records)
        all_subjects = sorted(records_df["subject"].unique().tolist())

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

        if clean_subjects:
            filtered = fold_subjects & clean_subjects
            if not filtered:
                warnings.warn(
                    f"CleanImageDataset ({fold}): no clean subjects found after "
                    "intersection. Using all fold subjects as fallback."
                )
            else:
                fold_subjects = filtered

        self.records = records_df[records_df["subject"].isin(fold_subjects)].to_dict(
            "records"
        )

        if not self.records:
            raise RuntimeError(
                f"CleanImageDataset ({fold}): no volumes remain after filtering."
            )

        self.transforms = build_image_only_transforms(
            spatial_size=self.spatial_size, stage=stage, use_to_tensor=False
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
