"""Task 1b CycleGAN dataset — unpaired domains A and B (RUN_0002).

Domain definitions (from Task 1a CSV):
    Domain A (artefacted): volumes where Noise >= noise_threshold OR
                           Motion >= motion_threshold.
    Domain B (clean):      volumes where Noise == 0 AND Motion == 0.

The dataset returns independent samples from domain A and domain B.
Since the two domains may be of different sizes, domain A is sampled
with replacement during training to match the length of the combined set.

Data lives in the Task 2 BIDS root (same *_ciso.nii.gz volumes) but the
artifact labels come from the Task 1a CSV.  Subjects present in the CSV
are intersected with subjects found in data_root.

The split is applied at subject level using the Task 2 split pickle
(train_subjects / val_subjects keys), since the volumes are the same.
If the Task 1a split pickle format is used instead, keys
train_indices / val_indices are resolved via the CSV.
"""

import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandAdjustContrastd,
    RandFlipd,
    RandShiftIntensityd,
    ScaleIntensityd,
    SpatialPadd,
)
from torch.utils.data import Dataset

NOISE_COL = "Noise"
MOTION_COL = "Motion"
SUBJECT_PATTERN = re.compile(r"(LISA_\d+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_subject(filename: str) -> Optional[str]:
    m = SUBJECT_PATTERN.search(str(filename))
    return m.group(1) if m else None


def _load_split_subjects(split_pkl: str, fold: str) -> Optional[List[str]]:
    """Load subject IDs from a split pickle.

    Supports two formats:
        * ``{"train_subjects": [...], "val_subjects": [...]}``
        * ``{"train_indices": [...], "val_indices": [...]}`` — returns None,
          caller must resolve via CSV index.
    """
    with open(split_pkl, "rb") as fh:
        split = pickle.load(fh)
    key = "train_subjects" if fold == "train" else "val_subjects"
    return split.get(key)  # None if key absent


def _build_transforms(spatial_size: Tuple[int, ...], stage: str) -> Compose:
    keys = ["img"]
    base = [
        LoadImaged(keys=keys, reader="nibabelreader"),
        EnsureChannelFirstd(keys=keys),
        NormalizeIntensityd(keys=keys, nonzero=False, channel_wise=True),
        # Scale to [-1, 1] to match Generator Tanh output range
        ScaleIntensityd(keys=keys, minv=-1.0, maxv=1.0),
        CenterSpatialCropd(keys=keys, roi_size=spatial_size),
        SpatialPadd(keys=keys, spatial_size=spatial_size, mode="symmetric"),
        EnsureTyped(keys=keys),
    ]
    if stage == "train":
        base.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=keys, prob=0.2,
                scale_range=(0.05, 0.05, 0.05),
                translate_range=(3, 3, 2),
                mode="bilinear",
            ),
            RandShiftIntensityd(keys=keys, prob=0.2, offsets=0.1),
            RandAdjustContrastd(keys=keys, prob=0.2, gamma=(0.8, 1.2)),
        ])
    return Compose(base)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Task1bCycleGANDataset(Dataset):
    """Unpaired CycleGAN dataset for Task 1b artifact removal.

    Each ``__getitem__`` returns a dict::

        {
            "img_A": Tensor[1, H, W, D],   # artefacted volume (domain A)
            "img_B": Tensor[1, H, W, D],   # clean volume     (domain B)
            "path_A": str,
            "path_B": str,
        }

    Domain A and B are sampled independently and cyclically, so the dataset
    length equals ``max(len_A, len_B)``.

    Args:
        data_root:         Root directory containing ``*_ciso.nii.gz`` files.
        csv_path:          Path to Task 1a CSV (artifact severity labels).
        split_pkl:         Subject-level split pickle (Task 2 format preferred).
        fold:              ``"train"`` or ``"val"``.
        stage:             ``"train"`` (augmented) or ``"val"``.
        image_suffix:      NIfTI file suffix (default ``"_ciso.nii.gz"``).
        spatial_size:      Crop/pad target spatial size.
        noise_threshold:   Minimum Noise score for domain A (default 1).
        motion_threshold:  Minimum Motion score for domain A (default 1).
        domain:            ``"A"``, ``"B"``, or ``"both"`` (default).
                           Use ``"A"`` or ``"B"`` for single-domain evaluation.
    """

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        split_pkl: str,
        fold: str,
        stage: str = "train",
        image_suffix: str = "_ciso.nii.gz",
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
        noise_threshold: int = 1,
        motion_threshold: int = 1,
        domain: str = "both",
    ) -> None:
        self.data_root = data_root
        self.spatial_size = tuple(spatial_size)
        self.domain = domain
        self.transforms = _build_transforms(self.spatial_size, stage)

        # ── Load CSV ──────────────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        df["subject"] = df["filename"].apply(_extract_subject)
        df = df.dropna(subset=["subject"])

        # ── Resolve split subjects ────────────────────────────────────────
        split_subjects = _load_split_subjects(split_pkl, fold)
        if split_subjects is not None:
            df = df[df["subject"].isin(split_subjects)]
        else:
            # Fallback: index-based split (Task 1a pkl format)
            with open(split_pkl, "rb") as fh:
                split = pickle.load(fh)
            idx_key = "train_indices" if fold == "train" else "val_indices"
            indices = split.get(idx_key, [])
            df = df.iloc[indices]

        df = df.reset_index(drop=True)

        # ── Deduplicate: one ciso file per subject ────────────────────────
        df = df.drop_duplicates(subset=["subject"]).reset_index(drop=True)

        # ── Domain partition ─────────────────────────────────────────────
        mask_noise = df[NOISE_COL] >= noise_threshold if NOISE_COL in df.columns \
            else pd.Series(False, index=df.index)
        mask_motion = df[MOTION_COL] >= motion_threshold if MOTION_COL in df.columns \
            else pd.Series(False, index=df.index)

        df_A = df[mask_noise | mask_motion].reset_index(drop=True)
        df_B = df[(~mask_noise) & (~mask_motion)].reset_index(drop=True)

        # ── Build file paths ──────────────────────────────────────────────
        self.paths_A = self._resolve_paths(df_A, image_suffix)
        self.paths_B = self._resolve_paths(df_B, image_suffix)

        if len(self.paths_A) == 0:
            raise RuntimeError(
                f"Task1bCycleGANDataset: domain A is empty for fold='{fold}'. "
                f"Check noise_threshold/motion_threshold or CSV columns."
            )
        if len(self.paths_B) == 0:
            raise RuntimeError(
                f"Task1bCycleGANDataset: domain B is empty for fold='{fold}'. "
                f"Ensure some subjects have Noise=0 AND Motion=0."
            )

        self._len = max(len(self.paths_A), len(self.paths_B))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_paths(self, df: pd.DataFrame, suffix: str) -> List[str]:
        """Return existing NIfTI paths for subjects in df.

        Files are stored flat in data_root as ``{subject}{suffix}``,
        e.g. ``/data/LISA_0001_ciso.nii.gz``.
        """
        paths = []
        for subject in df["subject"]:
            p = os.path.join(self.data_root, f"{subject}{suffix}")
            if os.path.isfile(p):
                paths.append(p)
        return paths

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self.domain == "A":
            return len(self.paths_A)
        if self.domain == "B":
            return len(self.paths_B)
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, object]:
        idx_A = idx % len(self.paths_A)
        idx_B = idx % len(self.paths_B)
        path_A = self.paths_A[idx_A]
        path_B = self.paths_B[idx_B]

        sample_A = self.transforms({"img": path_A})
        sample_B = self.transforms({"img": path_B})

        return {
            "img_A": sample_A["img"],
            "img_B": sample_B["img"],
            "path_A": path_A,
            "path_B": path_B,
        }
