"""
Task 1a Dataset and DataLoader
"""
import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, CenterSpatialCropd, SpatialPadd, ToTensord, RandRotated, RandAffined, RandShiftIntensityd, RandAdjustContrastd, EnsureTyped, RandFlipd, RandGaussianNoised, RandScaleIntensityd, RandCropByPosNegLabeld


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
        with open(split_pkl, 'rb') as f:
            self.split = pickle.load(f)
        
        # Filter by fold
        if fold == 'train':
            indices = self.split.get('train_indices', [])
        else:
            indices = self.split.get('val_indices', [])
        
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
                RandRotated(keys=["img"], prob=0.2, range_x=np.deg2rad(15), range_y=np.deg2rad(15), range_z=np.deg2rad(10), mode="bilinear"),
                RandAffined(keys=["img"], prob=0.2, scale_range=(0.05, 0.05, 0.05), translate_range=(3, 3, 2), mode="bilinear"),
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
        img_path = os.path.join(self.bids_root, row['filename'])
        label = int(row[self.task_name])
        
        data = {"img": img_path, "label": label}
        data = self.transforms(data)
        
        return {
            "img": data["img"],
            "label": torch.tensor(label, dtype=torch.long),
            "filename": row['filename']
        }


def get_dataloaders(csv_path, bids_root, split_pkl, task_name, batch_size=8, num_workers=2):
    """Get train and val dataloaders"""
    train_dataset = Task1aDataset(csv_path, bids_root, split_pkl, 'train', task_name, 'train')
    val_dataset = Task1aDataset(csv_path, bids_root, split_pkl, 'val', task_name, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


# ─── RUN_0002 – Multi-label ──────────────────────────────────────────────────

TASK_NAMES = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']


class Task1aMultiLabelDataset(Dataset):
    """Task 1a – returns all 7 artifact labels in a single sample (RUN_0002+)."""

    def __init__(self, csv_path, bids_root, split_pkl, fold, stage="train"):
        self.bids_root = bids_root
        self.fold = fold
        self.stage = stage

        self.df = pd.read_csv(csv_path)

        with open(split_pkl, 'rb') as f:
            split = pickle.load(f)

        indices = split.get('train_indices', []) if fold == 'train' else split.get('val_indices', [])
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
            base_transforms.extend([
                RandRotated(keys=["img"], prob=0.2, range_x=np.deg2rad(15), range_y=np.deg2rad(15), range_z=np.deg2rad(10), mode="bilinear"),
                RandAffined(keys=["img"], prob=0.2, scale_range=(0.05, 0.05, 0.05), translate_range=(3, 3, 2), mode="bilinear"),
                RandShiftIntensityd(keys=["img"], prob=0.2, offsets=0.1),
                RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.8, 1.2)),
            ])

        base_transforms.append(ToTensord(keys=["img"]))
        return Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.bids_root, row['filename'])
        labels = torch.tensor([int(row[t]) for t in TASK_NAMES], dtype=torch.long)  # [7]

        data = self.transforms({"img": img_path})

        return {
            "img": data["img"],
            "labels": labels,
            "filename": row['filename'],
        }


def get_multilabel_dataloaders(csv_path, bids_root, split_pkl, batch_size=8, num_workers=2):
    """Get train and val dataloaders for multi-label Task 1a (RUN_0002+)."""
    train_ds = Task1aMultiLabelDataset(csv_path, bids_root, split_pkl, 'train', 'train')
    val_ds = Task1aMultiLabelDataset(csv_path, bids_root, split_pkl, 'val', 'val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(train_ds), len(val_ds)


# ─── RUN_0003 – Task 2 Multi-structure Segmentation ─────────────────────────

def build_task2_records(data_root, image_suffix="_ciso.nii.gz", label_suffix="_LF_seg.nii.gz"):
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
        num_samples_per_volume=2,
    ):
        self.stage = stage
        self.fold = fold
        self.patch_size = tuple(patch_size)
        self.num_samples_per_volume = int(num_samples_per_volume)

        records = build_task2_records(data_root, image_suffix=image_suffix, label_suffix=label_suffix)
        records_df = pd.DataFrame(records)

        with open(split_pkl, "rb") as f:
            split = pickle.load(f)

        if fold == "train":
            keep_subjects = set(split.get("train_subjects", []))
        else:
            keep_subjects = set(split.get("val_subjects", []))

        self.records = records_df[records_df["subject"].isin(keep_subjects)].to_dict("records")
        self.transforms = self._build_transforms(stage)

    def _build_transforms(self, stage):
        base = [
            LoadImaged(keys=["img", "label"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img", "label"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            EnsureTyped(keys=["img", "label"]),
        ]

        if stage == "train":
            base.extend([
                SpatialPadd(keys=["img", "label"], spatial_size=self.patch_size, mode=("constant", "constant")),
                RandCropByPosNegLabeld(
                    keys=["img", "label"],
                    label_key="label",
                    spatial_size=self.patch_size,
                    pos=1,
                    neg=1,
                    num_samples=self.num_samples_per_volume,
                    image_key="img",
                    image_threshold=0,
                ),
                RandFlipd(keys=["img", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["img", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["img", "label"], prob=0.5, spatial_axis=2),
                RandGaussianNoised(keys=["img"], prob=0.15, mean=0.0, std=0.01),
                RandScaleIntensityd(keys=["img"], factors=0.1, prob=0.2),
            ])
        else:
            base.extend([
                SpatialPadd(keys=["img", "label"], spatial_size=self.patch_size, mode=("constant", "constant")),
            ])

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
    num_samples_per_volume=2,
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
