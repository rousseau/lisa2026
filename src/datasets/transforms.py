"""Shared MONAI transform pipelines for all LISA 2026 tasks.

All tasks use the same normalisation (zero-mean / unit-std, channel-wise,
including zero voxels) and the same centre-crop + symmetric-pad strategy.
Task-specific augmentations are added on top of this common base.
"""

from typing import Sequence, Tuple

import numpy as np
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
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    ToTensord,
)


from .nnunet_preprocessor import LoadAndPreprocessNnunetd


def build_image_only_transforms(
    spatial_size: Tuple[int, int, int],
    stage: str,
    use_to_tensor: bool = True,
    use_nnunet_preprocessing: bool = False,
) -> Compose:
    """Base pipeline for single-channel image volumes (Tasks 1a, 1b).

    Args:
        spatial_size:   Target spatial dimensions (H, W, D).
        stage:          ``"train"`` or ``"val"``.
        use_to_tensor:  Append ``ToTensord`` at the end (default True).
        use_nnunet_preprocessing:  If True, use exact nnU-Net DefaultPreprocessor
            instead of LoadImaged+NormalizeIntensityd.

    Returns:
        MONAI ``Compose`` transform pipeline.
    """
    keys = ["img"]
    if use_nnunet_preprocessing:
        base = [
            LoadAndPreprocessNnunetd(keys=["img", "label"]),
            EnsureTyped(keys=keys),
        ]
    else:
        base = [
            LoadImaged(keys=keys, reader="nibabelreader"),
            EnsureChannelFirstd(keys=keys),
            NormalizeIntensityd(keys=keys, nonzero=False, channel_wise=True),
            EnsureTyped(keys=keys),
        ]

    base.extend([
        CenterSpatialCropd(keys=keys, roi_size=spatial_size),
        SpatialPadd(keys=keys, spatial_size=spatial_size, mode="symmetric"),
    ])

    if stage == "train":
        base.extend(
            [
                RandRotated(
                    keys=keys,
                    prob=0.2,
                    range_x=np.deg2rad(15),
                    range_y=np.deg2rad(15),
                    range_z=np.deg2rad(10),
                    mode="bilinear",
                ),
                RandAffined(
                    keys=keys,
                    prob=0.2,
                    scale_range=(0.05, 0.05, 0.05),
                    translate_range=(3, 3, 2),
                    mode="bilinear",
                ),
                RandShiftIntensityd(keys=keys, prob=0.2, offsets=0.1),
                RandAdjustContrastd(keys=keys, prob=0.2, gamma=(0.8, 1.2)),
            ]
        )

    if use_to_tensor:
        base.append(ToTensord(keys=keys))

    return Compose(base)


def build_segmentation_transforms(
    spatial_size: Tuple[int, int, int],
    stage: str,
    num_classes: int,
    num_samples: int = 1,
    use_nnunet_preprocessing: bool = False,
) -> Compose:
    """Transform pipeline for image + label segmentation (Task 2).

    Args:
        spatial_size:  Target patch size (H, W, D).
        stage:         ``"train"`` or ``"val"``.
        num_classes:   Number of label classes (used for balanced crop ratios).
        num_samples:   Number of random crops per volume (train only, default 1).
        use_nnunet_preprocessing: If True, use exact nnU-Net DefaultPreprocessor.

    Returns:
        MONAI ``Compose`` transform pipeline.
    """
    keys = ["img", "label"]
    if use_nnunet_preprocessing:
        base = [
            LoadAndPreprocessNnunetd(keys=keys),
            EnsureTyped(keys=keys),
        ]
    else:
        base = [
            LoadImaged(keys=keys, reader="nibabelreader"),
            EnsureChannelFirstd(keys=keys),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            EnsureTyped(keys=keys),
        ]

    if stage == "train":
        base.extend(
            [
                SpatialPadd(
                    keys=keys,
                    spatial_size=spatial_size,
                    mode=("constant", "constant"),
                ),
                RandCropByLabelClassesd(
                    keys=keys,
                    label_key="label",
                    spatial_size=spatial_size,
                    num_classes=num_classes,
                    ratios=[0.25] + [1.0] * (num_classes - 1),
                    num_samples=num_samples,
                ),
                RandAffined(
                    keys=keys,
                    prob=0.25,
                    rotate_range=(
                        np.deg2rad(20),
                        np.deg2rad(20),
                        np.deg2rad(20),
                    ),
                    scale_range=(0.15, 0.15, 0.15),
                    translate_range=(8, 8, 8),
                    mode=("bilinear", "nearest"),
                ),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
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
        base.append(
            SpatialPadd(
                keys=keys,
                spatial_size=spatial_size,
                mode=("constant", "constant"),
            )
        )

    return Compose(base)
