"""Task 2 multi-structure segmentation model (DynUNet baseline, RUN_0003)."""

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet


class Task2DynUNetModel(nn.Module):
    """DynUNet baseline for Task 2 multi-structure segmentation (RUN_0003).

    Wraps MONAI ``DynUNet`` with the default filter sizes and kernel schedule
    used throughout the LISA 2026 experiments.

    Args:
        in_channels:          Input channels (default 1).
        out_channels:         Number of segmentation classes (default 12).
        kernel_size:          Per-level 3D kernel sizes.
        strides:              Per-level strides (first = 1, rest = 2).
        upsample_kernel_size: Per-level upsampling kernel sizes.
        filters:              Feature widths per level.
        norm_name:            Normalisation type (default "instance").
        deep_supervision:     Enable deep-supervision outputs (default False).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        upsample_kernel_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        filters=(32, 64, 128, 256, 320),
        norm_name: str = "instance",
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            norm_name=norm_name,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
