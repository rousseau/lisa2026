"""Task 1a models"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet264, DynUNet


class Task1aOrdinalModel(torch.nn.Module):
    """DenseNet264 for ordinal classification"""

    def __init__(self, num_classes=3, in_channels=1, spatial_dims=3):
        super().__init__()
        self.model = DenseNet264(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
        )

    def forward(self, x):
        return self.model(x)


# ─── RUN_0002 – Multi-head model ─────────────────────────────────────────────


class Task1aMultiLabelModel(nn.Module):
    """DenseNet264 shared backbone with 7 independent 3-class heads (RUN_0002+).

    The backbone is reused from DenseNet264 by setting out_channels = num_tasks *
    num_classes, then the final logits are reshaped to [B, num_tasks, num_classes].
    This is equivalent to a single linear layer serving as 7 joint heads while
    sharing all convolutional features.
    """

    def __init__(self, num_tasks: int = 7, num_classes: int = 3):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.backbone = DenseNet264(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_tasks * num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape [B, num_tasks, num_classes]."""
        out = self.backbone(x)  # [B, num_tasks * num_classes]
        return out.view(-1, self.num_tasks, self.num_classes)  # [B, 7, 3]


class Task2DynUNetModel(nn.Module):
    """DynUNet baseline for Task 2 multi-structure segmentation (RUN_0003)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        upsample_kernel_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        filters=(32, 64, 128, 256, 320),
        norm_name="instance",
        deep_supervision=False,
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


# ─── RUN_0004 – Task 1b Self-supervised Denoising ────────────────────────────

from monai.networks.nets import BasicUNet  # noqa: E402 (imported after DynUNet)


class Task1bUNetModel(nn.Module):
    """3D BasicUNet for self-supervised image denoising (Task 1b, RUN_0004).

    Trained to reconstruct clean MR volumes from synthetically degraded inputs.
    Architecture: MONAI BasicUNet with PRELU activations, instance normalisation,
    and transposed-convolution upsampling.  Input/output: single-channel 3D volumes.

    Note: MONAI BasicUNet expects exactly 6 feature values – the first 4 are
    encoder block widths, the 5th is the bottleneck width, the 6th is the output
    head width.  Default: (16, 32, 64, 128, 256, 16).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple = (16, 32, 64, 128, 256, 16),
    ):
        super().__init__()
        self.model = BasicUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            act=("PRELU", {}),
            norm=("INSTANCE", {}),
            dropout=0.0,
            upsample="deconv",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
