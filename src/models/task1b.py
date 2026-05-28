"""Task 1b image enhancement / denoising model (standalone baseline, RUN_0005)."""

import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet


class Task1bUNetModel(nn.Module):
    """3D BasicUNet for self-supervised image denoising (Task 1b, RUN_0005).

    Trained to reconstruct clean MR volumes from synthetically degraded inputs.
    Architecture: MONAI BasicUNet with PRELU activations, instance normalisation,
    and transposed-convolution upsampling.  Input/output: single-channel 3D volumes.

    Note: MONAI BasicUNet expects exactly 6 feature values – the first 4 are
    encoder block widths, the 5th is the bottleneck width, the 6th is the output
    head width.  Default: (16, 32, 64, 128, 256, 16).

    Args:
        in_channels:  Number of input channels (default 1).
        out_channels: Number of output channels (default 1).
        features:     6-tuple of feature widths for BasicUNet.
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
