"""Task 1a quality-assessment models."""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet264



class Task1aOrdinalModel(nn.Module):
    """DenseNet264 for single-artifact ordinal classification (RUN_0001).

    Predicts one of three severity levels (0 = none, 1 = mild, 2 = severe)
    for a single artifact type per forward pass.

    Args:
        num_classes:   Number of ordinal classes (default 3).
        in_channels:   Number of input image channels (default 1).
        spatial_dims:  Spatial dimensionality (default 3).
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.model = DenseNet264(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



