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


class Task1aMultiLabelModel(nn.Module):
    """DenseNet264 shared backbone with 7 independent 3-class heads (RUN_0002+).

    The backbone is reused from DenseNet264 by setting out_channels = num_tasks *
    num_classes, then the final logits are reshaped to [B, num_tasks, num_classes].
    This is equivalent to a single linear layer serving as 7 joint heads while
    sharing all convolutional features.

    Args:
        num_tasks:   Number of artifact types (default 7).
        num_classes: Number of severity levels per artifact (default 3).
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
        return out.view(-1, self.num_tasks, self.num_classes)
