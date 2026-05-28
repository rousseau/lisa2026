"""Shared building blocks for 3D U-Net style architectures."""

import torch
import torch.nn as nn


class DoubleConv3d(nn.Module):
    """Two consecutive Conv3d-InstanceNorm-LeakyReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3d(nn.Module):
    """Transposed-conv upsample + skip-connection concat + DoubleConv3d.

    Used in the Task 1b reconstruction decoder of ``DynUNetMultiHeadModel``.

    Architecture per block::

        x  (in_ch channels, spatial H)
        ↓  ConvTranspose3d(in_ch → in_ch, kernel=2, stride=2)  →  2H
        concat with skip (skip_ch channels, spatial 2H)
        ↓  DoubleConv3d(in_ch + skip_ch → out_ch)

    Args:
        in_ch:   Number of input channels (from the lower decoder level).
        skip_ch: Number of skip-connection channels (from the encoder).
        out_ch:  Number of output channels after the double-conv.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_ch, in_ch, kernel_size=2, stride=2, bias=False
        )
        self.conv = DoubleConv3d(in_ch + skip_ch, out_ch)

        # Small-weight initialisation to prevent fp16 overflow on fresh heads
        # when plugged into a pre-trained encoder (AMP).
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.upsample.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample *x*, concatenate with *skip*, then apply double-conv.

        Args:
            x:    Lower-resolution feature map  [B, in_ch, H, W, D].
            skip: Encoder skip feature map      [B, skip_ch, 2H, 2W, 2D].

        Returns:
            Feature map of shape [B, out_ch, 2H, 2W, 2D].
        """
        x = self.upsample(x)

        # Pad x to match skip size if rounding discrepancy (odd spatial dims).
        if x.shape != skip.shape:
            diff = [s - x_s for s, x_s in zip(skip.shape[2:], x.shape[2:])]
            x = nn.functional.pad(
                x, [0, diff[2], 0, diff[1], 0, diff[0]]
            )

        return self.conv(torch.cat([x, skip], dim=1))
