"""Task 1b CycleGAN models — unpaired artifact removal (RUN_0002).

Architecture
------------
Generator3D
    Lightweight 3D encoder-decoder (U-Net style) with residual blocks at the
    bottleneck.  Based on the standard CycleGAN generator adapted to 3-D.
    Downsampling via strided convolutions; upsampling via ConvTranspose3d.

Discriminator3D
    PatchGAN discriminator operating on 3-D patches.  Each output element
    corresponds to a receptive field covering a sub-volume of the input.
    Uses instance normalisation and LeakyReLU as in the original PatchGAN.

Both models work on single-channel 3-D volumes of arbitrary patch size.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResBlock3D(nn.Module):
    """3-D residual block with instance normalisation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator3D(nn.Module):
    """3-D U-Net-style CycleGAN generator.

    Encoder: 3 strided-conv downsampling stages.
    Bottleneck: ``n_res_blocks`` residual blocks.
    Decoder: 3 transposed-conv upsampling stages with skip connections.

    Args:
        in_channels:   Input/output channels (default 1 for MRI).
        base_filters:  Number of feature maps after the first conv (default 32).
        n_res_blocks:  Number of residual blocks in the bottleneck (default 6).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        n_res_blocks: int = 6,
    ) -> None:
        super().__init__()
        f = base_filters

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc0 = nn.Sequential(
            nn.ReplicationPad3d(3),
            nn.Conv3d(in_channels, f, kernel_size=7, bias=False),
            nn.InstanceNorm3d(f),
            nn.ReLU(inplace=True),
        )  # -> [B, f, H, W, D]

        self.enc1 = nn.Sequential(
            nn.Conv3d(f, f * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(f * 2),
            nn.ReLU(inplace=True),
        )  # -> [B, 2f, H/2, ...]

        self.enc2 = nn.Sequential(
            nn.Conv3d(f * 2, f * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(f * 4),
            nn.ReLU(inplace=True),
        )  # -> [B, 4f, H/4, ...]

        self.enc3 = nn.Sequential(
            nn.Conv3d(f * 4, f * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(f * 8),
            nn.ReLU(inplace=True),
        )  # -> [B, 8f, H/8, ...]

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResBlock3D(f * 8) for _ in range(n_res_blocks)]
        )

        # ── Decoder (with skip connections from encoder) ───────────────────
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(f * 8 + f * 8, f * 4, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm3d(f * 4),
            nn.ReLU(inplace=True),
        )  # skip from enc3

        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(f * 4 + f * 4, f * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm3d(f * 2),
            nn.ReLU(inplace=True),
        )  # skip from enc2

        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(f * 2 + f * 2, f, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm3d(f),
            nn.ReLU(inplace=True),
        )  # skip from enc1

        self.out_conv = nn.Sequential(
            nn.ReplicationPad3d(3),
            nn.Conv3d(f + f, in_channels, kernel_size=7),
            nn.Tanh(),
        )  # skip from enc0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.res_blocks(e3)
        d3 = self.dec3(torch.cat([b, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.out_conv(torch.cat([d1, e0], dim=1))


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class Discriminator3D(nn.Module):
    """3-D PatchGAN discriminator.

    Produces a feature map where each element corresponds to a sub-volume
    patch.  Uses LSGAN (no sigmoid — raw outputs, MSE loss in the trainer).

    Args:
        in_channels:  Input channels (default 1).
        base_filters: Feature maps in the first conv layer (default 64).
        n_layers:     Number of downsampling conv layers (default 4).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        f_in, f_out = base_filters, base_filters
        for i in range(1, n_layers):
            f_in = f_out
            f_out = min(base_filters * (2 ** i), 512)
            stride = 1 if i == n_layers - 1 else 2
            layers += [
                nn.Conv3d(f_in, f_out, kernel_size=4, stride=stride, padding=1,
                          bias=False),
                nn.InstanceNorm3d(f_out),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        # Final projection to 1 channel (no activation — LSGAN)
        layers.append(
            nn.Conv3d(f_out, 1, kernel_size=4, stride=1, padding=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
