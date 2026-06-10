"""MedSAM2 segmentation model: frozen encoder + trainable DynUNet-style decoder.

RUN_0003b — MedSAM2 Image Encoder (Hiera ViT, frozen) +
            DynUNet Decoder (trainable, 12-class output).
"""

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

from src.models.task2_medsam2_encoder import MedSAM2ImageEncoder


class Task2MedSAM2Model(nn.Module):
    """MedSAM2-based segmentation model for LISA Task 2.

    The image encoder is a frozen MedSAM2 Hiera ViT that extracts semantic
    features from 3D volumes (processed as stacked 2D slices).  A trainable
    3D DynUNet decoder then upsamples these features to the full spatial
    resolution and predicts 12-class segmentation maps.

    Args:
        medsam2_checkpoint: Path to MedSAM2_latest.pt.
        medsam2_config: Hydra config file for SAM2 (relative to MedSAM2 repo).
        num_classes: Number of segmentation classes (default 12).
        decoder_filters: Feature widths for the DynUNet decoder.
        device: Device on which the encoder is instantiated.
    """

    def __init__(
        self,
        medsam2_checkpoint: str,
        medsam2_config: str = "configs/sam2.1_hiera_t512.yaml",
        num_classes: int = 12,
        decoder_filters=(32, 64, 128, 256, 320),
        device: str = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Frozen MedSAM2 encoder ──────────────────────────────────────────
        self.encoder = MedSAM2ImageEncoder(
            checkpoint_path=medsam2_checkpoint,
            config_file=medsam2_config,
            device=device,
        )
        self.encoder.eval()

        # Determine encoder output channels dynamically
        with torch.no_grad():
            dummy_in = torch.zeros(1, 1, 256, 256, 2, device=device)
            enc_out = self.encoder(dummy_in)
            sam_ch = enc_out.shape[1]  # e.g. 256
            enc_h = enc_out.shape[2]

        # Project SAM features to first decoder filter size
        self.sam_proj = nn.Sequential(
            nn.Conv3d(sam_ch, decoder_filters[-1], kernel_size=1, bias=False),
            nn.InstanceNorm3d(decoder_filters[-1], affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # ── DynUNet Decoder (trainable) ─────────────────────────────────────
        # We build a shallow 3D U-Net that takes the projected SAM features
        # as its bottleneck and upsamples to input resolution.
        # Since SAM encoder is 2D-slice-based, the spatial resolution in H/W
        # is heavily downsampled (e.g. 256 -> 64 for 512 input, or 256 -> 32
        # depending on input size).  We add upsampling blocks to recover.
        self.decoder = DynUNet(
            spatial_dims=3,
            in_channels=decoder_filters[-1],
            out_channels=num_classes,
            kernel_size=[(3, 3, 3)] * len(decoder_filters),
            strides=[(1, 1, 1)] + [(2, 2, 2)] * (len(decoder_filters) - 1),
            upsample_kernel_size=[(2, 2, 2)] * (len(decoder_filters) - 1),
            filters=decoder_filters,
            norm_name="instance",
            deep_supervision=False,
        )

        # Initial input projection to match DynUNet expected input
        # The DynUNet internally builds skip connections from an encoder,
        # but since we only provide bottleneck features we rely on the
        # decoder path alone.
        # We'll manually upsample the SAM features before feeding to decoder.
        num_upsample = len(decoder_filters) - 1
        self.upsample_blocks = nn.ModuleList()
        current_ch = decoder_filters[-1]
        for i in range(num_upsample):
            out_ch = decoder_filters[-(i + 2)]
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(current_ch, out_ch, kernel_size=2, stride=2, bias=False),
                    nn.InstanceNorm3d(out_ch, affine=True),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(out_ch, affine=True),
                    nn.LeakyReLU(0.01, inplace=True),
                )
            )
            current_ch = out_ch

        self.final_conv = nn.Conv3d(current_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input volume [B, 1, H, W, D].

        Returns:
            Logits [B, num_classes, H, W, D].
        """
        B, _, H, W, D = x.shape

        # ── Encoder (frozen) ────────────────────────────────────────────────
        with torch.no_grad():
            enc_feat = self.encoder(x)  # [B, C_sam, H_s, W_s, D]

        # Project to decoder channels
        feat = self.sam_proj(enc_feat)  # [B, filter_last, H_s, W_s, D]

        # ── Decoder upsampling ──────────────────────────────────────────────
        for up_block in self.upsample_blocks:
            feat = up_block(feat)

        # Interpolate to exact input size (in case of mismatch)
        if feat.shape[2:] != (H, W, D):
            feat = torch.nn.functional.interpolate(
                feat, size=(H, W, D), mode="trilinear", align_corners=False
            )

        logits = self.final_conv(feat)  # [B, num_classes, H, W, D]
        return logits
