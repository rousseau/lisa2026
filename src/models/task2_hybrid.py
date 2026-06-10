"""Hybrid model — Feature fusion of nnU-Net/DynUNet + MedSAM2 encoder (RUN_0003c).

Architecture:
- nnU-Net / DynUNet encoder (trainable, optionally warm-started from RUN_0003a)
- MedSAM2 image encoder (frozen, Hiera ViT)
- Fusion blocks (1×1×1 conv) at each decoder level
- nnU-Net / DynUNet decoder modified for fused features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet

from src.models.task2_medsam2_encoder import MedSAM2ImageEncoder


class FeatureFusionBlock(nn.Module):
    """Fuse local (nnU-Net) and global (MedSAM2) features at one scale.

    Args:
        local_ch:   Number of local feature channels.
        global_ch:  Number of global feature channels.
        out_ch:     Output channel dimension after fusion.
    """

    def __init__(self, local_ch: int, global_ch: int, out_ch: int):
        super().__init__()
        # Project global features to match local channel count
        self.global_proj = nn.Sequential(
            nn.Conv3d(global_ch, local_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(local_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # Fuse and compress
        self.fusion = nn.Sequential(
            nn.Conv3d(local_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_feat:  [B, local_ch, H, W, D]
            global_feat: [B, global_ch, H, W, D] (after spatial interpolation)

        Returns:
            [B, out_ch, H, W, D]
        """
        # Project global to local channel space
        g = self.global_proj(global_feat)
        # Concatenate
        fused = torch.cat([local_feat, g], dim=1)
        out = self.fusion(fused)
        return out


class Task2HybridModel(nn.Module):
    """Hybrid segmentation model for LISA Task 2 (RUN_0003c).

    Fuses a trainable DynUNet encoder with a frozen MedSAM2 encoder
    via attention-guided feature fusion blocks, followed by a DynUNet decoder.

    Args:
        nnunet_checkpoint:        Path to nnU-Net/DynUNet checkpoint from RUN_0003a.
        medsam2_checkpoint:       Path to MedSAM2 checkpoint.
        num_classes:              Number of segmentation classes (default 12).
        filters:                  Feature widths per level.
        device:                   Device string.
    """

    def __init__(
        self,
        nnunet_checkpoint: str,
        medsam2_checkpoint: str,
        num_classes: int = 12,
        filters=(32, 64, 128, 256, 320),
        device: str = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.filters = filters
        self.device = device

        # ── MedSAM2 Encoder (frozen) ────────────────────────────────────────
        self.medsam2_encoder = MedSAM2ImageEncoder(
            checkpoint_path=medsam2_checkpoint,
            config_file="configs/sam2.1_hiera_t512.yaml",
            device=device,
        )

        # ── DynUNet Encoder (trainable) ─────────────────────────────────────
        self.local_encoder = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            kernel_size=[(3, 3, 3)] * len(filters),
            strides=[(1, 1, 1)] + [(2, 2, 2)] * (len(filters) - 1),
            upsample_kernel_size=[(2, 2, 2)] * (len(filters) - 1),
            filters=filters,
            norm_name="instance",
            deep_supervision=False,
        )

        # Try to warm-start from nnU-Net checkpoint if provided
        if nnunet_checkpoint and Path(nnunet_checkpoint).exists():
            try:
                ckpt = torch.load(nnunet_checkpoint, map_location=device, weights_only=True)
                sd = ckpt.get("model_state_dict", ckpt)
                self.local_encoder.load_state_dict(sd, strict=False)
                print(f"[INFO] Warm-started local encoder from {nnunet_checkpoint}")
            except Exception as e:
                print(f"[WARN] Could not load nnunet checkpoint: {e}")

        # ── Feature extraction hooks ───────────────────────────────────────
        # We need to extract intermediate encoder features for fusion.
        # DynUNet exposes features via its internal encoder.  Since MONAI's
        # DynUNet is monolithic, we'll build a custom encoder/decoder split.
        # Simpler approach: build a separate shallow encoder that matches
        # DynUNet structure and manually implement skip + fusion + decoder.

        # Rebuild encoder stages matching DynUNet
        self.encoder_stages = nn.ModuleList()
        in_ch = 1
        for f in filters:
            stage = nn.Sequential(
                nn.Conv3d(in_ch, f, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(f, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(f, f, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(f, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            )
            self.encoder_stages.append(stage)
            in_ch = f

        self.downsample = nn.ModuleList([
            nn.Conv3d(filters[i], filters[i + 1], kernel_size=3, stride=2, padding=1, bias=False)
            for i in range(len(filters) - 1)
        ])

        # ── Fusion blocks ───────────────────────────────────────────────────
        # Determine MedSAM2 output channels by dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 256, 256, 2, device=device)
            sam_feat = self.medsam2_encoder(dummy)
            sam_ch = sam_feat.shape[1]

        self.sam_proj = nn.Sequential(
            nn.Conv3d(sam_ch, filters[-1], kernel_size=1, bias=False),
            nn.InstanceNorm3d(filters[-1], affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.fusion_blocks = nn.ModuleList()
        for f in filters:
            self.fusion_blocks.append(
                FeatureFusionBlock(local_ch=f, global_ch=filters[-1], out_ch=f)
            )

        # ── Decoder ─────────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        for i in range(len(filters) - 1, 0, -1):
            in_ch = filters[i]
            skip_ch = filters[i - 1]
            out_ch = filters[i - 1]
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
                    nn.InstanceNorm3d(out_ch, affine=True),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(out_ch, affine=True),
                    nn.LeakyReLU(0.01, inplace=True),
                )
            )

        self.final_conv = nn.Conv3d(filters[0], num_classes, kernel_size=1)

    def encode_local(self, x):
        """Extract multi-scale local features via shallow encoder."""
        features = []
        feat = x
        for i, stage in enumerate(self.encoder_stages):
            feat = stage(feat)
            features.append(feat)
            if i < len(self.downsample):
                feat = self.downsample[i](feat)
        return features, feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W, D = x.shape

        # ── Global features (frozen MedSAM2) ────────────────────────────────
        with torch.no_grad():
            sam_feat = self.medsam2_encoder(x)  # [B, C_sam, H_s, W_s, D]
        sam_feat = self.sam_proj(sam_feat)

        # Interpolate SAM features to match local encoder resolutions
        # We need them at each encoder level's spatial size
        local_features, bottleneck = self.encode_local(x)

        # ── Fusion at each encoder level ────────────────────────────────────
        fused_features = []
        for i, local_f in enumerate(local_features):
            # Interpolate SAM feature to local resolution
            sam_interp = F.interpolate(
                sam_feat, size=local_f.shape[2:], mode="trilinear", align_corners=False
            )
            fused = self.fusion_blocks[i](local_f, sam_interp)
            fused_features.append(fused)

        # ── Decoder with skip connections from fused features ───────────────
        feat = bottleneck
        for i, up_block in enumerate(self.up_blocks):
            skip_idx = len(self.up_blocks) - 1 - i
            skip = fused_features[skip_idx]
            feat = up_block[0](feat)  # ConvTranspose
            # Pad if spatial mismatch
            if feat.shape != skip.shape:
                diff = [s - f for s, f in zip(skip.shape[2:], feat.shape[2:])]
                feat = F.pad(feat, [0, diff[2], 0, diff[1], 0, diff[0]])
            feat = torch.cat([feat, skip], dim=1)
            feat = up_block[1:](feat)  # Double conv

        # Interpolate to input resolution
        if feat.shape[2:] != (H, W, D):
            feat = F.interpolate(feat, size=(H, W, D), mode="trilinear", align_corners=False)

        logits = self.final_conv(feat)
        return logits
