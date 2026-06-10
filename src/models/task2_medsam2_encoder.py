"""MedSAM2 image encoder wrapper for 3D volumetric features (RUN_0003b).

Extracts the image encoder from MedSAM2 (Hiera ViT + FPN neck),
processes 3D volumes slice-by-slice (2.5D), and returns multi-scale
feature maps suitable for fusion with a 3D decoder.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add MedSAM2 to import path
_MEDSAM2_ROOT = Path(__file__).resolve().parent.parent.parent / "external" / "MedSAM2"
if str(_MEDSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEDSAM2_ROOT))

from sam2.build_sam import build_sam2


class MedSAM2ImageEncoder(nn.Module):
    """Frozen MedSAM2 image encoder producing 3D feature volumes.

    Processes a 3D medical image [B, 1, H, W, D] by treating depth as a
    batch dimension for the 2D SAM2 encoder, then reshaping back to 3D.

    Args:
        checkpoint_path: Path to MedSAM2_latest.pt or similar.
        config_file: Hydra config relative to MedSAM2 root
                     (default ``configs/sam2.1_hiera_t512.yaml``).
        device: Torch device string.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_file: str = "configs/sam2.1_hiera_t512.yaml",
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_file = config_file

        # Build full SAM2 model (we will extract only image_encoder)
        self.sam_model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=device,
            mode="eval",
        )
        self.image_encoder = self.sam_model.image_encoder

        # Freeze all parameters
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.sam_model.parameters():
            p.requires_grad = False

        self.image_encoder.eval()
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 3D feature volume from input image.

        Args:
            x: Input volume [B, 1, H, W, D] (MONAI convention).

        Returns:
            Feature volume [B, C, H', W', D'] where (H', W') are downsampled
            spatial dimensions and D' == D (depth preserved).
        """
        B, C, H, W, D = x.shape
        assert C == 1, f"Expected single channel, got {C}"

        # Normalize to [0, 255] range expected by SAM2 (it internally normalizes)
        # MONAI NormalizeIntensity already zero-mean unit-std'd the volume.
        # SAM2 expects uint8-like [0,255] then applies ImageNet normalization.
        # We need to denormalize and rescale.
        # However, for robust feature extraction we can just rescale to [0, 1]
        # and repeat to 3 channels.
        x_01 = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x_3ch = x_01.repeat(1, 3, 1, 1, 1)  # [B, 3, H, W, D]

        # Treat depth as batch: [B*D, 3, H, W]
        x_2d = x_3ch.permute(0, 4, 1, 2, 3).reshape(B * D, 3, H, W)

        # Forward through image encoder (returns dict with multi-scale features)
        enc_out = self.image_encoder(x_2d)

        # Use the final vision_features (highest semantic level)
        # Shape: [B*D, C_sam, H', W']
        vision_features = enc_out["vision_features"]
        _, C_sam, H_s, W_s = vision_features.shape

        # Reshape back to 3D: [B, D, C_sam, H_s, W_s] → [B, C_sam, H_s, W_s, D]
        feat_3d = vision_features.reshape(B, D, C_sam, H_s, W_s)
        feat_3d = feat_3d.permute(0, 2, 3, 4, 1)  # [B, C_sam, H_s, W_s, D]

        return feat_3d

    def get_output_channels(self) -> int:
        """Return the number of output channels of the encoder."""
        # Hiera-tiny typically outputs 256 channels after neck FPN
        # We'll determine dynamically by running a dummy forward
        dummy = torch.zeros(1, 1, 512, 512, 2, device=self.device)
        with torch.no_grad():
            out = self.forward(dummy)
        return out.shape[1]

    def get_output_spatial(self, input_shape) -> tuple:
        """Return (H_out, W_out, D_out) for a given (H_in, W_in, D_in)."""
        H, W, D = input_shape
        dummy = torch.zeros(1, 1, H, W, D, device=self.device)
        with torch.no_grad():
            out = self.forward(dummy)
        return out.shape[2], out.shape[3], out.shape[4]
