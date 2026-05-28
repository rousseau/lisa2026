"""Shared DynUNet multi-head model for joint training (RUN_0004)."""

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

from .blocks import UpBlock3d


class DynUNetMultiHeadModel(nn.Module):
    """Shared DynUNet backbone with three task-specific heads (RUN_0004).

    Unlike per-task models, this model wraps MONAI ``DynUNet`` directly as the
    Task 2 path, ensuring **full weight compatibility** with ``Task2DynUNetModel``
    (RUN_0003).

    Key compatibility
    -----------------
    All state-dict keys starting with ``model.`` are identical to
    ``Task2DynUNetModel``.  Only ``cls_*`` and ``recon_*`` keys are new and
    randomly initialised.  When partial-loading from a RUN_0003 checkpoint,
    all DynUNet keys will be matched.

    Architecture
    ------------
    Shared encoder (DynUNet internal blocks, filters [32,64,128,256,320]):
        input_block  (stride 1): 1×128³  → 32×128³
        downsamples[0] (stride 2):        → 64×64³
        downsamples[1] (stride 2):        → 128×32³
        downsamples[2] (stride 2):        → 256×16³
        bottleneck     (stride 2):        → 320×8³

    Head Task 2 — segmentation (full DynUNet forward, pretrained-compatible):
        upsamples + output_block → [B, num_seg_classes, H, W, D]

    Head Task 1a — artifact classification:
        GAP(bottleneck) → Linear(320,128) → LeakyReLU → Dropout(0.3)
                        → Linear(128, 7×3) → reshape [B,7,3]

    Head Task 1b — reconstruction decoder:
        4× UpBlock3d + Conv3d(32→1) → [B, 1, H, W, D]

    Args:
        in_channels:           Input channels (default 1).
        filters:               Encoder filter widths (default (32,64,128,256,320)).
        num_seg_classes:       Number of segmentation classes (default 12).
        num_artifact_tasks:    Number of artifact types (default 7).
        num_artifact_classes:  Number of severity levels (default 3).
        kernel_size:           Per-level 3D kernel sizes.
        strides:               Per-level strides.
        upsample_kernel_size:  Per-level upsampling kernel sizes.
        norm_name:             Normalisation type (default "instance").
    """

    def __init__(
        self,
        in_channels: int = 1,
        filters: tuple = (32, 64, 128, 256, 320),
        num_seg_classes: int = 12,
        num_artifact_tasks: int = 7,
        num_artifact_classes: int = 3,
        kernel_size=None,
        strides=None,
        upsample_kernel_size=None,
        norm_name: str = "instance",
    ):
        super().__init__()

        if kernel_size is None:
            kernel_size = [(3, 3, 3)] * 5
        if strides is None:
            strides = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        if upsample_kernel_size is None:
            upsample_kernel_size = [(2, 2, 2)] * 4

        f = tuple(int(x) for x in filters)
        self._filters = f
        self.num_artifact_tasks = num_artifact_tasks
        self.num_artifact_classes = num_artifact_classes
        self.num_seg_classes = num_seg_classes

        # ── Task 2 backbone (identical to Task2DynUNetModel — full key compat.) ─
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_seg_classes,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=f,
            norm_name=norm_name,
            deep_supervision=False,
        )

        # ── Task 1a: classification head ──────────────────────────────────────
        self.cls_gap = nn.AdaptiveAvgPool3d(1)
        self.cls_mlp = nn.Sequential(
            nn.Linear(f[4], 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_artifact_tasks * num_artifact_classes),
        )

        # ── Task 1b: reconstruction decoder ──────────────────────────────────
        # UpBlock3d(in_ch, skip_ch, out_ch) — mirrors the encoder in reverse
        self.recon_up4 = UpBlock3d(f[4], f[3], f[3])  # 320+256 → 256
        self.recon_up3 = UpBlock3d(f[3], f[2], f[2])  # 256+128 → 128
        self.recon_up2 = UpBlock3d(f[2], f[1], f[1])  # 128+64  → 64
        self.recon_up1 = UpBlock3d(f[1], f[0], f[0])  # 64+32   → 32
        self.recon_out = nn.Conv3d(f[0], 1, kernel_size=1)

        # ── Initialisation of new heads ───────────────────────────────────────
        # Small weights prevent fp16 overflow on the first forward passes when
        # the encoder is loaded from a pre-trained checkpoint (AMP).
        self._init_new_heads()

    # ── Head initialisation ──────────────────────────────────────────────────

    def _init_new_heads(self) -> None:
        """Initialise cls_mlp and recon_* with small weights.

        * cls_mlp  : Xavier uniform (gain=0.01) → initial logits ≈ 0.
        * recon_out: N(0, 1e-3)               → initial reconstruction ≈ 0.
        UpBlock3d blocks initialise their own weights in __init__.
        """
        for m in self.cls_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.recon_out.weight, mean=0.0, std=1e-3)
        if self.recon_out.bias is not None:
            nn.init.zeros_(self.recon_out.bias)

    # ── Shared encoder ───────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        """Run the DynUNet encoder and return ``(bottleneck, skips)``.

        ``skips`` ordered finest → coarsest:
            skips[0]: f[0]=32 ch, 128³  (input_block output)
            skips[1]: f[1]=64 ch, 64³   (downsamples[0])
            skips[2]: f[2]=128 ch, 32³  (downsamples[1])
            skips[3]: f[3]=256 ch, 16³  (downsamples[2])
        ``bottleneck``: f[4]=320 ch, 8³
        """
        out = self.model.input_block(x)
        skips = [out]
        for ds in self.model.downsamples:
            out = ds(out)
            skips.append(out)
        bottleneck = self.model.bottleneck(out)
        return bottleneck, skips

    # ── Task-specific forwards ───────────────────────────────────────────────

    def forward_task2(self, x: torch.Tensor) -> torch.Tensor:
        """Full DynUNet forward — identical to ``Task2DynUNetModel.forward()``."""
        return self.model(x)

    def forward_task1a(self, x: torch.Tensor) -> torch.Tensor:
        """Returns classification logits [B, num_artifact_tasks, num_artifact_classes]."""
        bottleneck, _ = self._encode(x)
        pooled = self.cls_gap(bottleneck).flatten(1)  # [B, f[4]]
        logits = self.cls_mlp(pooled)                 # [B, T*C]
        return logits.view(-1, self.num_artifact_tasks, self.num_artifact_classes)

    def forward_task1b(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed volume [B, 1, H, W, D]."""
        bottleneck, skips = self._encode(x)
        d = self.recon_up4(bottleneck, skips[3])  # 320@8³  → 256@16³
        d = self.recon_up3(d, skips[2])            # 256@16³ → 128@32³
        d = self.recon_up2(d, skips[1])            # 128@32³ → 64@64³
        d = self.recon_up1(d, skips[0])            # 64@64³  → 32@128³
        return self.recon_out(d)                    # 32@128³ → 1@128³

    def forward(self, x: torch.Tensor, task: str = "2") -> torch.Tensor:
        """Dispatch to the head for *task* ∈ {``"1a"``, ``"1b"``, ``"2"``}."""
        if task == "1a":
            return self.forward_task1a(x)
        if task == "1b":
            return self.forward_task1b(x)
        if task == "2":
            return self.forward_task2(x)
        raise ValueError(f"Unknown task '{task}'. Expected '1a', '1b', or '2'.")
