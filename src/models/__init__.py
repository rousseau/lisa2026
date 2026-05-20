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


# ─── RUN_0004 – Shared Encoder Multi-task Model ──────────────────────────────


class DoubleConv3d(nn.Module):
    """Two 3×3×3 conv + InstanceNorm3d (affine) + LeakyReLU(0.01).
    The stride is applied to the FIRST conv (like DynUNet encoder blocks).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock3d(nn.Module):
    """ConvTranspose3d(in_ch → out_ch, k=2, s=2) + concat skip + DoubleConv3d(out_ch+skip_ch → out_ch)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv3d(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class SharedEncoderMultiTaskModel(nn.Module):
    """Shared 3D U-Net encoder with three task-specific heads (RUN_0004).

    Architecture
    ------------
    Encoder (5 levels, default filters [32,64,128,256,320]):
      enc1 (stride=1) : 128³ → f[0]×128³
      enc2 (stride=2) : → f[1]×64³
      enc3 (stride=2) : → f[2]×32³
      enc4 (stride=2) : → f[3]×16³
      enc5 (stride=2) : → f[4]×8³  (bottleneck)

    Head 1a — artifact classification:
      GAP(bottleneck) → Linear(f[4],128) → LeakyReLU → Dropout(0.3)
                      → Linear(128, num_artifact_tasks*num_artifact_classes)
      output: [B, num_artifact_tasks, num_artifact_classes]

    Head 1b — reconstruction (autoencoder, trained on clean images):
      4× UpBlock3d + Conv3d(f[0]→1, k=1)
      output: [B, 1, H, W, D]

    Head 2 — segmentation:
      4× UpBlock3d + Conv3d(f[0]→num_seg_classes, k=1)
      output: [B, num_seg_classes, H, W, D]

    Notes
    -----
    * Input spatial size must be divisible by 2^4 = 16 (e.g. 128³).
    * No feature constraint is imposed (unconstrained baseline).
    * Both decoders share the same encoder output but have independent weights.
    """

    def __init__(
        self,
        in_channels: int = 1,
        filters: tuple = (32, 64, 128, 256, 320),
        num_seg_classes: int = 12,
        num_artifact_tasks: int = 7,
        num_artifact_classes: int = 3,
    ):
        super().__init__()
        f = filters
        self.filters = f
        self.num_artifact_tasks = num_artifact_tasks
        self.num_artifact_classes = num_artifact_classes
        self.num_seg_classes = num_seg_classes

        # ── Encoder ────────────────────────────────────────────────────────────────
        self.enc1 = DoubleConv3d(in_channels, f[0], stride=1)
        self.enc2 = DoubleConv3d(f[0], f[1], stride=2)
        self.enc3 = DoubleConv3d(f[1], f[2], stride=2)
        self.enc4 = DoubleConv3d(f[2], f[3], stride=2)
        self.enc5 = DoubleConv3d(f[3], f[4], stride=2)

        # ── Head 1a : artifact classification ──────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.cls_mlp = nn.Sequential(
            nn.Linear(f[4], 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_artifact_tasks * num_artifact_classes),
        )

        # ── Head 1b : reconstruction decoder ────────────────────────────────────────
        self.recon_up4 = UpBlock3d(f[4], f[3], f[3])
        self.recon_up3 = UpBlock3d(f[3], f[2], f[2])
        self.recon_up2 = UpBlock3d(f[2], f[1], f[1])
        self.recon_up1 = UpBlock3d(f[1], f[0], f[0])
        self.recon_out = nn.Conv3d(f[0], 1, kernel_size=1)

        # ── Head 2 : segmentation decoder ──────────────────────────────────────────
        self.seg_up4 = UpBlock3d(f[4], f[3], f[3])
        self.seg_up3 = UpBlock3d(f[3], f[2], f[2])
        self.seg_up2 = UpBlock3d(f[2], f[1], f[1])
        self.seg_up1 = UpBlock3d(f[1], f[0], f[0])
        self.seg_out = nn.Conv3d(f[0], num_seg_classes, kernel_size=1)

    # ── Encoder ────────────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor):
        """Run the shared encoder and return (bottleneck, skips).
        skips = (s1, s2, s3, s4) from coarsest to finest spatial resolution
        relative to the decoder order (s4 is used first by up4).
        """
        s1 = self.enc1(x)  # f[0] × 128³
        s2 = self.enc2(s1)  # f[1] × 64³
        s3 = self.enc3(s2)  # f[2] × 32³
        s4 = self.enc4(s3)  # f[3] × 16³
        bottleneck = self.enc5(s4)  # f[4] × 8³
        return bottleneck, (s1, s2, s3, s4)

    # ── Task-specific decoders ──────────────────────────────────────────────────────

    def forward_task1a(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, num_artifact_tasks, num_artifact_classes]."""
        bottleneck, _ = self.encode(x)
        pooled = self.gap(bottleneck).flatten(1)  # [B, f[4]]
        logits = self.cls_mlp(pooled)  # [B, T*C]
        return logits.view(-1, self.num_artifact_tasks, self.num_artifact_classes)

    def forward_task1b(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed volume [B, 1, H, W, D]."""
        bottleneck, (s1, s2, s3, s4) = self.encode(x)
        d = self.recon_up4(bottleneck, s4)
        d = self.recon_up3(d, s3)
        d = self.recon_up2(d, s2)
        d = self.recon_up1(d, s1)
        return self.recon_out(d)

    def forward_task2(self, x: torch.Tensor) -> torch.Tensor:
        """Returns segmentation logits [B, num_seg_classes, H, W, D]."""
        bottleneck, (s1, s2, s3, s4) = self.encode(x)
        d = self.seg_up4(bottleneck, s4)
        d = self.seg_up3(d, s3)
        d = self.seg_up2(d, s2)
        d = self.seg_up1(d, s1)
        return self.seg_out(d)

    def forward(self, x: torch.Tensor, task: str = "2") -> torch.Tensor:
        """Dispatch to the head for *task* ∈ {"1a", "1b", "2"}."""
        if task == "1a":
            return self.forward_task1a(x)
        if task == "1b":
            return self.forward_task1b(x)
        if task == "2":
            return self.forward_task2(x)
        raise ValueError(f"Unknown task '{task}'. Expected '1a', '1b', or '2'.")


# ─── RUN_0004 revised – DynUNet Multi-Head Model ─────────────────────────────


class DynUNetMultiHeadModel(nn.Module):
    """Shared DynUNet backbone with three task-specific heads (RUN_0004 v2).

    Unlike ``SharedEncoderMultiTaskModel`` (which uses custom ``DoubleConv3d``
    blocks), this model wraps MONAI ``DynUNet`` directly as the Task 2 path,
    ensuring **full weight compatibility** with ``Task2DynUNetModel`` (RUN_0003).

    Key compatibility
    -----------------
    All state-dict keys that start with ``model.`` are identical to
    ``Task2DynUNetModel``.  Only ``cls_*`` and ``recon_*`` keys are new and
    are randomly initialised.  When partial-loading from a RUN_0003 checkpoint,
    all 102 DynUNet keys will be matched (vs. 0/102 with the previous
    ``SharedEncoderMultiTaskModel``).

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

    Head Task 1b — reconstruction (clean-image autoencoder):
        4× UpBlock3d + Conv3d(32→1) → [B, 1, H, W, D]
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

        # ── Initialisation explicite des têtes nouvelles ──────────────────────
        # Les poids par défaut (Kaiming) peuvent produire des activations de
        # grande magnitude dès les premiers forward passes, causant des NaN avec
        # AMP (overflow fp16 > 65504).  On force de petits poids pour cls_mlp et
        # une sortie reconstruction proche de zéro.
        self._init_new_heads()

    # ── Head initialisation ─────────────────────────────────────────────────────

    def _init_new_heads(self) -> None:
        """Initialise cls_mlp et recon_* avec de petits poids.

        Objectif : éviter l'overflow AMP (fp16 > 65504) lors des premiers
        forward passes des têtes aléatoires sur l'encodeur pré-entraîné.

        * cls_mlp  : Xavier uniform avec gain=0.01 → logits initiaux ≈ 0.
        * recon_up*: Kaiming normal (fan_out, leaky_relu) + biais zéro.
        * recon_out: N(0, 1e-3) → reconstruction initiale ≈ 0 (pas d'overflow L1).
        """
        # Task 1a — très petits logits initiaux
        for m in self.cls_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Task 1b — décodeur : Kaiming standard, biais zéro
        for block in [self.recon_up4, self.recon_up3, self.recon_up2, self.recon_up1]:
            for m in block.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="leaky_relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.ConvTranspose3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Couche de sortie reconstruction : variance très faible
        # → recon ≈ 0 au départ → L1(recon, target) ≈ target  (fini, pas d'overflow)
        nn.init.normal_(self.recon_out.weight, mean=0.0, std=1e-3)
        if self.recon_out.bias is not None:
            nn.init.zeros_(self.recon_out.bias)

    # ── Shared encoder ──────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        """Run the DynUNet encoder stages and return ``(bottleneck, skips)``.

        ``skips`` is ordered finest → coarsest:
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

    # ── Task-specific forwards ──────────────────────────────────────────────────

    def forward_task2(self, x: torch.Tensor) -> torch.Tensor:
        """Full DynUNet forward — identical to ``Task2DynUNetModel.forward()``."""
        return self.model(x)

    def forward_task1a(self, x: torch.Tensor) -> torch.Tensor:
        """Returns classification logits [B, num_artifact_tasks, num_artifact_classes]."""
        bottleneck, _ = self._encode(x)
        pooled = self.cls_gap(bottleneck).flatten(1)  # [B, f[4]]
        logits = self.cls_mlp(pooled)  # [B, T*C]
        return logits.view(-1, self.num_artifact_tasks, self.num_artifact_classes)

    def forward_task1b(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed volume [B, 1, H, W, D]."""
        bottleneck, skips = self._encode(x)
        d = self.recon_up4(bottleneck, skips[3])  # 320@8³  → 256@16³
        d = self.recon_up3(d, skips[2])  # 256@16³ → 128@32³
        d = self.recon_up2(d, skips[1])  # 128@32³ → 64@64³
        d = self.recon_up1(d, skips[0])  # 64@64³  → 32@128³
        return self.recon_out(d)  # 32@128³ → 1@128³

    def forward(self, x: torch.Tensor, task: str = "2") -> torch.Tensor:
        """Dispatch to the head for *task* ∈ {``"1a"``, ``"1b"``, ``"2"``}."""
        if task == "1a":
            return self.forward_task1a(x)
        if task == "1b":
            return self.forward_task1b(x)
        if task == "2":
            return self.forward_task2(x)
        raise ValueError(f"Unknown task '{task}'. Expected '1a', '1b', or '2'.")
