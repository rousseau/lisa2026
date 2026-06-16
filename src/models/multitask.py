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
    When ``filters`` matches a previously-trained checkpoint (e.g. RUN_0003),
    all state-dict keys starting with ``model.`` can be reused.  Only
    ``cls_*`` and ``recon_*`` keys are new and randomly initialised.

    Architecture
    ------------
    Shared encoder (DynUNet internal blocks, ``filters`` dictating depth):
        input_block     : 1×128³ → f[0]×128³
        downsamples[i]  : → f[i+1]×128/2^(i+1)³
        bottleneck      : → f[-1]×...³

    Head Task 2 — segmentation (full DynUNet forward):
        upsamples + output_block → [B, num_seg_classes, H, W, D]

    Head Task 1a — artifact classification:
        GAP(bottleneck) → Linear(f[-1], 128) → LeakyReLU → Dropout(0.3)
                        → Linear(128, 7×3) → reshape [B, 7, 3]

    Head Task 1b — reconstruction decoder:
        len(filters)-1 × UpBlock3d + Conv3d(f[0]→1) → [B, 1, H, W, D]

    Args:
        in_channels:           Input channels (default 1).
        filters:               Encoder filter widths (default (32,64,128,256,320,320)).
        num_seg_classes:       Number of segmentation classes (default 12).
        num_artifact_tasks:    Number of artifact types (default 7).
        num_artifact_classes:  Number of severity levels (default 3).
        kernel_size:           Per-level 3D kernel sizes.
        strides:               Per-level strides.  If ``None``, inferred from *filters*.
        upsample_kernel_size:  Per-level upsampling kernel sizes.
        norm_name:             Normalisation type (default "instance").
    """

    def __init__(
        self,
        in_channels: int = 1,
        filters: tuple = (32, 64, 128, 256, 320, 320),
        num_seg_classes: int = 12,
        num_artifact_tasks: int = 7,
        num_artifact_classes: int = 3,
        kernel_size=None,
        strides=None,
        upsample_kernel_size=None,
        norm_name: str = "instance",
    ):
        super().__init__()

        n = len(filters)
        if kernel_size is None:
            kernel_size = [(3, 3, 3)] * n
        if strides is None:
            # Default: all (2,2,2) except first stride=1
            strides = [(1, 1, 1)] + [(2, 2, 2)] * (n - 1)
        if upsample_kernel_size is None:
            upsample_kernel_size = [(2, 2, 2)] * (n - 1)

        f = tuple(int(x) for x in filters)
        self._filters = f
        self.num_artifact_tasks = num_artifact_tasks
        self.num_artifact_classes = num_artifact_classes
        self.num_seg_classes = num_seg_classes

        # ── Task 2 backbone ───────────────────────────────────────────────────
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_seg_classes,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=f,
            norm_name=norm_name,
            deep_supervision=True,
        )

        # ── Task 1a: classification head ──────────────────────────────────────
        self.cls_gap = nn.AdaptiveAvgPool3d(1)
        self.cls_mlp = nn.Sequential(
            nn.Linear(f[-1], 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_artifact_tasks * num_artifact_classes),
        )

        # ── Task 1b: reconstruction decoder ──────────────────────────────────
        # Dynamically build as many UpBlock3d layers as needed.
        self.recon_ups = nn.ModuleList()
        for i in range(n - 1, 0, -1):
            # up from f[i] to f[i-1], skip f[i-1]
            self.recon_ups.append(UpBlock3d(f[i], f[i - 1], f[i - 1]))
        self.recon_out = nn.Conv3d(f[0], 1, kernel_size=1)

        # ── Initialisation of new heads ───────────────────────────────────────
        self._init_new_heads()

    def load_state_dict(self, state_dict, strict: bool = True):
        """Override to remap old ``recon_up{k}`` keys → ``recon_ups.{idx}``.

        Old checkpoints (commit before N-stage generalisation) named the
        reconstruction decoder blocks ``recon_up1`` … ``recon_up{n-1}``
        ordered *finest→coarsest*.  The current ModuleList orders them
        *coarsest→finest* (``recon_ups[0]`` … ``recon_ups[n-2]``).
        """
        n = len(self._filters)          # number of encoder stages
        mapped = {}
        remapped = False
        for k, v in state_dict.items():
            if k.startswith("recon_up") and not k.startswith("recon_ups."):
                suffix = k[len("recon_up"):]
                parts = suffix.split(".", 1)
                if parts[0].isdigit():
                    old_idx = int(parts[0])          # 1 … n-1
                    new_idx = (n - 1) - old_idx      # reverse order
                    rest = parts[1] if len(parts) > 1 else ""
                    new_k = f"recon_ups.{new_idx}"
                    if rest:
                        new_k += "." + rest
                    mapped[new_k] = v
                    remapped = True
                    continue
            mapped[k] = v
        if remapped:
            print("[INFO] Remapped old recon_upN keys -> recon_ups for backward compat.")
        super().load_state_dict(mapped, strict=strict)

    def reset_heads(self) -> None:
        """Re-initialise all three task-specific heads (1a, 1b, 2).

        Used after loading a pretrained encoder so that the heads start
        from random weights rather than inheriting potentially incompatible
        values from a previous training run (option B)."""
        self._init_new_heads()
        # Re-initialise DynUNet output_block (Task 2 head)
        with torch.no_grad():
            for m in self.model.output_block.modules():
                if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Re-initialise DynUNet upsample blocks
            for up in self.model.upsamples:
                for m in up.modules():
                    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        print("[INFO] All task heads reset to random initialisation.")

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
            skips[0]: f[0] ch, full spatial (input_block output)
            skips[1]: f[1] ch, /2
            ...
            skips[n-2]: f[n-2] ch, coarsest before bottleneck
        ``bottleneck``: f[-1] ch
        """
        out = self.model.input_block(x)
        skips = [out]
        for ds in self.model.downsamples:
            out = ds(out)
            skips.append(out)
        bottleneck = self.model.bottleneck(out)
        return bottleneck, skips

    # ── Task-specific forwards ───────────────────────────────────────────────

    def forward_task2(self, x: torch.Tensor):
        """Return raw DynUNet output — with deep supervision enabled this is
        a 6-D tensor ``[B, N_levels, C, H, W, D]``."""
        return self.model(x)

    def forward_task2_main(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the main (full-resolution) segmentation output [B, C, H, W, D]."""
        out = self.model(x)
        # MONAI DynUNet with deep_supervision=True returns [B, N_levels, C, H, W, D]
        if out.dim() == 6:
            return out[:, 0]  # first level = main output
        return out

    def forward_task1a(self, x: torch.Tensor) -> torch.Tensor:
        """Returns classification logits [B, num_artifact_tasks, num_artifact_classes]."""
        bottleneck, _ = self._encode(x)
        pooled = self.cls_gap(bottleneck).flatten(1)  # [B, f[-1]]
        logits = self.cls_mlp(pooled)                 # [B, T*C]
        return logits.view(-1, self.num_artifact_tasks, self.num_artifact_classes)

    def forward_task1b(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed volume [B, 1, H, W, D]."""
        bottleneck, skips = self._encode(x)
        d = bottleneck
        for i, up in enumerate(self.recon_ups):
            # skips[-1] = finest, ..., skips[0] = coarsest-before-bottleneck
            d = up(d, skips[-(i + 1)])
        return self.recon_out(d)

    def forward(self, x: torch.Tensor, task: str = "2") -> torch.Tensor:
        """Dispatch to the head for *task* ∈ {``"1a"``, ``"1b"``, ``"2"``}."""
        if task == "1a":
            return self.forward_task1a(x)
        if task == "1b":
            return self.forward_task1b(x)
        if task == "2":
            return self.forward_task2(x)
        raise ValueError(f"Unknown task '{task}'. Expected '1a', '1b', or '2'.")
