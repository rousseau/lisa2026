"""PlainConvUNet (nnU-Net v2) multi-head model for joint training (RUN_0006+).

This model replaces DynUNetMultiHeadModel with PlainConvUNet from
``dynamic_network_architectures`` (the exact backbone used by nnU-Net v2,
RUN_0003a, DSC=0.8220).

Architecture
------------
- Encoder : PlainConvEncoder  (6 stages, [32..320,320])
- Task 2  : UNetDecoder       (segmentation, with deep supervision)
- Task 1a : GAP + MLP         (artifact classification)
- Task 1b : Simple ConvTranspose decoder (reconstruction, no skip)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder


class PlainConvMultiHeadModel(nn.Module):
    """Shared PlainConvUNet encoder with three task-specific heads.

    Matches the exact nnU-Net v2 architecture used in RUN_0003a.
    """

    def __init__(
        self,
        input_channels: int = 1,
        n_stages: int = 6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
        num_seg_classes: int = 12,
        num_artifact_tasks: int = 7,
        num_artifact_classes: int = 3,
    ):
        super().__init__()

        self._f = tuple(int(x) for x in features_per_stage)
        self.num_artifact_tasks = num_artifact_tasks
        self.num_artifact_classes = num_artifact_classes
        self.num_seg_classes = num_seg_classes
        self.n_stages = n_stages

        # ── Encoder + Decoder (exact nnU-Net v2 params) ──────────────────────
        self.encoder = PlainConvEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=self._f,
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3]] * n_stages,
            strides=list(strides),
            n_conv_per_stage=[2] * n_stages,
            conv_bias=True,  # nnU-Net uses conv_bias=True
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            return_skips=True,
            nonlin_first=False,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes=num_seg_classes,
            n_conv_per_stage=[2] * (n_stages - 1),
            deep_supervision=True,
            nonlin_first=False,
        )

        # ── Task 1a: classification head ──────────────────────────────────────
        self.cls_gap = nn.AdaptiveAvgPool3d(1)
        self.cls_mlp = nn.Sequential(
            nn.Linear(self._f[-1], 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_artifact_tasks * num_artifact_classes),
        )

        # ── Task 1b: simple ConvTranspose reconstruction decoder (no skip) ───
        self.recon_ups = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            in_ch = self._f[i]
            out_ch = self._f[i - 1]
            self.recon_ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0),
                    nn.InstanceNorm3d(out_ch, affine=True),
                    nn.LeakyReLU(inplace=True),
                )
            )
        self.recon_out = nn.Conv3d(self._f[0], 1, kernel_size=3, padding=1)

        # ── Initialisation ────────────────────────────────────────────────────
        self._init_new_heads()

    def _init_new_heads(self):
        for m in self.cls_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.recon_ups.modules():
            if isinstance(m, (nn.ConvTranspose3d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.recon_out.weight, mean=0.0, std=1e-3)
        if self.recon_out.bias is not None:
            nn.init.zeros_(self.recon_out.bias)

    # ── Encoder freeze / unfreeze ────────────────────────────────────────────

    def freeze_encoder(self):
        """Freeze encoder parameters (backbone from nnU-Net)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("[INFO] Encoder frozen —", sum(1 for _ in self.encoder.parameters()), "params")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (for progressive unfreezing)."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("[INFO] Encoder unfrozen")

    # ── Load from nnU-Net v2 checkpoint ──────────────────────────────────────

    def load_pretrained_nnunet(self, ckpt_path: str, device="cpu") -> int:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        src_sd = ckpt.get("network_weights", ckpt)
        tgt_sd = self.state_dict()

        loaded = 0
        for k, v in src_sd.items():
            if k in tgt_sd:
                if v.shape == tgt_sd[k].shape:
                    tgt_sd[k] = v
                    loaded += 1
                else:
                    print(f"[WARN] Shape mismatch: {k} src={v.shape} tgt={tgt_sd[k].shape}")
        self.load_state_dict(tgt_sd, strict=False)
        print(f"[INFO] Loaded {loaded}/{len(tgt_sd)} keys from nnU-Net: {ckpt_path}")
        return loaded

    # ── Task-specific forwards ───────────────────────────────────────────────

    def forward_task2(self, x: torch.Tensor):
        """Return deep-supervised outputs [B, N_levels, C, H, W, D]."""
        skips = self.encoder(x)
        out_list = self.decoder(skips)
        if isinstance(out_list, (list, tuple)):
            main = out_list[0]
            batch, c, h, w, d = main.shape
            n = len(out_list)
            packed = torch.zeros(batch, n, c, h, w, d, device=main.device, dtype=main.dtype)
            for i, o in enumerate(out_list):
                if o.shape[2:] != (h, w, d):
                    o = F.interpolate(o, size=(h, w, d), mode="trilinear", align_corners=False)
                packed[:, i] = o
            return packed
        return out_list

    def forward_task2_main(self, x: torch.Tensor) -> torch.Tensor:
        """Main (full-res) segmentation [B, C, H, W, D]."""
        skips = self.encoder(x)
        out = self.decoder(skips)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def forward_task1a(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        bottleneck = skips[-1]
        pooled = self.cls_gap(bottleneck).flatten(1)
        logits = self.cls_mlp(pooled)
        return logits.view(-1, self.num_artifact_tasks, self.num_artifact_classes)

    def forward_task1b(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = x.shape[2:]
        skips = self.encoder(x)
        d = skips[-1]
        for up in self.recon_ups:
            d = up(d)
        out = self.recon_out(d)
        if out.shape[2:] != target_shape:
            out = F.interpolate(out, size=target_shape, mode="trilinear", align_corners=False)
        return torch.sigmoid(out)

    def forward(self, x: torch.Tensor, task: str = "2") -> torch.Tensor:
        if task == "1a":
            return self.forward_task1a(x)
        if task == "1b":
            return self.forward_task1b(x)
        if task == "2":
            return self.forward_task2(x)
        raise ValueError(f"Unknown task '{task}'")
