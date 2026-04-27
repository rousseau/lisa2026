#!/usr/bin/env python3
"""
Architecture du modèle joint LISA 2026.

Pipeline :
  [image 1×D×H×W]
      ↓
  SharedEncoder  (UNet encoder, 4 niveaux + bottleneck)
      ↓
  DisentanglementHead
      ├─ z_anat  [B, C_anat, d, h, w]
      ├─ z_mod   [B, C_mod,  d, h, w]
      └─ z_art   [B, C_art,  d, h, w]
      ↓ selon la tâche
  Task1aDecoder   z_art              → [B, 7, 3]  (sévérité artefact)
  Task1bDecoder   z_anat + z_mod     → [B, 1, D, H, W]  (reconstruction)
  Task2Decoder    z_anat             → [B, 14, D, H, W]  (segmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Blocs de base
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """2× (Conv3d 3³ → GroupNorm → ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        # GroupNorm est stable même avec batch_size=1
        n_groups = min(num_groups, out_ch)
        while out_ch % n_groups != 0:
            n_groups -= 1
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(n_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(n_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool3d(2) + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample ×2 + cat(skip) + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # ajustement de taille si nécessaire (arrondis de pooling)
        diff = [s - x.shape[i + 2] for i, s in enumerate(skip.shape[2:])]
        x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        return self.conv(torch.cat([x, skip], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Encodeur partagé
# ──────────────────────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """
    Encodeur UNet 4 niveaux.
    Canaux : 1 → b → 2b → 4b → 8b → 16b (bottleneck).
    Retourne (bottleneck, [skip1, skip2, skip3, skip4]).
    """

    def __init__(self, base: int = 16):
        super().__init__()
        b = base
        self.enc1       = ConvBlock(1,      b)       # [B, b,   S,   S,   S]
        self.enc2       = Down(b,    b * 2)           # [B, 2b,  S/2, ...]
        self.enc3       = Down(b*2,  b * 4)           # [B, 4b,  S/4, ...]
        self.enc4       = Down(b*4,  b * 8)           # [B, 8b,  S/8, ...]
        self.bottleneck = Down(b*8,  b * 16)          # [B, 16b, S/16, ...]
        self._base = base

    @property
    def bottleneck_ch(self) -> int:
        return self._base * 16

    @property
    def skip_chs(self) -> list[int]:
        b = self._base
        return [b, b * 2, b * 4, b * 8]  # skip1 … skip4

    def forward(self, x: torch.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        z  = self.bottleneck(s4)
        return z, [s1, s2, s3, s4]


# ──────────────────────────────────────────────────────────────────────────────
# Tête de désenchevêtrement (disentanglement)
# ──────────────────────────────────────────────────────────────────────────────

class DisentanglementHead(nn.Module):
    """
    Projette le bottleneck partagé en 3 sous-espaces spécialisés.
      z_anat : structure anatomique  (utilisée pour Task1b + Task2)
      z_mod  : contraste / modalité  (utilisée pour Task1b uniquement)
      z_art  : artefacts             (utilisée pour Task1a uniquement)
    """

    def __init__(self, in_ch: int, c_anat: int = 128, c_mod: int = 64, c_art: int = 64):
        super().__init__()
        self.proj_anat = nn.Sequential(nn.Conv3d(in_ch, c_anat, 1), nn.ReLU(inplace=True))
        self.proj_mod  = nn.Sequential(nn.Conv3d(in_ch, c_mod,  1), nn.ReLU(inplace=True))
        self.proj_art  = nn.Sequential(nn.Conv3d(in_ch, c_art,  1), nn.ReLU(inplace=True))
        self.c_anat, self.c_mod, self.c_art = c_anat, c_mod, c_art

    def forward(self, z: torch.Tensor):
        return self.proj_anat(z), self.proj_mod(z), self.proj_art(z)


# ──────────────────────────────────────────────────────────────────────────────
# Décodeur générique (UNet upsampling)
# ──────────────────────────────────────────────────────────────────────────────

class _UNetDecoder(nn.Module):
    """
    Décodeur UNet générique : bottleneck → ×4 upsamplings avec skip connections.
    Retourne des features [B, base, D, H, W] ; la couche de sortie est gérée par la tâche.
    """

    def __init__(self, in_ch: int, skip_chs: list[int], base: int = 16):
        super().__init__()
        b = base
        # skip_chs = [skip1_ch, skip2_ch, skip3_ch, skip4_ch]
        self.up4 = Up(in_ch,  skip_chs[3], b * 8)
        self.up3 = Up(b * 8,  skip_chs[2], b * 4)
        self.up2 = Up(b * 4,  skip_chs[1], b * 2)
        self.up1 = Up(b * 2,  skip_chs[0], b)

    def forward(self, z: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3, s4 = skips
        x = self.up4(z, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return x   # [B, base, D, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# Décodeurs spécifiques aux tâches
# ──────────────────────────────────────────────────────────────────────────────

class Task1aDecoder(nn.Module):
    """
    Task 1a – Quality Control.
    z_art → Global Avg Pool → FC → [B, N_art, N_sev].
    """

    def __init__(self, c_art: int = 64, n_artifacts: int = 7, n_severity: int = 3):
        super().__init__()
        self.n_artifacts = n_artifacts
        self.n_severity  = n_severity
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_art, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_artifacts * n_severity),
        )

    def forward(self, z_art: torch.Tensor) -> torch.Tensor:
        x = self.gap(z_art)
        x = self.fc(x)
        return x.view(-1, self.n_artifacts, self.n_severity)  # [B, 7, 3]


class Task1bDecoder(nn.Module):
    """
    Task 1b – Enhancement (reconstruction).
    (z_anat, z_mod) → image reconstruite [B, 1, D, H, W].
    """

    def __init__(self, c_anat: int, c_mod: int, skip_chs: list[int], base: int = 16):
        super().__init__()
        in_ch = c_anat + c_mod
        self.proj = nn.Conv3d(in_ch, base * 16, 1, bias=False)
        self.dec  = _UNetDecoder(base * 16, skip_chs, base=base)
        self.out  = nn.Sequential(
            nn.Conv3d(base, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_anat: torch.Tensor,
        z_mod:  torch.Tensor,
        skips:  list[torch.Tensor],
    ) -> torch.Tensor:
        z = self.proj(torch.cat([z_anat, z_mod], dim=1))
        x = self.dec(z, skips)
        return self.out(x)   # [B, 1, D, H, W]


class Task2Decoder(nn.Module):
    """
    Task 2 – Segmentation multi-structure.
    z_anat → logits de segmentation [B, N_classes, D, H, W].
    """

    def __init__(self, c_anat: int, n_classes: int, skip_chs: list[int], base: int = 16):
        super().__init__()
        self.proj = nn.Conv3d(c_anat, base * 16, 1, bias=False)
        self.dec  = _UNetDecoder(base * 16, skip_chs, base=base)
        self.out  = nn.Conv3d(base, n_classes, 1)

    def forward(self, z_anat: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        z = self.proj(z_anat)
        x = self.dec(z, skips)
        return self.out(x)   # [B, N_classes, D, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# Modèle principal
# ──────────────────────────────────────────────────────────────────────────────

class BrainFMLISA(nn.Module):
    """
    Modèle joint désenchevêtré pour le challenge LISA 2026.

    Paramètres :
        base           : largeur de base de l'UNet (défaut 16)
        c_anat/mod/art : canaux des sous-espaces disentangle
        n_artifacts    : nombre d'artefacts Task1a (7)
        n_severity     : niveaux de sévérité Task1a (3 : aucun/léger/sévère)
        n_seg_classes  : classes de segmentation Task2 (14 : fond + 13 structures)
    """

    def __init__(
        self,
        base:          int = 16,
        c_anat:        int = 128,
        c_mod:         int = 64,
        c_art:         int = 64,
        n_artifacts:   int = 7,
        n_severity:    int = 3,
        n_seg_classes: int = 14,
    ):
        super().__init__()
        self.encoder      = SharedEncoder(base=base)
        enc_ch            = self.encoder.bottleneck_ch   # base * 16
        skip_chs          = self.encoder.skip_chs        # [b, 2b, 4b, 8b]

        self.disentangle  = DisentanglementHead(enc_ch, c_anat, c_mod, c_art)

        self.task1a = Task1aDecoder(c_art, n_artifacts, n_severity)
        self.task1b = Task1bDecoder(c_anat, c_mod, skip_chs, base=base)
        self.task2  = Task2Decoder(c_anat, n_seg_classes, skip_chs, base=base)

    def forward(
        self,
        x:           torch.Tensor,
        run_task1a:  bool = True,
        run_task1b:  bool = True,
        run_task2:   bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Retourne un dict avec les clés 'task1a', 'task1b', 'task2'
        selon les flags run_taskXX.
        """
        z, skips                  = self.encoder(x)
        z_anat, z_mod, z_art      = self.disentangle(z)

        out = {}
        if run_task1a:
            out["task1a"] = self.task1a(z_art)
        if run_task1b:
            out["task1b"] = self.task1b(z_anat, z_mod, skips)
        if run_task2:
            out["task2"]  = self.task2(z_anat, skips)
        return out
