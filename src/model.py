#!/usr/bin/env python3
"""
Architecture du modèle joint LISA 2026 — backbone UNet complet partagé.

Pipeline :
  [image 1×D×H×W]
      ↓
  SharedUNet  (encodeur 4 niveaux + décodeur 4 niveaux)
      → feats        [B, base, D, H, W]         features pleine résolution
      → bottleneck   [B, 16·base, D/16, ...]    features basse résolution
      → dec_feats    list de 4 tenseurs intermédiaires du décodeur
      ↓ selon la tâche
  Task1aHead  (bottleneck + dec_feats) → [B, 7, 3]        (sévérité artefact)
  Task1bHead  feats                    → [B, 1, D, H, W]  (reconstruction)
  Task2Head   feats                    → [B, 14, D, H, W] (segmentation)

Nombre de paramètres (base=16, target_size=96) :
  SharedUNet    ≈ 5.89 M   (encodeur 3.53 M + décodeur 2.35 M)
  Task1aHead    ≈  418 K   (pooling multi-échelle 1488-dim → MLP 256→128→21)
  Task1bHead    ≈   14 K   (ConvBlock + Conv1×1 → Sigmoid)
  Task2Head     ≈   14 K   (ConvBlock + Conv1×1)
  ─────────────────────────
  Total         ≈ 6.33 M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Blocs de base
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Residual block: x + 2× (Conv3d 3³ → GroupNorm → ReLU)."""

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
        )
        self.relu = nn.ReLU(inplace=True)
        
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(n_groups, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.shortcut(x) + self.block(x))


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

    Paramètres (base=16) : ≈ 3.53 M
    """

    def __init__(self, base: int = 16):
        super().__init__()
        b = base
        self.enc1       = ConvBlock(1,     b)        # [B,  b,  S,   S,   S]   ≈   7 K
        self.enc2       = Down(b,    b * 2)           # [B, 2b,  S/2, ...]      ≈  42 K
        self.enc3       = Down(b*2,  b * 4)           # [B, 4b,  S/4, ...]      ≈ 166 K
        self.enc4       = Down(b*4,  b * 8)           # [B, 8b,  S/8, ...]      ≈ 664 K
        self.bottleneck = Down(b*8,  b * 16)          # [B, 16b, S/16, ...]     ≈ 2.65 M
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
# Décodeur partagé
# ──────────────────────────────────────────────────────────────────────────────

class SharedDecoder(nn.Module):
    """
    Décodeur UNet partagé entre toutes les tâches.
    bottleneck → 4× Up avec skip connections → features pleine résolution.

    Retourne :
      feats      [B, base, D, H, W]               features pleine résolution
      dec_feats  [f4, f3, f2, f1]                 features intermédiaires
                   f4 [B, 8b, D/8, ...]
                   f3 [B, 4b, D/4, ...]
                   f2 [B, 2b, D/2, ...]
                   f1 [B,  b, D,   ...]  (= feats)

    Paramètres (base=16) : ≈ 2.35 M
      up4 (256+128→128) ≈ 1.77 M
      up3 (128+ 64→ 64) ≈  443 K
      up2 ( 64+ 32→ 32) ≈  111 K
      up1 ( 32+ 16→ 16) ≈   28 K
    """

    def __init__(self, in_ch: int, skip_chs: list[int], base: int = 16):
        super().__init__()
        b = base
        self.up4 = Up(in_ch,  skip_chs[3], b * 8)   # 256+128 → 128
        self.up3 = Up(b * 8,  skip_chs[2], b * 4)   # 128+ 64 →  64
        self.up2 = Up(b * 4,  skip_chs[1], b * 2)   #  64+ 32 →  32
        self.up1 = Up(b * 2,  skip_chs[0], b)        #  32+ 16 →  16

    def forward(
        self,
        z: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1, s2, s3, s4 = skips
        f4 = self.up4(z,  s4)   # [B, 8b, D/8, ...]
        f3 = self.up3(f4, s3)   # [B, 4b, D/4, ...]
        f2 = self.up2(f3, s2)   # [B, 2b, D/2, ...]
        f1 = self.up1(f2, s1)   # [B,  b, D,   ...]
        return f1, [f4, f3, f2, f1]


# ──────────────────────────────────────────────────────────────────────────────
# Backbone UNet complet partagé
# ──────────────────────────────────────────────────────────────────────────────

class SharedUNet(nn.Module):
    """
    Backbone UNet complet (encodeur + décodeur) partagé entre les trois tâches.

    Retourne :
      feats        [B, base, D, H, W]      features pleine résolution (entrée Task1b/Task2)
      bottleneck   [B, 16·base, D/16, ...] features basse résolution
      dec_feats    [f4, f3, f2, f1]        features intermédiaires décodeur (pour Task1a)
    """

    def __init__(self, base: int = 16):
        super().__init__()
        self.encoder = SharedEncoder(base=base)
        self.decoder = SharedDecoder(
            in_ch    = self.encoder.bottleneck_ch,
            skip_chs = self.encoder.skip_chs,
            base     = base,
        )
        self._base = base

    @property
    def feat_ch(self) -> int:
        """Canaux des features pleine résolution (entrée des têtes Task1b/Task2)."""
        return self._base

    @property
    def bottleneck_ch(self) -> int:
        return self._base * 16

    @property
    def dec_feat_chs(self) -> list[int]:
        """Canaux à chaque niveau du décodeur : [8b, 4b, 2b, b]."""
        b = self._base
        return [b * 8, b * 4, b * 2, b]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        z, skips          = self.encoder(x)
        feats, dec_feats  = self.decoder(z, skips)
        return feats, z, dec_feats


# ──────────────────────────────────────────────────────────────────────────────
# Têtes spécifiques aux tâches
# ──────────────────────────────────────────────────────────────────────────────

class Task1aHead(nn.Module):
    """
    Task 1a – Quality Control (classification multi-artefact, 3 sévérités).

    Branche Spatiale  : Pooling multi-échelle (GAP + GMP + std) sur :
      - bottleneck (profond, sémantique)
      - dec_feats  (intermédiaires du décodeur)
      - extra_feats = concat(z_anat, z_art)  [optionnel, v8+]
    Branche Fréquentielle : FFT 3D sur l'image d'entrée (Zipper/Banding).
    Sortie Ordinale : 2 seuils binaires par artefact.

    extra_feat_ch (v8) : canaux de concat(z_anat, z_art) = c_anat + c_art.
    Si extra_feat_ch=0 (v7 compat), extra_feats est ignoré.
    """

    def __init__(
        self,
        bottleneck_ch:  int,
        dec_feat_chs:   list[int],
        n_artifacts:    int = 7,
        n_severity:     int = 3,
        extra_feat_ch:  int = 0,   # c_anat + c_art en v8, 0 en v7
    ):
        super().__init__()
        self.n_artifacts   = n_artifacts
        self.n_severity    = n_severity
        self.extra_feat_ch = extra_feat_ch

        # Features spatiales depuis bottleneck + décodeur + (optionnel) extra
        spatial_feats  = 3 * (bottleneck_ch + sum(dec_feat_chs))
        if extra_feat_ch > 0:
            spatial_feats += 3 * extra_feat_ch
        freq_feats = 128

        in_feats = spatial_feats + freq_feats

        self.mlp = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_artifacts * (n_severity - 1)),
        )

    @staticmethod
    def _pool(feat: torch.Tensor) -> torch.Tensor:
        """GAP + GMP + std sur les dimensions spatiales → [B, 3·C]."""
        v   = feat.flatten(2)
        gap = v.mean(dim=-1)
        gmp = v.max(dim=-1).values
        std = v.std(dim=-1, unbiased=False)
        return torch.cat([gap, gmp, std], dim=1)

    def _get_freq_feats(self, x: torch.Tensor) -> torch.Tensor:
        fft      = torch.fft.fftn(x, dim=(-3, -2, -1))
        mag      = torch.log1p(torch.abs(fft))
        flat_mag = mag.flatten(1)
        indices  = torch.linspace(0, flat_mag.shape[1] - 1, 128).long().to(x.device)
        feats    = flat_mag[:, indices]
        feats    = (feats - feats.mean(dim=1, keepdim=True)) / (feats.std(dim=1, keepdim=True) + 1e-6)
        return feats

    def forward(
        self,
        x:            torch.Tensor,
        bottleneck:   torch.Tensor,
        dec_feats:    list[torch.Tensor],
        extra_feats:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        pools = [self._pool(bottleneck)]
        for f in dec_feats:
            pools.append(self._pool(f))
        if extra_feats is not None and self.extra_feat_ch > 0:
            pools.append(self._pool(extra_feats))
        x_spatial = torch.cat(pools, dim=1)
        x_freq    = self._get_freq_feats(x)
        combined  = torch.cat([x_spatial, x_freq], dim=1)
        return self.mlp(combined).view(-1, self.n_artifacts, self.n_severity - 1)


class Task1bHead(nn.Module):
    """
    Task 1b – Enhancement (reconstruction d'image).

    ConvBlock(base, base) → Conv3d(base, 1, 1) → Sigmoid.

    Le ConvBlock permet d'affiner les features partagées vers une carte
    de reconstruction, sans un décodeur supplémentaire complet.

    Paramètres (base=16) : ≈ 14 K
      ConvBlock(16, 16) :  14 K
      Conv3d(16, 1, 1)  :  <1 K
    """

    def __init__(self, feat_ch: int):
        super().__init__()
        self.refine = ConvBlock(feat_ch, feat_ch)
        self.out    = nn.Sequential(
            nn.Conv3d(feat_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.out(self.refine(feats))   # [B, 1, D, H, W]


class Task2Head(nn.Module):
    """
    Task 2 – Segmentation multi-structure.

    ConvBlock(base, base) → Conv3d(base, n_classes, 1) → logits.

    Paramètres (base=16, n_classes=14) : ≈ 14 K
      ConvBlock(16, 16)    :  14 K
      Conv3d(16, 14, 1)    :  <1 K
    """

    def __init__(self, feat_ch: int, n_classes: int):
        super().__init__()
        self.refine = ConvBlock(feat_ch, feat_ch)
        self.out    = nn.Conv3d(feat_ch, n_classes, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.out(self.refine(feats))   # [B, n_classes, D, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# Modèle principal
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Décodeur factorisé (anatomie / modalité / artefact)
# ──────────────────────────────────────────────────────────────────────────────

class FactorizedProjection(nn.Module):
    """
    Projette les features pleine-résolution [B, feat_ch, D, H, W] en trois
    sous-espaces via trois convolutions 1×1 indépendantes :

        z_anat [B, c_anat, D, H, W]  — facteurs structurels / anatomiques
        z_mod  [B, c_mod,  D, H, W]  — facteurs d'apparence / modalité
        z_art  [B, c_art,  D, H, W]  — facteurs d'artefacts

    Routing vers les têtes :
        Task1a  ← concat(z_anat, z_art)  + bottleneck
        Task1b  ← concat(z_anat, z_mod)
        Task2   ← z_anat  (facteurs structurels uniquement)
    """

    def __init__(self, feat_ch: int, c_anat: int, c_mod: int, c_art: int):
        super().__init__()
        self.proj_anat = nn.Sequential(
            nn.Conv3d(feat_ch, c_anat, 1, bias=False),
            nn.GroupNorm(min(8, c_anat), c_anat),
            nn.ReLU(inplace=True),
        )
        self.proj_mod = nn.Sequential(
            nn.Conv3d(feat_ch, c_mod, 1, bias=False),
            nn.GroupNorm(min(8, c_mod), c_mod),
            nn.ReLU(inplace=True),
        )
        self.proj_art = nn.Sequential(
            nn.Conv3d(feat_ch, c_art, 1, bias=False),
            nn.GroupNorm(min(8, c_art), c_art),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.proj_anat(feats), self.proj_mod(feats), self.proj_art(feats)


class BackboneLISA(nn.Module):
    """
    Modèle joint LISA 2026 — backbone UNet + factorisation des features + têtes.

    Paramètres :
        base          : largeur de base de l'UNet (défaut 16)
        c_anat        : canaux sous-espace anatomique (défaut 16)
        c_mod         : canaux sous-espace modalité/apparence (défaut 8)
        c_art         : canaux sous-espace artefacts (défaut 8)
        n_artifacts   : nombre d'artefacts Task1a (7)
        n_severity    : niveaux de sévérité Task1a (3)
        n_seg_classes : classes de segmentation Task2 (14)

    Routing des sous-espaces :
        Task1a  ← concat(z_anat, z_art)  + bottleneck  (structure + artefacts)
        Task1b  ← concat(z_anat, z_mod)                (structure + apparence)
        Task2   ← z_anat                               (structure seule)
    """

    def __init__(
        self,
        base:          int = 16,
        n_artifacts:   int = 7,
        n_severity:    int = 3,
        n_seg_classes: int = 14,
        c_anat:        int = 16,
        c_mod:         int = 8,
        c_art:         int = 8,
    ):
        super().__init__()
        self.backbone   = SharedUNet(base=base)
        self.factorizer = FactorizedProjection(
            feat_ch=self.backbone.feat_ch,
            c_anat=c_anat, c_mod=c_mod, c_art=c_art,
        )

        self.task1a = Task1aHead(
            bottleneck_ch = self.backbone.bottleneck_ch,
            dec_feat_chs  = self.backbone.dec_feat_chs,
            n_artifacts   = n_artifacts,
            n_severity    = n_severity,
            extra_feat_ch = c_anat + c_art,   # remplace feat_ch dans le pooling spatial
        )
        self.task1b = Task1bHead(feat_ch=c_anat + c_mod)
        self.task2  = Task2Head(feat_ch=c_anat, n_classes=n_seg_classes)

    def forward(
        self,
        x:           torch.Tensor,
        run_task1a:  bool = True,
        run_task1b:  bool = True,
        run_task2:   bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Retourne un dict avec les clés 'task1a', 'task1b', 'task2'
        et optionnellement 'z_anat', 'z_mod', 'z_art' pour les losses de régularisation.
        """
        feats, bottleneck, dec_feats = self.backbone(x)
        z_anat, z_mod, z_art        = self.factorizer(feats)

        out = {
            "z_anat": z_anat,
            "z_mod":  z_mod,
            "z_art":  z_art,
        }
        if run_task1a:
            feat_qa = torch.cat([z_anat, z_art], dim=1)
            out["task1a"] = self.task1a(x, bottleneck, dec_feats, extra_feats=feat_qa)
        if run_task1b:
            feat_enh = torch.cat([z_anat, z_mod], dim=1)
            out["task1b"] = self.task1b(feat_enh)
        if run_task2:
            out["task2"] = self.task2(z_anat)
        return out
