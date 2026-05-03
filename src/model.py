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

    Architecture améliorée :
    1. Branche Spatiale : Pooling multi-échelle (GAP + GMP + std) sur bottleneck + décodeur.
    2. Branche Fréquentielle : FFT 3D sur l'image d'entrée pour capturer Zipper/Banding.
    3. Sortie Ordinale : Prédit 2 seuils binaires par artefact (Sévérité >= 1, Sévérité >= 2).

    Niveaux et dimensions (base=16) :
      bottleneck [B, 256, 6³]  → 3×256 =  768
      dec f4     [B, 128,12³]  → 3×128 =  384
      dec f3     [B,  64,24³]  → 3× 64 =  192
      dec f2     [B,  32,48³]  → 3× 32 =   96
      dec f1     [B,  16,96³]  → 3× 16 =   48
                                  Total = 1488
      FFT branch : [B, 128] (spectre de puissance moyen)
                                  Total final = 1616

    MLP : Linear(1616→256) + LN + ReLU + Dropout(0.3)
          Linear(256→128)  + ReLU + Dropout(0.2)
          Linear(128, n_artifacts * (n_severity - 1))
    """

    def __init__(
        self,
        bottleneck_ch: int,
        dec_feat_chs:  list[int],
        n_artifacts:   int = 7,
        n_severity:    int = 3,
    ):
        super().__init__()
        self.n_artifacts = n_artifacts
        self.n_severity  = n_severity
        
        # Features spatiales
        spatial_feats = 3 * (bottleneck_ch + sum(dec_feat_chs))
        # Features fréquentielles (on prend un vecteur résumé du spectre)
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
            # Sortie ordinale : (n_severity - 1) seuils par artefact
            nn.Linear(128, n_artifacts * (n_severity - 1)),
        )

    @staticmethod
    def _pool(feat: torch.Tensor) -> torch.Tensor:
        """GAP + GMP + std sur les dimensions spatiales → [B, 3·C]."""
        v   = feat.flatten(2)                       # [B, C, N_voxels]
        gap = v.mean(dim=-1)                        # [B, C]
        gmp = v.max(dim=-1).values                  # [B, C]
        std = v.std(dim=-1, unbiased=False)         # [B, C]
        return torch.cat([gap, gmp, std], dim=1)    # [B, 3·C]

    def _get_freq_feats(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait des features fréquentielles via FFT 3D.
        x : [B, 1, D, H, W]
        """
        # FFT 3D
        fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        # Spectre de puissance avec compression log pour éviter les NaNs
        mag = torch.log1p(torch.abs(fft))
        
        # On réduit le spectre en un vecteur de 128
        flat_mag = mag.flatten(1)
        indices = torch.linspace(0, flat_mag.shape[1]-1, 128).long().to(x.device)
        feats = flat_mag[:, indices]
        
        # Normalisation Z-score pour stabiliser l'entrée du MLP
        feats = (feats - feats.mean(dim=1, keepdim=True)) / (feats.std(dim=1, keepdim=True) + 1e-6)
        return feats

    def forward(
        self,
        x:           torch.Tensor,
        bottleneck:  torch.Tensor,
        dec_feats:   list[torch.Tensor],
    ) -> torch.Tensor:
        # 1. Features spatiales (Pooling multi-échelle)
        pools = [self._pool(bottleneck)]
        for f in dec_feats:
            pools.append(self._pool(f))
        x_spatial = torch.cat(pools, dim=1)                 # [B, 1488]
        
        # 2. Features fréquentielles
        x_freq = self._get_freq_feats(x)                     # [B, 128]
        
        # 3. Fusion et MLP
        combined = torch.cat([x_spatial, x_freq], dim=1)     # [B, 1616]
        logits = self.mlp(combined)
        
        # Sortie : [B, n_artifacts, n_severity - 1]
        return logits.view(-1, self.n_artifacts, self.n_severity - 1)


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

class BackboneLISA(nn.Module):
    """
    Modèle joint LISA 2026 — backbone UNet complet + têtes dédiées par tâche.

    Paramètres :
        base          : largeur de base de l'UNet (défaut 16)
        n_artifacts   : nombre d'artefacts Task1a (7)
        n_severity    : niveaux de sévérité Task1a (3 : aucun/léger/sévère)
        n_seg_classes : classes de segmentation Task2 (14 : fond + 13 structures)
        c_anat/mod/art: ignorés (conservés pour compatibilité YAML)
    """

    def __init__(
        self,
        base:          int = 16,
        n_artifacts:   int = 7,
        n_severity:    int = 3,
        n_seg_classes: int = 14,
        # anciens paramètres disentanglement — ignorés, conservés pour la
        # compatibilité avec les fichiers YAML et les checkpoints existants
        c_anat:        int = 16,
        c_mod:         int = 8,
        c_art:         int = 8,
    ):
        super().__init__()
        self.backbone = SharedUNet(base=base)

        self.task1a = Task1aHead(
            bottleneck_ch = self.backbone.bottleneck_ch,
            dec_feat_chs  = self.backbone.dec_feat_chs,
            n_artifacts   = n_artifacts,
            n_severity    = n_severity,
        )
        self.task1b = Task1bHead(feat_ch=self.backbone.feat_ch)
        self.task2  = Task2Head(feat_ch=self.backbone.feat_ch, n_classes=n_seg_classes)

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
        feats, bottleneck, dec_feats = self.backbone(x)

        out = {}
        if run_task1a:
            out["task1a"] = self.task1a(x, bottleneck, dec_feats)
        if run_task1b:
            out["task1b"] = self.task1b(feats)
        if run_task2:
            out["task2"]  = self.task2(feats)
        return out
