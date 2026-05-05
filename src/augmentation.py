#!/usr/bin/env python3
"""
augmentation.py — LISA 2026

Simulation d'artefacts IRM et augmentations géométriques.

Paramètres de simulation d'après Sundaresan 2024 / UPF LISA 2025 (3e place),
qui sont les paramètres de référence cités par BRIQA (2e place LISA 2025) :
  https://github.com/reitxel/LISA2025TeamUPF/blob/main/Task01/src/data/dataset.py

Ce module est utilisé par :
  - dataset.py   : augmentation de données en train (via augment_artifact)
  - da_check.py  : visualisation comparative propre | simulé | réel
  - train.py     : augmentations géométriques légères (via augment_geometric)

API publique :
  simulate_noise(img01, sev)       → Tensor [1,D,H,W]
  simulate_zipper(img01, sev)      → Tensor [1,D,H,W]
  simulate_positioning(img01, sev) → Tensor [1,D,H,W]
  simulate_banding(img01, sev)     → Tensor [1,D,H,W]
  simulate_motion(img01, sev)      → Tensor [1,D,H,W]
  simulate_contrast(img01, sev)    → Tensor [1,D,H,W]
  simulate_distortion(img01, sev)  → Tensor [1,D,H,W]

  augment_artifact(img01, labels)  → (Tensor, np.ndarray)  # pour dataset.py
  augment_geometric(x_batch)       → Tensor                # pour train.py
"""

import random

import numpy as np
import torch
import torchio as tio

ARTIFACT_NAMES = [
    "Noise", "Zipper", "Positioning",
    "Banding", "Motion", "Contrast", "Distortion",
]
N_ARTIFACTS = 7


# ──────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ──────────────────────────────────────────────────────────────────────────────

def _to_subj(img: torch.Tensor) -> tio.Subject:
    """Tenseur [1, D, H, W] → tio.Subject (avec permutation D,H,W → W,H,D)."""
    return tio.Subject(image=tio.ScalarImage(tensor=img.permute(0, 3, 2, 1)))


def _from_subj(subj: tio.Subject) -> torch.Tensor:
    """tio.Subject → Tenseur [1, D, H, W]."""
    return subj.image.data.permute(0, 3, 2, 1).float()


def _to_zscore(img: torch.Tensor) -> torch.Tensor:
    """[0,1] → z-score (μ=0, σ=1) sur les voxels non nuls.

    UPF/Sundaresan 2024 appliquent leurs transforms directement sur des
    images z-score.  Tous les std UPF (ex. 0.18–0.28 pour Noise) sont donc
    exprimés en unités de déviations standards.
    """
    nz = img[img > 0]
    if nz.numel() < 10:
        return img.float()
    mu  = nz.mean()
    sig = nz.std()
    if sig < 1e-6:
        return (img - mu).float()
    return ((img - mu) / sig).float()


def _from_zscore(z: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """z-score → [0,1] en utilisant les stats de l'image de référence [0,1]."""
    nz  = ref[ref > 0]
    mu  = nz.mean()
    sig = nz.std()
    out = z * sig + mu
    lo, hi = float(out.min()), float(out.max())
    if hi > lo:
        out = (out - lo) / (hi - lo)
    return out.clamp(0.0, 1.0)


# RandomBlur(0.4) — appliqué après chaque artefact (UPF : sur masque cerveau
# uniquement, approximé ici sur tout le volume).
_blur = tio.RandomBlur(0.4)


# ──────────────────────────────────────────────────────────────────────────────
# Classe ZipperArtifactSimulator  (UPF exact)
# ──────────────────────────────────────────────────────────────────────────────

class ZipperArtifactSimulator:
    """
    Replication exacte de la classe UPF ZipperArtifactSimulator.

    Applique des bandes discrètes verticales ou horizontales slice-by-slice
    le long de la dimension de plus petite taille (axe d'acquisition).

    sev=1 : 2-4 bandes, amplitude_factor = 0.20–0.40 × std(image)
    sev=2 : 4-7 bandes, amplitude_factor = 0.40–0.70 × std(image)
    """

    def generate_random_zipper(self, image: np.ndarray, severity: int = 1) -> np.ndarray:
        return self.generate_zipper_artifact(image, severity=severity)

    def generate_zipper_artifact(
        self, image: np.ndarray, severity: int = 1,
        direction=None, num_bands=None, band_width=None, amplitude_factor=None,
    ) -> np.ndarray:
        art = image.copy().astype(np.float64)
        if direction is None:
            direction = random.choice(["vertical", "horizontal"])
        if num_bands is None:
            num_bands = (np.random.randint(2, 5) if severity == 1
                         else np.random.randint(4, 8))
        if band_width is None:
            band_width = (np.random.uniform(0.15, 0.4) if severity == 1
                          else np.random.uniform(0.10, 0.3))
        if amplitude_factor is None:
            amplitude_factor = (np.random.uniform(0.2, 0.4) if severity == 1
                                else np.random.uniform(0.4, 0.7))

        if len(image.shape) == 3:
            ax = int(np.argmin(image.shape))
            for i in range(image.shape[ax]):
                sl = [slice(None)] * 3
                sl[ax] = i
                art[tuple(sl)] = self._apply_2d_zipper(
                    art[tuple(sl)], direction, num_bands, band_width, amplitude_factor
                )
        else:
            art = self._apply_2d_zipper(
                art, direction, num_bands, band_width, amplitude_factor
            )
        return art.astype(image.dtype)

    def _apply_2d_zipper(self, img2d, direction, num_bands, band_width, amplitude_factor):
        h, w = img2d.shape
        if direction == "vertical":
            coord      = np.tile(np.linspace(0, 1, w), (h, 1))
            coord_size = w
        else:
            coord      = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
            coord_size = h

        pattern = np.zeros_like(coord)
        bt      = max(2, int(coord_size * 0.01))   # épaisseur de bande ≥ 2 px

        for _ in range(num_bands):
            bp = np.random.uniform(0.1, 0.9)
            if direction == "vertical":
                p = int(bp * w)
                s, e = max(0, p - bt // 2), min(w, p + bt // 2)
                pattern[:, s:e] = 1.0
            else:
                p = int(bp * h)
                s, e = max(0, p - bt // 2), min(h, p + bt // 2)
                pattern[s:e, :] = 1.0

        if np.any(pattern > 0):
            variation = np.random.normal(1.0, (1.0 - band_width) * 0.3, pattern.shape)
            pattern  *= variation

        noise   = np.random.normal(0, 0.2, pattern.shape)
        pattern += noise * np.abs(pattern)

        amp = amplitude_factor * np.std(img2d)
        out = img2d + pattern * amp + np.random.normal(0, amp * 0.05, img2d.shape)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Classe MRIDistortionSimulator  (UPF exact)
# ──────────────────────────────────────────────────────────────────────────────

class MRIDistortionSimulator:
    """
    Replication exacte de la classe UPF MRIDistortionSimulator.

    mild  (sev=1) : elastic(4 pts, 8 mm) + BiasField(0.2, p=0.7)
                    + Ghosting(0.1, p=0.4) + Spike(0.1, p=0.3)
    severe (sev=2): elastic(6 pts, 15 mm) + BiasField(0.4, p=0.7)
                    + Ghosting(0.2, p=0.4) + Spike(0.2, p=0.3)
    """

    _PARAMS = {
        "mild": dict(
            elastic_num_control_points=4,
            elastic_max_displacement=8.0,
            bias_coefficients=0.2,
            ghosting_intensity=0.1,
            spike_intensity=0.1,
        ),
        "severe": dict(
            elastic_num_control_points=6,
            elastic_max_displacement=15.0,
            bias_coefficients=0.4,
            ghosting_intensity=0.2,
            spike_intensity=0.2,
        ),
    }

    def __init__(self, severity: str = "mild"):
        self.p = self._PARAMS[severity]

    def apply_distortion(self, vol: np.ndarray) -> np.ndarray:
        """Entrée / sortie : np.ndarray [D, H, W]."""
        img4d = vol[np.newaxis, ...]   # [1, D, H, W]
        pipeline = tio.Compose([
            tio.RandomElasticDeformation(
                num_control_points=self.p["elastic_num_control_points"],
                max_displacement=self.p["elastic_max_displacement"],
                locked_borders=1,
                image_interpolation="bspline",
                p=1.0,
            ),
            tio.RandomBiasField(coefficients=self.p["bias_coefficients"], p=0.7),
            tio.RandomGhosting(
                num_ghosts=(1, 2), axes=(0, 1, 2),
                intensity=self.p["ghosting_intensity"], p=0.4,
            ),
            tio.RandomSpike(
                num_spikes=1, intensity=self.p["spike_intensity"], p=0.3,
            ),
        ])
        out = pipeline(img4d)
        if isinstance(out, torch.Tensor):
            return out.squeeze(0).numpy()
        return np.squeeze(np.array(out), axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions de simulation individuelle
# ──────────────────────────────────────────────────────────────────────────────
# Entrée / sortie : Tensor [1, D, H, W], valeurs dans [0, 1].
# Les transforms UPF sont appliquées en espace z-score puis reconverties.

def simulate_noise(img01: torch.Tensor, sev: int = 1) -> torch.Tensor:
    """
    Noise UPF/Sundaresan 2024 :
      sev=1 : RandomNoise(mean=0, std=U(0.18, 0.22)) en z-score
      sev=2 : RandomNoise(mean=0, std=U(0.22, 0.28)) en z-score
    """
    z    = _to_zscore(img01)
    subj = _to_subj(z)
    if sev == 1:
        subj = tio.RandomNoise(0, (0.18, 0.22))(subj)
    else:
        subj = tio.RandomNoise(0, (0.22, 0.28))(subj)
    z_aug = _blur(_from_subj(subj))
    return _from_zscore(z_aug, img01)


def simulate_zipper(img01: torch.Tensor, sev: int = 1) -> torch.Tensor:
    """
    Zipper UPF : ZipperArtifactSimulator slice-by-slice.
      sev=1 : 2-4 bandes, amplitude 0.20–0.40 × std_image
      sev=2 : 4-7 bandes, amplitude 0.40–0.70 × std_image
    """
    z   = _to_zscore(img01)
    vol = z.squeeze(0).numpy()
    aug = ZipperArtifactSimulator().generate_random_zipper(vol, severity=sev)
    z_aug = _blur(torch.from_numpy(aug[None].astype(np.float32)))
    return _from_zscore(z_aug, img01)


def simulate_positioning(img01: torch.Tensor, sev: int = 1) -> torch.Tensor:
    """
    Positioning UPF :
      sev=1 : RandomAffine translation ±10 mm (axe LR)
      sev=2 : RandomAffine translation ±20 mm (axe LR)
    Note : simule un décalage de la tête dans le FOV, pas un vrai artefact
    de positionnement LISA, mais c'est la définition UPF.
    """
    z    = _to_zscore(img01)
    subj = _to_subj(z)
    if sev == 1:
        subj = tio.RandomAffine(
            scales=0, degrees=0,
            translation=(-10, 10, 0, 0, 0, 0),
            isotropic=False, center="image",
        )(subj)
    else:
        subj = tio.RandomAffine(
            scales=0, degrees=0,
            translation=(-20, 20, 0, 0, 0, 0),
            isotropic=False, center="image",
        )(subj)
    z_aug = _blur(_from_subj(subj))
    return _from_zscore(z_aug, img01)


def simulate_banding(img01: torch.Tensor, sev: int = 1) -> torch.Tensor:
    """
    Banding UPF : décalage additif moyen sur une bande aléatoire du volume.
      sev=1 : mean_noise = U(1.0, 2.0) σ, std=0.1 σ, bande de 10–50 vx
      sev=2 : mean_noise = U(2.0, 3.0) σ, std=0.1 σ, bande de 10–50 vx
    L'axe de la bande est choisi aléatoirement (D, H ou W).
    """
    z      = _to_zscore(img01).clone()   # [1, D, H, W]
    ax     = np.random.randint(1, 4)     # 1=D, 2=H, 3=W
    dim_sz = z.shape[ax]
    max_bsz = max(1, min(50, dim_sz // 3))
    min_bsz = max(1, min(10, max_bsz))
    bsz     = np.random.randint(min_bsz, max_bsz + 1)
    bstart  = np.random.randint(0, max(1, dim_sz - bsz + 1))

    sl = [slice(None)] * 4
    sl[ax] = slice(bstart, bstart + bsz)

    mean_noise = (np.random.uniform(1.0, 2.0) if sev == 1
                  else np.random.uniform(2.0, 3.0))
    band_np = z[tuple(sl)].numpy()
    noise   = np.random.normal(mean_noise, 0.1, band_np.shape).astype(np.float32)
    z[tuple(sl)] = torch.from_numpy(band_np + noise)
    z_aug = _blur(z)
    return _from_zscore(z_aug, img01)


def simulate_motion(img01: torch.Tensor, sev: int = 1) -> torch.Tensor:
    """
    Motion UPF/Sundaresan 2024 :
      sev=1 : RandomMotion(degrees=10, translation=(0,3),  num_transforms=2)
      sev=2 : RandomMotion(degrees=20, translation=(0,7),  num_transforms=4)
    """
    z    = _to_zscore(img01)
    subj = _to_subj(z)
    if sev == 1:
        subj = tio.RandomMotion(degrees=10, translation=(0, 3),  num_transforms=2)(subj)
    else:
        subj = tio.RandomMotion(degrees=20, translation=(0, 7),  num_transforms=4)(subj)
    z_aug = _blur(_from_subj(subj))
    return _from_zscore(z_aug, img01)


def simulate_contrast(img01: torch.Tensor, sev: int = 1) -> torch.Tensor:
    """
    Contrast UPF (sur masque cerveau uniquement, approximé sur tout le volume):
      sev=1 : RandomGamma(0.1–0.2) + RandomBiasField(-0.05–0.05, ord=3)
              + rescaling vers [0.3, 0.7] du range z-score
      sev=2 : RandomGamma(0.2–0.3) + RandomBiasField(-0.3–0.3, ord=4)
              + rescaling vers [0.4, 0.6] + compression histogramme (×0.4)
    """
    z    = _to_zscore(img01)
    subj = _to_subj(z)

    if sev == 1:
        subj = tio.Compose([
            tio.RandomGamma(log_gamma=(0.1, 0.2)),
            tio.RandomBiasField(coefficients=(-0.05, 0.05), order=3),
        ])(subj)
        x  = _from_subj(subj).numpy()
        lo, hi = float(x.min()), float(x.max())
        if hi > lo:
            x = 0.3 + (x - lo) / (hi - lo + 1e-8) * 0.4 * (hi - lo)
    else:
        subj = tio.Compose([
            tio.RandomGamma(log_gamma=(0.2, 0.3)),
            tio.RandomBiasField(coefficients=(-0.3, 0.3), order=4),
        ])(subj)
        x  = _from_subj(subj).numpy()
        lo, hi = float(x.min()), float(x.max())
        if hi > lo:
            x = 0.4 + (x - lo) / (hi - lo + 1e-8) * 0.2 * (hi - lo)
        mean_val = float(x.mean())
        x = mean_val + (x - mean_val) * 0.4   # compression histogramme

    z_aug = _blur(torch.from_numpy(x.astype(np.float32)))
    return _from_zscore(z_aug, img01)


def simulate_distortion(img01: torch.Tensor, sev: int = 1) -> torch.Tensor:
    """
    Distortion UPF : MRIDistortionSimulator.
      sev=1 (mild)   : elastic(4pt, 8mm) + BiasField(0.2, p=0.7)
                       + Ghosting(0.1, p=0.4) + Spike(0.1, p=0.3)
      sev=2 (severe) : elastic(6pt, 15mm) + BiasField(0.4, p=0.7)
                       + Ghosting(0.2, p=0.4) + Spike(0.2, p=0.3)
    """
    z       = _to_zscore(img01)
    vol     = z.squeeze(0).numpy()
    sev_str = "mild" if sev == 1 else "severe"
    aug     = MRIDistortionSimulator(severity=sev_str).apply_distortion(vol)
    z_aug   = _blur(torch.from_numpy(aug[None].astype(np.float32)))
    return _from_zscore(z_aug, img01)


# ──────────────────────────────────────────────────────────────────────────────
# Probabilités d'application par artefact (calibrées pour LISA 2026)
# ──────────────────────────────────────────────────────────────────────────────
# Format : (p_sev2, p_sev1_ou_sev2)
# On simule uniquement sur les slots vides (label == 0) pour ne pas écraser
# les annotations réelles.

_ARTIFACT_PROBS = {
    "Noise":       (0.12, 0.35),
    "Zipper":      (0.10, 0.28),
    "Positioning": (0.00, 0.00),   # non simulé (besoin info positionnement)
    "Banding":     (0.10, 0.30),
    "Motion":      (0.10, 0.25),
    "Contrast":    (0.10, 0.25),
    "Distortion":  (0.08, 0.20),
}

_SIMULATE_FN = {
    "Noise":      simulate_noise,
    "Zipper":     simulate_zipper,
    "Banding":    simulate_banding,
    "Motion":     simulate_motion,
    "Contrast":   simulate_contrast,
    "Distortion": simulate_distortion,
}


def augment_artifact(
    img01:  torch.Tensor,   # [1, D, H, W] float32, valeurs dans [0, 1]
    labels: np.ndarray,     # [N_ARTIFACTS] int64, valeurs 0/1/2
) -> tuple:
    """
    Applique des artefacts simulés (TorchIO, paramètres UPF/Sundaresan 2024)
    uniquement sur les slots où ``label == 0`` (aucun artefact réel annoté).

    Retourne (aug_image, new_labels).

    Usage dans __getitem__ :
        if self.simulate_artifacts and self.split == 'train':
            image, task1a_labels = augment_artifact(image, task1a_labels)
    """
    img        = img01
    new_labels = labels.copy()

    for a, name in enumerate(ARTIFACT_NAMES):
        if new_labels[a] != 0:
            continue   # artefact réel présent → on ne simule pas par-dessus
        p2, p12 = _ARTIFACT_PROBS.get(name, (0.0, 0.0))
        if p2 == 0.0 and p12 == 0.0:
            continue   # Positioning : non simulé

        fn = _SIMULATE_FN.get(name)
        if fn is None:
            continue

        r = torch.rand(1).item()
        if r < p2:
            img = fn(img, sev=2)
            new_labels[a] = 2
        elif r < p12:
            img = fn(img, sev=1)
            new_labels[a] = 1

    return img, new_labels


# ──────────────────────────────────────────────────────────────────────────────
# Augmentations géométriques légères (batch-level, pour train.py)
# ──────────────────────────────────────────────────────────────────────────────

def augment_geometric(x: torch.Tensor) -> torch.Tensor:
    """
    Augmentations légères sur un batch [B, 1, D, H, W] ∈ [0, 1].

      - Flip gauche-droite (axe W, dim=4), p=0.5
      - Bruit gaussien additif, σ ~ U(0, 0.02)
      - Scaling d'intensité, scale ~ U(0.9, 1.1)

    Ces augmentations géométriques et d'intensité légères sont complémentaires
    à la simulation d'artefacts (augment_artifact) qui agit au niveau item.
    """
    if torch.rand(1).item() < 0.5:
        x = x.flip(4)
    sigma = torch.rand(1).item() * 0.02
    x     = x + sigma * torch.randn_like(x)
    scale = 0.9 + torch.rand(1).item() * 0.2
    return (x * scale).clamp(0.0, 1.0)
