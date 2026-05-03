#!/usr/bin/env python3
"""
Dataset joint pour le challenge LISA 2026.

Chaque item est un dict contenant :
  image           : FloatTensor [1, D, H, W]  normalisé dans [0,1]
  subject         : str
  orientation     : str  ('axi' | 'cor' | 'sag' | 'ciso')
  is_isotropic    : bool  (True uniquement pour 'ciso')
  has_task1a      : bool
  task1a_labels   : LongTensor [7]  sévérité 0/1/2 par artefact (ou zéros)
  is_artifact_free: bool  (présent dans Task_1b_NoNoise_NoMotion.csv)
  has_seg         : bool
  seg             : LongTensor [D, H, W]  0=fond, 1-13=structures (ou zéros)
"""

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT_DEFAULT = "/home/rousseau/Data/LISA2026"
TARGET_SIZE       = (96, 96, 96)   # taille fixe en entraînement
TARGET_SPACING_MM = 1.0            # résolution isotrope cible (mm)

ARTIFACT_COLS = [
    "Noise", "Zipper", "Positioning",
    "Banding", "Motion", "Contrast", "Distortion",
]
N_ARTIFACTS    = 7
N_SEG_CLASSES  = 14   # 0 = fond, 1–13 = structures

# Regex pour fichiers LF anisotropes (axi/cor/sag)
_FILE_RE = re.compile(
    r"^(LISA_(?:VALIDATION_)?(\d+))_LF_(axi|cor|sag)\.nii\.gz$"
)
# Regex pour ciso (IDs numériques uniquement – pas de VALIDATION pour les ciso annotés)
_CISO_RE = re.compile(r"^(LISA_(\d+))_ciso\.nii\.gz$")


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires de traitement d'image
# ──────────────────────────────────────────────────────────────────────────────

def load_nifti(path: str):
    """Charge un NIfTI → (data float32, zooms np.array(3))."""
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    return data, zooms


def resample_to_isotropic(data: np.ndarray, zooms: np.ndarray,
                           order: int = 1) -> np.ndarray:
    """Resampling vers TARGET_SPACING_MM mm isotrope via scipy.ndimage.zoom."""
    factors = zooms / TARGET_SPACING_MM
    if np.allclose(factors, 1.0, atol=0.02):
        return data
    return zoom(data, factors, order=order, mode="nearest")


def crop_or_pad(data: np.ndarray, target_size: tuple) -> np.ndarray:
    """Crop centré ou zero-padding pour atteindre target_size (D, H, W)."""
    result = np.zeros(target_size, dtype=data.dtype)
    src_slices, dst_slices = [], []
    for i, (s, t) in enumerate(zip(data.shape, target_size)):
        if s >= t:
            start = (s - t) // 2
            src_slices.append(slice(start, start + t))
            dst_slices.append(slice(0, t))
        else:
            pad = (t - s) // 2
            src_slices.append(slice(0, s))
            dst_slices.append(slice(pad, pad + s))
    result[tuple(dst_slices)] = data[tuple(src_slices)]
    return result


def normalize(data: np.ndarray) -> np.ndarray:
    """Clip au 99e percentile des voxels non nuls, puis scale → [0, 1]."""
    nonzero = data[data > 0]
    if len(nonzero) == 0:
        return data
    p99 = np.percentile(nonzero, 99)
    if p99 <= 0:
        return data
    return np.clip(data, 0, p99) / p99


import torch as _torch  # noqa: E402 — import local pour éviter une dépendance circulaire


def compute_task1a_weights(
    items: list,
    n_artifacts: int = N_ARTIFACTS,
    n_severity:  int = 3,
) -> tuple:
    """
    Calcule deux tenseurs de pondération pour corriger le déséquilibre de la
    loss Task 1a :

    sev_weights : Tensor [N_art, N_sev]
        Poids par sévérité pour chaque artefact (inverse de fréquence,
        normalisé par artefact → somme = N_sev).
        À passer comme ``weight`` à ``F.cross_entropy``.

    art_weights : Tensor [N_art]
        Poids par type d'artefact (inverse du taux d'activité, normalisé
        → moyenne = 1).
        Les artefacts rares (ex. Banding) obtiennent un poids plus élevé.

    Args:
        items       : liste des items du split d'entraînement (``train_ds.items``)
        n_artifacts : nombre de types d'artefacts (défaut : 7)
        n_severity  : nombre de classes de sévérité (défaut : 3)

    Returns:
        (sev_weights, art_weights)
    """
    task1a_items = [it for it in items if it.get("has_task1a", False)]
    N = len(task1a_items)

    if N == 0:
        return (
            _torch.ones(n_artifacts, n_severity, dtype=_torch.float32),
            _torch.ones(n_artifacts,              dtype=_torch.float32),
        )

    labels = np.stack([it["task1a_labels"] for it in task1a_items])
    # labels : [N, N_art], valeurs 0 / 1 / 2

    # ── poids de sévérité (intra-artefact) ───────────────────────────────────
    sev_w = np.zeros((n_artifacts, n_severity), dtype=np.float32)
    for a in range(n_artifacts):
        for s in range(n_severity):
            cnt = int((labels[:, a] == s).sum())
            sev_w[a, s] = N / (max(cnt, 1) * n_severity)
        # normaliser → somme = n_severity (convention PyTorch "weight" pour CE)
        sev_w[a] *= n_severity / sev_w[a].sum()
    # Plancher : évite les poids quasi-nuls (ex. Banding sev=0 → 0.02)
    # qui bloquent le gradient et provoquent des modes dégénérés.
    sev_w = np.clip(sev_w, a_min=0.5, a_max=None)
    # Renormaliser après clamp pour conserver la convention somme = n_severity
    for a in range(n_artifacts):
        sev_w[a] *= n_severity / sev_w[a].sum()

    # ── poids inter-artefact ──────────────────────────────────────────────────
    art_w = np.zeros(n_artifacts, dtype=np.float32)
    for a in range(n_artifacts):
        active = int((labels[:, a] > 0).sum())
        art_w[a] = N / max(active, 1) / n_artifacts
    # normaliser → moyenne = 1
    art_w = art_w / art_w.mean()

    return (
        _torch.from_numpy(sev_w),
        _torch.from_numpy(art_w),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class LISAJointDataset(Dataset):
    """Dataset joint couvrant toutes les données du challenge LISA 2026."""

    def __init__(
        self,
        data_root:   str   = DATA_ROOT_DEFAULT,
        target_size: tuple = TARGET_SIZE,
        split:       str   = "train",   # 'train' | 'val' | 'all'
        val_fraction: float = 0.2,
        simulate_artifacts: bool = False,
    ):
        self.data_root          = Path(data_root)
        self.target_size        = target_size
        self.split              = split
        self.simulate_artifacts = simulate_artifacts
        self.items: list[dict] = []
        self._build_index()
        self._apply_split(split, val_fraction)

    # ── construction de l'index ───────────────────────────────────────────────

    def _build_index(self):
        dr = self.data_root

        # --- chargement des CSVs ---
        task1a_df = pd.read_csv(dr / "LISA_Task1a_2026.csv").set_index("filename")
        artifact_free_set = set(
            pd.read_csv(dr / "Task_1b_NoNoise_NoMotion.csv")["filename"].tolist()
        )

        # --- images anisotropes (axi / cor / sag) ---
        for f in sorted(dr.iterdir()):
            m = _FILE_RE.match(f.name)
            if not m:
                continue
            subject = m.group(1)
            sid     = int(m.group(2))
            orient  = m.group(3)

            fname    = f.name
            has1a    = fname in task1a_df.index
            t1a_labs = np.zeros(N_ARTIFACTS, dtype=np.int64)
            if has1a:
                t1a_labs = (
                    task1a_df.loc[fname, ARTIFACT_COLS].values.astype(np.int64)
                )

            self.items.append({
                "filepath":        str(f),
                "subject":         subject,
                "sid":             sid,
                "orientation":     orient,
                "is_isotropic":    False,
                "has_task1a":      has1a,
                "task1a_labels":   t1a_labs,
                "is_artifact_free": fname in artifact_free_set,
                "has_seg":         False,
                "seg_filepath":    None,
            })

        # --- images isotropes (ciso) ---
        for f in sorted(dr.iterdir()):
            m2 = _CISO_RE.match(f.name)
            if not m2:
                continue
            subject  = m2.group(1)
            sid      = int(m2.group(2))
            seg_path = dr / f"{subject}_LF_seg.nii.gz"

            self.items.append({
                "filepath":        str(f),
                "subject":         subject,
                "sid":             sid,
                "orientation":     "ciso",
                "is_isotropic":    True,
                "has_task1a":      False,
                "task1a_labels":   np.zeros(N_ARTIFACTS, dtype=np.int64),
                "is_artifact_free": False,
                "has_seg":         seg_path.exists(),
                "seg_filepath":    str(seg_path) if seg_path.exists() else None,
            })

    # ── split train / val ─────────────────────────────────────────────────────

    def _apply_split(self, split: str, val_fraction: float):
        """
        Split train/val par sujet, stratifié par artefact.

        Pour chaque type d'artefact, au moins ceil(val_fraction * n_actifs)
        sujets actifs sont réservés dans la val, ce qui garantit que tous les
        artefacts présents dans les données sont représentés en validation
        (évite qu'un artefact rare, ex. Banding, se retrouve entièrement dans
        le train à cause d'un simple tri par ID).

        Algorithme :
          1. Pour chaque artefact, trier les sujets actifs et en prélever
             ceil(val_fraction) dans la val (échantillonnage uniforme).
          2. Compléter la val avec des sujets neutres (sans aucun artefact actif)
             pour atteindre le quota global val_fraction * n_sujets.
          3. Tous les sujets VALIDATION sont exclus des deux splits.
        """
        if split == "all":
            return

        import math

        # Sujets supervisés (hors set VALIDATION et hors ciso-only)
        supervised_subjects = sorted({
            it["subject"]
            for it in self.items
            if 1 <= it["sid"] <= 1999 and "VALIDATION" not in it["subject"]
        })
        n_total = len(supervised_subjects)
        n_val_target = max(1, int(val_fraction * n_total))

        # Carte sujet → labels Task1a (on prend les 3 orientations aniso)
        subj_labels: dict[str, np.ndarray] = {}
        for it in self.items:
            if it["has_task1a"] and it["subject"] in set(supervised_subjects):
                if it["subject"] not in subj_labels:
                    subj_labels[it["subject"]] = it["task1a_labels"].copy()
                else:
                    # max par artefact (même sujet, 3 orientations identiques)
                    subj_labels[it["subject"]] = np.maximum(
                        subj_labels[it["subject"]], it["task1a_labels"]
                    )

        val_subjects: set[str] = set()

        # ── 1. garantir la représentation de chaque artefact ─────────────────
        for a in range(N_ARTIFACTS):
            actifs = sorted(
                s for s, lbl in subj_labels.items() if lbl[a] > 0
            )
            if not actifs:
                continue
            n_val_art = max(1, math.ceil(val_fraction * len(actifs)))
            # prélèvement uniforme pour couvrir toute la gamme des sévérités
            step = max(1, len(actifs) // n_val_art)
            for k in range(0, len(actifs), step):
                if len(val_subjects) < n_val_target:
                    val_subjects.add(actifs[k])

        # ── 2. compléter avec des sujets neutres ──────────────────────────────
        all_active = {s for s, lbl in subj_labels.items() if lbl.any()}
        neutral = [s for s in supervised_subjects if s not in all_active]
        # on complète en prenant les derniers (tri par ID croissant → stable)
        for s in reversed(neutral):
            if len(val_subjects) >= n_val_target:
                break
            val_subjects.add(s)

        # ── 3. filtrage ───────────────────────────────────────────────────────
        if split == "train":
            self.items = [
                it for it in self.items if it["subject"] not in val_subjects
            ]
        elif split == "val":
            self.items = [
                it for it in self.items if it["subject"] in val_subjects
            ]

    # ── __len__ / __getitem__ ─────────────────────────────────────────────────

    def __len__(self):
        return len(self.items)

    # ── Simulation d'artefacts (TorchIO) ─────────────────────────────────────

    @staticmethod
    def _tio_subject(image: torch.Tensor) -> tio.Subject:
        """Crée un Subject TorchIO depuis un tenseur [1, D, H, W].
        TorchIO attend [C, W, H, D], on permute avant et après."""
        return tio.Subject(image=tio.ScalarImage(tensor=image.permute(0, 3, 2, 1)))

    @staticmethod
    def _from_subject(subj: tio.Subject) -> torch.Tensor:
        """Reconvertit Subject TorchIO → tenseur [1, D, H, W]."""
        return subj.image.data.permute(0, 3, 2, 1).float().clamp(0.0, 1.0)

    @staticmethod
    def _simulate_zipper(
        image: torch.Tensor, sev: int
    ) -> torch.Tensor:
        """
        Simule l'artefact Zipper : bandes étroites discrètes (bright ou dark)
        perpéndiculàres à l'axe d'encodage de phase.

        Inspiré de l'équipe UPF LISA 2025 (ZipperArtifactSimulator) :
          sev1 : 2-4 bandes, amplitude 0.2-0.4 (×std image)
          sev2 : 4-8 bandes, amplitude 0.4-0.7 (×std image)

        Le vrai artefact Zipper en IRM apparaît comme des raies claires/sombres
        parallèles (interférence RF extérieure ou spike k-space discret).
        """
        vol = image[0].numpy().copy().astype(np.float64)  # [D, H, W]
        D, H, W = vol.shape

        direction = np.random.choice(['W', 'H'])
        n_bands   = np.random.randint(2, 5) if sev == 1 else np.random.randint(4, 9)
        img_std   = float(np.std(vol))
        amp       = np.random.uniform(0.20, 0.40) if sev == 1 else np.random.uniform(0.40, 0.70)
        amplitude = max(amp * img_std, 0.02)

        # Positions des bandes constantes sur toutes les slices (caractéristique du Zipper)
        coord_size    = W if direction == 'W' else H
        band_thickness = max(1, int(coord_size * 0.012))
        positions = [np.random.randint(0, max(1, coord_size - band_thickness)) for _ in range(n_bands)]
        signs     = [1.0 if np.random.rand() > 0.4 else -1.0 for _ in range(n_bands)]

        for pos, sign in zip(positions, signs):
            noise = np.random.normal(0, amplitude * 0.15, vol.shape)
            if direction == 'W':
                vol[:, :, pos:pos + band_thickness] += sign * amplitude + noise[:, :, pos:pos + band_thickness]
            else:
                vol[:, pos:pos + band_thickness, :] += sign * amplitude + noise[:, pos:pos + band_thickness, :]

        return torch.from_numpy(np.clip(vol, 0.0, 1.0).astype(np.float32)).unsqueeze(0)

    @staticmethod
    def _simulate_banding(
        image: torch.Tensor, sev: int
    ) -> torch.Tensor:
        """
        Simule l'artefact Banding : bandes localisées de sur/sous-intensité.

        Calibration sur données LISA (images [0,1], std≈0.18) :
          sev1 : 1-2 bandes, amplitude ±15-30 % dans 8-20 % de la dimension
          sev2 : 2-4 bandes, amplitude ±30-55 % dans 8-20 % de la dimension

        L'artefact Banding en IRM ULF (64mT) apparaît comme une ou plusieurs
        bandes d'intensité uniformément sombre/lumineuse perpendiculaires à
        l'axe de lecture (inhomogénéité B0/B1). L'implémentation UPF (LISA 2025)
        remplace la région par du bruit haute intensité ; ici on utilise une
        modulation multiplicative lisse (transition gaussienne aux bords) qui
        préserve la structure anatomique tout en créant un contraste local.
        """
        img = image.numpy().copy()  # [1, D, H, W]

        n_bands    = np.random.randint(1, 3) if sev == 1 else np.random.randint(2, 5)
        width_frac = np.random.uniform(0.08, 0.20)
        amp        = np.random.uniform(0.15, 0.30) if sev == 1 else np.random.uniform(0.30, 0.55)

        # Axe de banding : 1=D (axial), 2=H (coronal), 3=W (sagittal)
        ax  = np.random.randint(1, 4)
        dim = img.shape[ax]
        band_w = max(3, int(dim * width_frac))

        for _ in range(n_bands):
            pos = np.random.randint(0, max(1, dim - band_w))
            # Profil gaussien (transition douce aux bords de la bande)
            center    = pos + band_w / 2.0
            sigma     = band_w / 4.0
            coords    = np.arange(dim)
            envelope  = np.exp(-0.5 * ((coords - center) / sigma) ** 2)

            # Amplitude finale : bande sombre ou lumineuse
            sign      = 1.0 if np.random.rand() > 0.5 else -1.0
            modulator = 1.0 + sign * amp * envelope  # shape [dim]

            # Broadcast selon l'axe choisi
            shape = [1, 1, 1, 1]
            shape[ax] = dim
            modulator = modulator.reshape(shape)
            img = np.clip(img * modulator, 0.0, 1.0)

        return torch.from_numpy(img.astype(np.float32))

    @staticmethod
    def _simulate_contrast(
        image: torch.Tensor, sev: int
    ) -> torch.Tensor:
        """
        Simule l'artefact Contrast : réduction de la dynamique + biais de champ.

        Calibration sur données LISA (images [0,1], std≈0.18, mean≈0.44) :
          - real Contrast sev>0 : std≈0.162, mean≈0.514 (range plus étroit, décalé)
          sev1 : RandomGamma(±0.3) + RandomBiasField(0.2) + compression 65 %
          sev2 : RandomGamma(±0.5) + RandomBiasField(0.35) + compression 40 %

        L'artefact Contrast en IRM ULF se manifeste par un manque de contraste
        T1/T2 : les structures semblent « lavées », la dynamique est réduite,
        et une inhomogénéité de champ ajoute un gradient d'intensité.
        """
        subj = LISAJointDataset._tio_subject(image)

        if sev == 1:
            subj = tio.RandomGamma(log_gamma=(-0.3, 0.3))(subj)
            subj = tio.RandomBiasField(coefficients=0.20)(subj)
            scale = np.random.uniform(0.55, 0.75)
        else:
            subj = tio.RandomGamma(log_gamma=(-0.5, 0.5))(subj)
            subj = tio.RandomBiasField(coefficients=0.35)(subj)
            scale = np.random.uniform(0.30, 0.50)

        img_out = subj.image.data.permute(0, 3, 2, 1).float()
        # Compression du range autour d'un pivot légèrement au-dessus de la moyenne
        # (reproduit le décalage de mean observé dans les vraies images de Contrast)
        pivot = img_out.mean() + 0.04
        img_out = (pivot + (img_out - pivot) * scale).clamp(0.0, 1.0)
        return img_out

    def _simulate_artifact(
        self,
        image:  torch.Tensor,  # [1, D, H, W] float32, valeurs dans [0, 1]
        labels: np.ndarray,    # [N_art] int64, valeurs 0/1/2
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Applique des transformations simulant des artefacts MRI.
        Seuls les artefacts dont le label courant est 0 sont candidats.

        Mapping ARTIFACT_COLS :
          0=Noise  1=Zipper  2=Positioning  3=Banding  4=Motion  5=Contrast  6=Distortion

        Sources :
          - Noise/Zipper/Motion : paramètres BRIQA (LISA 2025, 2ème place)
            BRIQA sev1 Motion = 5°, sev2 = 10° (vs Sundaresan 2024 : 3°/7°)
          - Banding : bandes localisées, inspiré UPF (LISA 2025, 5ème place)
          - Contrast : RandomGamma + RandomBiasField + rescaling, inspiré UPF
          - Distortion : elastic + ghosting + spike, inspiré UPF
        """
        new_labels = labels.copy()

        # ── Noise (idx=0) ─────────────────────────────────────────────────────
        # Calibration BRIQA/Sundaresan : leurs images sont z-score (std≈1),
        # std_noise=0.18-0.28 → équivalent [0,1] (std≈0.18) : std≈0.03-0.05
        # Real Noise sev>0 dans LISA : std≈0.170 (< clean 0.179) → bruit réduit le contraste
        # On ajoute donc un bruit modéré + légère dérivée de RandomBlur pour les sev élevés
        if labels[0] == 0:
            r = torch.rand(1).item()
            if r < 0.12:    # sev=2
                subj = self._tio_subject(image)
                subj = tio.RandomNoise(std=(0.04, 0.08))(subj)
                image = self._from_subject(subj)
                new_labels[0] = 2
            elif r < 0.35:  # sev=1
                subj = self._tio_subject(image)
                subj = tio.RandomNoise(std=(0.015, 0.04))(subj)
                image = self._from_subject(subj)
                new_labels[0] = 1

        # ── Motion (idx=4) ────────────────────────────────────────────────────
        # Paramètres BRIQA : sev1=5°/2mm, sev2=10°/5mm
        # UPF utilise des valeurs plus fortes (10°/20°) mais BRIQA est plus proche
        # de ce qui est observé dans ces IRM basse résolution
        if labels[4] == 0:
            r = torch.rand(1).item()
            if r < 0.10:    # sev=2 : 7-10°
                subj = self._tio_subject(image)
                subj = tio.RandomMotion(
                    degrees=10, translation=5, num_transforms=3
                )(subj)
                image = self._from_subject(subj)
                new_labels[4] = 2
            elif r < 0.25:  # sev=1 : 3-5°
                subj = self._tio_subject(image)
                subj = tio.RandomMotion(
                    degrees=5, translation=2, num_transforms=2
                )(subj)
                image = self._from_subject(subj)
                new_labels[4] = 1

        # ── Zipper / Spike k-space (idx=1) ────────────────────────────────────
        # Bandes discrètes perpéndiculàres à l'axe de phase (UPF ZipperArtifactSimulator).
        # TorchIO RandomSpike crée un motif en damier (spike k-space = sinusoide 2D)
        # qui ne représente pas correctement le zipper LISA (bandes parallèles discrètes).
        if labels[1] == 0:
            r = torch.rand(1).item()
            if r < 0.10:    # sev=2
                image = self._simulate_zipper(image, sev=2)
                new_labels[1] = 2
            elif r < 0.28:  # sev=1
                image = self._simulate_zipper(image, sev=1)
                new_labels[1] = 1

        # ── Banding (idx=3) ───────────────────────────────────────────────────
        # Bandes localisées (UPF) — PLUS RÉALISTE que RandomBiasField global
        # Corrige le problème de v5 : le modèle ne voyait jamais de Banding sev>0
        if labels[3] == 0:
            r = torch.rand(1).item()
            if r < 0.10:    # sev=2
                image = self._simulate_banding(image, sev=2)
                new_labels[3] = 2
            elif r < 0.30:  # sev=1
                image = self._simulate_banding(image, sev=1)
                new_labels[3] = 1

        # ── Contrast (idx=5) ──────────────────────────────────────────────────
        # RandomGamma + RandomBiasField + compression du range (UPF)
        if labels[5] == 0:
            r = torch.rand(1).item()
            if r < 0.10:    # sev=2
                image = self._simulate_contrast(image, sev=2)
                new_labels[5] = 2
            elif r < 0.25:  # sev=1
                image = self._simulate_contrast(image, sev=1)
                new_labels[5] = 1

        # ── Distortion / ElasticDeformation (idx=6) ───────────────────────────
        # Elastic + Ghosting + Spike (UPF) — plus réaliste que elastic seul
        if labels[6] == 0:
            r = torch.rand(1).item()
            if r < 0.08:    # sev=2
                subj = self._tio_subject(image)
                subj = tio.Compose([
                    tio.RandomElasticDeformation(num_control_points=7, max_displacement=15.0),
                    tio.RandomBiasField(coefficients=0.3, p=0.7),
                    tio.RandomGhosting(num_ghosts=(1, 2), intensity=(0.1, 0.2), p=0.4),
                    tio.RandomSpike(num_spikes=1, intensity=(0.1, 0.2), p=0.3),
                ])(subj)
                image = self._from_subject(subj)
                new_labels[6] = 2
            elif r < 0.20:  # sev=1
                subj = self._tio_subject(image)
                subj = tio.Compose([
                    tio.RandomElasticDeformation(num_control_points=5, max_displacement=8.0),
                    tio.RandomBiasField(coefficients=0.15, p=0.7),
                    tio.RandomGhosting(num_ghosts=(1, 2), intensity=(0.05, 0.1), p=0.4),
                ])(subj)
                image = self._from_subject(subj)
                new_labels[6] = 1

        # Repositioning (idx=2) : non simulable sans infos de positionnement, ignoré.

        return image, new_labels

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]

        # chargement et prétraitement de l'image — pleine résolution
        data, zooms = load_nifti(it["filepath"])
        data = resample_to_isotropic(data, zooms, order=1)
        data = normalize(data)
        image = torch.from_numpy(data[None])  # [1, D, H, W] pleine résolution

        # ── Simulation d'artefacts sur le volume pleine résolution (avant crop) ─
        has_task1a    = it["has_task1a"]
        task1a_labels = it["task1a_labels"].copy()   # np.ndarray [N_art]

        if self.simulate_artifacts and self.split == "train":
            image, task1a_labels = self._simulate_artifact(image, task1a_labels)
            # Activer has_task1a si des artefacts ont été simulés sur une image
            # qui n'était pas dans le CSV task1a (labels synthétiques valides)
            if not has_task1a and task1a_labels.any():
                has_task1a = True
            # Si l'image était déjà annotée, ses labels synthétiques s'ajoutent
            # aux vrais labels (on ne simule que les artefacts absents : label==0)

        # crop/pad vers la taille cible (après simulation)
        data = image.squeeze(0).numpy()
        data = crop_or_pad(data, self.target_size)
        image = torch.from_numpy(data[None])  # [1, D, H, W]

        # segmentation (images ciso uniquement)
        seg = torch.zeros(self.target_size, dtype=torch.long)
        if it["has_seg"] and it["seg_filepath"]:
            seg_data, seg_zooms = load_nifti(it["seg_filepath"])
            seg_data = resample_to_isotropic(seg_data, seg_zooms, order=0)
            seg_data = crop_or_pad(seg_data.astype(np.int32), self.target_size)
            seg = torch.from_numpy(seg_data).long()

        return {
            "image":            image,
            "subject":          it["subject"],
            "orientation":      it["orientation"],
            "is_isotropic":     torch.tensor(it["is_isotropic"],    dtype=torch.bool),
            "has_task1a":       torch.tensor(has_task1a,            dtype=torch.bool),
            "task1a_labels":    torch.from_numpy(task1a_labels),
            "is_artifact_free": torch.tensor(it["is_artifact_free"], dtype=torch.bool),
            "has_seg":          torch.tensor(it["has_seg"],          dtype=torch.bool),
            "seg":              seg,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    data_root:   str   = DATA_ROOT_DEFAULT,
    target_size: tuple = TARGET_SIZE,
    batch_size:  int   = 1,
    num_workers: int   = 2,
):
    train_ds = LISAJointDataset(data_root, target_size, split="train")
    val_ds   = LISAJointDataset(data_root, target_size, split="val")
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} items | Val: {len(val_ds)} items")
    return train_dl, val_dl
