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
    ):
        self.data_root   = Path(data_root)
        self.target_size = target_size
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
        """Split 80/20 par sujet, stratifié par groupe de sujets avec GT."""
        if split == "all":
            return

        # groupes ayant de la supervision (train_seg et train_seg_hf)
        supervised_subjects = sorted({
            it["subject"]
            for it in self.items
            if 1 <= it["sid"] <= 1999 and "VALIDATION" not in it["subject"]
        })
        n_val = max(1, int(val_fraction * len(supervised_subjects)))
        val_subjects = set(supervised_subjects[-n_val:])

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

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]

        # chargement et prétraitement de l'image
        data, zooms = load_nifti(it["filepath"])
        data = resample_to_isotropic(data, zooms, order=1)
        data = crop_or_pad(data, self.target_size)
        data = normalize(data)
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
            "has_task1a":       torch.tensor(it["has_task1a"],      dtype=torch.bool),
            "task1a_labels":    torch.from_numpy(it["task1a_labels"]),
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
