#!/usr/bin/env python3
"""
Visualisation comparative des artefacts simulés vs réels — LISA 2026.

Génère un PNG par artefact (6 fichiers) :
  Colonnes : Original (propre) | Simulé sev=1 | Simulé sev=2 | Réel (sev>0)
  Lignes   : Axiale centrale | Coronale centrale | Sagittale centrale

Les simulations sont effectuées sur le volume pleine résolution (après
resampling isotropique 1mm), identiquement à ce qui est fait à l'entraînement.
Les coupes affichées sont celles du volume pleine résolution (non croppé).

Usage :
  cd /home/rousseau/Exp/lisa2026
  python src/visualize_artifacts.py \
      --data_root /home/rousseau/Data/LISA2026 \
      --out_dir results/plots/artifacts
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dataset import (  # noqa: E402
    LISAJointDataset,
    DATA_ROOT_DEFAULT,
    load_nifti,
    resample_to_isotropic,
    normalize,
)

SIM_ARTIFACTS = [
    (0, "Noise"),
    (1, "Zipper"),
    (3, "Banding"),
    (4, "Motion"),
    (5, "Contrast"),
    (6, "Distortion"),
]

FIXED_SEED = 42
COL_TITLES = ["Original (propre)", "Simulé sev=1", "Simulé sev=2", "Réel (sev>0)"]
ROW_LABELS = ["Axiale", "Coronale", "Sagittale"]


# ─────────────────────────────────────────────────────────────────────────────
# Chargement d'un volume NIfTI en pleine résolution (resampling + normalize)
# ─────────────────────────────────────────────────────────────────────────────

def _load_full_res(filepath: str) -> tuple:
    """
    Charge un NIfTI, resample à 1mm isotropique, normalise.
    Retourne (image_tensor [1,D,H,W], vol_numpy [D,H,W]).
    """
    data, zooms = load_nifti(filepath)
    data = resample_to_isotropic(data, zooms, order=1)
    data = normalize(data)
    vol  = data.astype(np.float32)
    img  = torch.from_numpy(vol[None])  # [1, D, H, W]
    return img, vol


# ─────────────────────────────────────────────────────────────────────────────
# Extraction des trois coupes centrales d'un volume [D, H, W]
# ─────────────────────────────────────────────────────────────────────────────

def _slices(vol: np.ndarray) -> tuple:
    """Renvoie (axiale, coronale, sagittale) — coupes centrales, float32 [0,1]."""
    D, H, W = vol.shape
    ax  = vol[D // 2, :, :]   # [H, W]
    cor = vol[:, H // 2, :]   # [D, W]
    sag = vol[:, :, W // 2]   # [D, H]
    return ax.astype(np.float32), cor.astype(np.float32), sag.astype(np.float32)


def _display_range(vol: np.ndarray) -> tuple:
    """Calcule vmin/vmax sur le percentile 1-99 du volume entier."""
    lo = float(np.percentile(vol, 1))
    hi = float(np.percentile(vol, 99))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Simulation forcée (seed fixe pour reproductibilité)
# ─────────────────────────────────────────────────────────────────────────────

def _force_simulate(art_idx: int, sev: int, image: torch.Tensor) -> np.ndarray:
    """
    Applique la simulation de l'artefact `art_idx` à sévérité `sev` sur `image`
    [1, D, H, W] pleine résolution.
    Retourne le volume simulé [D, H, W] numpy float32.
    """
    import torchio as tio

    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    random.seed(FIXED_SEED)

    if art_idx == 0:  # Noise
        std = (0.015, 0.04) if sev == 1 else (0.04, 0.08)
        subj = LISAJointDataset._tio_subject(image)
        subj = tio.RandomNoise(std=std)(subj)
        out = LISAJointDataset._from_subject(subj)

    elif art_idx == 1:  # Zipper — bandes discrètes (UPF)
        out = LISAJointDataset._simulate_zipper(image, sev=sev)

    elif art_idx == 3:  # Banding
        out = LISAJointDataset._simulate_banding(image, sev=sev)

    elif art_idx == 4:  # Motion (BRIQA : sev1=5 deg, sev2=10 deg)
        deg, tra, n = (5, 2, 2) if sev == 1 else (10, 5, 3)
        subj = LISAJointDataset._tio_subject(image)
        subj = tio.RandomMotion(degrees=deg, translation=tra, num_transforms=n)(subj)
        out = LISAJointDataset._from_subject(subj)

    elif art_idx == 5:  # Contrast
        out = LISAJointDataset._simulate_contrast(image, sev=sev)

    elif art_idx == 6:  # Distortion (UPF : elastic + biasfield + ghosting)
        subj = LISAJointDataset._tio_subject(image)
        if sev == 1:
            subj = tio.Compose([
                tio.RandomElasticDeformation(num_control_points=5, max_displacement=5.0),
                tio.RandomBiasField(coefficients=0.15, p=1.0),
                tio.RandomGhosting(num_ghosts=(1, 2), intensity=(0.05, 0.1), p=0.6),
            ])(subj)
        else:
            subj = tio.Compose([
                tio.RandomElasticDeformation(num_control_points=7, max_displacement=10.0),
                tio.RandomBiasField(coefficients=0.30, p=1.0),
                tio.RandomGhosting(num_ghosts=(1, 2), intensity=(0.1, 0.2), p=0.8),
                tio.RandomSpike(num_spikes=1, intensity=(0.1, 0.2), p=0.5),
            ])(subj)
        out = LISAJointDataset._from_subject(subj)

    else:
        out = image

    return out.squeeze(0).numpy()  # [D, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Generation d'un PNG par artefact
# ─────────────────────────────────────────────────────────────────────────────

def make_artifact_pngs(
    data_root: str      = DATA_ROOT_DEFAULT,
    out_dir: str        = None,
):
    """
    Pour chaque artefact simulable, genere un PNG 4 colonnes x 3 lignes.
    Les images sont chargees en pleine resolution (aucun crop), la simulation
    est appliquee sur le volume entier exactement comme a l'entrainement.
    """
    out_path = Path(out_dir) if out_dir else Path("results/plots/artifacts")
    out_path.mkdir(parents=True, exist_ok=True)

    # Phase 1 : indexation par metadonnees (pas de chargement NIfTI)
    print("Indexation des metadonnees...")
    ds = LISAJointDataset(
        data_root=data_root,
        target_size=(96, 96, 96),
        split="all",
        simulate_artifacts=False,
    )

    clean_per_art = {}
    real_per_art  = {}
    art_indices   = {a for a, _ in SIM_ARTIFACTS}

    for meta in ds.items:
        if not meta["has_task1a"]:
            continue
        lbl = meta["task1a_labels"]   # np.ndarray [N_art] depuis _build_index
        for art_idx, _ in SIM_ARTIFACTS:
            if art_idx not in clean_per_art and int(lbl[art_idx]) == 0 and int(lbl.sum()) == 0:
                clean_per_art[art_idx] = meta
            if art_idx not in real_per_art and int(lbl[art_idx]) > 0:
                real_per_art[art_idx] = meta
        if (len(clean_per_art) == len(art_indices)
                and len(real_per_art) == len(art_indices)):
            break

    # Phase 2 : un PNG par artefact
    for art_idx, art_name in SIM_ARTIFACTS:
        clean_meta = clean_per_art.get(art_idx)
        real_meta  = real_per_art.get(art_idx)

        if clean_meta is None:
            print(f"[{art_name}] Pas d'image propre disponible.")
            continue

        print(f"  [{art_name}] chargement pleine resolution...", end="", flush=True)

        clean_img, clean_vol = _load_full_res(clean_meta["filepath"])
        print(f" {clean_vol.shape}", end="", flush=True)

        sim1_vol = _force_simulate(art_idx, sev=1, image=clean_img)
        sim2_vol = _force_simulate(art_idx, sev=2, image=clean_img)

        if real_meta is not None:
            _, real_vol = _load_full_res(real_meta["filepath"])
            real_sev   = int(real_meta["task1a_labels"][art_idx])
            real_label = f"Reel sev={real_sev}"
        else:
            real_vol   = np.zeros_like(clean_vol)
            real_sev   = -1
            real_label = "Reel (non dispo)"

        vlo,  vhi  = _display_range(clean_vol)
        rvlo, rvhi = _display_range(real_vol) if real_meta else (0.0, 1.0)

        fig, axes = plt.subplots(
            3, 4,
            figsize=(16, 10),
            facecolor="#111111",
            gridspec_kw={"hspace": 0.06, "wspace": 0.04},
        )
        fig.suptitle(
            f"Artefact : {art_name}  --  volume pleine resolution {clean_vol.shape}",
            color="white", fontsize=13, fontweight="bold", y=0.99,
        )

        planes_clean = _slices(clean_vol)
        planes_sim1  = _slices(sim1_vol)
        planes_sim2  = _slices(sim2_vol)
        planes_real  = _slices(real_vol)

        col_specs = [
            (planes_clean, vlo,  vhi,  COL_TITLES[0]),
            (planes_sim1,  vlo,  vhi,  COL_TITLES[1]),
            (planes_sim2,  vlo,  vhi,  COL_TITLES[2]),
            (planes_real,  rvlo, rvhi, real_label),
        ]

        for row, plane_name in enumerate(ROW_LABELS):
            for col, (planes, vmin, vmax, ctitle) in enumerate(col_specs):
                ax = axes[row, col]
                ax.imshow(
                    planes[row],
                    cmap="gray",
                    vmin=vmin, vmax=vmax,
                    interpolation="bilinear",
                    aspect="equal",
                    origin="lower",
                )
                ax.axis("off")
                if row == 0:
                    col_color = "white" if col < 3 else "gold"
                    ax.set_title(ctitle, color=col_color, fontsize=10,
                                 fontweight="bold", pad=5)
                if col == 0:
                    ax.text(
                        -0.04, 0.5, plane_name,
                        transform=ax.transAxes,
                        va="center", ha="right",
                        color="white", fontsize=9, fontweight="bold",
                        rotation=90,
                    )

        fpath = out_path / f"{art_name.lower()}.png"
        plt.savefig(str(fpath), dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f" -> {fpath.name}")

    print("Termine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genere des PNG par artefact : propre / sim sev1 / sim sev2 / reel."
    )
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--out_dir",   default=None,
                        help="Repertoire de sortie (defaut: results/plots/artifacts)")
    args = parser.parse_args()

    make_artifact_pngs(
        data_root=args.data_root,
        out_dir=args.out_dir,
    )
