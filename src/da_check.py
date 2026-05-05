#!/usr/bin/env python3
"""
Data augmentation visual check — LISA 2026 (paramètres UPF / Sundaresan 2024)

Génère 7 figures PNG de comparaison : propre | simulé | réel
pour chaque type d'artefact LISA (Noise, Zipper, Positioning, Banding,
Motion, Contrast, Distortion).

La simulation utilise les paramètres exacts de Sundaresan 2024 / UPF LISA 2025,
qui sont les paramètres de référence cités par BRIQA (2e place LISA 2025).
Toute la logique de simulation est dans augmentation.py.

Usage :
  cd /home/rousseau/Exp/lisa2026
  conda run -n mf python src/da_check.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_nifti, resample_to_isotropic, normalize  # noqa: E402
from augmentation import (                                          # noqa: E402
    simulate_noise, simulate_zipper, simulate_positioning,
    simulate_banding, simulate_motion, simulate_contrast,
    simulate_distortion,
)

# --- Chemins ------------------------------------------------------------------

home      = os.path.expanduser("~")
data_path = os.path.join(home, "Data", "LISA2026")

clean_image_file       = os.path.join(data_path, "LISA_0001_LF_axi.nii.gz")
noise_image_file       = os.path.join(data_path, "LISA_1041_LF_axi.nii.gz")
zipper_image_file      = os.path.join(data_path, "LISA_0033_LF_sag.nii.gz")
positioning_image_file = os.path.join(data_path, "LISA_0037_LF_cor.nii.gz")
banding_image_file     = os.path.join(data_path, "LISA_2005_LF_sag.nii.gz")
motion_image_file      = os.path.join(data_path, "LISA_0033_LF_cor.nii.gz")
contrast_image_file    = os.path.join(data_path, "LISA_2063_LF_cor.nii.gz")
distortion_image_file  = os.path.join(data_path, "LISA_0002_LF_cor.nii.gz")

OUT_DIR = ROOT / "results" / "plots"
SEED    = 42

ARTIFACTS = [
    ("Noise",       simulate_noise,       noise_image_file),
    ("Zipper",      simulate_zipper,      zipper_image_file),
    ("Positioning", simulate_positioning, positioning_image_file),
    ("Banding",     simulate_banding,     banding_image_file),
    ("Motion",      simulate_motion,      motion_image_file),
    ("Contrast",    simulate_contrast,    contrast_image_file),
    ("Distortion",  simulate_distortion,  distortion_image_file),
]


# --- Chargement ---------------------------------------------------------------

def load_vol(filepath):
    data, zooms = load_nifti(filepath)
    data = resample_to_isotropic(data, zooms, order=1)
    data = normalize(data)
    return torch.from_numpy(data[None].astype(np.float32))


# --- Visualisation ------------------------------------------------------------

def _slices(vol):
    D, H, W = vol.shape
    return vol[D // 2, :, :], vol[:, H // 2, :], vol[:, :, W // 2]


def _display_range(vol):
    nz = vol[vol > 0]
    lo = float(np.percentile(nz, 1)) if nz.size > 0 else 0.0
    hi = float(np.percentile(vol, 99))
    return lo, hi


def make_comparison_png(artifact_name, clean_img, sim_img, real_img,
                        out_dir, real_available=True):
    clean_vol = clean_img.squeeze(0).numpy()
    sim_vol   = sim_img.squeeze(0).numpy()
    real_vol  = real_img.squeeze(0).numpy()

    vlo, vhi = _display_range(clean_vol)

    col_titles = [
        "Propre (clean)",
        "Simule  --  " + artifact_name + "  (UPF/BRIQA)",
        "Reel (avec artefact)" if real_available else "Reel (non disponible)",
    ]
    col_colors = ["white", "#88ccff", "gold" if real_available else "#888888"]
    row_labels = ["Axiale", "Coronale", "Sagittale"]
    planes_list = [_slices(clean_vol), _slices(sim_vol), _slices(real_vol)]

    fig, axes = plt.subplots(
        3, 3, figsize=(13, 9), facecolor="#111111",
        gridspec_kw={"hspace": 0.05, "wspace": 0.04},
    )
    fig.suptitle(
        "Artefact : " + artifact_name +
        "   |   UPF/Sundaresan 2024 params   |   vol " + str(tuple(clean_vol.shape)),
        color="white", fontsize=12, fontweight="bold", y=0.99,
    )

    for row in range(3):
        for col, planes in enumerate(planes_list):
            ax = axes[row, col]
            ax.imshow(planes[row], cmap="gray", vmin=vlo, vmax=vhi,
                      interpolation="bilinear", aspect="equal", origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], color=col_colors[col],
                             fontsize=10, fontweight="bold", pad=5)
            if col == 0:
                ax.text(-0.04, 0.5, row_labels[row],
                        transform=ax.transAxes, va="center", ha="right",
                        color="white", fontsize=9, fontweight="bold", rotation=90)

    out_path = out_dir / ("da_" + artifact_name.lower() + ".png")
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  -> " + out_path.name)


# --- Main ---------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Simulation : UPF / Sundaresan 2024 (source BRIQA)")
    print("Chargement de l'image propre...")
    clean = load_vol(clean_image_file)
    print("  shape apres resample iso :", tuple(clean.shape))

    for art_name, sim_fn, real_file in ARTIFACTS:
        print("[" + art_name + "]", end="  ", flush=True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        print("simulation...", end=" ", flush=True)
        sim = sim_fn(clean, sev=1)

        real_ok = os.path.isfile(real_file)
        if real_ok:
            real = load_vol(real_file)
            print("reel", tuple(real.shape), end="  ", flush=True)
        else:
            print("MANQUANT (" + real_file + ")", end="  ", flush=True)
            real = torch.zeros_like(clean)

        make_comparison_png(art_name, clean, sim, real, OUT_DIR,
                            real_available=real_ok)

    print("Termine.")


if __name__ == "__main__":
    main()
