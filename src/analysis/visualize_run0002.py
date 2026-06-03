#!/usr/bin/env python3
"""Visualization script for RUN_0002 — CycleGAN Task 1b.

Generates mosaics showing: Original | Reconstructed | Residue
For categories: Clean, Noise, Motion.
Saves results as PNG plots.
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# Add project root to sys.path to allow imports from src.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.datasets.task1b import Task1bCycleGANDataset
from src.models.task1b import Generator3D


def normalize_image(img, percentile=98):
    """Normalize image to [0, 1] based on percentiles for better contrast."""
    img = img.astype(np.float32)
    low = np.percentile(img, 2)
    high = np.percentile(img, percentile)
    img = np.clip((img - low) / (high - low + 1e-5), 0, 1)
    return img


def get_central_slice(vol, axis=0):
    """Extract the central slice along the given axis."""
    shape = vol.shape
    idx = shape[axis] // 2
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]


def run_inference(model, device, img_path, dataset_transforms):
    """Run G_AB on a single image path and return volumes."""
    sample = dataset_transforms({"img": img_path})
    img = sample["img"].unsqueeze(0).to(device).float()
    with torch.no_grad():
        fake_B = model(img)
    orig_vol = img.squeeze().cpu().numpy()
    out_vol = fake_B.squeeze().cpu().numpy()
    residue_vol = np.abs(orig_vol - out_vol)
    return orig_vol, out_vol, residue_vol


def save_category_mosaics(model, device, img_paths, dataset_transforms,
                          category_name, subj_labels, output_dir,
                          domain_key="A"):
    """Generate mosaic plots for a list of image paths."""
    views = {'Axial': 0, 'Coronal': 1, 'Sagittal': 2}
    n_samples = min(len(img_paths), 5)  # max 5 samples per mosaic

    if n_samples == 0:
        print(f"[!] No samples for category '{category_name}', skipping.")
        return

    print(f"-- Generating {category_name} ({n_samples} samples) --")

    for view, axis_idx in views.items():
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = [axes]
        fig.suptitle(
            f"Category: {category_name} — {view} View",
            fontsize=14
        )

        for i, img_path in enumerate(img_paths[:n_samples]):
            m = re.search(r'(LISA_\d+)', Path(img_path).name)
            subj_id = m.group(1) if m else "unknown"
            labels = subj_labels.get(subj_id, {})
            label_str = f"N={labels.get('Noise', '?')} M={labels.get('Motion', '?')}"

            orig_vol, out_vol, residue_vol = run_inference(
                model, device, img_path, dataset_transforms
            )

            ax_row = axes[i]
            for j, (vol_slice, title, cmap) in enumerate([
                (orig_vol, f"{subj_id} ({label_str})\nOriginal", 'gray'),
                (out_vol, f"Reconstructed\n(domain A→B)", 'gray'),
                (residue_vol, f"Residue\n|orig - recon|", 'magma'),
            ]):
                s = get_central_slice(vol_slice, axis_idx)
                ax_row[j].imshow(normalize_image(s), cmap=cmap)
                ax_row[j].set_title(title, fontsize=9)
                ax_row[j].axis('off')

        plt.tight_layout()
        out_path = output_dir / f"visual_proofs_{category_name}_{view.lower()}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved {view}: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Visual proofs for RUN_0002 CycleGAN.")
    parser.add_argument("--config", default="configs/run_0002_cyclegan_task1b.yaml")
    parser.add_argument("--output_dir", default="results/runs/RUN_0002/plots/visuals")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Model ───
    ckpt_dir = config["output"].get("checkpoint_dir", "outputs/checkpoints/RUN_0002")
    ckpt_path = os.path.join(ckpt_dir, "G_AB_best.pt")

    cfg_m = config.get("model", {})
    model = Generator3D(
        base_filters=int(cfg_m.get("base_filters", 32)),
        n_res_blocks=int(cfg_m.get("n_res_blocks", 6)),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # ── Load Datasets (val + train) ───
    cfg_d = config["data"]
    data_root = os.getenv("LISA_DATA_ROOT", cfg_d["data_root"])
    csv_path = os.getenv("LISA_CSV_PATH", cfg_d.get("csv_path", ""))
    split_pkl = cfg_d["split_pkl"]
    image_suffix = cfg_d.get("image_suffix", "_ciso.nii.gz")
    spatial_size = tuple(int(v) for v in cfg_d.get("spatial_size", [96, 96, 96]))
    noise_thr = int(cfg_d.get("noise_threshold", 1))
    motion_thr = int(cfg_d.get("motion_threshold", 1))

    # Val dataset for categories available in val
    val_ds = Task1bCycleGANDataset(
        data_root=data_root, csv_path=csv_path, split_pkl=split_pkl,
        fold="val", stage="val", image_suffix=image_suffix,
        spatial_size=spatial_size, noise_threshold=noise_thr,
        motion_threshold=motion_thr, domain="both",
    )

    # Train dataset for categories missing in val (e.g., noise)
    train_ds = Task1bCycleGANDataset(
        data_root=data_root, csv_path=csv_path, split_pkl=split_pkl,
        fold="train", stage="val", image_suffix=image_suffix,
        spatial_size=spatial_size, noise_threshold=noise_thr,
        motion_threshold=motion_thr, domain="both",
    )

    # ── CSV labels for all subjects ───
    df = pd.read_csv(csv_path)
    df["subject"] = df["filename"].str.extract(r'(LISA_\d+)')
    with open(split_pkl, "rb") as f:
        split = pickle.load(f)

    def get_subj_labels(indices):
        df_split = df.iloc[indices].drop_duplicates("subject").reset_index(drop=True)
        return {
            row["subject"]: {"Noise": row["Noise"], "Motion": row["Motion"]}
            for _, row in df_split.iterrows()
        }

    val_labels = get_subj_labels(split.get("val_indices", []))
    train_labels = get_subj_labels(split.get("train_indices", []))

    def categorize_paths(paths, labels, cat):
        """Return paths matching a category."""
        result = []
        for p in paths:
            m = re.search(r'(LISA_\d+)', Path(p).name)
            subj = m.group(1) if m else None
            labs = labels.get(subj, {})
            if cat == "clean" and labs.get("Noise", 0) == 0 and labs.get("Motion", 0) == 0:
                result.append(p)
            elif cat == "noise" and labs.get("Noise", 0) >= noise_thr:
                result.append(p)
            elif cat == "motion" and labs.get("Motion", 0) >= motion_thr:
                result.append(p)
        return result

    # ── Category selection ───
    # Clean: from domain B (val set)
    clean_paths = categorize_paths(val_ds.paths_B, val_labels, "clean")

    # Motion: from domain A (val set if available)
    motion_paths_val = categorize_paths(val_ds.paths_A, val_labels, "motion")
    motion_paths = motion_paths_val if motion_paths_val else categorize_paths(
        train_ds.paths_A, train_labels, "motion"
    )[:5]

    # Noise: from domain A (val if available, else train)
    noise_paths_val = categorize_paths(val_ds.paths_A, val_labels, "noise")
    noise_paths = noise_paths_val if noise_paths_val else categorize_paths(
        train_ds.paths_A, train_labels, "noise"
    )[:5]

    categories = {
        "clean": (clean_paths, val_ds.transforms),
        "motion": (motion_paths, val_ds.transforms),
        "noise": (noise_paths, val_ds.transforms),
    }

    for cat, (paths, transforms) in categories.items():
        save_category_mosaics(
            model, device, paths, transforms,
            cat, {**val_labels, **train_labels}, out_dir
        )

    print(f"\nVisual proofs generated in {out_dir}")


if __name__ == "__main__":
    main()
