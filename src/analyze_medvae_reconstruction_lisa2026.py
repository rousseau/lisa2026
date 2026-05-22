#!/usr/bin/env python3
"""MedVAE reconstruction analysis on LISA 2026 data.

What this script does:
1. Loads LISA Task1a annotations (artifact labels per acquisition).
2. Selects a subset using optional fixed split indices (train/val/all).
3. Loads each 3D volume, resamples to 1mm isotropic, and robustly normalizes.
4. Runs MedVAE reconstruction, prioritizing full-volume encode/decode.
5. Falls back to patch-based reconstruction on OOM.
6. Computes MAE/MSE/SSIM and orientation-wise SSIM.
7. Produces CSV + summary + plots by image type and artifact type.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.ndimage import zoom as scipy_zoom


ARTIFACT_COLS = [
    "Noise",
    "Zipper",
    "Positioning",
    "Banding",
    "Motion",
    "Contrast",
    "Distortion",
]


def normalize_percentile(vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    v = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (v * 2.0 - 1.0).astype(np.float32)


def resample_to_1mm(vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    spacing = np.abs(np.diag(affine)[:3]).astype(np.float32)
    factors = spacing / np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if np.allclose(factors, 1.0, atol=0.05):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def _sample_indices(n: int, max_slices: int) -> np.ndarray:
    if n <= max_slices:
        return np.arange(n)
    return np.linspace(0, n - 1, max_slices).astype(int)


def ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01, 0.03
    mu1 = ndimage.gaussian_filter(a, sigma=1.5)
    mu2 = ndimage.gaussian_filter(b, sigma=1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = ndimage.gaussian_filter(a * a, sigma=1.5) - mu1_sq
    sigma2_sq = ndimage.gaussian_filter(b * b, sigma=1.5) - mu2_sq
    sigma12 = ndimage.gaussian_filter(a * b, sigma=1.5) - mu12

    num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    return float(np.mean(num / den))


def ssim_3d(x: np.ndarray, y: np.ndarray, max_slices: int = 24) -> float:
    idxs = _sample_indices(x.shape[2], max_slices=max_slices)
    vals = [ssim_2d(x[:, :, k], y[:, :, k]) for k in idxs]
    return float(np.mean(vals)) if vals else float("nan")


def orientation_ssim(x: np.ndarray, y: np.ndarray, max_slices_per_axis: int = 16) -> Dict[str, float]:
    ixs = _sample_indices(x.shape[0], max_slices=max_slices_per_axis)
    iys = _sample_indices(x.shape[1], max_slices=max_slices_per_axis)
    izs = _sample_indices(x.shape[2], max_slices=max_slices_per_axis)

    sag = [ssim_2d(x[i, :, :], y[i, :, :]) for i in ixs]
    cor = [ssim_2d(x[:, j, :], y[:, j, :]) for j in iys]
    axi = [ssim_2d(x[:, :, k], y[:, :, k]) for k in izs]
    return {
        "ssim_sagittal": float(np.mean(sag)) if sag else float("nan"),
        "ssim_coronal": float(np.mean(cor)) if cor else float("nan"),
        "ssim_axial": float(np.mean(axi)) if axi else float("nan"),
    }


def _patch_starts(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, max(size - patch + 1, 1), stride))
    last = size - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def reconstruct_with_patches(
    model: torch.nn.Module,
    x: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float,
) -> torch.Tensor:
    _, _, h, w, d = x.shape
    ph, pw, pd = patch_size
    sh = max(1, int(ph * (1.0 - overlap)))
    sw = max(1, int(pw * (1.0 - overlap)))
    sd = max(1, int(pd * (1.0 - overlap)))

    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)

    hs = _patch_starts(h, ph, sh)
    ws = _patch_starts(w, pw, sw)
    ds = _patch_starts(d, pd, sd)

    for i in hs:
        for j in ws:
            for k in ds:
                patch = x[:, :, i : i + ph, j : j + pw, k : k + pd]
                with torch.no_grad():
                    z = model.encode(patch)
                    if isinstance(z, tuple):
                        z = z[0]
                    rec = model.decode(z)
                    if rec.shape[1] > 1:
                        rec = rec[:, :1]
                out[:, :, i : i + ph, j : j + pw, k : k + pd] += rec
                weight[:, :, i : i + ph, j : j + pw, k : k + pd] += 1.0

    return out / torch.clamp(weight, min=1e-6)


def run_reconstruction(
    model: torch.nn.Module,
    x: torch.Tensor,
    prefer_full: bool,
    patch_size: Tuple[int, int, int],
    patch_overlap: float,
) -> Tuple[torch.Tensor, str]:
    if prefer_full:
        try:
            with torch.no_grad():
                z = model.encode(x)
                if isinstance(z, tuple):
                    z = z[0]
                rec = model.decode(z)
                if rec.shape[1] > 1:
                    rec = rec[:, :1]
            return rec, "full"
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rec = reconstruct_with_patches(model, x, patch_size=patch_size, overlap=patch_overlap)
    return rec, "patch"


def dominant_artifact(row: pd.Series) -> str:
    vals = {c: int(row[c]) for c in ARTIFACT_COLS}
    if all(v == 0 for v in vals.values()):
        return "clean"
    m = max(vals.values())
    names = sorted([k for k, v in vals.items() if v == m])
    return names[0]


def image_type(row: pd.Series) -> str:
    # Acquisition orientation from filename suffix in Task1a CSV.
    fn = str(row["filename"])
    if "_LF_axi" in fn:
        return "axi"
    if "_LF_cor" in fn:
        return "cor"
    if "_LF_sag" in fn:
        return "sag"
    return "unknown"


def select_rows(df: pd.DataFrame, split_pkl: Path | None, fold: str) -> pd.DataFrame:
    if split_pkl is None:
        return df.copy().reset_index(drop=True)
    with open(split_pkl, "rb") as f:
        split = pickle.load(f)
    if fold == "train":
        idx = split.get("train_indices", [])
    elif fold == "val":
        idx = split.get("val_indices", [])
    else:
        idx = list(split.get("train_indices", [])) + list(split.get("val_indices", []))
    return df.iloc[idx].reset_index(drop=True)


def save_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_png: Path) -> None:
    groups = df.groupby(x_col)[y_col].apply(list).to_dict()
    labels = sorted(groups.keys())
    data = [groups[k] for k in labels]
    plt.figure(figsize=(max(8, 0.9 * len(labels)), 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(y_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_bar_mean(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_png: Path) -> None:
    means = df.groupby(x_col)[y_col].mean().sort_index()
    plt.figure(figsize=(max(8, 0.8 * len(means)), 4.5))
    plt.bar(means.index.tolist(), means.values.tolist())
    plt.title(title)
    plt.ylabel(f"mean {y_col}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="MedVAE reconstruction analysis for LISA 2026")
    ap.add_argument("--csv", default="/home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv")
    ap.add_argument("--bids-root", default="/home/rousseau/Data/LISA2026")
    ap.add_argument("--split-pkl", default="results/splits/task1a_fixed.pkl")
    ap.add_argument("--fold", choices=["train", "val", "all"], default="val")
    ap.add_argument("--max-samples", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", default="medvae_4_1_3d")
    ap.add_argument("--device", default=None)
    ap.add_argument("--prefer-full-image", action="store_true")
    ap.add_argument("--patch-size", nargs=3, type=int, default=[96, 96, 96])
    ap.add_argument("--patch-overlap", type=float, default=0.25)
    ap.add_argument("--max-ssim-slices", type=int, default=24)
    ap.add_argument("--max-orientation-slices", type=int, default=16)
    ap.add_argument("--output-dir", default="results/medvae_lisa2026_reconstruction")
    args = ap.parse_args()

    np.random.seed(args.seed)

    csv_path = Path(args.csv)
    bids_root = Path(args.bids_root)
    split_pkl = Path(args.split_pkl) if args.split_pkl else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for c in ARTIFACT_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing artifact column: {c}")

    df = select_rows(df, split_pkl=split_pkl, fold=args.fold)
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Rows selected: {len(df)}")

    from medvae import MVAE

    model = MVAE(model_name=args.model_name, modality="mri").to(device)
    model.eval()

    records: List[Dict[str, object]] = []
    n_full, n_patch = 0, 0

    for i, row in df.iterrows():
        fn = str(row["filename"])
        path = bids_root / fn
        if not path.exists():
            continue

        img = nib.load(str(path))
        vol = img.get_fdata(dtype=np.float32)
        vol = resample_to_1mm(vol, img.affine)
        vol = normalize_percentile(vol)

        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        rec_t, mode = run_reconstruction(
            model,
            x,
            prefer_full=args.prefer_full_image,
            patch_size=tuple(args.patch_size),
            patch_overlap=args.patch_overlap,
        )
        rec = rec_t.squeeze().detach().cpu().numpy().astype(np.float32)

        if mode == "full":
            n_full += 1
        else:
            n_patch += 1

        if rec.shape != vol.shape:
            d0 = min(rec.shape[0], vol.shape[0])
            d1 = min(rec.shape[1], vol.shape[1])
            d2 = min(rec.shape[2], vol.shape[2])
            rec = rec[:d0, :d1, :d2]
            vol = vol[:d0, :d1, :d2]

        mae = float(np.mean(np.abs(vol - rec)))
        mse = float(np.mean((vol - rec) ** 2))
        ssim = ssim_3d(vol, rec, max_slices=args.max_ssim_slices)
        ssim_axes = orientation_ssim(vol, rec, max_slices_per_axis=args.max_orientation_slices)

        r: Dict[str, object] = {
            "filename": fn,
            "image_type": image_type(row),
            "is_clean": int((row[ARTIFACT_COLS] == 0).all()),
            "dominant_artifact": dominant_artifact(row),
            "artifact_sum": int(sum(int(row[c]) for c in ARTIFACT_COLS)),
            "reconstruction_mode": mode,
            "mae": mae,
            "mse": mse,
            "ssim": ssim,
            **ssim_axes,
        }
        for c in ARTIFACT_COLS:
            r[c] = int(row[c])
        records.append(r)

        print(f"[{i + 1:03d}/{len(df):03d}] {fn} | {mode} | MAE={mae:.4f} | SSIM={ssim:.4f}")

    out_csv = out_dir / "medvae_lisa2026_reconstruction_metrics.csv"
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False)

    # Plot 1: quality by image type (axi/cor/sag)
    save_boxplot(
        out_df,
        x_col="image_type",
        y_col="ssim",
        title="MedVAE SSIM by image type (acquisition orientation)",
        out_png=out_dir / "ssim_by_image_type.png",
    )

    # Plot 2: quality by clean vs artifact
    out_df["image_group"] = out_df["is_clean"].map({1: "clean", 0: "with_artifact"})
    save_boxplot(
        out_df,
        x_col="image_group",
        y_col="ssim",
        title="MedVAE SSIM: clean vs with artifact",
        out_png=out_dir / "ssim_clean_vs_artifact.png",
    )

    # Plot 3: quality by dominant artifact type
    save_bar_mean(
        out_df,
        x_col="dominant_artifact",
        y_col="ssim",
        title="MedVAE mean SSIM by dominant artifact type",
        out_png=out_dir / "ssim_by_dominant_artifact.png",
    )

    # Summary
    mean_mae = float(out_df["mae"].mean()) if len(out_df) else float("nan")
    mean_ssim = float(out_df["ssim"].mean()) if len(out_df) else float("nan")
    clean_df = out_df[out_df["is_clean"] == 1]
    art_df = out_df[out_df["is_clean"] == 0]

    summary = out_dir / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write("MedVAE reconstruction analysis on LISA 2026\n")
        f.write(f"n_samples_processed: {len(out_df)}\n")
        f.write(f"mode_full: {n_full}\n")
        f.write(f"mode_patch: {n_patch}\n")
        f.write(f"mean_mae: {mean_mae:.6f}\n")
        f.write(f"mean_ssim: {mean_ssim:.6f}\n")
        if len(clean_df):
            f.write(f"mean_ssim_clean: {float(clean_df['ssim'].mean()):.6f}\n")
        if len(art_df):
            f.write(f"mean_ssim_with_artifact: {float(art_df['ssim'].mean()):.6f}\n")
        f.write(
            "orientation_ssim_mean: "
            f"sagittal={float(out_df['ssim_sagittal'].mean()):.6f}, "
            f"coronal={float(out_df['ssim_coronal'].mean()):.6f}, "
            f"axial={float(out_df['ssim_axial'].mean()):.6f}\n"
        )

    print("\nDone.")
    print(f"- Metrics CSV: {out_csv}")
    print(f"- Summary: {summary}")
    print(f"- Plots: {out_dir}")


if __name__ == "__main__":
    main()
