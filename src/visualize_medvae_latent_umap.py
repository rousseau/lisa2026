#!/usr/bin/env python3
"""UMAP visualization of MedVAE latent codes for LISA 2026 Task 1a.

What this script does
---------------------
1. Loads LISA Task1a CSV annotations and selects a train/val/all split.
2. Loads MedVAE latent codes either from a directory of NIfTI files
   or by encoding each volume on-the-fly with a MedVAE model.
3. Flattens each latent volume to a 1D vector.
4. Runs UMAP to produce a 2D embedding.
5. Saves two figures:
   - Figure 1: coloured by clean vs artifact (binary).
   - Figure 2: coloured by artifact degree (sum of all artifact columns).
6. Saves a summary CSV: filename, artifact_degree, is_clean, umap_x, umap_y.

Usage
-----
  # Encode on-the-fly (no pre-computed latents)
  python src/visualize_medvae_latent_umap.py --fold val

  # Use pre-computed latent NIfTI files
  python src/visualize_medvae_latent_umap.py --latents-dir /path/to/latents --fold val

  # Full example with explicit paths
  python src/visualize_medvae_latent_umap.py \\
      --csv /home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv \\
      --bids-root /home/rousseau/Data/LISA2026 \\
      --split-pkl results/splits/task1a_fixed.pkl \\
      --fold val \\
      --latents-dir results/latents \\
      --device cuda

Author: LISA 2026
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom as scipy_zoom

# ---------------------------------------------------------------------------
# Artifact columns (must match the CSV and the reference script)
# ---------------------------------------------------------------------------

ARTIFACT_COLS = [
    "Noise",
    "Zipper",
    "Positioning",
    "Banding",
    "Motion",
    "Contrast",
    "Distortion",
]


# ---------------------------------------------------------------------------
# Preprocessing  (identical to analyze_medvae_reconstruction_lisa2026.py)
# ---------------------------------------------------------------------------


def normalize_percentile(
    vol: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> np.ndarray:
    """Robust percentile normalisation -> [-1, 1]."""
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    v = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (v * 2.0 - 1.0).astype(np.float32)


def resample_to_1mm(vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Resample volume to 1 mm isotropic spacing via scipy zoom (order=1)."""
    spacing = np.abs(np.diag(affine)[:3]).astype(np.float32)
    factors = spacing / np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if np.allclose(factors, 1.0, atol=0.05):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Split selection  (identical to analyze_medvae_reconstruction_lisa2026.py)
# ---------------------------------------------------------------------------


def select_rows(
    df: pd.DataFrame,
    split_pkl: Optional[Path],
    fold: str,
) -> pd.DataFrame:
    """Return the subset of *df* corresponding to *fold*."""
    if split_pkl is None or not split_pkl.exists():
        if split_pkl is not None:
            print(f"[WARN] split file not found: {split_pkl}  -> using all rows")
        return df.copy().reset_index(drop=True)
    with open(split_pkl, "rb") as f:
        split = pickle.load(f)
    if fold == "train":
        idx = split.get("train_indices", [])
    elif fold == "val":
        idx = split.get("val_indices", [])
    else:  # "all"
        idx = list(split.get("train_indices", [])) + list(split.get("val_indices", []))
    return df.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Latent loading / on-the-fly encoding
# ---------------------------------------------------------------------------


def _latent_path_for(filename: str, latents_dir: Path) -> Optional[Path]:
    """Return the latent NIfTI path that corresponds to a BIDS filename.

    Convention: strip leading BIDS sub-directories and look for a file
    in *latents_dir* whose stem starts with the NIfTI base-name.
    E.g. ``sub-001/anat/sub-001_T1w.nii.gz`` -> ``<dir>/sub-001_T1w_medvae_latent.nii.gz``
    or the base itself ``sub-001_T1w.nii.gz`` without the latent suffix.
    """
    base = Path(filename).name  # e.g. sub-001_T1w.nii.gz
    # Strip double extension for .nii.gz
    if base.endswith(".nii.gz"):
        stem = base[:-7]
    elif base.endswith(".nii"):
        stem = base[:-4]
    else:
        stem = Path(base).stem

    # Prefer the explicit latent suffix produced by extract_medvae_latent.py
    candidate_latent = latents_dir / f"{stem}_medvae_latent.nii.gz"
    if candidate_latent.exists():
        return candidate_latent
    # Fallback: bare NIfTI with same name already in latents_dir
    for ext in (".nii.gz", ".nii"):
        candidate_bare = latents_dir / f"{stem}{ext}"
        if candidate_bare.exists():
            return candidate_bare
    return None


def encode_volume(
    vol_path: Path,
    model: "torch.nn.Module",
    device: torch.device,
) -> np.ndarray:
    """Load, preprocess and encode a single NIfTI volume. Returns flat float32 array."""
    img = nib.load(str(vol_path))
    vol = img.get_fdata(dtype=np.float32)
    vol = resample_to_1mm(vol, img.affine)
    vol = normalize_percentile(vol)

    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encode(x)
        if isinstance(z, tuple):
            z = z[0]
    return z.squeeze().detach().cpu().numpy().astype(np.float32).ravel()


def load_latent_nifti(latent_path: Path) -> np.ndarray:
    """Load a pre-computed latent NIfTI and return a flat float32 array."""
    img = nib.load(str(latent_path))
    return img.get_fdata(dtype=np.float32).ravel()


# ---------------------------------------------------------------------------
# UMAP embedding
# ---------------------------------------------------------------------------


def run_umap(
    vectors: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """Fit UMAP on *vectors* (N x D) and return 2D embedding (N x 2)."""
    try:
        import umap  # umap-learn
    except ImportError as e:
        raise ImportError(
            "umap-learn is required. Install with: pip install umap-learn"
        ) from e

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_umap_binary(
    embedding: np.ndarray,
    is_clean: np.ndarray,
    out_path: Path,
    title: str = "MedVAE Latents — Clean vs Artifact",
) -> None:
    """Figure 1: binary clean (0) / artifact (1) colouring."""
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#e41a1c", "#377eb8"]  # red=artifact, blue=clean  (Set1 inspired)

    for val, label in [(1, "clean"), (0, "artifact")]:
        mask = is_clean == val
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=palette[val],
            label=label,
            s=18,
            alpha=0.75,
            linewidths=0,
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Group", fontsize=10, markerscale=1.8)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_umap_degree(
    embedding: np.ndarray,
    artifact_degree: np.ndarray,
    out_path: Path,
    title: str = "MedVAE Latents — Artifact Degree",
) -> None:
    """Figure 2: continuous artifact degree colouring (viridis)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=artifact_degree,
        cmap="viridis",
        s=18,
        alpha=0.80,
        linewidths=0,
        vmin=0,
        vmax=max(int(artifact_degree.max()), 1),
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Artifact degree (sum of artifact columns)", fontsize=9)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="UMAP visualisation of MedVAE latent codes for LISA 2026 Task 1a."
    )
    ap.add_argument(
        "--csv",
        default="/home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv",
        help="Path to LISA_Task1a_2026.csv.",
    )
    ap.add_argument(
        "--bids-root",
        default="/home/rousseau/Data/LISA2026",
        help="Root of the BIDS dataset (volumes are loaded from here when encoding on-the-fly).",
    )
    ap.add_argument(
        "--split-pkl",
        default="results/splits/task1a_fixed.pkl",
        help="Path to the fixed train/val split pickle.",
    )
    ap.add_argument(
        "--fold",
        choices=["train", "val", "all"],
        default="val",
        help="Which fold to visualise (default: val).",
    )
    ap.add_argument(
        "--latents-dir",
        default=None,
        help=(
            "Directory containing pre-computed latent NIfTI files "
            "(produced by extract_medvae_latent.py). "
            "If omitted, volumes are encoded on-the-fly."
        ),
    )
    ap.add_argument(
        "--model-name",
        default="medvae_4_1_3d",
        help="MedVAE model name used for on-the-fly encoding (default: medvae_4_1_3d).",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="PyTorch device (cuda / cpu / cuda:N). Auto-detected if omitted.",
    )
    ap.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15).",
    )
    ap.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1).",
    )
    ap.add_argument(
        "--umap-metric",
        default="euclidean",
        help="UMAP distance metric (default: euclidean).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP (default: 42).",
    )
    ap.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory for output figures and CSV (default: results/plots).",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to output filenames (e.g. fold name).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for no limit).",
    )
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    csv_path = Path(args.csv)
    bids_root = Path(args.bids_root)
    split_pkl = Path(args.split_pkl) if args.split_pkl else None
    latents_dir = Path(args.latents_dir) if args.latents_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tag = args.tag if args.tag else args.fold

    # ------------------------------------------------------------------
    # 1. Load CSV and select split
    # ------------------------------------------------------------------
    print(f"Loading annotations: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ARTIFACT_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing artifact column in CSV: {col}")

    df = select_rows(df, split_pkl=split_pkl, fold=args.fold)
    print(f"Rows after split selection ({args.fold}): {len(df)}")

    # Subsample if max-samples is specified
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed)
        print(f"Subsampled to {args.max_samples} random samples")

    # ------------------------------------------------------------------
    # 2. Optionally load MedVAE model for on-the-fly encoding
    # ------------------------------------------------------------------
    model = None
    if latents_dir is None:
        print(f"[MODE] On-the-fly encoding with {args.model_name} on {device}")
        from medvae import MVAE

        model = MVAE(model_name=args.model_name, modality="mri").to(device)
        model.eval()
    else:
        print(f"[MODE] Loading pre-computed latents from {latents_dir}")

    # ------------------------------------------------------------------
    # 3. Collect latent vectors + metadata
    # ------------------------------------------------------------------
    filenames: List[str] = []
    artifact_degrees: List[int] = []
    is_cleans: List[int] = []
    vectors: List[np.ndarray] = []
    skipped = 0

    for i, row in df.iterrows():
        fn = str(row["filename"])
        vol_path = bids_root / fn

        # Compute metadata regardless of loading mode
        degree = int(sum(int(row[c]) for c in ARTIFACT_COLS))
        clean = int((row[ARTIFACT_COLS] == 0).all())

        # --- Obtain flat latent vector ---
        if latents_dir is not None:
            latent_path = _latent_path_for(fn, latents_dir)
            if latent_path is None:
                print(f"  [SKIP] no latent found for {fn}")
                skipped += 1
                continue
            try:
                vec = load_latent_nifti(latent_path)
            except Exception as exc:
                print(f"  [SKIP] error loading {latent_path}: {exc}")
                skipped += 1
                continue
        else:
            if not vol_path.exists():
                print(f"  [SKIP] volume not found: {vol_path}")
                skipped += 1
                continue
            try:
                vec = encode_volume(vol_path, model, device)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"  [SKIP] OOM encoding {fn}")
                    skipped += 1
                    continue
                raise

        filenames.append(fn)
        artifact_degrees.append(degree)
        is_cleans.append(clean)
        vectors.append(vec)

        label = "clean" if clean else f"degree={degree}"
        print(
            f"[{len(vectors):03d}/{len(df):03d}] {fn}  {label}  latent_size={vec.shape[0]}"
        )

    print(f"\nLoaded: {len(vectors)}  |  Skipped: {skipped}")

    if len(vectors) < 5:
        raise RuntimeError(
            f"Too few samples to run UMAP ({len(vectors)}). "
            "Check your --latents-dir / --bids-root / --split-pkl paths."
        )

    # ------------------------------------------------------------------
    # 4. Normalise vector lengths (pad/truncate to the minimum size so
    #    that UMAP receives a rectangular matrix even if latent shapes differ)
    # ------------------------------------------------------------------
    sizes = [v.shape[0] for v in vectors]
    min_size = min(sizes)
    max_size = max(sizes)
    if min_size != max_size:
        print(
            f"[WARN] Latent sizes vary ({min_size}..{max_size}). "
            f"Truncating all to {min_size}."
        )
        vectors = [v[:min_size] for v in vectors]

    X = np.stack(vectors, axis=0).astype(np.float32)  # (N, D)
    print(f"Feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # 5. UMAP
    # ------------------------------------------------------------------
    print(
        f"\nRunning UMAP (n_neighbors={args.umap_neighbors}, "
        f"min_dist={args.umap_min_dist}, metric={args.umap_metric}) ..."
    )
    embedding = run_umap(
        X,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )  # (N, 2)
    print(f"Embedding shape: {embedding.shape}")

    is_clean_arr = np.array(is_cleans, dtype=int)
    degree_arr = np.array(artifact_degrees, dtype=int)

    # ------------------------------------------------------------------
    # 6. Figure 1 — binary clean vs artifact
    # ------------------------------------------------------------------
    fig1_path = out_dir / f"medvae_latent_umap_clean_vs_artifact_{tag}.png"
    plot_umap_binary(
        embedding,
        is_clean_arr,
        out_path=fig1_path,
        title=f"MedVAE Latents — Clean vs Artifact  [{args.fold}]",
    )

    # ------------------------------------------------------------------
    # 7. Figure 2 — artifact degree
    # ------------------------------------------------------------------
    fig2_path = out_dir / f"medvae_latent_umap_artifact_degree_{tag}.png"
    plot_umap_degree(
        embedding,
        degree_arr,
        out_path=fig2_path,
        title=f"MedVAE Latents — Artifact Degree  [{args.fold}]",
    )

    # ------------------------------------------------------------------
    # 8. Summary CSV
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(
        {
            "filename": filenames,
            "artifact_degree": artifact_degrees,
            "is_clean": is_cleans,
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
        }
    )
    csv_out = out_dir / f"medvae_latent_umap_summary{tag}.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # ------------------------------------------------------------------
    # 9. Console summary
    # ------------------------------------------------------------------
    n_clean = int(is_clean_arr.sum())
    n_artifact = len(is_clean_arr) - n_clean
    print(
        f"\nSummary\n"
        f"  Total samples : {len(embedding)}\n"
        f"  Clean         : {n_clean}\n"
        f"  With artifact : {n_artifact}\n"
        f"  Degree range  : {degree_arr.min()}..{degree_arr.max()}\n"
        f"\nOutputs\n"
        f"  Figure 1 (binary) : {fig1_path}\n"
        f"  Figure 2 (degree) : {fig2_path}\n"
        f"  Summary CSV       : {csv_out}\n"
    )
    print("Done.")


if __name__ == "__main__":
    main()
