#!/usr/bin/env python3
"""Extract MedVAE latent code from a single 3D MRI volume and save as NIfTI.

This script loads a 3D MRI volume, preprocesses it (resampling + normalization),
encodes it through a MedVAE model, and saves the resulting latent code as a
NIfTI file.

Usage
-----
  python src/extract_medvae_latent.py --input sub-001_T1w.nii.gz
  python src/extract_medvae_latent.py --input sub-001_T1w.nii.gz --output latent.nii.gz
  python src/extract_medvae_latent.py --input sub-001_T1w.nii.gz --device cpu

What this script does
---------------------
1. Loads a 3D MRI volume from a NIfTI file.
2. Resamples to 1mm isotropic spacing.
3. Robustly normalizes intensity (percentile clip -> [-1, 1]).
4. Encodes the volume with MedVAE (medvae_4_1_3d, modality mri).
5. Saves the latent code as a NIfTI file with preserved orientation and updated spacing.

Author: LISA 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom

# ------ Preprocessing ------


def normalize_percentile(
    vol: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> np.ndarray:
    """Robust percentile normalization -> [-1, 1].

    Clips the volume to the given percentiles, linearly maps to [0, 1],
    then shifts to [-1, 1].  Volumes with flat intensity (hi == lo) become
    zero-filled arrays.

    Args:
        vol:      Input 3D volume (any dtype, will be cast to float32).
        lo_pct:   Lower percentile (default 0.5).
        hi_pct:   Upper percentile (default 99.5).

    Returns:
        Normalized volume as float32 in [-1, 1].
    """
    vol_f = vol.astype(np.float32)
    lo = np.percentile(vol_f, lo_pct)
    hi = np.percentile(vol_f, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol_f)
    v = np.clip((vol_f - lo) / (hi - lo), 0.0, 1.0)
    return (v * 2.0 - 1.0).astype(np.float32)


def resample_to_1mm(vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Resample volume to 1mm isotropic spacing.

    Uses the affine matrix diagonal to infer the current voxel spacing and
    applies scipy.zoom with linear interpolation (order=1).

    Args:
        vol:      3D volume array.
        affine:   4x4 NIfTI affine matrix.

    Returns:
        Resampled volume as float32.
    """
    spacing = np.abs(np.diag(affine)[:3]).astype(np.float32)
    factors = spacing / np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if np.allclose(factors, 1.0, atol=0.05):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


# ------ Core extraction ------


def extract_latent(
    input_path: str,
    output_path: str,
    device: str,
    model_name: str = "medvae_4_1_3d",
) -> None:
    """Extract MedVAE latent code from an MRI volume and save as NIfTI.

    Args:
        input_path:  Path to the input NIfTI file.
        output_path: Path for the output latent NIfTI file.
        device:      PyTorch device string (e.g. "cuda", "cpu", "cuda:0").
        model_name:  MedVAE model name (default "medvae_4_1_3d").
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # --- 1. Load volume ---
    print(f"[1/5] Loading {input_path}")
    img = nib.load(str(input_path))
    vol = img.get_fdata(dtype=np.float32)
    affine = img.affine.copy()
    print(f"  Input shape: {vol.shape}, spacing: {np.diag(affine)[:3]}")

    # --- 2. Resample to 1mm isotropic ---
    print("[2/5] Resampling to 1mm isotropic")
    vol = resample_to_1mm(vol, affine)
    print(f"  Resampled shape: {vol.shape}")

    # --- 3. Normalize ---
    print("[3/5] Normalizing (percentile 0.5-99.5)")
    vol = normalize_percentile(vol)
    print(f"  Intensity range: [{vol.min():.3f}, {vol.max():.3f}]")

    # --- 4. Encode with MedVAE ---
    print("[4/5] Encoding with MedVAE")
    from medvae import MVAE

    model = MVAE(model_name=model_name, modality="mri").to(device)
    model.eval()

    # Tensor shape: (batch=1, channel=1, depth, height, width)
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        z = model.encode(x)
        # For medvae_4_1_3d, encode returns a plain tensor: (1, 1, D/4, H/4, W/4)
        if isinstance(z, tuple):
            z = z[0]

    z_np = z.squeeze().detach().cpu().numpy().astype(np.float32)
    print(f"  Latent shape: {z_np.shape}")

    # --- 5. Save as NIfTI ---
    print("[5/5] Saving latent to NIfTI")
    # Update voxel spacing: original spacing x 4 (because latent is 4x smaller)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * 4  # scale diagonal by 4
    new_img = nib.Nifti1Image(z_np, new_affine)
    nib.save(new_img, str(output_path))
    print(f"  Output: {output_path}")
    print(f"  Output spacing: {np.diag(new_affine)[:3]}")


# ------ CLI ------


def parse_args(argv=None):
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Extract MedVAE latent code from a 3D MRI volume (NIfTI).",
    )
    ap.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input MRI NIfTI file.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path for the output latent NIfTI file. "
            "Default: <input_basename>_medvae_latent.nii.gz"
        ),
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device (cuda, cpu, cuda:N). Auto-detects GPU if available.",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="medvae_4_1_3d",
        help="MedVAE model name (default: medvae_4_1_3d).",
    )
    return ap.parse_args(argv)


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)

    # Resolve output path
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = input_path.suffix
        # Handle .nii.gz -> basename without both extensions
        if suffix == ".gz" and input_path.stem.endswith(".nii"):
            base = input_path.stem[:-4]
            output_path = input_path.with_name(f"{base}_medvae_latent.nii.gz")
        else:
            output_path = input_path.with_name(f"{input_path.stem}_medvae_latent.nii")

    # Resolve device
    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"MedVAE model: {args.model_name}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("-" * 50)

    try:
        extract_latent(
            input_path=str(input_path),
            output_path=str(output_path),
            device=device,
            model_name=args.model_name,
        )
    except Exception as exc:
        print(f"\n[ERROR] Extraction failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
