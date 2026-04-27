#!/usr/bin/env python3
"""
Script d'inférence — Modèle joint LISA 2026.

Usage :
    # Task 1a – quality control (CSV de prédictions de sévérité)
    python src/inference.py --checkpoint outputs/checkpoints/epoch_0100.pt \\
        --task task1a --input-dir /data/LISA2026 \\
        --output results/predictions/task1a_pred.csv

    # Task 1b – enhancement (NIfTI images améliorées)
    python src/inference.py --checkpoint outputs/checkpoints/epoch_0100.pt \\
        --task task1b --input-dir /data/LISA2026 \\
        --output-dir results/predictions/task1b

    # Task 2 – segmentation (NIfTI masques de segmentation)
    python src/inference.py --checkpoint outputs/checkpoints/epoch_0100.pt \\
        --task task2 --input-dir /data/LISA2026 \\
        --output-dir results/predictions/task2
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from dataset import (
    ARTIFACT_COLS,
    TARGET_SIZE,
    TARGET_SPACING_MM,
    crop_or_pad,
    load_nifti,
    normalize,
    resample_to_isotropic,
)
from model import BrainFMLISA


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inférence LISA 2026")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument(
        "--task", required=True,
        choices=["task1a", "task1b", "task2"],
    )
    p.add_argument("--input-dir",    required=True, help="Dossier contenant les NIfTI")
    p.add_argument(
        "--glob", default=None,
        help=(
            "Pattern glob pour filtrer les fichiers. "
            "Défaut : *_LF_*.nii.gz pour task1a/1b, *_ciso.nii.gz pour task2."
        ),
    )
    p.add_argument("--output",     default=None, help="CSV de sortie (task1a)")
    p.add_argument("--output-dir", default=None, help="Répertoire de sortie NIfTI (task1b/2)")
    p.add_argument("--target-size",    type=int, default=96)
    p.add_argument("--base-channels",  type=int, default=16)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(filepath: str, target_size: tuple) -> tuple[torch.Tensor, dict]:
    """
    Charge, resamplé et normalise une image.
    Retourne (tensor [1, 1, D, H, W], metadata dict).
    """
    data, zooms = load_nifti(filepath)
    img_nib      = nib.load(str(filepath))
    orig_shape   = data.shape
    orig_affine  = img_nib.affine

    data_rs = resample_to_isotropic(data, zooms, order=1)
    rs_shape = data_rs.shape
    data_cp  = crop_or_pad(data_rs, target_size)
    data_n   = normalize(data_cp)

    tensor = torch.from_numpy(data_n[None, None])   # [1, 1, D, H, W]
    meta   = {
        "orig_shape":  orig_shape,
        "rs_shape":    rs_shape,
        "orig_affine": orig_affine,
        "zooms":       zooms,
        "target_size": target_size,
    }
    return tensor, meta


def _unpad_uncrop(pred: np.ndarray, meta: dict) -> np.ndarray:
    """
    Inverse de crop_or_pad : remet les prédictions dans l'espace resampleé.
    L'espace original (avant resample) n'est pas restauré ici – l'affine NIfTI
    est conservée pour que la résolution de sortie soit cohérente avec l'entrée.
    """
    target_size = meta["target_size"]
    rs_shape    = meta["rs_shape"]
    result = np.zeros(rs_shape, dtype=pred.dtype)
    src_slices, dst_slices = [], []
    for i, (t, s) in enumerate(zip(target_size, rs_shape)):
        if s >= t:
            pad = (s - t) // 2
            src_slices.append(slice(0, t))
            dst_slices.append(slice(pad, pad + t))
        else:
            crop = (t - s) // 2
            src_slices.append(slice(crop, crop + s))
            dst_slices.append(slice(0, s))
    result[tuple(dst_slices)] = pred[tuple(src_slices)]
    return result


def _isotropic_affine(orig_affine: np.ndarray, orig_zooms: np.ndarray) -> np.ndarray:
    """Ajuste l'affine pour une résolution isotrope 1 mm."""
    scale = orig_zooms / TARGET_SPACING_MM
    # l'affine isotrope conserve l'orientation et l'origine
    aff = orig_affine.copy()
    for i in range(3):
        aff[:3, i] = orig_affine[:3, i] / np.linalg.norm(orig_affine[:3, i])
    return aff


# ──────────────────────────────────────────────────────────────────────────────
# Inférence par tâche
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_task1a(model, files, target_size, device):
    """Retourne un générateur de (filename, preds[7]) pour Task 1a."""
    for fp in files:
        tensor, _ = preprocess(str(fp), target_size)
        tensor = tensor.to(device)
        out    = model(tensor, run_task1a=True, run_task1b=False, run_task2=False)
        logits = out["task1a"][0]              # [7, 3]
        preds  = logits.argmax(dim=-1).cpu().numpy()
        yield fp.name, preds


@torch.no_grad()
def infer_task1b(model, files, target_size, device):
    """Retourne un générateur de (fp, recon_np, meta) pour Task 1b."""
    for fp in files:
        tensor, meta = preprocess(str(fp), target_size)
        tensor = tensor.to(device)
        out    = model(tensor, run_task1a=False, run_task1b=True, run_task2=False)
        recon  = out["task1b"][0, 0].cpu().numpy()   # [D, H, W]
        recon  = _unpad_uncrop(recon, meta)
        yield fp, recon, meta


@torch.no_grad()
def infer_task2(model, files, target_size, device):
    """Retourne un générateur de (fp, seg_np, meta) pour Task 2."""
    for fp in files:
        tensor, meta = preprocess(str(fp), target_size)
        tensor = tensor.to(device)
        out    = model(tensor, run_task1a=False, run_task1b=False, run_task2=True)
        seg    = out["task2"][0].argmax(dim=0).cpu().numpy().astype(np.int16)
        seg    = _unpad_uncrop(seg, meta)
        yield fp, seg, meta


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    device      = torch.device(args.device)
    target_size = (args.target_size,) * 3

    # ── chargement modèle ────────────────────────────────────────────────────
    model = BrainFMLISA(base=args.base_channels).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint chargé : {args.checkpoint}")

    input_dir = Path(args.input_dir)

    # ── fichiers d'entrée ────────────────────────────────────────────────────
    if args.glob:
        pattern = args.glob
    elif args.task in ("task1a", "task1b"):
        pattern = "*_LF_*.nii.gz"
    else:
        pattern = "*_ciso.nii.gz"

    files = sorted(input_dir.glob(pattern))
    print(f"Fichiers trouvés ({pattern}) : {len(files)}")
    if not files:
        print("Aucun fichier correspondant — vérifier --input-dir et --glob.")
        return

    # ── Task 1a ──────────────────────────────────────────────────────────────
    if args.task == "task1a":
        out_path = Path(args.output or "results/predictions/task1a_pred.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filename"] + ARTIFACT_COLS)
            for fname, preds in infer_task1a(model, files, target_size, device):
                writer.writerow([fname] + preds.tolist())
                print(f"  {fname} → {dict(zip(ARTIFACT_COLS, preds.tolist()))}")

        print(f"Task 1a prédictions → {out_path}")

    # ── Task 1b ──────────────────────────────────────────────────────────────
    elif args.task == "task1b":
        out_dir = Path(args.output_dir or "results/predictions/task1b")
        out_dir.mkdir(parents=True, exist_ok=True)

        for fp, recon, meta in infer_task1b(model, files, target_size, device):
            stem     = fp.name.replace(".nii.gz", "")
            out_path = out_dir / f"{stem}_enhanced.nii.gz"
            affine   = _isotropic_affine(meta["orig_affine"], meta["zooms"])
            nib.save(
                nib.Nifti1Image(recon.astype(np.float32), affine),
                str(out_path),
            )
            print(f"  {out_path.name}")

        print(f"Task 1b images améliorées → {out_dir}")

    # ── Task 2 ───────────────────────────────────────────────────────────────
    elif args.task == "task2":
        out_dir = Path(args.output_dir or "results/predictions/task2")
        out_dir.mkdir(parents=True, exist_ok=True)

        for fp, seg, meta in infer_task2(model, files, target_size, device):
            stem     = fp.name.replace(".nii.gz", "")
            out_path = out_dir / f"{stem}_seg_pred.nii.gz"
            affine   = _isotropic_affine(meta["orig_affine"], meta["zooms"])
            nib.save(
                nib.Nifti1Image(seg, affine),
                str(out_path),
            )
            print(f"  {out_path.name}")

        print(f"Task 2 segmentations → {out_dir}")


if __name__ == "__main__":
    main()
