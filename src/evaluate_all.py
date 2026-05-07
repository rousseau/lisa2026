#!/usr/bin/env python3
"""
Evaluation complète LISA 2026 sur le split validation local.

Tâches évaluées :
  - Task 1a : QA multi-artefact (métriques weighted 3 classes LISA 2025)
  - Task 1b : reconstruction sur images sans artefact (MAE, MSE, PSNR, SSIM)
  - Task 2  : segmentation multi-structure (Dice par structure + sous-ensembles)

Comparaisons intégrées :
  - LISA 2024 Task 1 : weighted accuracy = 0.823
  - LISA 2025 Task 1 : Mean5 gagnant = 0.799
  - LISA 2024 Task 2 hippocampe : Dice = 0.61
  - LISA 2025 Task 2a hippocampe : Dice = 0.72
  - LISA 2025 Task 2b ganglions de la base : Dice = 0.87

Notes :
  - Task 1b n'a pas d'équivalent officiel dans les challenges LISA 2024/2025.
  - La comparaison Task 2b est une approximation locale sur les labels qui se
    recouvrent le mieux avec le challenge (caudés + putamens + globus pallidus).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

from dataset import ARTIFACT_COLS, DATA_ROOT_DEFAULT, LISAJointDataset
from evaluate import compute_metrics_per_artifact, print_results, collect_predictions
from model import BackboneLISA


TASK1A_REF_2024_WEIGHTED_ACC = 0.823
TASK1A_REF_2025_MEAN5 = 0.799
TASK1A_REF_2025_UPF_MEAN5 = 0.777

TASK2_REF_2024_HIP_DICE = 0.61
TASK2_REF_2025_HIP_DICE = 0.72
TASK2_REF_2025_BG_DICE = 0.87

SEG_LABELS = {
    1: "hippo_G",
    2: "hippo_D",
    3: "caude_G",
    4: "caude_D",
    5: "putamen_G",
    6: "putamen_D",
    7: "globus_G",
    8: "globus_D",
    9: "thalamus_G",
    10: "thalamus_D",
    11: "corps_calleux",
    12: "ventricule_G",
    13: "ventricule_D",
}

HIPPO_LABELS = [1, 2]
BG_PROXY_LABELS = [3, 4, 5, 6, 7, 8]
ALL_STRUCT_LABELS = list(SEG_LABELS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation complète LISA 2026 sur le split validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_model_and_val_ds(args):
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("config", {})
    cfg.setdefault("data_root", DATA_ROOT_DEFAULT)
    cfg.setdefault("target_size", 96)
    cfg.setdefault("val_fraction", 0.2)
    cfg.setdefault("base_channels", 16)
    cfg.setdefault("c_anat", 16)
    cfg.setdefault("c_mod", 8)
    cfg.setdefault("c_art", 8)
    cfg.setdefault("n_artifacts", 7)
    cfg.setdefault("n_severity", 3)
    cfg.setdefault("n_seg_classes", 14)

    if args.data_root:
        cfg["data_root"] = args.data_root

    device_str = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)

    ts = (cfg["target_size"],) * 3
    val_ds = LISAJointDataset(
        cfg["data_root"], target_size=ts, split="val", val_fraction=cfg["val_fraction"]
    )

    model = BackboneLISA(
        base=cfg["base_channels"],
        c_anat=cfg["c_anat"],
        c_mod=cfg["c_mod"],
        c_art=cfg["c_art"],
        n_artifacts=cfg["n_artifacts"],
        n_severity=cfg["n_severity"],
        n_seg_classes=cfg["n_seg_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch_num = ckpt.get("epoch", -1) + 1
    return ckpt_path, epoch_num, model, val_ds, device


def safe_dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    pred_mask = pred == label
    gt_mask = gt == label
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return np.nan
    inter = np.logical_and(pred_mask, gt_mask).sum()
    return float((2.0 * inter) / denom)


def safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


@torch.no_grad()
def evaluate_task1b(model, val_ds, device, batch_size=2, num_workers=2):
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    maes, mses, psnrs, ssims = [], [], [], []
    n_items = 0

    for batch in loader:
        mask = batch["is_artifact_free"]
        if not mask.any():
            continue

        x = batch["image"].to(device)
        recon = model(x, run_task1a=False, run_task1b=True, run_task2=False)["task1b"]

        recon_np = recon[mask].cpu().numpy()
        target_np = batch["image"][mask].cpu().numpy()

        for pred, gt in zip(recon_np, target_np):
            pred = pred[0]
            gt = gt[0]
            diff = pred - gt
            maes.append(float(np.mean(np.abs(diff))))
            mses.append(float(np.mean(diff ** 2)))
            psnrs.append(float(peak_signal_noise_ratio(gt, pred, data_range=1.0)))
            ssims.append(float(structural_similarity(gt, pred, data_range=1.0)))
            n_items += 1

    return {
        "n_items": n_items,
        "mae": float(np.mean(maes)) if maes else np.nan,
        "mse": float(np.mean(mses)) if mses else np.nan,
        "psnr": float(np.mean(psnrs)) if psnrs else np.nan,
        "ssim": float(np.mean(ssims)) if ssims else np.nan,
    }


@torch.no_grad()
def evaluate_task2(model, val_ds, device, batch_size=1, num_workers=2):
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    dices = {label: [] for label in ALL_STRUCT_LABELS}
    n_items = 0

    for batch in loader:
        mask = (batch["is_isotropic"] & batch["has_seg"])
        if not mask.any():
            continue

        x = batch["image"].to(device)
        logits = model(x, run_task1a=False, run_task1b=False, run_task2=True)["task2"]
        preds = logits.argmax(dim=1).cpu().numpy()
        gts = batch["seg"].cpu().numpy()

        for pred, gt, keep in zip(preds, gts, mask.cpu().numpy()):
            if not keep:
                continue
            n_items += 1
            for label in ALL_STRUCT_LABELS:
                dices[label].append(safe_dice(pred, gt, label))

    per_label = {
        SEG_LABELS[label]: safe_nanmean(values)
        for label, values in dices.items()
    }

    def subset_mean(labels):
        vals = []
        for label in labels:
            vals.extend(dices[label])
        return safe_nanmean(vals)

    return {
        "n_items": n_items,
        "per_label": per_label,
        "mean_all": subset_mean(ALL_STRUCT_LABELS),
        "mean_hippo": subset_mean(HIPPO_LABELS),
        "mean_bg_proxy": subset_mean(BG_PROXY_LABELS),
    }


def print_task1a_summary(results: dict):
    means = {k: [] for k in ["accuracy", "f1", "f2", "precision", "recall", "mean5"]}
    for metrics in results.values():
        for key in means:
            means[key].append(metrics[key])

    mean_acc = float(np.mean(means["accuracy"]))
    mean5 = float(np.mean(means["mean5"]))

    print("  Comparaison historique Task 1a :")
    print(f"    - Nous (val local)      : mean accuracy={mean_acc:.4f} | Mean5={mean5:.4f}")
    print(f"    - LISA 2024 gagnant     : weighted accuracy={TASK1A_REF_2024_WEIGHTED_ACC:.4f}")
    print(f"    - LISA 2025 gagnant     : Mean5={TASK1A_REF_2025_MEAN5:.4f}")
    print(f"    - LISA 2025 UPF (5e)    : Mean5={TASK1A_REF_2025_UPF_MEAN5:.4f}")
    print("    - Note : comparaison indicative, car val local ≠ test officiel et la métrique 2024 publiée est une weighted accuracy.")
    print()


def print_task1b_summary(metrics: dict):
    print("  Task 1b — reconstruction sur images sans artefact")
    print("  ---------------------------------------------------")
    print(f"  Items évalués : {metrics['n_items']}")
    print(f"  MAE           : {metrics['mae']:.6f}")
    print(f"  MSE           : {metrics['mse']:.6f}")
    print(f"  PSNR          : {metrics['psnr']:.4f}")
    print(f"  SSIM          : {metrics['ssim']:.4f}")
    print()
    print("  Comparaison historique Task 1b :")
    print("    - Aucun benchmark officiel LISA 2024/2025 directement comparable : l'amélioration d'image n'était pas une tâche classée séparée.")
    print("    - Les scores ci-dessus sont donc des métriques locales internes, utiles pour suivre la reconstruction mais non comparables au leaderboard.")
    print()


def print_task2_summary(metrics: dict):
    print("  Task 2 — segmentation multi-structure")
    print("  --------------------------------------")
    print(f"  Items évalués           : {metrics['n_items']}")
    print(f"  Dice moyen toutes structures : {metrics['mean_all']:.4f}")
    print(f"  Dice moyen hippocampes       : {metrics['mean_hippo']:.4f}")
    print(f"  Dice moyen proxy ganglions   : {metrics['mean_bg_proxy']:.4f}")
    print()
    print("  Dice par structure :")
    for name, value in metrics["per_label"].items():
        print(f"    - {name:14s} {value:.4f}")
    print()
    print("  Comparaison historique Task 2 :")
    print(f"    - LISA 2024 hippocampes : Dice={TASK2_REF_2024_HIP_DICE:.4f}")
    print(f"    - LISA 2025 Task 2a     : Dice={TASK2_REF_2025_HIP_DICE:.4f}")
    print(f"    - LISA 2025 Task 2b     : Dice={TASK2_REF_2025_BG_DICE:.4f}")
    print("    - Note : la comparaison 2025 Task 2b est approximative, car notre proxy local utilise caudés + putamens + globus pallidus, alors que le challenge 2b ciblait caudé + lentiforme.")
    print()


def main():
    args = parse_args()
    ckpt_path, epoch_num, model, val_ds, device = load_model_and_val_ds(args)

    print(f"Device : {device}")
    print(f"Checkpoint : {ckpt_path.name}")
    print(f"Modèle chargé — epoch {epoch_num}")
    print(f"Val items : {len(val_ds)}")
    print()

    y_true, y_pred = collect_predictions(
        model,
        val_ds,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    task1a_results = compute_metrics_per_artifact(y_true.numpy(), y_pred.numpy(), mode="weighted")
    print_results(task1a_results, checkpoint_name=ckpt_path.name, mode="weighted")
    print_task1a_summary(task1a_results)

    task1b_metrics = evaluate_task1b(
        model, val_ds, device, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print_task1b_summary(task1b_metrics)

    task2_metrics = evaluate_task2(
        model, val_ds, device, batch_size=1, num_workers=args.num_workers
    )
    print_task2_summary(task2_metrics)


if __name__ == "__main__":
    main()