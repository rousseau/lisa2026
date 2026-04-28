#!/usr/bin/env python3
"""
Évaluation Task 1a sur le split de validation — métriques LISA 2025.

Les 5 métriques officielles (accuracy, F1, F2, precision, recall) sont
calculées en mode BINAIRE par artefact :
    positif  = artefact présent (sévérité > 0)
    négatif  = artefact absent  (sévérité = 0)

Le score final est la moyenne des 5 métriques, moyennée sur les 7 artefacts.

Usage :
    python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
    python src/evaluate.py --checkpoint outputs/checkpoints/epoch_0010.pt --config configs/train_default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

# ── import du projet ──────────────────────────────────────────────────────────
SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from dataset import LISAJointDataset, ARTIFACT_COLS, DATA_ROOT_DEFAULT
from model import BrainFMLISA

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "train_default.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Inférence sur le jeu de validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, val_ds, device, batch_size=4, num_workers=4):
    """
    Retourne (y_true, y_pred) tenseurs [N, 7] long.
    Seuls les items ayant has_task1a=True sont conservés.
    """
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    model.eval()

    all_true, all_pred = [], []

    for batch in loader:
        mask = batch["has_task1a"]
        if not mask.any():
            continue
        x = batch["image"].to(device)
        preds = model(x)
        logits = preds["task1a"]          # [B, 7, 3]
        pred_sev = logits.argmax(dim=2)   # [B, 7]  classe prédite

        all_true.append(batch["task1a_labels"][mask].cpu())
        all_pred.append(pred_sev[mask].cpu())

    if not all_true:
        return torch.zeros(0, 7, dtype=torch.long), torch.zeros(0, 7, dtype=torch.long)

    return torch.cat(all_true, dim=0), torch.cat(all_pred, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Calcul des métriques
# ──────────────────────────────────────────────────────────────────────────────

def fbeta_score(precision, recall, beta=1.0):
    """F-beta score à partir de precision/recall scalaires."""
    denom = beta ** 2 * precision + recall
    return (1 + beta ** 2) * precision * recall / denom if denom > 0 else 0.0


def compute_metrics_per_artifact(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calcule les 5 métriques binaires (présence/absence artefact) pour chaque
    des 7 artefacts.

    Args:
        y_true : [N, 7]  sévérité GT   (0/1/2)
        y_pred : [N, 7]  sévérité pred (0/1/2)

    Returns:
        dict {artifact_name: {acc, f1, f2, prec, rec, mean5}}
    """
    results = {}
    for a, name in enumerate(ARTIFACT_COLS):
        gt_bin   = (y_true[:, a] > 0).astype(int)   # 1 = artefact présent
        pred_bin = (y_pred[:, a] > 0).astype(int)

        # Éviter zéro division si une classe est absente en val
        kw = dict(zero_division=0)

        acc  = accuracy_score(gt_bin, pred_bin)
        prec = precision_score(gt_bin, pred_bin, **kw)
        rec  = recall_score(gt_bin, pred_bin, **kw)
        f1   = f1_score(gt_bin, pred_bin, **kw)
        f2   = fbeta_score(prec, rec, beta=2.0)
        mean5 = (acc + f1 + f2 + prec + rec) / 5.0

        results[name] = {
            "accuracy":  acc,
            "f1":        f1,
            "f2":        f2,
            "precision": prec,
            "recall":    rec,
            "mean5":     mean5,
            "n_pos_gt":  int(gt_bin.sum()),
            "n_pos_pred":int(pred_bin.sum()),
            "n_total":   len(gt_bin),
        }
    return results


def print_results(results: dict, checkpoint_name: str = ""):
    """Affiche un tableau lisible et un résumé global."""
    header = f"{'Artifact':14s}  {'Acc':6s}  {'F1':6s}  {'F2':6s}  {'Prec':6s}  {'Rec':6s}  {'Mean5':6s}  {'Pos/GT':8s}"
    sep    = "-" * len(header)
    title  = f"  Task 1a — métriques LISA (binaire par artefact)"
    if checkpoint_name:
        title += f"  [{checkpoint_name}]"

    print()
    print(title)
    print(sep)
    print(header)
    print(sep)

    means = {k: [] for k in ["accuracy", "f1", "f2", "precision", "recall", "mean5"]}

    for name, m in results.items():
        pos_info = f"{m['n_pos_gt']:3d}/{m['n_total']:3d}"
        print(
            f"  {name:12s}  "
            f"{m['accuracy']:.4f}  {m['f1']:.4f}  {m['f2']:.4f}  "
            f"{m['precision']:.4f}  {m['recall']:.4f}  {m['mean5']:.4f}  "
            f"{pos_info}"
        )
        for k in means:
            means[k].append(m[k])

    print(sep)
    avg_line = (
        f"  {'MEAN':12s}  "
        f"{np.mean(means['accuracy']):.4f}  "
        f"{np.mean(means['f1']):.4f}  "
        f"{np.mean(means['f2']):.4f}  "
        f"{np.mean(means['precision']):.4f}  "
        f"{np.mean(means['recall']):.4f}  "
        f"{np.mean(means['mean5']):.4f}"
    )
    print(avg_line)
    print(sep)
    print(
        f"\n  Score global (mean of 5 metrics, mean over artifacts) : "
        f"{np.mean(means['mean5']):.4f}"
    )
    print()

    # ── LISA 2025 best scores (scores numériques non publiés dans le HTML)
    print(
        "  Référence LISA 2025 — top-3 équipes (Task 1) :\n"
        "    🥇 CGP  🥈 MBZ  🥉 UPF\n"
        "    (scores numériques non accessibles via l'API Synapse statique)\n"
        "    cf. https://www.synapse.org/Synapse:syn65670170/wiki/631796"
    )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Évaluation Task 1a — métriques LISA 2025",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Chemin vers le fichier .pt (checkpoint ou best_model.pt)",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # ── chargement checkpoint ──────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint introuvable : {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg  = ckpt.get("config", {})

    # Surcharges CLI
    if args.data_root:
        cfg["data_root"] = args.data_root
    cfg.setdefault("data_root",     DATA_ROOT_DEFAULT)
    cfg.setdefault("target_size",   96)
    cfg.setdefault("val_fraction",  0.2)
    cfg.setdefault("base_channels", 16)
    cfg.setdefault("c_anat",        16)
    cfg.setdefault("c_mod",          8)
    cfg.setdefault("c_art",          8)
    cfg.setdefault("n_artifacts",    7)
    cfg.setdefault("n_severity",     3)
    cfg.setdefault("n_seg_classes", 14)

    # ── device ────────────────────────────────────────────────────────────
    device_str = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)
    print(f"Device : {device}")

    # ── dataset validation ────────────────────────────────────────────────
    ts = (cfg["target_size"],) * 3
    val_ds = LISAJointDataset(
        cfg["data_root"], target_size=ts, split="val",
        val_fraction=cfg["val_fraction"],
    )
    task1a_items = [it for it in val_ds.items if it.get("has_task1a", False)]
    print(f"Val items Task1a : {len(task1a_items)}")

    # ── modèle ────────────────────────────────────────────────────────────
    model = BrainFMLISA(
        base=cfg["base_channels"],
        c_anat=cfg["c_anat"],
        c_mod=cfg["c_mod"],
        c_art=cfg["c_art"],
        n_artifacts=cfg["n_artifacts"],
        n_severity=cfg["n_severity"],
        n_seg_classes=cfg["n_seg_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    epoch_num = ckpt.get("epoch", -1) + 1
    print(f"Modèle chargé — epoch {epoch_num}")

    # ── inférence ─────────────────────────────────────────────────────────
    y_true, y_pred = collect_predictions(
        model, val_ds, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Prédictions collectées : {len(y_true)} items")

    # ── métriques ─────────────────────────────────────────────────────────
    results = compute_metrics_per_artifact(y_true.numpy(), y_pred.numpy())
    print_results(results, checkpoint_name=ckpt_path.name)

    # ── distribution des prédictions ──────────────────────────────────────
    print("  Distribution des sévérités prédites vs GT :")
    for a, name in enumerate(ARTIFACT_COLS):
        gt_counts   = [(y_true[:, a] == s).sum().item() for s in range(3)]
        pred_counts = [(y_pred[:, a] == s).sum().item() for s in range(3)]
        print(
            f"    {name:12s}  GT  [0:{gt_counts[0]:3d} 1:{gt_counts[1]:3d} 2:{gt_counts[2]:3d}]  "
            f"Pred [0:{pred_counts[0]:3d} 1:{pred_counts[1]:3d} 2:{pred_counts[2]:3d}]"
        )
    print()


if __name__ == "__main__":
    main()
