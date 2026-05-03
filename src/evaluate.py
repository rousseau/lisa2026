#!/usr/bin/env python3
"""
Évaluation Task 1a sur le split de validation — métriques LISA 2025.

Deux modes disponibles (--mode) :

  weighted  [défaut] : métriques officielles LISA 2025
    → F1, F2, Acc, Precision, Recall calculés avec average='weighted'
      sur les 3 classes de sévérité (0/1/2) par artefact
    → Mean5 = moyenne des 5 métriques, moyennée sur les 7 artefacts
    → Comparable aux scores publiés (CGP=0.799, BRIQA≈0.799, UPF=0.777)

  binary    : ancienne métrique (présence/absence artefact)
    → positif = sévérité > 0, négatif = sévérité = 0

Usage :
    python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
    python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pt --mode binary
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
from model import BackboneLISA

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "train_default.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Inférence sur le jeu de validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, val_ds, device, batch_size=4, num_workers=4,
                        logit_adj_tau: float = 0.0,
                        log_prior: "torch.Tensor | None" = None):
    """
    Retourne (y_true, y_pred) tenseurs [N, 7] long.
    Seuls les items ayant has_task1a=True sont conservés.

    Si logit_adj_tau > 0 et log_prior est fourni, applique le Logit Adjustment :
        logits[b, a, s] -= tau * log_prior[a, s]
    avant le argmax, ce qui corrige le biais vers les classes majoritaires.
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
        logits = preds["task1a"]          # [B, 7, 2] (ordinal thresholds)

        # Logit Adjustment
        if logit_adj_tau > 0.0 and log_prior is not None:
            # On ajuste les logits des seuils. 
            # Pour simplifier, on applique l'ajustement sur les seuils 1 et 2.
            # On suppose que log_prior est [N_art, 3]
            # On utilise log_prior[a, 1] et log_prior[a, 2] pour ajuster les seuils.
            adj = log_prior[:, 1:].to(logits.device) # [N_art, 2]
            logits = logits - logit_adj_tau * adj.unsqueeze(0)

        # Conversion des seuils en classes (0, 1, 2)
        # pred_sev = sum(sigmoid(logits) > 0.5)
        probs = torch.sigmoid(logits)
        pred_sev = (probs > 0.5).sum(dim=2)   # [B, 7]  classe prédite (0, 1 ou 2)

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


# Scores de référence LISA 2025 — Task 1 (test set officiel)
# Métrique : weighted average sur 3 classes (0/1/2), Mean5 = moy(F1,F2,Acc,Prec,Rec)
# Source : proceedings LISA 2025
LISA2025_REFERENCE = {
    # rang : (équipe, institution, Mean5, weighted_F1)
    1: ("CGP",   "Tsinghua Univ.",           0.799, 0.781),
    2: ("BRIQA", "MBZ Univ.",                0.799, None),
    5: ("UPF",   "Univ. Pompeu Fabra",       0.777, 0.771),
    # BRIQA macro F1 par artefact (pour référence) :
    # Noise=0.725 Zipper=0.731 Positioning=0.732 Banding=0.605
    # Motion=0.625 Contrast=0.698 Distortion=0.657 → mean_macro_F1=0.706
}


def compute_metrics_per_artifact(y_true: np.ndarray, y_pred: np.ndarray,
                                  mode: str = "weighted"):
    """
    Calcule les 5 métriques pour chaque des 7 artefacts.

    Args:
        y_true : [N, 7]  sévérité GT   (0/1/2)
        y_pred : [N, 7]  sévérité pred (0/1/2)
        mode   : 'weighted' (LISA 2025 officiel, 3 classes) ou 'binary'

    Returns:
        dict {artifact_name: {acc, f1, f2, prec, rec, mean5}}
    """
    results = {}
    for a, name in enumerate(ARTIFACT_COLS):
        kw = dict(zero_division=0)

        if mode == "binary":
            gt   = (y_true[:, a] > 0).astype(int)
            pred = (y_pred[:, a] > 0).astype(int)
            avg  = "binary"
        else:  # weighted — métrique officielle LISA 2025
            gt   = y_true[:, a]
            pred = y_pred[:, a]
            avg  = "weighted"

        acc  = accuracy_score(gt, pred)
        prec = precision_score(gt, pred, average=avg, **kw)
        rec  = recall_score(gt, pred, average=avg, **kw)
        f1   = f1_score(gt, pred, average=avg, **kw)
        f2   = fbeta_score(prec, rec, beta=2.0)
        mean5 = (acc + f1 + f2 + prec + rec) / 5.0

        if mode == "binary":
            n_pos_gt   = int((gt   > 0).sum())
            n_pos_pred = int((pred > 0).sum())
        else:
            n_pos_gt   = int((gt   > 0).sum())
            n_pos_pred = int((pred > 0).sum())

        results[name] = {
            "accuracy":  acc,
            "f1":        f1,
            "f2":        f2,
            "precision": prec,
            "recall":    rec,
            "mean5":     mean5,
            "n_pos_gt":  n_pos_gt,
            "n_pos_pred":n_pos_pred,
            "n_total":   len(gt),
        }
    return results


def print_results(results: dict, checkpoint_name: str = "", mode: str = "weighted"):
    """Affiche un tableau lisible et un résumé global."""
    header = f"{'Artifact':14s}  {'Acc':6s}  {'F1':6s}  {'F2':6s}  {'Prec':6s}  {'Rec':6s}  {'Mean5':6s}  {'Pos/GT':8s}"
    sep    = "-" * len(header)
    mode_label = "weighted 3-classes (LISA 2025 officiel)" if mode == "weighted" else "binaire (présent/absent)"
    title  = f"  Task 1a — métriques {mode_label}"
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
    our_mean5 = np.mean(means['mean5'])
    print(sep)
    print(
        f"\n  Score global (Mean5) : {our_mean5:.4f}"
    )

    if mode == "weighted":
        print()
        print("  ── Comparaison LISA 2025 (test set officiel, même métrique weighted) ──")
        print(f"  {'Rang':5s}  {'Équipe':8s}  {'Institution':26s}  {'Mean5':6s}  {'wF1':6s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*26}  {'-'*6}  {'-'*6}")
        for rang, (team, inst, m5, wf1) in sorted(LISA2025_REFERENCE.items()):
            wf1_str = f"{wf1:.3f}" if wf1 is not None else "  — "
            print(f"  {rang:<5d}  {team:<8s}  {inst:<26s}  {m5:.3f}   {wf1_str}")
        print(f"  {'—':5s}  {'NOUS':8s}  {'(val set, pas test)':26s}  {our_mean5:.3f}")
        print()
        print("  Note : nos scores sont sur le VAL set local (≠ test set officiel LISA).")
        print("         Les scores LISA 2025 ci-dessus sont sur le TEST set officiel.")
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
    p.add_argument(
        "--logit-adj-tau", type=float, default=0.0,
        help="Logit Adjustment : tau > 0 soustrait tau*log(prior) des logits avant argmax."
             " tau=1.0 = correction complète du biais de classe (recommandé : 0.5–1.0).",
    )
    p.add_argument(
        "--mode", choices=["weighted", "binary"], default="weighted",
        help="Mode de calcul des métriques : "
             "'weighted' = LISA 2025 officiel (average='weighted' sur 3 classes 0/1/2) ; "
             "'binary' = présence/absence artefact (ancienne métrique).",
    )
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
    epoch_num = ckpt.get("epoch", -1) + 1
    print(f"Modèle chargé — epoch {epoch_num}")

    # ── inférence ─────────────────────────────────────────────────────────
    # Logit Adjustment : calcul du log-prior depuis le train set
    log_prior = None
    if args.logit_adj_tau > 0.0:
        from dataset import compute_task1a_weights  # import local pour éviter la dépendance
        train_ds_tmp = LISAJointDataset(
            cfg["data_root"], target_size=ts, split="train",
            val_fraction=cfg["val_fraction"],
        )
        n_art = cfg["n_artifacts"]
        n_sev = cfg["n_severity"]
        counts = torch.zeros(n_art, n_sev)
        for it in train_ds_tmp.items:
            if not it.get("has_task1a", False):
                continue
            for a in range(n_art):
                s = int(it["task1a_labels"][a].item())
                counts[a, s] += 1
        counts = counts + 1e-6   # lissage de Laplace pour éviter log(0)
        prior     = counts / counts.sum(dim=1, keepdim=True)
        log_prior = torch.log(prior)   # [N_art, N_sev]
        print(f"Logit Adjustment tau={args.logit_adj_tau} — log-prior par artefact :")
        for a, name in enumerate(ARTIFACT_COLS):
            lp = log_prior[a].numpy()
            print(f"  {name:12s}  sev0={lp[0]:+.2f}  sev1={lp[1]:+.2f}  sev2={lp[2]:+.2f}")

    y_true, y_pred = collect_predictions(
        model, val_ds, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logit_adj_tau=args.logit_adj_tau,
        log_prior=log_prior,
    )
    print(f"Prédictions collectées : {len(y_true)} items")

    # ── métriques ─────────────────────────────────────────────────────────
    results = compute_metrics_per_artifact(y_true.numpy(), y_pred.numpy(), mode=args.mode)
    print_results(results, checkpoint_name=ckpt_path.name, mode=args.mode)

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
