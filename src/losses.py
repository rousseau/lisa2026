#!/usr/bin/env python3
"""
Fonctions de loss multi-tâches pour LISA 2026.

Task 1a  – Quality Control     : cross-entropy par artefact (3 classes de sévérité)
Task 1b  – Enhancement         : L1 de reconstruction (masqué sur images sans artefact)
Task 2   – Segmentation         : cross-entropy + Dice (14 classes : fond + 13 structures)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ARTIFACTS   = 7
N_SEVERITY    = 3
N_SEG_CLASSES = 14


# ──────────────────────────────────────────────────────────────────────────────
# Task 1a
# ──────────────────────────────────────────────────────────────────────────────

def task1a_loss(
    logits:          torch.Tensor,               # [B, N_art, N_sev]
    labels:          torch.Tensor,               # [B, N_art]  long, valeurs 0/1/2
    mask:            torch.Tensor,               # [B] bool – échantillons avec labels Task1a
    sev_weights:     torch.Tensor | None = None, # [N_art, N_sev] poids de sévérité par artefact
    art_weights:     torch.Tensor | None = None, # [N_art] poids inter-artefact
    label_smoothing: float = 0.0,                # lissage de label (0 → désactivé)
) -> torch.Tensor:
    """
    Cross-entropy pondérée par artefact.

    Pour chaque type d'artefact ``a`` :
      - ``sev_weights[a]`` est passé comme ``weight`` à ``F.cross_entropy`` afin
        de corriger le déséquilibre de sévérité (sev=0 très majoritaire).
      - ``art_weights[a]`` pondère la contribution de chaque artefact à la loss
        totale (les artefacts rares, ex. Banding, reçoivent un poids plus élevé).

    Les poids sont calculés par ``dataset.compute_task1a_weights`` à partir du
    jeu d'entraînement et passés ici depuis ``multi_task_loss``.
    """
    if not mask.any():
        return logits.sum() * 0.0   # gradient nul mais graph connecté

    dev      = logits.device
    logits_m = logits[mask]   # [M, N_art, N_sev]
    labels_m = labels[mask]   # [M, N_art]
    N        = logits_m.shape[1]

    ce_per_art = []
    for a in range(N):
        w = sev_weights[a].to(dev) if sev_weights is not None else None
        ce_a = F.cross_entropy(
            logits_m[:, a, :], labels_m[:, a],
            weight=w, label_smoothing=label_smoothing,
        )
        ce_per_art.append(ce_a)

    ce_per_art = torch.stack(ce_per_art)       # [N_art]
    if art_weights is not None:
        ce_per_art = ce_per_art * art_weights.to(dev)

    return ce_per_art.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Task 1b
# ──────────────────────────────────────────────────────────────────────────────

def task1b_loss(
    recon:  torch.Tensor,   # [B, 1, D, H, W]  reconstruction
    target: torch.Tensor,   # [B, 1, D, H, W]  image originale (normalisée)
    mask:   torch.Tensor,   # [B] bool – images sans artefact
) -> torch.Tensor:
    """
    L1 de reconstruction, calculée uniquement sur les images sans artefact.
    """
    if not mask.any():
        return recon.sum() * 0.0

    return F.l1_loss(recon[mask], target[mask])


# ──────────────────────────────────────────────────────────────────────────────
# Task 2
# ──────────────────────────────────────────────────────────────────────────────

def _soft_dice_loss(
    prob:     torch.Tensor,   # [B, C, D, H, W]  après softmax
    one_hot:  torch.Tensor,   # [B, C, D, H, W]  float
    eps: float = 1e-5,
) -> torch.Tensor:
    """Dice loss souple (soft) moyenné sur toutes les classes et le batch."""
    B, C = prob.shape[:2]
    p = prob.view(B, C, -1)
    t = one_hot.view(B, C, -1)
    intersection = (p * t).sum(-1)
    cardinality  = p.sum(-1) + t.sum(-1)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def task2_loss(
    logits: torch.Tensor,   # [B, C, D, H, W]
    seg_gt: torch.Tensor,   # [B, D, H, W]  long, 0=fond
    mask:   torch.Tensor,   # [B] bool – images ciso avec GT segmentation
    n_classes: int = N_SEG_CLASSES,
) -> torch.Tensor:
    """
    Cross-entropy + Dice soft, calculé uniquement sur les images ciso annotées.
    """
    if not mask.any():
        return logits.sum() * 0.0

    logits_m = logits[mask]   # [M, C, D, H, W]
    seg_m    = seg_gt[mask]   # [M, D, H, W]

    ce   = F.cross_entropy(logits_m, seg_m)
    prob = F.softmax(logits_m, dim=1)
    one_hot = (
        F.one_hot(seg_m, n_classes)
        .permute(0, 4, 1, 2, 3)
        .float()
    )
    dc = _soft_dice_loss(prob, one_hot)
    return ce + dc


# ──────────────────────────────────────────────────────────────────────────────
# Loss combinée
# ──────────────────────────────────────────────────────────────────────────────

def multi_task_loss(
    preds:          dict,
    batch:          dict,
    lam:            tuple[float, float, float] = (1.0, 1.0, 1.0),
    device:         torch.device | None = None,
    task1a_weights: tuple | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Agrège les trois losses.

    Args:
        preds          : sortie de BrainFMLISA.forward()
        batch          : dict du DataLoader
        lam            : (λ_1a, λ_1b, λ_2)  poids des tâches
        device         : device cible (inféré depuis preds si None)
        task1a_weights : (sev_weights [N_art,N_sev], art_weights [N_art]) ou None

    Returns:
        (total_loss, dict des losses individuelles)
    """
    if device is None:
        device = next(iter(preds.values())).device

    losses: dict = {}
    total = torch.tensor(0.0, device=device)

    # ── Task 1a : quality control ────────────────────────────────────────────
    if "task1a" in preds:
        mask_1a  = batch["has_task1a"].to(device)
        sev_w, art_w = task1a_weights if task1a_weights is not None else (None, None)
        l1a = task1a_loss(
            preds["task1a"],
            batch["task1a_labels"].to(device),
            mask_1a,
            sev_weights=sev_w,
            art_weights=art_w,
            label_smoothing=label_smoothing,
        )
        losses["task1a"] = l1a
        total = total + lam[0] * l1a

    # ── Task 1b : reconstruction (sans artefact uniquement) ──────────────────
    if "task1b" in preds:
        mask_1b = batch["is_artifact_free"].to(device)
        l1b = task1b_loss(
            preds["task1b"],
            batch["image"].to(device),
            mask_1b,
        )
        losses["task1b"] = l1b
        total = total + lam[1] * l1b

    # ── Task 2 : segmentation (ciso avec GT) ─────────────────────────────────
    if "task2" in preds:
        mask_2 = (batch["is_isotropic"] & batch["has_seg"]).to(device)
        l2 = task2_loss(
            preds["task2"],
            batch["seg"].to(device),
            mask_2,
        )
        losses["task2"] = l2
        total = total + lam[2] * l2

    losses["total"] = total
    return total, losses
