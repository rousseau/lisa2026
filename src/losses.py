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

def _ordinal_emd_loss(
    logits: torch.Tensor,               # [M, N_sev]
    labels: torch.Tensor,               # [M]  long 0/1/2
    weight: torch.Tensor | None = None, # [N_sev]  poids de sévérité
) -> torch.Tensor:
    """
    Earth Mover's Distance (L1 sur la CDF cumulée) pour classification ordinale.

    Pour K=3 sévérités (0/1/2), deux seuils de CDF :
        label=0  →  CDF cible = [1, 1]   (P(Y≤0)=1, P(Y≤1)=1)
        label=1  →  CDF cible = [0, 1]   (P(Y≤0)=0, P(Y≤1)=1)
        label=2  →  CDF cible = [0, 0]   (P(Y≤0)=0, P(Y≤1)=0)

    Pénalise proportionnellement à la distance ordinale : une erreur 0→2
    coûte 2× plus qu'une erreur 0→1.
    """
    M, K = logits.shape
    prob     = F.softmax(logits, dim=1)               # [M, K]
    pred_cdf = torch.cumsum(prob, dim=1)[:, :-1]      # [M, K-1]

    # CDF cible : target_cdf[i, k] = 1 si labels[i] <= k, sinon 0
    target_cdf = torch.zeros(M, K - 1, device=logits.device, dtype=torch.float32)
    for k in range(K - 1):
        target_cdf[:, k] = (labels <= k).float()

    emd = torch.abs(pred_cdf - target_cdf)            # [M, K-1]

    if weight is not None:
        # Pondérer chaque exemple par le poids de sa vraie classe
        sample_w = weight[labels.clamp(0, K - 1)]     # [M]
        emd = emd * sample_w.unsqueeze(1)

    return emd.mean()


def task1a_loss(
    logits:          torch.Tensor,               # [B, N_art, N_sev-1]
    labels:          torch.Tensor,               # [B, N_art]  long, valeurs 0/1/2
    mask:            torch.Tensor,               # [B] bool – échantillons avec labels Task1a
    sev_weights:     torch.Tensor | None = None, # [N_art, N_sev] poids de sévérité par artefact
    art_weights:     torch.Tensor | None = None, # [N_art] poids inter-artefact,
    label_smoothing: float = 0.0,                # lissage de label (0 → désactivé),
    ordinal_weight:  float = 0.0,                # ignoré ici, on utilise la BCE ordinale
    focal_gamma:     float = 0.0,                # focus factor (0 = désactivé)
    focal_alpha:     torch.Tensor | None = None, # [N_sev] poids par sévérité pour focal
    ohem_ratio:      float = 0.0,                # fraction d'exemples durs (0 = désactivé)
    ohem_anneal:     bool = False,               # annealing du ratio OHEM
    ohem_epoch:      int = 0,                    # epoch courant pour annealing
    ohem_max_epoch:  int = 200,                  # epoch max pour annealing
    ohem_min_ratio:  float = 0.1,                # ratio minimum OHEM
) -> torch.Tensor:
    """
    Loss pour classification ordinale via Binary Cross Entropy sur les seuils.

    On transforme le label y ∈ {0, 1, 2} en un vecteur binaire de seuils :
        y=0  → [0, 0]
        y=1  → [1, 0]
        y=2  → [1, 1]
    
    La loss est la moyenne des BCE sur ces seuils.
    """
    if not mask.any():
        return logits.sum() * 0.0

    dev      = logits.device
    logits_m = logits[mask]   # [M, N_art, N_sev-1]
    labels_m = labels[mask]   # [M, N_art]
    N_art    = logits_m.shape[1]
    N_sev_minus_1 = logits_m.shape[2]

    # Transformation des labels en cibles binaires (seuils)
    # target[m, a, k] = 1 si labels[m, a] >= k+1
    target = torch.zeros_like(logits_m)
    for k in range(N_sev_minus_1):
        target[:, :, k] = (labels_m >= (k + 1)).float()

    # BCE pondérée (avec logits pour compatibilité AMP)
    loss_val = F.binary_cross_entropy_with_logits(logits_m, target, reduction='none') # [M, N_art, N_sev-1]
    
    # Moyenne par artefact et par échantillon
    loss_per_art = loss_val.mean(dim=-1) # [M, N_art]
    
    if art_weights is not None:
        aw = art_weights.to(dev)
        loss_per_art = loss_per_art * aw
        
    return loss_per_art.mean()


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

def _generalized_dice_loss(
    prob: torch.Tensor,      # [B, C, D, H, W] après softmax
    one_hot: torch.Tensor,   # [B, C, D, H, W] float
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Generalized Dice Loss (Sudre et al.) robuste au fort déséquilibre de classes.

    Les poids de classes sont inversement proportionnels au volume GT au carré,
    ce qui renforce les petites structures (ex: hippocampes).
    """
    p = prob.reshape(prob.shape[0], prob.shape[1], -1)
    t = one_hot.reshape(one_hot.shape[0], one_hot.shape[1], -1)

    # Somme sur batch + voxels pour obtenir les volumes par classe.
    class_vol = t.sum(dim=(0, 2))
    weights = torch.where(
        class_vol > 0,
        1.0 / (class_vol * class_vol + eps),
        torch.zeros_like(class_vol),
    )

    inter = (p * t).sum(dim=(0, 2))
    denom = p.sum(dim=(0, 2)) + t.sum(dim=(0, 2))

    numerator = 2.0 * (weights * inter).sum()
    denominator = (weights * denom).sum() + eps
    dice = numerator / denominator
    return 1.0 - dice


def _weighted_cross_entropy_loss(
    logits: torch.Tensor,    # [B, C, D, H, W]
    target: torch.Tensor,    # [B, D, H, W]
    n_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Cross-Entropy pondérée par classe (poids inverse fréquence sur le batch).
    """
    flat = target.reshape(-1)
    hist = torch.bincount(flat, minlength=n_classes).float()

    # Poids inverse fréquence, 0 pour classes absentes.
    weights = torch.where(hist > 0, 1.0 / (hist + eps), torch.zeros_like(hist))

    # Normalisation pour garder une échelle stable (moyenne des classes présentes = 1).
    present = weights > 0
    if present.any():
        weights[present] = weights[present] / weights[present].mean()
    else:
        weights = torch.ones_like(weights)

    return F.cross_entropy(logits, target, weight=weights.to(logits.device))


def task2_loss(
    logits: torch.Tensor,   # [B, C, D, H, W]
    seg_gt: torch.Tensor,   # [B, D, H, W]  long, 0=fond
    mask:   torch.Tensor,   # [B] bool – images ciso avec GT segmentation
    n_classes: int = N_SEG_CLASSES,
) -> torch.Tensor:
    """
    Weighted Cross-Entropy + Generalized Dice,
    calculé uniquement sur les images ciso annotées.
    """
    if not mask.any():
        return logits.sum() * 0.0

    logits_m = logits[mask]   # [M, C, D, H, W]
    seg_m    = seg_gt[mask]   # [M, D, H, W]

    ce = _weighted_cross_entropy_loss(logits_m, seg_m, n_classes=n_classes)
    prob = F.softmax(logits_m, dim=1)
    one_hot = (
        F.one_hot(seg_m, n_classes)
        .permute(0, 4, 1, 2, 3)
        .float()
    )
    dc = _generalized_dice_loss(prob, one_hot)
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
    ordinal_weight:  float = 0.0,
    focal_gamma:     float = 0.0,
    focal_alpha:     torch.Tensor | None = None,
    ohem_ratio:      float = 0.0,
    ohem_anneal:     bool = False,
    ohem_epoch:      int = 0,
    ohem_max_epoch:  int = 200,
    ohem_min_ratio:  float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Agrège les trois losses.

    Args:
        preds          : sortie de BackboneLISA.forward()
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
            ordinal_weight=ordinal_weight,
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


# ──────────────────────────────────────────────────────────────────────────────
# Régularisation de factorisation (v8+)
# ──────────────────────────────────────────────────────────────────────────────

def orthogonality_loss(
    z_a: torch.Tensor,   # [B, c_a, D, H, W]
    z_b: torch.Tensor,   # [B, c_b, D, H, W]
) -> torch.Tensor:
    """
    Contrainte d'orthogonalité douce entre deux sous-espaces latents.

    1. Global Average Pooling sur les dimensions spatiales → [B, c_X]
    2. Calcule la norme de Frobenius du produit matriciel Z_a^T Z_b

    L_orth = ||Z_a^T Z_b||_F / (B * c_a * c_b)

    → Zéro si les projections moyennes sont orthogonales, pénalise
      proportionnellement à leur redondance.
    """
    # GAP : [B, c_X, D, H, W] → [B, c_X]
    a = z_a.flatten(2).mean(dim=-1)   # [B, c_a]
    b = z_b.flatten(2).mean(dim=-1)   # [B, c_b]

    # Normalisation L2 par échantillon pour comparer les directions
    a = F.normalize(a, dim=-1)        # [B, c_a]
    b = F.normalize(b, dim=-1)        # [B, c_b]

    # Produit croisé : [B, c_a] × [B, c_b]^T  →  [B, B]  (corrélation inter-batch)
    # On veut les corrélations dans l'espace des features, pas dans le batch :
    # cov [c_a, c_b] = A^T @ B  /  B
    cov = (a.T @ b) / a.shape[0]      # [c_a, c_b]
    return (cov ** 2).sum().sqrt() / (z_a.shape[1] * z_b.shape[1])


def vicreg_cov_loss(
    z: torch.Tensor,          # [B, c, D, H, W]
    gamma: float = 1.0,       # seuil de variance cible
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Pénalité de variance + covariance (VICReg, Bardes et al. 2022).

    Appliquée sur une seule vue (pas besoin de 2 augmentations) :
      - Variance  : max(0, γ − std(z_d)) pour chaque dimension d
        → force chaque dimension à être informative (évite le collapse)
      - Covariance: pénalise les éléments hors-diagonale de cov(z)/B
        → force les dimensions à être décorrélées

    z : [B, c, D, H, W] — GAP spatial → [B, c]
    """
    # GAP spatial
    z_flat = z.flatten(2).mean(dim=-1)   # [B, c]
    B, C   = z_flat.shape

    # Cas limite : batch trop petit pour estimer correctement cov/std.
    if B < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    # Centrage
    z_flat = z_flat - z_flat.mean(dim=0, keepdim=True)

    # Variance : on pénalise les dimensions sous γ
    # unbiased=False évite les NaN quand B est faible.
    std = z_flat.std(dim=0, unbiased=False) + eps             # [C]
    loss_v = F.relu(gamma - std).mean()

    # Covariance : pénalise les éléments hors-diagonale
    cov = (z_flat.T @ z_flat) / max(B - 1, 1)                # [C, C]
    diag = torch.eye(C, device=z.device, dtype=z.dtype)
    loss_c = (cov[~diag.bool()] ** 2).sum() / C

    out = loss_v + loss_c
    if not torch.isfinite(out):
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    return out


def factorization_loss(
    preds:      dict,
    lam_orth:   float = 0.05,
    lam_vicreg: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Agrège les pertes de régularisation de factorisation :
      - Orthogonalité entre z_anat et z_mod
      - Orthogonalité entre z_anat et z_art
      - VICReg (var + cov) sur chacun des 3 sous-espaces

    Retourne (total_reg_loss, dict des composantes).
    """
    z_anat = preds.get("z_anat")
    z_mod  = preds.get("z_mod")
    z_art  = preds.get("z_art")

    reg_losses = {}
    total_reg  = torch.tensor(0.0, device=z_anat.device)

    if lam_orth > 0:
        lo1 = orthogonality_loss(z_anat, z_mod)
        lo2 = orthogonality_loss(z_anat, z_art)
        reg_losses["orth_anat_mod"] = lo1
        reg_losses["orth_anat_art"] = lo2
        total_reg = total_reg + lam_orth * (lo1 + lo2)

    if lam_vicreg > 0:
        lv1 = vicreg_cov_loss(z_anat)
        lv2 = vicreg_cov_loss(z_mod)
        lv3 = vicreg_cov_loss(z_art)
        reg_losses["vicreg_anat"] = lv1
        reg_losses["vicreg_mod"]  = lv2
        reg_losses["vicreg_art"]  = lv3
        total_reg = total_reg + lam_vicreg * (lv1 + lv2 + lv3)

    reg_losses["total_reg"] = total_reg
    return total_reg, reg_losses
