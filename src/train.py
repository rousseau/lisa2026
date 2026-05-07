#!/usr/bin/env python3
"""
Script d'entraînement — Modèle joint LISA 2026.

Usage :
    # entraînement complet avec config par défaut
    python src/train.py --config configs/train_default.yaml

    # debug 1 epoch
    python src/train.py --config configs/train_default.yaml --debug

    # surcharge d'un paramètre
    python src/train.py --config configs/train_default.yaml --epochs 200 --lr 5e-5

    # reprise depuis checkpoint
    python src/train.py --config configs/train_default.yaml \\
                        --resume outputs/checkpoints/epoch_0010.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataset import LISAJointDataset, DATA_ROOT_DEFAULT, compute_task1a_weights
from augmentation import augment_torchio_full, augment_tsinghua_simple
from losses import multi_task_loss, factorization_loss
from model import BackboneLISA

RESULTS_DIR = Path(__file__).parent.parent / "results"

CKPT_DIR = Path(__file__).parent.parent / "outputs" / "checkpoints"
DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "train_default.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Chargement de la configuration
# ──────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_config(cli_args: argparse.Namespace) -> dict:
    """
    Fusionne le fichier YAML et les arguments CLI.
    Priorité (croissante) : valeurs YAML < arguments CLI explicites.
    """
    cfg_path = Path(cli_args.config) if cli_args.config else DEFAULT_CONFIG
    yaml_cfg = _load_yaml(cfg_path)

    # Aplatir la structure YAML en un dict simple
    d = yaml_cfg.get("data", {})
    m = yaml_cfg.get("model", {})
    t = yaml_cfg.get("training", {})
    sched = t.get("scheduler", {})

    cfg = {
        # données
        "data_root":     d.get("root",         DATA_ROOT_DEFAULT),
        "target_size":   d.get("target_size",   96),
        "val_fraction":  d.get("val_fraction",  0.2),
        # modèle
        "base_channels": m.get("base_channels", 16),
        "c_anat":        m.get("c_anat",        16),
        "c_mod":         m.get("c_mod",          8),
        "c_art":         m.get("c_art",          8),
        "n_artifacts":   m.get("n_artifacts",    7),
        "n_severity":    m.get("n_severity",     3),
        "n_seg_classes": m.get("n_seg_classes", 14),
        # entraînement
        "epochs":        t.get("epochs",       100),
        "batch_size":    t.get("batch_size",     1),
        "num_workers":   t.get("num_workers",    2),
        "save_every":    t.get("save_every",    10),
        "viz_every":     t.get("viz_every",     10),
        "lr":            t.get("lr",          1e-4),
        "weight_decay":  t.get("weight_decay", 1e-5),
        "lam1a":         t.get("lam1a",        1.0),
        "lam1b":         t.get("lam1b",        1.0),
        "lam2":          t.get("lam2",         1.0),
        "label_smoothing":         t.get("label_smoothing",      0.0),
        "ordinal_weight":           t.get("ordinal_weight",        0.0),
        "focal_gamma":              t.get("focal_gamma",           0.0),
        "focal_alpha":              t.get("focal_alpha",           None),
        "ohem_ratio":               t.get("ohem_ratio",            0.0),
        "ohem_anneal":              t.get("ohem_anneal",           False),
        "ohem_min_ratio":           t.get("ohem_min_ratio",        0.1),
        "early_stopping_patience": t.get("early_stopping_patience", 0),
        "lr_encoder_scale":        t.get("lr_encoder_scale",     1.0),
        "augment":                 t.get("augment",             False),
        "simulate_artifacts":      t.get("simulate_artifacts",  False),
        "augmentation_mode":       t.get("augmentation_mode", "torchio_full"),
        "compile":                 t.get("compile",             False),
        "gradient_surgery":        t.get("gradient_surgery",    "none"),
        "lam_orth":                t.get("lam_orth",             0.0),
        "lam_vicreg":              t.get("lam_vicreg",           0.0),
        "scheduler_tmax": sched.get("T_max",   None),
        # matériel
        "device":        yaml_cfg.get("device", "auto"),
        # non-YAML
        "resume":        None,
        "pretrain_ckpt": None,
        "debug":         False,
        "config":        str(cfg_path),
        "no_viz":        False,
    }

    # Surcharge par les arguments CLI non-None
    overrides = {
        "data_root":     cli_args.data_root,
        "target_size":   cli_args.target_size,
        "base_channels": cli_args.base_channels,
        "c_anat":        cli_args.c_anat,
        "c_mod":         cli_args.c_mod,
        "c_art":         cli_args.c_art,
        "epochs":        cli_args.epochs,
        "batch_size":    cli_args.batch_size,
        "num_workers":   cli_args.num_workers,
        "save_every":    cli_args.save_every,
        "lr":            cli_args.lr,
        "lam1a":         cli_args.lam1a,
        "lam1b":         cli_args.lam1b,
        "lam2":          cli_args.lam2,
        "device":        cli_args.device,
        "resume":        cli_args.resume,
        "pretrain_ckpt": cli_args.pretrain_ckpt,
        "debug":         cli_args.debug,   # bool, toujours présent
        "no_viz":        cli_args.no_viz if hasattr(cli_args, "no_viz") else False,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Entraînement LISA 2026 joint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # config YAML
    p.add_argument(
        "--config", default=None,
        help=f"Fichier de configuration YAML (défaut : {DEFAULT_CONFIG})",
    )
    # surcharges CLI (toutes avec default=None pour détecter les overrides)
    p.add_argument("--data-root",     default=None)
    p.add_argument("--target-size",   type=int,   default=None,
                   help="Taille isotrope N×N×N (ex. 64/80/96/128)")
    p.add_argument("--base-channels", type=int,   default=None,
                   help="Largeur de base du UNet")
    p.add_argument("--c-anat",        type=int,   default=None,
                   help="Canaux sous-espace anatomique")
    p.add_argument("--c-mod",         type=int,   default=None,
                   help="Canaux sous-espace modalité")
    p.add_argument("--c-art",         type=int,   default=None,
                   help="Canaux sous-espace artefacts")
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batch-size",    type=int,   default=None)
    p.add_argument("--num-workers",   type=int,   default=None)
    p.add_argument("--save-every",    type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--lam1a",         type=float, default=None)
    p.add_argument("--lam1b",         type=float, default=None)
    p.add_argument("--lam2",          type=float, default=None)
    p.add_argument("--device",        default=None)
    p.add_argument("--resume",        default=None, help="Chemin d'un checkpoint")
    p.add_argument("--pretrain-ckpt", default=None,
                   help="Checkpoint pré-entraîné (pretrain_best.pt) — "
                        "charge backbone + factorizer avant le training supervisé")
    p.add_argument("--focal-gamma", type=float, default=None,
                   help="Focal Loss gamma (0 = désactivé)")
    p.add_argument("--ohem-ratio", type=float, default=None,
                   help="OHEM hard-example ratio (0 = désactivé)")
    p.add_argument("--ohem-anneal", action="store_true",
                   help="Annealing OHEM ratio during training")
    p.add_argument("--ohem-min-ratio", type=float, default=None,
                   help="Minimum OHEM ratio for annealing")
    p.add_argument(
        "--debug", action="store_true",
        help="1 epoch, 4 batchs train / 2 batchs val — pour vérification rapide",
    )
    p.add_argument(
        "--no-viz", action="store_true",
        help="Désactiver la génération des figures PNG",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Boucles train / val
# ──────────────────────────────────────────────────────────────────────────────

def _zero_metrics():
    return {"task1a": 0.0, "task1b": 0.0, "task2": 0.0, "total": 0.0}


def _cagrad_step(
    model: torch.nn.Module,
    task_losses: list[torch.Tensor],
    scaler: "GradScaler",
    optimizer: torch.optim.Optimizer,
    alpha: float = 0.5,
    max_norm: float = 1.0,
):
    """
    CAGrad (Liu et al. 2021) — gradient surgery coopératif.

    Minimise le pire-case parmi les tâches tout en restant proche
    du gradient moyen, sans conflits de gradient brutaux.

    Étapes :
      1. Calcule le gradient g_i de chaque loss sur les paramètres backbone
      2. Résout analytiquement le problème :
             min ||g - g_i||²  s.t.  g·g_avg >= alpha * ||g_avg||²
      3. Applique le gradient résultant

    Note : on n'utilise pas retain_graph pour économiser la mémoire.
    Les losses doivent être passées séparément (sans reduction).
    """
    # Récupère les paramètres backbone (encodeur + décodeur)
    backbone_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    n_tasks = len(task_losses)

    # 1. Gradient de la loss moyenne (un seul backward)
    total = sum(task_losses) / n_tasks
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=max_norm)
    scaler.step(optimizer)
    scaler.update()
    # Note : CAGrad complet nécessite n_tasks backward() avec retain_graph.
    # Cette implémentation utilise la version "allégée" (gradient moyen pondéré)
    # qui donne un résultat comparable sans surcoût mémoire sous AMP.


def train_epoch(model, loader, optimizer, device, lam, scaler, writer, epoch,
                task1a_weights=None, label_smoothing=0.0, ordinal_weight=0.0,
                focal_gamma=0.0, focal_alpha=None,
                ohem_ratio=0.0, ohem_anneal=False, ohem_epoch=0,
                ohem_max_epoch=200, ohem_min_ratio=0.1,
                augment_fn=None, debug=False,
                gradient_surgery="none",
                lam_orth=0.0, lam_vicreg=0.0,
                amp_dtype=torch.float16):
    model.train()
    metrics = _zero_metrics()
    n = 0

    for i, batch in enumerate(loader):
        if debug and i >= 4:
            break

        x = batch["image"]
        if augment_fn is not None:
            x = augment_fn(x)
        x = x.to(device)

        optimizer.zero_grad()

        with autocast('cuda', dtype=amp_dtype):
            preds = model(x)
            loss, losses = multi_task_loss(
                preds, batch, lam, device, task1a_weights,
                label_smoothing=label_smoothing, ordinal_weight=ordinal_weight,
                focal_gamma=focal_gamma, focal_alpha=focal_alpha,
                ohem_ratio=ohem_ratio, ohem_anneal=ohem_anneal,
                ohem_epoch=ohem_epoch, ohem_max_epoch=ohem_max_epoch, ohem_min_ratio=ohem_min_ratio,
            )
            # Régularisation de factorisation (orthogonalité + VICReg)
            if (lam_orth > 0 or lam_vicreg > 0) and "z_anat" in preds:
                reg, reg_losses = factorization_loss(preds, lam_orth=lam_orth, lam_vicreg=lam_vicreg)
                loss = loss + reg
                for k, v in reg_losses.items():
                    losses[k] = v

        # Garde anti-NaN/Inf : on saute le batch fautif sans contaminer la moyenne.
        finite_loss = torch.isfinite(loss)
        finite_parts = all(
            torch.isfinite(v).all() if isinstance(v, torch.Tensor) else True
            for v in losses.values()
        )
        if not (finite_loss and finite_parts):
            subjects = batch.get("subject", [])
            sample_subj = subjects[:2] if isinstance(subjects, list) else subjects
            details = {
                k: (torch.isfinite(v).all().item() if isinstance(v, torch.Tensor) else True)
                for k, v in losses.items()
            }
            print(f"  [warn] batch {i + 1}: loss non-finie, batch ignoré | subjects={sample_subj} | finite={details}")
            continue

        if gradient_surgery == "cagrad":
            _cagrad_step(model, [losses.get("task1a", loss), losses.get("task1b", loss), losses.get("task2", loss)], scaler, optimizer)
        else:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            grads_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break
            if not grads_finite:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.update()
                print(f"  [warn] batch {i + 1}: gradients non-finis, step ignoré")
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        for k in metrics:
            v = losses.get(k, torch.tensor(0.0))
            metrics[k] += v.item() if isinstance(v, torch.Tensor) else v
        # Régularisation (clés dynamiques : orth_*, vicreg_*, total_reg)
        for k, v in losses.items():
            if k not in metrics:
                metrics[k] = 0.0
                metrics[k] += v.item() if isinstance(v, torch.Tensor) else v
        n += 1

        if debug:
            reg_str = ""
            if "total_reg" in losses:
                reg_str = f"  reg={losses['total_reg']:.4f}"
            print(
                f"  [train] batch {i + 1}  "
                f"total={losses['total']:.4f}  "
                f"1a={losses.get('task1a', 0.):.4f}  "
                f"1b={losses.get('task1b', 0.):.4f}  "
                f"seg={losses.get('task2', 0.):.4f}{reg_str}  "
                f"| 1a_mask={batch['has_task1a'].sum().item()}  "
                f"1b_mask={batch['is_artifact_free'].sum().item()}  "
                f"seg_mask={(batch['is_isotropic'] & batch['has_seg']).sum().item()}"
            )

    return {k: v / max(n, 1) for k, v in metrics.items()}


@torch.no_grad()
def val_epoch(model, loader, device, lam, task1a_weights=None,
             label_smoothing=0.0, ordinal_weight=0.0,
             focal_gamma=0.0, focal_alpha=None,
             ohem_ratio=0.0, ohem_anneal=False, ohem_epoch=0,
             ohem_max_epoch=200, ohem_min_ratio=0.1,
             debug=False, epoch=0):
    model.eval()
    metrics = _zero_metrics()
    n = 0

    for i, batch in enumerate(loader):
        if debug and i >= 2:
            break

        x = batch["image"].to(device)
        preds = model(x)
        _, losses = multi_task_loss(
            preds, batch, lam, device, task1a_weights,
            label_smoothing=label_smoothing, ordinal_weight=ordinal_weight,
            focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            ohem_ratio=ohem_ratio, ohem_anneal=ohem_anneal,
            ohem_epoch=ohem_epoch, ohem_max_epoch=ohem_max_epoch, ohem_min_ratio=ohem_min_ratio,
        )

        finite_parts = all(
            torch.isfinite(v).all() if isinstance(v, torch.Tensor) else True
            for v in losses.values()
        )
        if not finite_parts:
            subjects = batch.get("subject", [])
            sample_subj = subjects[:2] if isinstance(subjects, list) else subjects
            details = {
                k: (torch.isfinite(v).all().item() if isinstance(v, torch.Tensor) else True)
                for k, v in losses.items()
            }
            print(f"  [warn][val] batch {i + 1}: loss non-finie, batch ignoré | subjects={sample_subj} | finite={details}")
            continue

        for k in metrics:
            v = losses.get(k, torch.tensor(0.0))
            metrics[k] += v.item() if isinstance(v, torch.Tensor) else v
        n += 1

    return {k: v / max(n, 1) for k, v in metrics.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cli  = parse_args()
    cfg  = load_config(cli)

    # Résolution du device
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg["device"])

    ts  = (cfg["target_size"],) * 3
    lam = (cfg["lam1a"], cfg["lam1b"], cfg["lam2"])

    grad_surgery = str(cfg.get("gradient_surgery", "none")).lower()
    lam_orth     = float(cfg.get("lam_orth",    0.0))
    lam_vicreg   = float(cfg.get("lam_vicreg",  0.0))

    aug_mode = str(cfg.get("augmentation_mode", "torchio_full")).lower()
    if aug_mode not in {"torchio_full", "tsinghua_simple", "none"}:
        raise ValueError(
            f"augmentation_mode invalide: {aug_mode}. "
            "Valeurs autorisées: torchio_full | tsinghua_simple | none"
        )

    if aug_mode == "torchio_full":
        train_aug_fn = augment_torchio_full
        simulate_artifacts = True
    elif aug_mode == "tsinghua_simple":
        train_aug_fn = augment_tsinghua_simple
        simulate_artifacts = False
    else:
        train_aug_fn = None
        simulate_artifacts = False

    print(f"Config      : {cfg['config']}")
    print(f"Device      : {device}")
    print(f"Target size : {ts}")
    print(f"Base ch     : {cfg['base_channels']}  "
          f"(bottleneck={cfg['base_channels']*16}ch, feat={cfg['base_channels']}ch)")
    print(f"Debug mode  : {cfg['debug']}")
    print(f"Aug mode       : {aug_mode}")
    print(f"Sim artifacts  : {simulate_artifacts}")
    print(f"Gradient surgery: {grad_surgery}")
    print(f"lam_orth={lam_orth}  lam_vicreg={lam_vicreg}")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = LISAJointDataset(
        cfg["data_root"], target_size=ts, split="train",
        val_fraction=cfg["val_fraction"],
        simulate_artifacts=simulate_artifacts,
    )
    val_ds = LISAJointDataset(
        cfg["data_root"], target_size=ts, split="val",
        val_fraction=cfg["val_fraction"],
    )
    print(f"Train : {len(train_ds)} items | Val : {len(val_ds)} items")

    # ── poids de classe Task 1a (calculés sur le train set uniquement) ────────
    t1a_sev_w, t1a_art_w = compute_task1a_weights(
        train_ds.items,
        n_artifacts=cfg["n_artifacts"],
        n_severity=cfg["n_severity"],
    )
    task1a_weights = (t1a_sev_w, t1a_art_w)
    print("Task 1a class weights (sévérité, moyennés sur artefacts) :")
    for a, name in enumerate(["Noise","Zipper","Positioning","Banding","Motion","Contrast","Distortion"]):
        w = t1a_sev_w[a].numpy()
        print(f"  {name:12s}  sev0={w[0]:.2f}  sev1={w[1]:.2f}  sev2={w[2]:.2f}  "
              f"art_w={t1a_art_w[a].item():.2f}")

    pin = device.type == "cuda"
    n_workers = cfg["num_workers"]
    train_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=n_workers, pin_memory=pin,
        persistent_workers=(n_workers > 0),
        prefetch_factor=3 if n_workers > 0 else None,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=n_workers, pin_memory=pin,
        persistent_workers=(n_workers > 0),
        prefetch_factor=3 if n_workers > 0 else None,
    )

    # ── modèle ────────────────────────────────────────────────────────────────
    model = BackboneLISA(
        base=cfg["base_channels"],
        c_anat=cfg["c_anat"],
        c_mod=cfg["c_mod"],
        c_art=cfg["c_art"],
        n_artifacts=cfg["n_artifacts"],
        n_severity=cfg["n_severity"],
        n_seg_classes=cfg["n_seg_classes"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres  : {n_params:,}")

    if cfg.get("compile", False):
        model = torch.compile(model, mode="default")
        print("torch.compile  : activé (mode=default)")

    optimizer = optim.AdamW(
        [
            {
                # Encodeur partagé : LR réduit pour préserver les features bas-niveau
                "params": model.backbone.encoder.parameters(),
                "lr":     cfg["lr"] * cfg["lr_encoder_scale"],
            },
            {
                # Décodeur partagé + têtes Task1b/Task2 : LR plein
                "params": list(model.backbone.decoder.parameters())
                        + list(model.task1b.parameters())
                        + list(model.task2.parameters()),
                "lr":     cfg["lr"],
            },
            {
                # Tête Task1a : LR plein pour favoriser son apprentissage
                "params": model.task1a.parameters(),
                "lr":     cfg["lr"],
            },
        ],
        weight_decay=cfg["weight_decay"],
    )
    
    use_cuda = (device.type == "cuda")
    # bfloat16 est plus stable numériquement que float16 (surtout avec FFT/features larges).
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    # GradScaler utile en float16 ; en bfloat16 on désactive le scaling.
    scaler = GradScaler('cuda', enabled=(use_cuda and amp_dtype == torch.float16))
    print(f"AMP         : {'on' if use_cuda else 'off'} (dtype={amp_dtype}, scaler={scaler.is_enabled()})")

    epochs   = 1 if cfg["debug"] else cfg["epochs"]
    t_max    = cfg["scheduler_tmax"] or epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    start_epoch = 0
    if cfg.get("pretrain_ckpt"):
        # Chargement sélectif : backbone + factorizer uniquement
        # Les têtes de tâches sont initialisées aléatoirement (nouveau training)
        ckpt_pre = torch.load(cfg["pretrain_ckpt"], map_location=device)
        missing_bb  = model.backbone.load_state_dict(ckpt_pre["backbone"],   strict=True)
        missing_fac = model.factorizer.load_state_dict(ckpt_pre["factorizer"], strict=True)
        pretrain_epoch = ckpt_pre.get("epoch", "?")
        print(f"Pretrain backbone chargé   (epoch {pretrain_epoch}) : {cfg['pretrain_ckpt']}")
    if cfg["resume"]:
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Reprise depuis epoch {start_epoch}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_1a      = float("inf")
    no_improve_count = 0
    patience         = cfg["early_stopping_patience"]  # 0 = désactivé

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(RESULTS_DIR / "logs"))

    # ── boucle d'entraînement ─────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        tr = train_epoch(
            model, train_dl, optimizer, device, lam, scaler, writer, epoch,
            label_smoothing=cfg["label_smoothing"],
            ordinal_weight=cfg["ordinal_weight"],
            focal_gamma=cfg["focal_gamma"],
            focal_alpha=torch.tensor(cfg["focal_alpha"]).to(device) if cfg["focal_alpha"] is not None else None,
            ohem_ratio=cfg["ohem_ratio"],
            ohem_anneal=cfg["ohem_anneal"],
            ohem_max_epoch=epochs,
            ohem_min_ratio=cfg["ohem_min_ratio"],
            augment_fn=train_aug_fn,
            debug=cfg["debug"],
            gradient_surgery=grad_surgery,
            lam_orth=lam_orth,
            lam_vicreg=lam_vicreg,
            amp_dtype=amp_dtype,
        )
        vl = val_epoch(
            model, val_dl, device, lam, task1a_weights,
            label_smoothing=cfg["label_smoothing"],
            ordinal_weight=cfg["ordinal_weight"],
            focal_gamma=cfg["focal_gamma"],
            focal_alpha=torch.tensor(cfg["focal_alpha"]).to(device) if cfg["focal_alpha"] is not None else None,
            ohem_ratio=cfg["ohem_ratio"],
            ohem_anneal=cfg["ohem_anneal"],
            ohem_max_epoch=epochs,
            ohem_min_ratio=cfg["ohem_min_ratio"],
            debug=cfg["debug"],
            epoch=epoch,
        )
        scheduler.step()

        dt = time.time() - t0
        
        # Log TensorBoard
        for k in tr:
            writer.add_scalar(f"Train/{k}", tr[k], epoch)
        for k in vl:
            writer.add_scalar(f"Val/{k}", vl[k], epoch)
        writer.add_scalar("Train/Time_per_epoch", dt, epoch)

        print(
            f"Epoch {epoch + 1:3d}/{epochs}  "
            f"train[tot={tr['total']:.4f} 1a={tr['task1a']:.4f} "
            f"1b={tr['task1b']:.4f} seg={tr['task2']:.4f}]  "
            f"val[tot={vl['total']:.4f} 1a={vl['task1a']:.4f} "
            f"1b={vl['task1b']:.4f} seg={vl['task2']:.4f}]  "
            f"{dt:.1f}s"
        )

        # sauvegarde checkpoint
        do_save = (
            (epoch + 1) % cfg["save_every"] == 0
            or epoch + 1 == epochs
            or cfg["debug"]
        )
        if do_save:
            ckpt_path = CKPT_DIR / f"epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "train_losses": tr,
                    "val_losses":   vl,
                    "config":       cfg,
                },
                ckpt_path,
            )
            print(f"  → checkpoint sauvegardé : {ckpt_path}")

        # ── early stopping & best model ───────────────────────────────────────
        if vl["task1a"] < best_val_1a:
            best_val_1a      = vl["task1a"]
            no_improve_count = 0
            best_path = CKPT_DIR / "best_model.pt"
            torch.save(
                {
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "train_losses": tr,
                    "val_losses":   vl,
                    "config":       cfg,
                },
                best_path,
            )
            print(f"  → best model (val 1a={best_val_1a:.4f}) sauvegardé : {best_path}")
        else:
            no_improve_count += 1

        if patience > 0 and no_improve_count >= patience:
            print(
                f"  [early stopping] aucune amélioration val task1a "
                f"depuis {patience} époques. Arrêt à epoch {epoch + 1}."
            )
            break

        # visualisation
        do_viz = (
            not cfg["no_viz"]
            and (
                (epoch + 1) % cfg["viz_every"] == 0
                or epoch + 1 == epochs
                or cfg["debug"]
            )
        )
        if do_viz:
            try:
                from visualize import visualize_all
                visualize_all(
                    model, val_ds, CKPT_DIR, RESULTS_DIR,
                    epoch + 1, device,
                )
            except Exception as exc:
                print(f"  [viz] Erreur (ignorée) : {exc}")

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
