#!/usr/bin/env python3
"""
Pré-entraînement VICReg auto-supervisé — LISA 2026  (Phase 4)

Pipeline :
  1. Pour chaque volume, générer 2 vues augmentées (view1, view2)
  2. Forward backbone → FactorizedProjection → z_anat, z_mod, z_art × 2 vues
  3. Loss VICReg complète par sous-espace :
       L_inv  = MSE(GAP(z1), GAP(z2))     (invariance — les 2 vues convergent)
       L_var  = VICReg variance            (évite le collapse par dimension)
       L_cov  = VICReg covariance          (décore les dimensions)
       L_orth = orthogonalité entre sous-espaces (z_anat ⊥ z_mod, z_anat ⊥ z_art)
  4. Sauvegarde backbone + factorizer → rechargé dans train.py via --pretrain-ckpt

Différences vs vicreg_cov_loss dans losses.py :
  - losses.py : pénalités variance + covariance sur UNE vue (pas d'invariance)
  - pretrain.py : TROIS termes complets sur DEUX vues (VICReg original)

Usage :
    python src/pretrain.py --config configs/pretrain_v8.yaml
    python src/pretrain.py --config configs/pretrain_v8.yaml --debug

Reprise :
    python src/pretrain.py --config configs/pretrain_v8.yaml \\
                           --resume outputs/checkpoints/pretrain_epoch_0020.pt

Lancer le training supervisé après :
    python src/train.py --config configs/train_v8.yaml \\
                        --pretrain-ckpt outputs/checkpoints/pretrain_best.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from dataset import LISAJointDataset, DATA_ROOT_DEFAULT
from augmentation import augment_torchio_full
from losses import orthogonality_loss, vicreg_cov_loss
from model import SharedUNet, FactorizedProjection

RESULTS_DIR = Path(__file__).parent.parent / "results"
CKPT_DIR    = Path(__file__).parent.parent / "outputs" / "checkpoints"
DEFAULT_CFG = Path(__file__).parent.parent / "configs" / "pretrain_v8.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 2 vues (self-supervised)
# ──────────────────────────────────────────────────────────────────────────────

class TwoViewDataset(Dataset):
    """
    Wrapper autour de LISAJointDataset qui retourne 2 vues augmentées
    du même volume pour l'entraînement auto-supervisé.

    __getitem__ retourne (view1, view2), chacun [1, D, H, W] ∈ [0, 1].
    Les 2 vues sont indépendantes (appel séparé de aug_fn) pour maximiser
    la diversité tout en conservant l'identité du volume.
    """

    def __init__(self, base_ds: LISAJointDataset, aug_fn):
        self.ds     = base_ds
        self.aug_fn = aug_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        x    = item["image"]                            # [1, D, H, W]
        # augment_torchio_full attend un batch [B, 1, D, H, W]
        view1 = self.aug_fn(x.unsqueeze(0)).squeeze(0)  # [1, D, H, W]
        view2 = self.aug_fn(x.unsqueeze(0)).squeeze(0)  # [1, D, H, W]
        return view1, view2


# ──────────────────────────────────────────────────────────────────────────────
# Modèle pré-entraînable (backbone + factorizer, sans têtes de tâches)
# ──────────────────────────────────────────────────────────────────────────────

class PretrainModel(nn.Module):
    """
    SharedUNet + FactorizedProjection.

    Identique aux composants de BackboneLISA mais sans les têtes Task1a/1b/2,
    ce qui réduit les paramètres et focalise le pretrain sur les sous-espaces.
    """

    def __init__(self, base: int, c_anat: int, c_mod: int, c_art: int):
        super().__init__()
        self.backbone   = SharedUNet(base=base)
        self.factorizer = FactorizedProjection(
            feat_ch=self.backbone.feat_ch,
            c_anat=c_anat,
            c_mod=c_mod,
            c_art=c_art,
        )

    def forward(self, x: torch.Tensor):
        feats, _, _ = self.backbone(x)
        return self.factorizer(feats)    # (z_anat, z_mod, z_art)


# ──────────────────────────────────────────────────────────────────────────────
# Loss VICReg complète (2 vues, 1 sous-espace)
# ──────────────────────────────────────────────────────────────────────────────

def vicreg_full_loss(
    z1:      torch.Tensor,    # [B, C, D, H, W]
    z2:      torch.Tensor,    # [B, C, D, H, W]
    lam_inv: float = 25.0,
    lam_var: float = 25.0,
    lam_cov: float = 1.0,
) -> torch.Tensor:
    """
    VICReg complet (Bardes et al. 2022) entre 2 vues d'un sous-espace.

    Après GAP spatial [B, C, D, H, W] → [B, C] :
      L_inv = lam_inv × MSE(z1, z2)
      L_var = lam_var × (VICReg_var(z1) + VICReg_var(z2)) / 2
      L_cov = lam_cov × (VICReg_cov(z1) + VICReg_cov(z2)) / 2

    vicreg_cov_loss dans losses.py fait déjà L_var + L_cov sur une vue.
    Ce terme L_inv est l'apport unique du pretrain à 2 vues.
    """
    a = z1.flatten(2).mean(-1)   # GAP → [B, C]
    b = z2.flatten(2).mean(-1)   # GAP → [B, C]

    loss_inv = F.mse_loss(a, b)
    # vicreg_cov_loss : variance (relu(γ-std)) + covariance hors-diag
    loss_vc  = (vicreg_cov_loss(z1) + vicreg_cov_loss(z2)) * 0.5

    return lam_inv * loss_inv + lam_var * loss_vc


def pretrain_loss(
    z_anat1, z_mod1, z_art1,
    z_anat2, z_mod2, z_art2,
    lam_inv:  float = 25.0,
    lam_var:  float = 25.0,
    lam_cov:  float = 1.0,
    lam_orth: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """
    Loss totale de pré-entraînement sur les 3 sous-espaces :
      - VICReg complet sur z_anat, z_mod, z_art (invariance + var + cov)
      - Orthogonalité (z_anat, z_mod) + (z_anat, z_art)
    """
    lv_anat = vicreg_full_loss(z_anat1, z_anat2, lam_inv, lam_var, lam_cov)
    lv_mod  = vicreg_full_loss(z_mod1,  z_mod2,  lam_inv, lam_var, lam_cov)
    lv_art  = vicreg_full_loss(z_art1,  z_art2,  lam_inv, lam_var, lam_cov)
    total   = lv_anat + lv_mod + lv_art

    losses = {
        "vicreg_anat": lv_anat.detach(),
        "vicreg_mod":  lv_mod.detach(),
        "vicreg_art":  lv_art.detach(),
    }

    if lam_orth > 0:
        # Vue 1 uniquement pour l'orthogonalité (stable, pas de variance inter-vues)
        lo1 = orthogonality_loss(z_anat1, z_mod1)
        lo2 = orthogonality_loss(z_anat1, z_art1)
        total = total + lam_orth * (lo1 + lo2)
        losses["orth_anat_mod"] = lo1.detach()
        losses["orth_anat_art"] = lo2.detach()

    losses["total"] = total.detach()
    return total, losses


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

def load_config(args) -> dict:
    cfg_path = Path(args.config) if args.config else DEFAULT_CFG
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    d = raw.get("data", {})
    m = raw.get("model", {})
    t = raw.get("training", {})

    return {
        # données
        "data_root":     d.get("root",          DATA_ROOT_DEFAULT),
        "target_size":   d.get("target_size",    96),
        # modèle
        "base_channels": m.get("base_channels",  32),
        "c_anat":        m.get("c_anat",         16),
        "c_mod":         m.get("c_mod",           8),
        "c_art":         m.get("c_art",           8),
        # entraînement
        "epochs":        t.get("epochs",         50),
        "batch_size":    t.get("batch_size",      4),
        "num_workers":   t.get("num_workers",     4),
        "save_every":    t.get("save_every",     10),
        "lr":            t.get("lr",           1e-4),
        "weight_decay":  t.get("weight_decay", 1e-5),
        "lam_inv":       t.get("lam_inv",      25.0),
        "lam_var":       t.get("lam_var",      25.0),
        "lam_cov":       t.get("lam_cov",       1.0),
        "lam_orth":      t.get("lam_orth",      0.05),
        "compile":       t.get("compile",      False),
        "device":        raw.get("device",     "auto"),
        # CLI
        "debug":         getattr(args, "debug",  False),
        "resume":        getattr(args, "resume", None),
        "config":        str(cfg_path),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Pré-entraînement VICReg auto-supervisé — LISA 2026",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",  default=None,
                   help=f"Config YAML (défaut : {DEFAULT_CFG})")
    p.add_argument("--resume",  default=None,
                   help="Checkpoint pretrain pour reprendre (pretrain_epoch_XXXX.pt)")
    p.add_argument("--debug",   action="store_true",
                   help="1 epoch, 4 batchs — vérification rapide")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde du checkpoint pretrain
# ──────────────────────────────────────────────────────────────────────────────

def _get_state(model: nn.Module, key: str) -> dict:
    """Gère le cas torch.compile (model._orig_mod) pour extraire state_dict."""
    m = getattr(model, "_orig_mod", model)
    return getattr(m, key).state_dict()


def save_pretrain_ckpt(model, optimizer, epoch, losses, path: Path):
    torch.save(
        {
            "epoch":      epoch,
            "backbone":   _get_state(model, "backbone"),
            "factorizer": _get_state(model, "factorizer"),
            "optimizer":  optimizer.state_dict(),
            "losses":     {k: float(v) for k, v in losses.items()},
        },
        path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args)

    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg["device"])

    # Activation TF32 pour les matmuls (pas d'impact sur la précision VICReg)
    torch.set_float32_matmul_precision("high")

    ts = (cfg["target_size"],) * 3
    print(f"Config      : {cfg['config']}")
    print(f"Device      : {device}")
    print(f"Target size : {ts}")
    print(f"lam_inv={cfg['lam_inv']}  lam_var={cfg['lam_var']}  "
          f"lam_cov={cfg['lam_cov']}  lam_orth={cfg['lam_orth']}")

    # ── Dataset 2 vues ────────────────────────────────────────────────────────
    # split="all" : pas de split train/val, on utilise toutes les images
    # simulate_artifacts=False : on veut apprendre les structures, pas les artefacts
    base_ds = LISAJointDataset(
        cfg["data_root"],
        target_size=ts,
        split="all",
        val_fraction=0.0,
        simulate_artifacts=False,
    )
    ds = TwoViewDataset(base_ds, augment_torchio_full)
    n_w = cfg["num_workers"]
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=n_w,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_w > 0),
        prefetch_factor=2 if n_w > 0 else None,
    )
    print(f"Dataset     : {len(ds)} volumes → {len(dl)} batchs/epoch")

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = PretrainModel(
        base   = cfg["base_channels"],
        c_anat = cfg["c_anat"],
        c_mod  = cfg["c_mod"],
        c_art  = cfg["c_art"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres  : {n_params:,}  (backbone + factorizer, sans têtes de tâches)")

    if cfg.get("compile", False):
        model = torch.compile(model, mode="default")
        print("torch.compile  : activé")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    use_amp  = (device.type == "cuda")
    scaler   = GradScaler("cuda") if use_amp else None
    epochs   = 1 if cfg["debug"] else cfg["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    if cfg["resume"]:
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.backbone.load_state_dict(ckpt["backbone"])
        model.factorizer.load_state_dict(ckpt["factorizer"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Reprise depuis epoch {start_epoch}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(RESULTS_DIR / "logs_pretrain"))

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        accum: dict[str, float] = {}
        n  = 0
        t0 = time.time()

        for i, (v1, v2) in enumerate(dl):
            if cfg["debug"] and i >= 4:
                break

            v1 = v1.to(device)
            v2 = v2.to(device)

            optimizer.zero_grad()

            ctx = autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)
            with ctx:
                z_anat1, z_mod1, z_art1 = model(v1)
                z_anat2, z_mod2, z_art2 = model(v2)
                loss, losses = pretrain_loss(
                    z_anat1, z_mod1, z_art1,
                    z_anat2, z_mod2, z_art2,
                    lam_inv  = cfg["lam_inv"],
                    lam_var  = cfg["lam_var"],
                    lam_cov  = cfg["lam_cov"],
                    lam_orth = cfg["lam_orth"],
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k, v in losses.items():
                accum[k] = accum.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                )
            n += 1

            if cfg["debug"]:
                print(
                    f"  [pretrain] batch {i+1:2d}  "
                    f"total={losses['total']:.4f}  "
                    f"anat={losses['vicreg_anat']:.4f}  "
                    f"mod={losses['vicreg_mod']:.4f}  "
                    f"art={losses['vicreg_art']:.4f}  "
                    f"orth={losses.get('orth_anat_mod', float('nan')):.4f}"
                )

        scheduler.step()
        dt   = time.time() - t0
        avgs = {k: v / max(n, 1) for k, v in accum.items()}

        for k, v in avgs.items():
            writer.add_scalar(f"Pretrain/{k}", v, epoch)

        print(
            f"Pretrain  {epoch+1:3d}/{epochs}  "
            f"total={avgs.get('total', 0):.4f}  "
            f"anat={avgs.get('vicreg_anat', 0):.4f}  "
            f"mod={avgs.get('vicreg_mod', 0):.4f}  "
            f"art={avgs.get('vicreg_art', 0):.4f}  "
            f"orth_am={avgs.get('orth_anat_mod', 0):.4f}  "
            f"{dt:.1f}s"
        )

        do_save = (
            (epoch + 1) % cfg["save_every"] == 0
            or epoch + 1 == epochs
            or cfg["debug"]
        )
        if do_save:
            ckpt_path = CKPT_DIR / f"pretrain_epoch_{epoch+1:04d}.pt"
            save_pretrain_ckpt(model, optimizer, epoch, avgs, ckpt_path)
            print(f"  → checkpoint : {ckpt_path}")

    # ── Checkpoint final "best" ───────────────────────────────────────────────
    final_path = CKPT_DIR / "pretrain_best.pt"
    save_pretrain_ckpt(model, optimizer, epochs - 1, avgs, final_path)
    print(f"\nPré-entraînement terminé.")
    print(f"Backbone sauvegardé : {final_path}")
    print(f"\nLancer le training supervisé :")
    print(f"  python src/train.py --config configs/train_v8.yaml \\")
    print(f"                      --pretrain-ckpt {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
