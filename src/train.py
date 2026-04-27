#!/usr/bin/env python3
"""
Script d'entraînement — Modèle joint LISA 2026.

Usage :
    # entraînement complet
    python src/train.py

    # debug 1 epoch (4 batchs train + 2 val)
    python src/train.py --debug

    # options principales
    python src/train.py --epochs 100 --batch-size 1 --target-size 96 \\
                        --lr 1e-4 --lam1a 1.0 --lam1b 1.0 --lam2 1.0 \\
                        --device cuda --resume outputs/checkpoints/epoch_0010.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LISAJointDataset, DATA_ROOT_DEFAULT, TARGET_SIZE
from losses import multi_task_loss
from model import BrainFMLISA

CKPT_DIR = Path(__file__).parent.parent / "outputs" / "checkpoints"


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entraînement LISA 2026 joint")
    p.add_argument("--data-root",   default=DATA_ROOT_DEFAULT)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=1)
    p.add_argument(
        "--target-size", type=int, default=96,
        help="Taille spatiale isotrope (ex. 96 ou 128 selon la VRAM disponible)",
    )
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lam1a",       type=float, default=1.0, help="Poids Task1a")
    p.add_argument("--lam1b",       type=float, default=1.0, help="Poids Task1b")
    p.add_argument("--lam2",        type=float, default=1.0, help="Poids Task2")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-every",  type=int, default=10)
    p.add_argument("--resume",      default=None, help="Chemin d'un checkpoint")
    p.add_argument(
        "--debug", action="store_true",
        help="1 epoch, 4 batchs train / 2 batchs val — pour vérification rapide",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Boucles train / val
# ──────────────────────────────────────────────────────────────────────────────

def _zero_metrics():
    return {"task1a": 0.0, "task1b": 0.0, "task2": 0.0, "total": 0.0}


def train_epoch(model, loader, optimizer, device, lam, debug=False):
    model.train()
    metrics = _zero_metrics()
    n = 0

    for i, batch in enumerate(loader):
        if debug and i >= 4:
            break

        x = batch["image"].to(device)
        preds = model(x)
        loss, losses = multi_task_loss(preds, batch, lam, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in metrics:
            v = losses.get(k, torch.tensor(0.0))
            metrics[k] += v.item() if isinstance(v, torch.Tensor) else v
        n += 1

        if debug:
            print(
                f"  [train] batch {i + 1}  "
                f"total={losses['total']:.4f}  "
                f"1a={losses.get('task1a', 0.):.4f}  "
                f"1b={losses.get('task1b', 0.):.4f}  "
                f"seg={losses.get('task2', 0.):.4f}  "
                f"| 1a_mask={batch['has_task1a'].sum().item()}  "
                f"1b_mask={batch['is_artifact_free'].sum().item()}  "
                f"seg_mask={(batch['is_isotropic'] & batch['has_seg']).sum().item()}"
            )

    return {k: v / max(n, 1) for k, v in metrics.items()}


@torch.no_grad()
def val_epoch(model, loader, device, lam, debug=False):
    model.eval()
    metrics = _zero_metrics()
    n = 0

    for i, batch in enumerate(loader):
        if debug and i >= 2:
            break

        x = batch["image"].to(device)
        preds = model(x)
        _, losses = multi_task_loss(preds, batch, lam, device)

        for k in metrics:
            v = losses.get(k, torch.tensor(0.0))
            metrics[k] += v.item() if isinstance(v, torch.Tensor) else v
        n += 1

    return {k: v / max(n, 1) for k, v in metrics.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)
    ts     = (args.target_size,) * 3
    lam    = (args.lam1a, args.lam1b, args.lam2)

    print(f"Device      : {device}")
    print(f"Target size : {ts}")
    print(f"Debug mode  : {args.debug}")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = LISAJointDataset(args.data_root, target_size=ts, split="train")
    val_ds   = LISAJointDataset(args.data_root, target_size=ts, split="val")
    print(f"Train : {len(train_ds)} items | Val : {len(val_ds)} items")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    # ── modèle ────────────────────────────────────────────────────────────────
    model = BrainFMLISA(base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres  : {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    epochs    = 1 if args.debug else args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Reprise depuis epoch {start_epoch}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── boucle d'entraînement ─────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        tr = train_epoch(model, train_dl, optimizer, device, lam, args.debug)
        vl = val_epoch(model, val_dl, device, lam, args.debug)
        scheduler.step()

        dt = time.time() - t0
        print(
            f"Epoch {epoch + 1:3d}/{epochs}  "
            f"train[tot={tr['total']:.4f} 1a={tr['task1a']:.4f} "
            f"1b={tr['task1b']:.4f} seg={tr['task2']:.4f}]  "
            f"val[tot={vl['total']:.4f} 1a={vl['task1a']:.4f} "
            f"1b={vl['task1b']:.4f} seg={vl['task2']:.4f}]  "
            f"{dt:.1f}s"
        )

        # sauvegarde checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == epochs or args.debug:
            ckpt_path = CKPT_DIR / f"epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch":         epoch,
                    "model":         model.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "train_losses":  tr,
                    "val_losses":    vl,
                    "args":          vars(args),
                },
                ckpt_path,
            )
            print(f"  → checkpoint sauvegardé : {ckpt_path}")

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
