#!/usr/bin/env python3
"""
Visualisation de l'entraînement — LISA 2026.

Génère dans results/plots/ :
  loss_curves.png     — courbes train/val pour chaque tâche + total
  task1b_samples.png  — exemples de reconstruction (input vs recon, 3 coupes)
  task2_samples.png   — exemples de segmentation (ciso + overlay prédit)

Usage :
    # standalone depuis un checkpoint
    python src/visualize.py --checkpoint outputs/checkpoints/epoch_0010.pt \\
                            --config configs/train_default.yaml

    # appelé automatiquement depuis train.py
    visualize_all(model, val_ds, ckpt_dir, results_dir, cfg, epoch, device)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from dataset import LISAJointDataset, ARTIFACT_COLS, DATA_ROOT_DEFAULT
from model import BrainFMLISA
from train import load_config, parse_args as _parse_train_args

# ─────────────────────────────────────────────────────────────────────────────
# Couleurs segmentation (14 classes : 0=fond + 1–13)
# ─────────────────────────────────────────────────────────────────────────────
SEG_COLORS = np.array([
    [0.00, 0.00, 0.00, 0.00],  # 0 fond (transparent)
    [0.89, 0.10, 0.11, 0.60],  # 1  hippocampe_G
    [0.22, 0.49, 0.72, 0.60],  # 2  hippocampe_D
    [0.30, 0.69, 0.29, 0.60],  # 3  caude_G
    [0.60, 0.80, 0.20, 0.60],  # 4  caude_D
    [0.99, 0.75, 0.04, 0.60],  # 5  putamen_G
    [1.00, 0.50, 0.00, 0.60],  # 6  putamen_D
    [0.65, 0.34, 0.16, 0.60],  # 7  globus_G
    [0.96, 0.51, 0.75, 0.60],  # 8  globus_D
    [0.60, 0.20, 0.80, 0.60],  # 9  thalamus_G
    [0.30, 0.00, 0.50, 0.60],  # 10 thalamus_D
    [0.00, 0.90, 0.90, 0.60],  # 11 corps_calleux
    [0.10, 0.70, 0.80, 0.60],  # 12 ventricule_G
    [0.00, 0.40, 0.90, 0.60],  # 13 ventricule_D
], dtype=np.float32)

SEG_NAMES = [
    "fond", "hippo_G", "hippo_D", "caude_G", "caude_D",
    "putamen_G", "putamen_D", "globus_G", "globus_D",
    "thalamus_G", "thalamus_D", "corps_calleux", "ventricule_G", "ventricule_D",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers visuels
# ─────────────────────────────────────────────────────────────────────────────

def _mid_slices(vol: np.ndarray):
    """Retourne les 3 coupes centrales d'un volume 3D (axi, cor, sag)."""
    d, h, w = vol.shape
    return vol[d // 2, :, :], vol[:, h // 2, :], vol[:, :, w // 2]


def _seg_rgb(seg: np.ndarray) -> np.ndarray:
    """Convertit un tableau de labels (H, W) en RGBA float32."""
    out = SEG_COLORS[np.clip(seg.astype(np.int32), 0, 13)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Courbes de loss
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(ckpt_dir: Path, out_path: Path) -> None:
    """
    Lit tous les checkpoints epoch_XXXX.pt et trace les courbes de loss.
    """
    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        print("  [viz] Aucun checkpoint trouvé pour les courbes de loss.")
        return

    epochs, train, val = [], {}, {}
    for k in ("total", "task1a", "task1b", "task2"):
        train[k] = []
        val[k]   = []

    for ck in ckpt_files:
        d = torch.load(ck, map_location="cpu", weights_only=False)
        ep = d.get("epoch", 0) + 1
        tl = d.get("train_losses", {})
        vl = d.get("val_losses",   {})
        epochs.append(ep)
        for k in ("total", "task1a", "task1b", "task2"):
            train[k].append(tl.get(k, 0.0))
            val[k].append(vl.get(k, 0.0))

    n_tasks = 4
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4))
    titles = ["Total", "Task 1a (QC)", "Task 1b (Enhancement)", "Task 2 (Seg)"]
    keys   = ["total", "task1a", "task1b", "task2"]

    for ax, key, title in zip(axes, keys, titles):
        ax.plot(epochs, train[key], label="train", color="steelblue", linewidth=1.5)
        ax.plot(epochs, val[key],   label="val",   color="tomato",    linewidth=1.5,
                linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    fig.suptitle("Courbes de loss — LISA 2026", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Loss curves → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Exemples Task 1b (reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_task1b_samples(
    model:    BrainFMLISA,
    val_ds:   LISAJointDataset,
    out_path: Path,
    device:   torch.device,
    n_samples: int = 4,
) -> None:
    """
    Sélectionne des items anisotropes sans artefact, lance la reconstruction
    et sauvegarde une grille input / recon (3 coupes) + diff abs.
    """
    items = [
        it for it in val_ds.items
        if not it["is_isotropic"] and it["is_artifact_free"]
    ][:n_samples]

    if not items:
        print("  [viz] Aucun item Task1b sans artefact dans la val — skip.")
        return

    n_rows = len(items)
    # colonnes : axi_in, cor_in, sag_in, axi_rec, cor_rec, sag_rec, axi_diff, cor_diff, sag_diff
    n_cols = 9
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Input axi", "Input cor", "Input sag",
        "Recon axi", "Recon cor", "Recon sag",
        "|Diff| axi", "|Diff| cor", "|Diff| sag",
    ]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=7)

    model.eval()
    for ri, it in enumerate(items):
        # chargement
        from dataset import load_nifti, resample_to_isotropic, crop_or_pad, normalize
        data, zooms = load_nifti(it["filepath"])
        data = resample_to_isotropic(data, zooms)
        data = crop_or_pad(data, val_ds.target_size)
        data = normalize(data)
        x = torch.from_numpy(data[None, None]).to(device)  # [1,1,D,H,W]

        out   = model(x, run_task1a=False, run_task1b=True, run_task2=False)
        recon = out["task1b"][0, 0].cpu().numpy()  # [D,H,W]
        inp   = data                               # [D,H,W]
        diff  = np.abs(inp - recon)

        for ci, (vol, cmap) in enumerate(
            [(inp, "gray"), (inp, "gray"), (inp, "gray"),
             (recon, "gray"), (recon, "gray"), (recon, "gray"),
             (diff, "hot"), (diff, "hot"), (diff, "hot")]
        ):
            axi, cor, sag = _mid_slices(vol)
            slices = [axi, cor, sag]
            grp = ci // 3
            idx_within = ci % 3
            sl = slices[idx_within]

            ax = axes[ri, ci]
            vmax = 1.0 if grp < 2 else diff.max() + 1e-6
            ax.imshow(sl.T, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
            ax.axis("off")
            if ci == 0:
                ax.set_ylabel(f"{it['subject']}\n{it['orientation']}", fontsize=6)

        # vertical separator après col 2 et col 5
        for sep in (2, 5):
            axes[ri, sep].spines["right"].set_visible(True)
            axes[ri, sep].spines["right"].set_linewidth(2)
            axes[ri, sep].spines["right"].set_color("white")

    fig.suptitle("Task 1b — Input vs Reconstruction", fontsize=12, y=1.01)
    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Task1b samples → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Exemples Task 2 (segmentation)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_task2_samples(
    model:    BrainFMLISA,
    val_ds:   LISAJointDataset,
    out_path: Path,
    device:   torch.device,
    n_samples: int = 3,
) -> None:
    """
    Sélectionne des images ciso avec GT segmentation, lance la segmentation
    et affiche image + overlay GT + overlay prédit (3 coupes).
    """
    items = [
        it for it in val_ds.items
        if it["is_isotropic"] and it["has_seg"]
    ][:n_samples]

    if not items:
        print("  [viz] Aucun item Task2 (ciso+seg) dans la val — skip.")
        return

    n_rows = len(items)
    # colonnes : axi_img, cor_img, sag_img | axi_gt, cor_gt, sag_gt | axi_pred, cor_pred, sag_pred
    n_cols = 9
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Image axi", "Image cor", "Image sag",
        "GT axi",    "GT cor",    "GT sag",
        "Pred axi",  "Pred cor",  "Pred sag",
    ]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=7)

    model.eval()
    for ri, it in enumerate(items):
        from dataset import load_nifti, resample_to_isotropic, crop_or_pad, normalize
        import numpy as np

        # image
        data, zooms = load_nifti(it["filepath"])
        data = resample_to_isotropic(data, zooms)
        data = crop_or_pad(data, val_ds.target_size)
        data = normalize(data)
        x = torch.from_numpy(data[None, None]).to(device)

        # GT segmentation
        seg_data, seg_zooms = load_nifti(it["seg_filepath"])
        seg_data = resample_to_isotropic(seg_data, seg_zooms, order=0)
        from dataset import crop_or_pad as _cpp
        seg_data = _cpp(seg_data.astype(np.int32), val_ds.target_size)

        # prédiction
        out = model(x, run_task1a=False, run_task1b=False, run_task2=True)
        pred_seg = out["task2"][0].argmax(dim=0).cpu().numpy()  # [D,H,W]

        for ci in range(9):
            ax = axes[ri, ci]
            grp = ci // 3
            idx = ci % 3

            axi_img, cor_img, sag_img = _mid_slices(data)
            img_slices = [axi_img, cor_img, sag_img]

            axi_gt,  cor_gt,  sag_gt  = _mid_slices(seg_data.astype(np.int32))
            gt_slices  = [axi_gt,  cor_gt,  sag_gt]

            axi_p, cor_p, sag_p = _mid_slices(pred_seg)
            pred_slices = [axi_p, cor_p, sag_p]

            img_sl = img_slices[idx]
            ax.imshow(img_sl.T, cmap="gray", origin="lower", vmin=0, vmax=1)

            if grp == 1:
                overlay = _seg_rgb(gt_slices[idx].T)
                ax.imshow(overlay, origin="lower", aspect="auto")
            elif grp == 2:
                overlay = _seg_rgb(pred_slices[idx].T)
                ax.imshow(overlay, origin="lower", aspect="auto")

            ax.axis("off")
            if ci == 0:
                ax.set_ylabel(it["subject"], fontsize=6)

    # légende des structures
    patches = [
        mpatches.Patch(color=SEG_COLORS[i, :3], label=SEG_NAMES[i])
        for i in range(1, 14)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=6,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Task 2 — Image | GT seg | Prédiction seg", fontsize=12, y=1.01)
    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Task2 samples → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Distribution des prédictions Task 1a (artefacts)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_task1a_confusion(
    model:    BrainFMLISA,
    val_ds:   LISAJointDataset,
    out_path: Path,
    device:   torch.device,
) -> None:
    """Matrice de confusion (gt vs pred) pour chaque type d'artefact."""
    items = [it for it in val_ds.items if it["has_task1a"]]
    if not items:
        print("  [viz] Aucun item Task1a dans la val — skip.")
        return

    from dataset import load_nifti, resample_to_isotropic, crop_or_pad, normalize
    import numpy as np

    n_art = len(ARTIFACT_COLS)
    conf  = np.zeros((n_art, 3, 3), dtype=np.int32)  # [art, gt, pred]

    model.eval()
    for it in items:
        data, zooms = load_nifti(it["filepath"])
        data = resample_to_isotropic(data, zooms)
        data = crop_or_pad(data, val_ds.target_size)
        data = normalize(data)
        x = torch.from_numpy(data[None, None]).to(device)

        out    = model(x, run_task1a=True, run_task1b=False, run_task2=False)
        preds  = out["task1a"][0].argmax(dim=-1).cpu().numpy()  # [7]
        labels = it["task1a_labels"]                            # np [7]
        for a in range(n_art):
            gt, pr = int(labels[a]), int(preds[a])
            if 0 <= gt <= 2 and 0 <= pr <= 2:
                conf[a, gt, pr] += 1

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for a, art in enumerate(ARTIFACT_COLS):
        ax = axes[a]
        mat = conf[a]
        im  = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_title(art, fontsize=10)
        ax.set_xlabel("Prédit", fontsize=8)
        ax.set_ylabel("GT", fontsize=8)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["0", "1", "2"])
        ax.set_yticklabels(["0", "1", "2"])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > mat.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    axes[-1].axis("off")  # 8e case vide
    fig.suptitle("Task 1a — Confusion GT vs Prédit (val)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Task1a confusion → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────────────────────────────────────

def visualize_all(
    model:       BrainFMLISA,
    val_ds:      LISAJointDataset,
    ckpt_dir:    Path,
    results_dir: Path,
    epoch:       int,
    device:      torch.device,
    n_samples_1b: int = 4,
    n_samples_2:  int = 3,
) -> None:
    """
    Appelé depuis train.py à chaque epoch de visualisation.
    Génère les 4 figures dans results_dir/plots/.
    """
    plots_dir = results_dir / "plots"

    plot_loss_curves(ckpt_dir, plots_dir / "loss_curves.png")
    plot_task1a_confusion(model, val_ds, plots_dir / f"task1a_confusion_ep{epoch:04d}.png", device)
    plot_task1b_samples(model, val_ds, plots_dir / f"task1b_samples_ep{epoch:04d}.png", device, n_samples_1b)
    plot_task2_samples(model, val_ds, plots_dir / f"task2_seg_ep{epoch:04d}.png", device, n_samples_2)


# ─────────────────────────────────────────────────────────────────────────────
# Script standalone
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Visualisation LISA 2026")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/train_default.yaml")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    import sys, types

    args = _parse_args()

    # Charger la config via le même mécanisme que train.py
    # On instancie un Namespace minimal pour load_config
    import argparse as _ap
    fake_cli = _ap.Namespace(
        config=args.config,
        data_root=None, target_size=None, base_channels=None,
        c_anat=None, c_mod=None, c_art=None,
        epochs=None, batch_size=None, num_workers=None, save_every=None,
        lr=None, lam1a=None, lam1b=None, lam2=None,
        device=args.device, resume=None, debug=False,
    )
    cfg = load_config(fake_cli)
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg["device"])
    ts     = (cfg["target_size"],) * 3

    # Dataset val
    val_ds = LISAJointDataset(cfg["data_root"], target_size=ts, split="val",
                               val_fraction=cfg["val_fraction"])
    print(f"Val dataset : {len(val_ds)} items")

    # Modèle
    model = BrainFMLISA(
        base=cfg["base_channels"],
        c_anat=cfg["c_anat"], c_mod=cfg["c_mod"], c_art=cfg["c_art"],
        n_artifacts=cfg["n_artifacts"], n_severity=cfg["n_severity"],
        n_seg_classes=cfg["n_seg_classes"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", 0) + 1

    ckpt_dir    = Path(args.checkpoint).parent
    results_dir = Path(__file__).parent.parent / "results"

    visualize_all(model, val_ds, ckpt_dir, results_dir, epoch, device)
    print("Visualisation terminée.")
