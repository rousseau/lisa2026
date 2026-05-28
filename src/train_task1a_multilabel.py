#!/usr/bin/env python
"""
RUN_0002 – Multi-label Training Script for Task 1a
Single DenseNet264 with 7 independent heads trained simultaneously.
Loss = ordinal EMD + Focal  (inspired by UPF LISA 2025, Table 1).
"""
import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets import get_multilabel_dataloaders, TASK_NAMES
from src.models import Task1aMultiLabelModel
from src.utils.config import apply_env_overrides


# ─── Loss functions ───────────────────────────────────────────────────────────

def ordinal_emd_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Earth Mover's Distance (Cramér distance) for ordinal classification.

    Captures the ordinal nature of severity labels (0 < 1 < 2) by penalising
    predictions proportionally to their distance from the true class in
    CDF space.

    Args:
        logits:  [B, 7, 3]  raw model output
        targets: [B, 7]     integer class labels in {0, 1, 2}

    Returns:
        Scalar mean EMD loss over batch and tasks.
    """
    num_classes = logits.shape[-1]
    probs = torch.softmax(logits, dim=-1)                      # [B, 7, 3]
    targets_oh = F.one_hot(targets, num_classes).float()       # [B, 7, 3]
    pred_cdf = torch.cumsum(probs, dim=-1)                     # [B, 7, 3]
    target_cdf = torch.cumsum(targets_oh, dim=-1)              # [B, 7, 3]
    # Last CDF bin is always 1 for both – drop it (no information).
    return torch.mean((pred_cdf[..., :-1] - target_cdf[..., :-1]) ** 2)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: tuple = (0.25, 0.5, 1.0),
) -> torch.Tensor:
    """Focal loss to address class imbalance (e.g. Banding: 96% class-0).

    Uses per-class weights as in UPF: alpha=[0.25, 0.5, 1.0] for classes
    [none, moderate, severe].

    Args:
        logits:  [B, 7, 3]
        targets: [B, 7]
        gamma:   focusing parameter
        alpha:   per-class weights (length = num_classes)

    Returns:
        Scalar mean focal loss.
    """
    B, T, C = logits.shape
    logits_2d = logits.reshape(B * T, C)
    targets_1d = targets.reshape(B * T)

    weight = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
    ce = F.cross_entropy(logits_2d, targets_1d, weight=weight, reduction='none')  # [B*T]
    pt = torch.exp(-ce)
    return torch.mean((1.0 - pt) ** gamma * ce)


# ─── Trainer ─────────────────────────────────────────────────────────────────

class MultiLabelTrainer:
    """Single-model trainer for multi-label Task 1a (RUN_0002)."""

    def __init__(self, config: dict, smoke_test: bool = False):
        self.config = config
        self.smoke_test = smoke_test
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Paths
        self.ckpt_dir = config['output']['checkpoint_dir']
        self.results_dir = config['output']['results_dir']
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Data
        data_cfg = config['data']
        train_loader, val_loader, n_train, n_val = get_multilabel_dataloaders(
            csv_path=data_cfg['csv_path'],
            bids_root=data_cfg['bids_root'],
            split_pkl=data_cfg['split_pkl'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 2),
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        print(f"  Train: {n_train} samples | Val: {n_val} samples")

        # Model
        self.model = Task1aMultiLabelModel().to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")

        # Optimiser
        train_cfg = config['training']
        self.optimizer = Adam(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 1e-5),
        )

        # Loss hyperparameters
        loss_cfg = train_cfg.get('loss', {})
        self.emd_weight = loss_cfg.get('emd_weight', 1.0)
        self.focal_weight = loss_cfg.get('focal_weight', 1.0)
        self.focal_gamma = loss_cfg.get('focal_gamma', 2.0)
        self.focal_alpha = tuple(loss_cfg.get('focal_alpha', [0.25, 0.5, 1.0]))

        # Scheduler
        num_epochs = 2 if smoke_test else train_cfg['num_epochs']
        self.num_epochs = num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        # Early stopping
        es_cfg = config.get('early_stopping', {})
        self.patience = es_cfg.get('patience', 10)
        self.best_score = -float('inf')
        self.no_improve = 0

    # ── Epoch helpers ────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool) -> dict:
        """Run one epoch; return dict with loss and per-task F1."""
        self.model.train(train)
        context = torch.enable_grad() if train else torch.no_grad()

        total_loss = 0.0
        all_preds = [[] for _ in TASK_NAMES]
        all_labels = [[] for _ in TASK_NAMES]

        with context:
            for batch_idx, batch in enumerate(loader):
                if self.smoke_test and batch_idx >= 2:
                    break

                imgs = batch['img'].to(self.device).float()
                labels = batch['labels'].to(self.device)  # [B, 7]

                logits = self.model(imgs)                  # [B, 7, 3]

                loss = (
                    self.emd_weight * ordinal_emd_loss(logits, labels)
                    + self.focal_weight * focal_loss(
                        logits, labels,
                        gamma=self.focal_gamma,
                        alpha=self.focal_alpha,
                    )
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1).cpu().numpy()  # [B, 7]
                lbl_np = labels.cpu().numpy()                       # [B, 7]
                for t in range(len(TASK_NAMES)):
                    all_preds[t].extend(preds[:, t].tolist())
                    all_labels[t].extend(lbl_np[:, t].tolist())

        n_batches = min(2, len(loader)) if self.smoke_test else len(loader)
        avg_loss = total_loss / max(n_batches, 1)

        per_task_f1 = {}
        for t, task in enumerate(TASK_NAMES):
            f1 = f1_score(all_labels[t], all_preds[t], average='macro', zero_division=0)
            per_task_f1[task] = float(f1)
        mean_f1 = float(np.mean(list(per_task_f1.values())))

        return {'loss': avg_loss, 'mean_f1': mean_f1, 'per_task_f1': per_task_f1}

    # ── Main training loop ───────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"RUN_0002 – Multi-label Training ({self.num_epochs} epochs)")
        print(f"Loss: EMD (w={self.emd_weight}) + Focal (w={self.focal_weight})")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        best_ckpt = os.path.join(self.ckpt_dir, "multilabel_best.pt")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:03d}/{self.num_epochs:03d} "
                f"| train_loss={train_metrics['loss']:.4f} "
                f"| val_f1={val_metrics['mean_f1']:.4f} "
                f"| {elapsed:.0f}s"
            )

            # Per-task breakdown every 5 epochs
            if epoch % 5 == 0 or epoch == self.num_epochs:
                for task, f1 in val_metrics['per_task_f1'].items():
                    print(f"    {task:<16} F1={f1:.3f}")

            # Early stopping + checkpoint
            if val_metrics['mean_f1'] > self.best_score:
                self.best_score = val_metrics['mean_f1']
                self.no_improve = 0
                torch.save(self.model.state_dict(), best_ckpt)
                print(f"  ✓ New best val_f1={self.best_score:.4f} – checkpoint saved")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience and not self.smoke_test:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        print(f"\n✓ Training complete. Best val_f1={self.best_score:.4f}")
        print(f"  Checkpoint: {best_ckpt}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RUN_0002 – Multi-label Task 1a Training")
    parser.add_argument('--config', type=str, default='configs/run_0002_upf.yaml')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Quick sanity check: 2 epochs, 2 batches each')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = apply_env_overrides(config)

    if args.smoke_test:
        print("⚡ SMOKE TEST MODE – limited epochs and batches")

    trainer = MultiLabelTrainer(config, smoke_test=args.smoke_test)
    trainer.train()


if __name__ == '__main__':
    main()
