"""Task 2 hybrid trainer — two-phase training (RUN_0003c).

Phase 1: Train fusion blocks + decoder only (encoders frozen).
Phase 2: Fine-tune everything except MedSAM2 encoder.
"""

import json
import os

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import get_task2_seg_dataloaders
from src.models import Task2HybridModel
from src.training.base import BaseTrainer


def _to_3tuple(values):
    return tuple(int(v) for v in values)


class Task2HybridTrainer(BaseTrainer):
    """Hybrid model trainer with 2-phase curriculum (RUN_0003c)."""

    val_metric_key = "val_dice"
    val_metric_direction = "max"

    def __init__(self, config: dict, smoke_test: bool = False):
        super().__init__(config, smoke_test)
        self._build_dataloaders()
        self._build_model()
        self._setup_phases()

        self.num_epochs = 1 if smoke_test else int(config["training"]["num_epochs"])
        self.phase1_epochs = int(config["training"].get("phase1_epochs", 20))
        self.ckpt_path = os.path.join(self.ckpt_dir, "task2_hybrid_best.pt")
        self.history_path = os.path.join(self.results_dir, "training_history.json")
        self.current_phase = 1

        self.run_meta = {
            "run_id": config.get("run_id", "0003c"),
            "phase1_epochs": self.phase1_epochs,
            "device": self.device,
            "num_classes": self.num_classes,
        }

        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"Task2 Hybrid train: {self.n_train}  val: {self.n_val}")
        print(f"Phase 1: {self.phase1_epochs} epochs (fusion+decoder only)")
        print(f"Phase 2: {self.num_epochs - self.phase1_epochs} epochs (full model except SAM)")

    def _build_model(self) -> None:
        cfg = self.config["model"]
        self.num_classes = int(cfg["out_channels"])
        self.model = Task2HybridModel(
            nnunet_checkpoint=cfg.get("nnunet_checkpoint", ""),
            medsam2_checkpoint=cfg["medsam2_checkpoint"],
            num_classes=self.num_classes,
            filters=tuple(int(x) for x in cfg.get("filters", [32, 64, 128, 256, 320])),
            device=self.device,
        ).to(self.device)

        loss_cfg = self.config["training"].get("loss", {})
        self.loss_type = loss_cfg.get("type", "dice_ce")
        if self.loss_type == "dice_focal":
            self.loss_fn = DiceFocalLoss(
                include_background=False, to_onehot_y=True, softmax=True,
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                lambda_focal=float(loss_cfg.get("lambda_focal", 1.0)),
                gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            )
        else:
            self.loss_fn = DiceCELoss(
                include_background=False, to_onehot_y=True, softmax=True,
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
            )

    def _setup_phases(self) -> None:
        """Configure parameter groups for 2-phase training."""
        self._phase1_params = []
        self._phase2_params = []

        # Phase 1: fusion blocks + decoder + final conv + SAM projection
        for name, p in self.model.named_parameters():
            if p.requires_grad and "medsam2_encoder" not in name:
                if "fusion" in name or "up_blocks" in name or "final_conv" in name or "sam_proj" in name or "encoder_stages" not in name:
                    self._phase1_params.append(p)
                else:
                    self._phase2_params.append(p)
            elif not p.requires_grad:
                # MedSAM2 encoder parameters are frozen
                pass

        # Phase 2: everything trainable except MedSAM2 encoder
        self._phase2_params = [p for p in self.model.parameters() if p.requires_grad]

    def _build_dataloaders(self) -> None:
        cfg = self.config
        data_root = os.getenv("LISA_DATA_ROOT", cfg["data"]["data_root"])
        split_pkl = os.getenv("LISA_TASK2_SPLIT_PKL", cfg["data"]["split_pkl"])
        patch_size = _to_3tuple(cfg["data"]["patch_size"])
        self.val_roi_size = _to_3tuple(cfg["data"]["val_roi_size"])
        batch_size = 1 if self.smoke_test else int(cfg["training"]["batch_size"])
        train_num_samples = 1 if self.smoke_test else int(cfg["data"].get("train_num_samples", 1))

        self.train_loader, self.val_loader, self.n_train, self.n_val = get_task2_seg_dataloaders(
            data_root=data_root,
            split_pkl=split_pkl,
            batch_size=batch_size,
            num_workers=int(cfg["training"]["num_workers"]),
            image_suffix=cfg["data"].get("image_suffix", "_ciso.nii.gz"),
            label_suffix=cfg["data"].get("label_suffix", "_LF_seg.nii.gz"),
            patch_size=patch_size,
            num_samples_per_volume=train_num_samples,
            num_classes=int(cfg["model"]["out_channels"]),
            collapse_labels=bool(cfg["data"].get("collapse_labels", False)),
        )

    def _build_optimizer(self, lr: float, params) -> None:
        cfg = self.config["training"]
        self.optimizer = AdamW(params, lr=float(lr), weight_decay=float(cfg["weight_decay"]))
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(cfg.get("num_epochs", 1))),
            eta_min=float(cfg.get("min_learning_rate", 1e-6)),
        )

    def train_one_epoch(self) -> dict:
        self.model.train()
        if hasattr(self.model, "medsam2_encoder"):
            self.model.medsam2_encoder.eval()

        total_loss = 0.0
        last_idx = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train P{self.current_phase}")):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += float(loss.item())
            last_idx = batch_idx
            if self.smoke_test and batch_idx >= 1:
                break

        return {"train_loss": total_loss / max(1, last_idx + 1)}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        num_classes = int(self.config["model"]["out_channels"])

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Val")):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = sliding_window_inference(
                    images,
                    roi_size=self.val_roi_size,
                    sw_batch_size=int(self.config["inference"]["sw_batch_size"]),
                    predictor=self.model,
                    overlap=float(self.config["inference"]["overlap"]),
                )
                loss = self.loss_fn(logits, labels)
            total_loss += float(loss.item())
            pred = torch.argmax(logits, dim=1, keepdim=True)
            pred_oh = F.one_hot(pred.squeeze(1), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            lbl_oh = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            dice_metric(y_pred=pred_oh, y=lbl_oh)
            if self.smoke_test and batch_idx >= 0:
                break

        val_dice = float(dice_metric.aggregate().item())
        dice_metric.reset()
        return {self.val_metric_key: val_dice, "val_loss": total_loss / max(1, batch_idx + 1)}

    def train(self) -> None:
        history = []
        for epoch in range(self.num_epochs):
            # ── Phase switch ────────────────────────────────────────────────
            if epoch == 0:
                self.current_phase = 1
                lr = float(self.config["training"].get("phase1_lr", 1e-3))
                self._build_optimizer(lr, self._phase1_params)
                print(f"\n>>> Starting Phase 1 (epochs 1-{self.phase1_epochs}, LR={lr})")
            elif epoch == self.phase1_epochs:
                self.current_phase = 2
                lr = float(self.config["training"].get("phase2_lr", 5e-4))
                # Rebuild optimizer with all trainable params
                self._build_optimizer(lr, self._phase2_params)
                print(f"\n>>> Starting Phase 2 (epochs {self.phase1_epochs+1}-{self.num_epochs}, LR={lr})")

            train_m = self.train_one_epoch()
            val_m = self.validate()

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()

            row = {"epoch": epoch + 1, "phase": self.current_phase, **train_m, **val_m}
            history.append(row)
            self._print_epoch(epoch + 1, train_m, val_m)

            if self.is_improvement(val_m[self.val_metric_key]):
                self.update_best(val_m[self.val_metric_key])
                self.save_checkpoint(epoch + 1, val_m[self.val_metric_key], self.ckpt_path)
                print(f"  -> New best checkpoint ({self.ckpt_path})")
            elif self.increment_patience():
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break

            if self.smoke_test:
                break

        self.save_history(history)
        print(f"Training complete. Best {self.val_metric_key}: {self.best_val_metric:.4f}")

    def _print_epoch(self, epoch: int, train_m: dict, val_m: dict) -> None:
        print(
            f"Epoch {epoch:03d}/{self.num_epochs:03d} | "
            f"Phase {self.current_phase} | "
            f"train_loss={train_m['train_loss']:.4f} "
            f"val_loss={val_m['val_loss']:.4f} "
            f"val_dice={val_m['val_dice']:.4f}"
        )

    def save_history(self, history: list) -> None:
        with open(self.history_path, "w") as f:
            import json as _json
            _json.dump([self.run_meta] + history, f, indent=2)
