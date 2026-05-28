#!/usr/bin/env python
"""Train DynUNet baseline for LISA 2026 Task 2 (RUN_0003)."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import get_task2_seg_dataloaders
from src.models import Task2DynUNetModel
from src.utils.seed import set_seed


def to_3tuple(values):
    return tuple(int(v) for v in values)


class Task2Trainer:
    def __init__(self, config: dict, smoke_test: bool = False):
        self.config = config
        self.smoke_test = smoke_test

        seed = int(config["environment"]["seed"])
        set_seed(seed)

        if config["environment"].get("deterministic", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (
            bool(config["environment"].get("mixed_precision", True))
            and self.device == "cuda"
        )
        self.num_classes = int(config["model"]["out_channels"])

        data_root = os.getenv("LISA_DATA_ROOT", config["data"]["data_root"])
        split_pkl = os.getenv("LISA_TASK2_SPLIT_PKL", config["data"]["split_pkl"])

        patch_size = to_3tuple(config["data"]["patch_size"])
        self.val_roi_size = to_3tuple(config["data"]["val_roi_size"])

        batch_size = int(config["training"]["batch_size"])
        num_workers = int(config["training"]["num_workers"])

        train_num_samples = int(config["data"].get("train_num_samples", 2))
        if smoke_test:
            train_num_samples = 1
            batch_size = 1

        self.train_loader, self.val_loader, n_train, n_val = get_task2_seg_dataloaders(
            data_root=data_root,
            split_pkl=split_pkl,
            batch_size=batch_size,
            num_workers=num_workers,
            image_suffix=config["data"].get("image_suffix", "_ciso.nii.gz"),
            label_suffix=config["data"].get("label_suffix", "_LF_seg.nii.gz"),
            patch_size=patch_size,
            num_samples_per_volume=train_num_samples,
            num_classes=int(config["model"]["out_channels"]),
            collapse_labels=bool(config["data"].get("collapse_labels", False)),
        )

        self.n_train = n_train
        self.n_val = n_val

        model_cfg = config["model"]
        self.model = Task2DynUNetModel(
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            kernel_size=tuple(
                tuple(int(x) for x in ks) for ks in model_cfg["kernel_size"]
            ),
            strides=tuple(tuple(int(x) for x in st) for st in model_cfg["strides"]),
            upsample_kernel_size=tuple(
                tuple(int(x) for x in st) for st in model_cfg["upsample_kernel_size"]
            ),
            filters=tuple(int(x) for x in model_cfg["filters"]),
            norm_name=model_cfg.get("norm_name", "instance"),
            deep_supervision=bool(model_cfg.get("deep_supervision", False)),
        ).to(self.device)

        loss_cfg = config["training"].get("loss", {})
        self.loss_type = loss_cfg.get("type", "dice_ce")
        if self.loss_type == "dice_focal":
            self.loss_fn = DiceFocalLoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True,
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                lambda_focal=float(loss_cfg.get("lambda_focal", 1.0)),
                gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            )
        else:
            self.loss_fn = DiceCELoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True,
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
            )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(config["training"].get("num_epochs", 1))),
            eta_min=float(config["training"].get("min_learning_rate", 1.0e-6)),
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        out_dir = config["output"]
        self.ckpt_dir = out_dir["checkpoint_dir"]
        self.log_dir = out_dir["log_dir"]
        self.results_dir = out_dir["results_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.ckpt_path = os.path.join(self.ckpt_dir, "task2_dynunet_best.pt")
        self.history_path = os.path.join(self.results_dir, "training_history.json")

        pretrained = config["training"].get("pretrained_checkpoint", "")
        self.pretrained_loaded = False
        if pretrained and os.path.exists(pretrained):
            state = torch.load(pretrained, map_location=self.device)
            try:
                self.model.load_state_dict(state["model_state_dict"])
                self.pretrained_loaded = True
                print(f"[INFO] Loaded pretrained checkpoint: {pretrained}")
            except RuntimeError as err:
                import warnings

                warnings.warn(
                    f"[WARNING] Pretrained checkpoint SKIPPED — architecture mismatch: {err}\n"
                    f"  Checkpoint: {pretrained}\n"
                    f"  Model filters: {config['model'].get('filters')}\n"
                    "  Training will proceed from scratch (random init).",
                    stacklevel=2,
                )
                print(
                    f"[WARNING] Skipping pretrained checkpoint (incompatible architecture)."
                )
                print(f"  Checkpoint : {pretrained}")
                print(f"  Model will be trained from scratch (random initialisation).")
        elif pretrained and not os.path.exists(pretrained):
            import warnings

            warnings.warn(
                f"[WARNING] Pretrained checkpoint path not found: {pretrained}\n"
                "  Training will proceed from scratch.",
                stacklevel=2,
            )

        self.symmetry_weight = float(
            config["training"].get("symmetry_consistency_weight", 0.0)
        )
        self.symmetry_flip_axes = [
            int(ax) for ax in config["training"].get("symmetry_flip_axes", [])
        ]
        self.symmetry_channel_permutation = self._build_symmetry_channel_permutation(
            config["training"].get("symmetry_channel_pairs", []),
            self.num_classes,
        )

        self.best_val_dice = -1.0
        self.patience_counter = 0

        self.num_epochs = int(config["training"]["num_epochs"])
        self.patience = int(config["early_stopping"]["patience"])

        if self.smoke_test:
            self.num_epochs = 1

        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"Task2 train subjects: {self.n_train}, val subjects: {self.n_val}")
        print(f"Loss: {self.loss_type}")
        print(
            f"Symmetry consistency: weight={self.symmetry_weight} axes={self.symmetry_flip_axes}"
        )

    @staticmethod
    def _build_symmetry_channel_permutation(channel_pairs, num_classes):
        permutation = list(range(num_classes))
        for left_class, right_class in channel_pairs:
            left_index = int(left_class)
            right_index = int(right_class)
            if left_index < num_classes and right_index < num_classes:
                permutation[left_index] = right_index
                permutation[right_index] = left_index
        return permutation

    def _symmetry_consistency_loss(self, images):
        if self.symmetry_weight <= 0.0 or len(self.symmetry_flip_axes) == 0:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            ref_logits = self.model(images)
            ref_probs = torch.softmax(ref_logits, dim=1)

        total = 0.0
        for axis in self.symmetry_flip_axes:
            flipped_images = torch.flip(images, dims=[axis])
            flip_logits = self.model(flipped_images)
            flip_probs = torch.softmax(torch.flip(flip_logits, dims=[axis]), dim=1)
            flip_probs = flip_probs[:, self.symmetry_channel_permutation]
            total = total + F.mse_loss(flip_probs, ref_probs)

        return total / max(1, len(self.symmetry_flip_axes))

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train")):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                seg_loss = self.loss_fn(logits, labels)
                sym_loss = self._symmetry_consistency_loss(images)
                loss = seg_loss + self.symmetry_weight * sym_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())

            if self.smoke_test and batch_idx >= 1:
                break

        denom = max(1, batch_idx + 1)
        return total_loss / denom

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        dice_metric = DiceMetric(include_background=False, reduction="mean")

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

            total_val_loss += float(loss.item())

            pred = torch.argmax(logits, dim=1, keepdim=True)
            pred_one_hot = (
                torch.nn.functional.one_hot(
                    pred.squeeze(1),
                    num_classes=int(self.config["model"]["out_channels"]),
                )
                .permute(0, 4, 1, 2, 3)
                .float()
            )
            label_one_hot = (
                torch.nn.functional.one_hot(
                    labels.squeeze(1).long(),
                    num_classes=int(self.config["model"]["out_channels"]),
                )
                .permute(0, 4, 1, 2, 3)
                .float()
            )
            dice_metric(y_pred=pred_one_hot, y=label_one_hot)

            if self.smoke_test and batch_idx >= 0:
                break

        mean_dice = float(dice_metric.aggregate().item())
        dice_metric.reset()

        denom = max(1, batch_idx + 1)
        return total_val_loss / denom, mean_dice

    def save_checkpoint(self, epoch, val_dice):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_dice": val_dice,
                "config": self.config,
            },
            self.ckpt_path,
        )

    def train(self):
        # Record run metadata at the top of the history for traceability
        run_meta = {
            "run_id": self.config.get("run_id", "unknown"),
            "pretrained_checkpoint_requested": self.config["training"].get(
                "pretrained_checkpoint", ""
            ),
            "pretrained_checkpoint_loaded": self.pretrained_loaded,
            "device": self.device,
            "num_classes": self.num_classes,
            "filters": list(self.config["model"].get("filters", [])),
        }
        history = [run_meta]

        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_dice = self.validate()

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
            }
            history.append(row)

            print(
                f"Epoch {epoch + 1:03d}/{self.num_epochs:03d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
            )

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.patience_counter = 0
                self.save_checkpoint(epoch=epoch + 1, val_dice=val_dice)
                print(f"  -> New best checkpoint saved ({self.ckpt_path})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"  -> Early stopping at epoch {epoch + 1}")
                    break

            self.scheduler.step()

            if self.smoke_test:
                break

        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Training completed. Best val Dice: {self.best_val_dice:.4f}")


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/run_0003_task2_dynunet.yaml"
    )
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Task2Trainer(config=config, smoke_test=args.smoke_test)
    trainer.train()


if __name__ == "__main__":
    main()
