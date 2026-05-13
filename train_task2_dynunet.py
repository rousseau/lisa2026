#!/usr/bin/env python
"""Train DynUNet baseline for LISA 2026 Task 2 (RUN_0003)."""
import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from tqdm import tqdm

from src.datasets import get_task2_seg_dataloaders
from src.models import Task2DynUNetModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        self.use_amp = bool(config["environment"].get("mixed_precision", True)) and self.device == "cuda"

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
        )

        self.n_train = n_train
        self.n_val = n_val

        model_cfg = config["model"]
        self.model = Task2DynUNetModel(
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            kernel_size=tuple(tuple(int(x) for x in ks) for ks in model_cfg["kernel_size"]),
            strides=tuple(tuple(int(x) for x in st) for st in model_cfg["strides"]),
            upsample_kernel_size=tuple(tuple(int(x) for x in st) for st in model_cfg["upsample_kernel_size"]),
            filters=tuple(int(x) for x in model_cfg["filters"]),
            norm_name=model_cfg.get("norm_name", "instance"),
            deep_supervision=bool(model_cfg.get("deep_supervision", False)),
        ).to(self.device)

        self.loss_fn = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=float(config["training"]["loss"].get("lambda_dice", 1.0)),
            lambda_ce=float(config["training"]["loss"].get("lambda_ce", 1.0)),
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
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

        self.best_val_dice = -1.0
        self.patience_counter = 0

        self.num_epochs = int(config["training"]["num_epochs"])
        self.patience = int(config["early_stopping"]["patience"])

        if self.smoke_test:
            self.num_epochs = 1

        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"Task2 train subjects: {self.n_train}, val subjects: {self.n_val}")

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train")):
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
            pred_one_hot = torch.nn.functional.one_hot(
                pred.squeeze(1), num_classes=int(self.config["model"]["out_channels"])
            ).permute(0, 4, 1, 2, 3).float()
            label_one_hot = torch.nn.functional.one_hot(
                labels.squeeze(1).long(), num_classes=int(self.config["model"]["out_channels"])
            ).permute(0, 4, 1, 2, 3).float()
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
        history = []

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
    parser.add_argument("--config", type=str, default="configs/run_0003_task2_dynunet.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Task2Trainer(config=config, smoke_test=args.smoke_test)
    trainer.train()


if __name__ == "__main__":
    main()
