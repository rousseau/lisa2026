"""Task 2 segmentation trainer — DynUNet (RUN_0003)."""

import json
import os
import warnings

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import get_task2_seg_dataloaders
from src.models import Task2DynUNetModel
from src.training.base import BaseTrainer


def _to_3tuple(values):
    return tuple(int(v) for v in values)


class Task2Trainer(BaseTrainer):
    """DynUNet segmentation trainer for Task 2 (RUN_0003).

    Supports optional symmetry consistency regularisation and pretrained
    encoder warm-start.  Early stopping monitors validation Dice (higher =
    better).
    """

    val_metric_key = "val_dice"
    val_metric_direction = "max"

    def __init__(self, config: dict, smoke_test: bool = False):
        super().__init__(config, smoke_test)
        self._build_dataloaders()
        self._build_model()
        self._build_optimizer()
        self._load_pretrained()

        self.num_epochs = 1 if smoke_test else int(config["training"]["num_epochs"])
        self.ckpt_path = os.path.join(self.ckpt_dir, "task2_dynunet_best.pt")
        self.history_path = os.path.join(self.results_dir, "training_history.json")

        # Run metadata (prepended to training history)
        self.run_meta = {
            "run_id": self.config.get("run_id", "0003"),
            "pretrained_checkpoint_loaded": self.pretrained_loaded,
            "device": self.device,
            "num_classes": self.num_classes,
            "filters": list(self.config["model"].get("filters", [])),
        }

        # Symmetry consistency
        self.symmetry_weight = float(config["training"].get("symmetry_consistency_weight", 0.0))
        self.symmetry_flip_axes = [int(ax) for ax in config["training"].get("symmetry_flip_axes", [])]
        self.symmetry_channel_permutation = self._build_symmetry_permutation(
            config["training"].get("symmetry_channel_pairs", []),
            self.num_classes,
        )

        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"Task2 train: {self.n_train}  val: {self.n_val}")
        print(f"Loss: {self.loss_type} | Symmetry w={self.symmetry_weight}")

    # ── BaseTrainer interface ────────────────────────────────────────────────

    def _build_model(self) -> None:
        cfg = self.config["model"]
        self.num_classes = int(cfg["out_channels"])
        self.model = Task2DynUNetModel(
            in_channels=int(cfg["in_channels"]),
            out_channels=self.num_classes,
            kernel_size=tuple(tuple(int(x) for x in ks) for ks in cfg["kernel_size"]),
            strides=tuple(tuple(int(x) for x in st) for st in cfg["strides"]),
            upsample_kernel_size=tuple(tuple(int(x) for x in st) for st in cfg["upsample_kernel_size"]),
            filters=tuple(int(x) for x in cfg["filters"]),
            norm_name=cfg.get("norm_name", "instance"),
            deep_supervision=bool(cfg.get("deep_supervision", False)),
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

    def _build_dataloaders(self) -> None:
        cfg = self.config
        data_root = os.getenv("LISA_DATA_ROOT", cfg["data"]["data_root"])
        split_pkl = os.getenv("LISA_TASK2_SPLIT_PKL", cfg["data"]["split_pkl"])
        patch_size = _to_3tuple(cfg["data"]["patch_size"])
        self.val_roi_size = _to_3tuple(cfg["data"]["val_roi_size"])
        batch_size = 1 if self.smoke_test else int(cfg["training"]["batch_size"])
        train_num_samples = 1 if self.smoke_test else int(cfg["data"].get("train_num_samples", 2))

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

    def _build_optimizer(self) -> None:
        cfg = self.config["training"]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(cfg.get("num_epochs", 1))),
            eta_min=float(cfg.get("min_learning_rate", 1e-6)),
        )

    # ── Pretrained loading ───────────────────────────────────────────────────

    def _load_pretrained(self) -> None:
        pretrained = self.config["training"].get("pretrained_checkpoint", "")
        self.pretrained_loaded = False
        if not pretrained:
            return
        if not os.path.exists(pretrained):
            warnings.warn(f"[WARNING] Pretrained checkpoint not found: {pretrained}. Training from scratch.", stacklevel=2)
            return
        state = torch.load(pretrained, map_location=self.device)
        try:
            self.model.load_state_dict(state["model_state_dict"])
            self.pretrained_loaded = True
            print(f"[INFO] Loaded pretrained checkpoint: {pretrained}")
        except RuntimeError as err:
            warnings.warn(f"[WARNING] Pretrained checkpoint skipped — architecture mismatch: {err}", stacklevel=2)
            print("[WARNING] Skipping pretrained checkpoint. Training from scratch.")

    # ── Symmetry helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _build_symmetry_permutation(channel_pairs, num_classes):
        perm = list(range(num_classes))
        for left, right in channel_pairs:
            l, r = int(left), int(right)
            if l < num_classes and r < num_classes:
                perm[l], perm[r] = r, l
        return perm

    def _symmetry_consistency_loss(self, images: torch.Tensor) -> torch.Tensor:
        if self.symmetry_weight <= 0.0 or not self.symmetry_flip_axes:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            ref_probs = torch.softmax(self.model(images), dim=1)
        total = torch.tensor(0.0, device=self.device)
        for axis in self.symmetry_flip_axes:
            flip_logits = self.model(torch.flip(images, dims=[axis]))
            flip_probs = torch.softmax(torch.flip(flip_logits, dims=[axis]), dim=1)
            flip_probs = flip_probs[:, self.symmetry_channel_permutation]
            total = total + F.mse_loss(flip_probs, ref_probs)
        return total / max(1, len(self.symmetry_flip_axes))

    # ── Epoch methods ────────────────────────────────────────────────────────

    def train_one_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        last_idx = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train")):
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels) + self.symmetry_weight * self._symmetry_consistency_loss(images)
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

    def _print_epoch(self, epoch: int, train_m: dict, val_m: dict) -> None:
        print(
            f"Epoch {epoch:03d}/{self.num_epochs:03d} | "
            f"train_loss={train_m['train_loss']:.4f} "
            f"val_loss={val_m['val_loss']:.4f} "
            f"val_dice={val_m['val_dice']:.4f}"
        )

    def save_history(self, history: list) -> None:
        """Override to prepend run metadata."""
        with open(self.history_path, "w") as f:
            import json as _json
            _json.dump([self.run_meta] + history, f, indent=2)
