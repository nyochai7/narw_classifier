"""EfficientNet-B3 baseline LightningModule for NARW upcall binary classification.

Supports two freezing modes:
- ``linear_probe``: freeze all backbone weights, train only the classification head.
- ``full_finetune``: train every parameter.

Per-epoch metrics (val + test): AUROC, AP, accuracy / precision / recall / F1 / FPR
at threshold 0.5, recall @ 1% FPR, recall @ 5% FPR, best F1, plus raw TP/FP/FN/TN
counts. All logged under ``val/*`` and ``test/*`` so W&B has everything needed for
FP/FN error analysis without re-running inference.
"""

from __future__ import annotations

from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from ..utils.epoch_metrics import EpochMetricsMixin
from .preprocess import MelImagePreprocessor

FreezeMode = Literal["linear_probe", "full_finetune"]


def _build_backbone(pretrained: bool) -> nn.Module:
    """Pretrained EfficientNet-B3 with the final ``Linear(1536, 1000)`` replaced
    by a binary head (``Linear(1536, 1)``)."""
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    net = efficientnet_b3(weights=weights)
    in_features = net.classifier[1].in_features  # 1536 for B3
    net.classifier[1] = nn.Linear(in_features, 1)
    return net


class BaselineEfficientNet(EpochMetricsMixin, pl.LightningModule):
    def __init__(
        self,
        preprocessor: MelImagePreprocessor,
        freeze_mode: FreezeMode = "linear_probe",
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str | None = None,
        max_epochs: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["preprocessor"])
        self.preprocessor = preprocessor
        self.net = _build_backbone(pretrained=pretrained)
        self._apply_freeze_mode(freeze_mode)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_auroc = tm.AUROC(task="binary")

    def _apply_freeze_mode(self, mode: FreezeMode) -> None:
        if mode == "linear_probe":
            for p in self.net.parameters():
                p.requires_grad = False
            for p in self.net.classifier.parameters():
                p.requires_grad = True
        elif mode == "full_finetune":
            for p in self.net.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze_mode: {mode}")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """``waveform``: ``(B, n_samples)``. Returns raw logits ``(B,)``."""
        img = self.preprocessor(waveform)
        return self.net(img).squeeze(-1)

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveform, label = batch
        logits = self(waveform)
        loss = self.loss_fn(logits, label.float())
        return loss, logits, label

    # ---- train ---------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, logits, label = self._step(batch)
        probs = torch.sigmoid(logits)
        self.train_auroc.update(probs, label)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    # ---- val / test ----------------------------------------------------

    def validation_step(self, batch, batch_idx):
        loss, logits, label = self._step(batch)
        probs = torch.sigmoid(logits)
        self._val_buffer.append(probs, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def test_step(self, batch, batch_idx):
        loss, logits, label = self._step(batch)
        probs = torch.sigmoid(logits)
        self._test_buffer.append(probs, label)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    # ---- optimizer -----------------------------------------------------

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        if self.hparams.scheduler is None:
            return optimizer
        if self.hparams.scheduler == "cosine":
            if self.hparams.max_epochs is None:
                raise ValueError("scheduler=cosine requires max_epochs to be set")
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs
            )
            return {"optimizer": optimizer, "lr_scheduler": sched}
        raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
