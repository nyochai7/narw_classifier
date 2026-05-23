"""Linear probe on top of frozen Perch v2 embeddings.

Single ``Linear(1536, 1)`` + ``BCEWithLogitsLoss`` + AdamW. Same per-epoch
metrics as the EfficientNet baseline (AUROC, AP, accuracy / precision / recall
/ F1 / FPR at 0.5, recall@1%FPR, recall@5%FPR, best F1, TP/FP/FN/TN counts) so
runs are directly comparable in W&B.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from ..utils.epoch_metrics import EpochMetricsMixin


class PerchLinearProbe(EpochMetricsMixin, pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 1536,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(embedding_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_auroc = tm.AUROC(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, embedding_dim)``. Returns logits ``(B,)``."""
        return self.net(x).squeeze(-1)

    def _step(self, batch):
        emb, label = batch
        logits = self(emb)
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
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
