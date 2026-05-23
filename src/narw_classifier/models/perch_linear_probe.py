"""Linear probe on top of frozen Perch v2 embeddings.

Single ``Linear(1536, 1)`` + ``BCEWithLogitsLoss`` + AdamW. Same metrics as the
EfficientNet baseline so runs are directly comparable in W&B.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm


class PerchLinearProbe(pl.LightningModule):
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
        self.val_auroc = tm.AUROC(task="binary")
        self.val_ap = tm.AveragePrecision(task="binary")
        self.val_acc = tm.Accuracy(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, embedding_dim)``. Returns logits ``(B,)``."""
        return self.net(x).squeeze(-1)

    def _step(self, batch):
        emb, label = batch
        logits = self(emb)
        loss = self.loss_fn(logits, label.float())
        return loss, logits, label

    def training_step(self, batch, batch_idx):
        loss, logits, label = self._step(batch)
        probs = torch.sigmoid(logits)
        self.train_auroc.update(probs, label)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def validation_step(self, batch, batch_idx):
        loss, logits, label = self._step(batch)
        probs = torch.sigmoid(logits)
        self.val_auroc.update(probs, label)
        self.val_ap.update(probs, label)
        self.val_acc.update(probs, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/ap", self.val_ap.compute(), prog_bar=True)
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.val_auroc.reset()
        self.val_ap.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
