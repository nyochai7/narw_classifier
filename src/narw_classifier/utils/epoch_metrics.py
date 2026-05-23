"""Per-epoch eval-metrics mixin for binary LightningModules.

Buffers probs + labels during validation/test steps, then computes the full
:func:`narw_classifier.analysis.artifacts.compute_summary` dict at epoch end and
logs each entry under ``{split}/{key}``. This way the W&B run history contains
recall@FPR, F1, TP/FP/FN/TN counts, etc. — not just AUROC — for every epoch.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch

from ..analysis.artifacts import compute_summary

# Keys we log to the progress bar — keep it short.
_PROG_BAR_KEYS = ("auroc", "ap", "f1_t0.5")


class _Buffer:
    """Append-only buffer for (probs, labels) tensors across an epoch."""

    def __init__(self) -> None:
        self.probs: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def append(self, probs: torch.Tensor, labels: torch.Tensor) -> None:
        self.probs.append(probs.detach().cpu())
        self.labels.append(labels.detach().cpu())

    def drain(self) -> tuple[np.ndarray, np.ndarray]:
        probs = torch.cat(self.probs).numpy()
        labels = torch.cat(self.labels).numpy()
        self.probs.clear()
        self.labels.clear()
        return probs, labels


class EpochMetricsMixin(pl.LightningModule):
    """Adds val + test buffers and a ``log_epoch_metrics`` helper.

    Subclasses must call ``self._val_buffer.append(probs, labels)`` from
    ``validation_step`` and ``self._test_buffer.append(...)`` from ``test_step``,
    then call ``self._log_epoch_metrics(split)`` in the corresponding epoch-end
    hook.
    """

    def __init__(self) -> None:
        super().__init__()
        self._val_buffer = _Buffer()
        self._test_buffer = _Buffer()

    def _log_epoch_metrics(self, split: str) -> dict[str, float]:
        buf = self._val_buffer if split == "val" else self._test_buffer
        if not buf.probs:
            return {}
        probs, labels = buf.drain()
        summary = compute_summary(probs, labels)
        for key, value in summary.items():
            self.log(
                f"{split}/{key}",
                float(value),
                prog_bar=(key in _PROG_BAR_KEYS),
            )
        return summary
