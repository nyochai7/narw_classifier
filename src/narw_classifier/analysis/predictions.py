"""Run a trained model on a DataLoader, save the predictions for offline analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_predictions(
    model: pl.LightningModule,
    dataloader: DataLoader,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` on every batch from ``dataloader`` and return ``(probs, labels)``.

    Assumes the model's ``forward`` returns 1-D logits ``(B,)`` for binary classification
    (matches both :class:`BaselineEfficientNet` and :class:`PerchLinearProbe`).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval().to(device)

    probs_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        logits = model(x)
        probs_chunks.append(torch.sigmoid(logits).cpu().numpy())
        label_chunks.append(y.cpu().numpy())
    return np.concatenate(probs_chunks), np.concatenate(label_chunks)


def save_predictions(
    out_path: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    files: list[str] | None = None,
) -> None:
    """Save ``(probs, labels, files)`` to a single .npz so the analysis notebook
    can reload them without re-running inference."""
    if probs.shape != labels.shape:
        raise ValueError(f"probs / labels shape mismatch: {probs.shape} vs {labels.shape}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"probs": probs.astype(np.float32), "labels": labels.astype(np.int64)}
    if files is not None:
        if len(files) != len(probs):
            raise ValueError(f"files length {len(files)} does not match predictions ({len(probs)})")
        payload["files"] = np.array(files, dtype=object)
    np.savez_compressed(out_path, **payload)


def load_predictions(
    in_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Inverse of :func:`save_predictions`. Returns ``(probs, labels, files_or_None)``."""
    z = np.load(in_path, allow_pickle=True)
    files = list(z["files"]) if "files" in z.files else None
    return z["probs"], z["labels"], files
