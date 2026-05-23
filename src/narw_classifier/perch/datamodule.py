"""LightningDataModule that serves cached Perch embeddings + labels.

Cache layout (per ``cache_dir``):
    cache_dir/train.npz
    cache_dir/val.npz

The train loader uses a WeightedRandomSampler so each batch is class-balanced
(addresses the ~8:1 imbalance, matching the EfficientNet baseline).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .cache import load_split


def _make_balanced_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"Need both classes present; got counts={counts.tolist()}")
    class_weight = 1.0 / counts
    weights = torch.tensor([class_weight[y] for y in labels], dtype=torch.double)
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True, generator=g
    )


class PerchEmbeddingsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cache_dir: str | Path,
        batch_size: int = 256,
        num_workers: int = 0,
        balanced_train_sampler: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self._train_ds: TensorDataset | None = None
        self._val_ds: TensorDataset | None = None
        self._train_labels: np.ndarray | None = None

    def prepare_data(self) -> None:
        for name in ("train.npz", "val.npz"):
            if not (self.cache_dir / name).exists():
                raise FileNotFoundError(
                    f"Embedding cache not found: {self.cache_dir / name}. "
                    "Run `uv run python -m narw_classifier.perch.extract` first."
                )

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None and self._val_ds is not None:
            return
        tr_emb, tr_lab, _ = load_split(self.cache_dir / "train.npz")
        va_emb, va_lab, _ = load_split(self.cache_dir / "val.npz")
        self._train_labels = tr_lab
        self._train_ds = TensorDataset(
            torch.from_numpy(tr_emb).float(),
            torch.from_numpy(tr_lab).long(),
        )
        self._val_ds = TensorDataset(
            torch.from_numpy(va_emb).float(),
            torch.from_numpy(va_lab).long(),
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None and self._train_labels is not None
        sampler = (
            _make_balanced_sampler(self._train_labels, seed=self.hparams.seed)
            if self.hparams.balanced_train_sampler
            else None
        )
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.hparams.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
