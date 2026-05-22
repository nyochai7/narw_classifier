"""LightningDataModule for the NARW upcall dataset.

Train loader uses a ``WeightedRandomSampler`` so each batch is class-balanced
(addresses the ~8:1 imbalance). Val loader uses the natural class distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import NARWAudioDataset
from .manifest import build_train_manifest
from .splits import stratified_split


def _make_balanced_sampler(labels: list[int], seed: int) -> WeightedRandomSampler:
    """Return a sampler whose draws are uniform over classes (not over examples)."""
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


class NARWDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        train_subdir: str = "train2",
        target_sample_rate: int = 2000,
        target_duration_s: float = 2.0,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        balanced_train_sampler: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = Path(data_root).expanduser().resolve()
        self.train_dir = self.data_root / train_subdir
        self._train_ds: NARWAudioDataset | None = None
        self._val_ds: NARWAudioDataset | None = None
        self._train_labels: list[int] = []

    def prepare_data(self) -> None:
        if not self.train_dir.is_dir():
            raise FileNotFoundError(
                f"Train directory not found: {self.train_dir} "
                f"(set data.root / data.train_subdir in your Hydra config)"
            )

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None and self._val_ds is not None:
            return
        files, labels = build_train_manifest(self.train_dir)
        if len(files) == 0:
            raise RuntimeError(f"No labeled .aif files found in {self.train_dir}")
        tr_files, tr_labels, va_files, va_labels = stratified_split(
            files=files,
            labels=labels,
            val_fraction=self.hparams.val_fraction,
            seed=self.hparams.split_seed,
        )
        self._train_labels = tr_labels
        self._train_ds = NARWAudioDataset(
            files=tr_files,
            labels=tr_labels,
            data_dir=self.train_dir,
            target_sample_rate=self.hparams.target_sample_rate,
            target_duration_s=self.hparams.target_duration_s,
        )
        self._val_ds = NARWAudioDataset(
            files=va_files,
            labels=va_labels,
            data_dir=self.train_dir,
            target_sample_rate=self.hparams.target_sample_rate,
            target_duration_s=self.hparams.target_duration_s,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup() first"
        sampler = (
            _make_balanced_sampler(self._train_labels, seed=self.hparams.split_seed)
            if self.hparams.balanced_train_sampler
            else None
        )
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "Call setup() first"
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            persistent_workers=self.hparams.num_workers > 0,
        )
