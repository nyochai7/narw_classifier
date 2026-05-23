"""LightningDataModule that serves cached Perch embeddings + labels with a 70/15/15
train/val/test split — reusing existing extraction caches.

Loads *all* cached embeddings from ``cache_dir`` (transparently supporting both the
new ``all.npz`` layout and the legacy ``train.npz`` + ``val.npz`` layout), then
re-splits them according to ``split_strategy`` + ``(val_fraction, test_fraction)``.

This means swapping split strategies or fractions does NOT require re-extracting
the (slow) ONNX embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils.samplers import make_balanced_sampler
from .embeddings_cache import load_full_cache
from .splits import day_stratified_split_3way, stratified_split_3way

_SPLITTERS_3WAY = {
    "day_stratified": day_stratified_split_3way,
    "random": stratified_split_3way,
}


class PerchEmbeddingsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cache_dir: str | Path,
        batch_size: int = 256,
        num_workers: int = 0,
        balanced_train_sampler: bool = True,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        split_seed: int = 42,
        split_strategy: str = "day_stratified",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self._train_ds: TensorDataset | None = None
        self._val_ds: TensorDataset | None = None
        self._test_ds: TensorDataset | None = None
        self._train_labels: np.ndarray | None = None
        self.val_files: list[str] = []
        self.test_files: list[str] = []

    def prepare_data(self) -> None:
        # Triggers FileNotFoundError early if the cache is missing.
        load_full_cache(self.cache_dir)

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None and self._val_ds is not None and self._test_ds is not None:
            return

        all_emb, all_lab, all_files = load_full_cache(self.cache_dir)
        idx_by_file = {f: i for i, f in enumerate(all_files)}

        splitter = _SPLITTERS_3WAY.get(self.hparams.split_strategy)
        if splitter is None:
            raise ValueError(
                f"Unknown split_strategy: {self.hparams.split_strategy!r} "
                f"(expected one of {sorted(_SPLITTERS_3WAY)!r})"
            )
        tr_f, tr_y, va_f, va_y, te_f, te_y = splitter(
            files=all_files,
            labels=list(all_lab),
            val_fraction=self.hparams.val_fraction,
            test_fraction=self.hparams.test_fraction,
            seed=self.hparams.split_seed,
        )

        def take(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
            idx = np.array([idx_by_file[f] for f in files], dtype=np.int64)
            return all_emb[idx], all_lab[idx]

        tr_emb, tr_lab = take(tr_f)
        va_emb, va_lab = take(va_f)
        te_emb, te_lab = take(te_f)

        self._train_labels = tr_lab
        self.val_files = list(va_f)
        self.test_files = list(te_f)

        def make_ds(emb: np.ndarray, lab: np.ndarray) -> TensorDataset:
            return TensorDataset(
                torch.from_numpy(emb).float(),
                torch.from_numpy(lab).long(),
            )

        self._train_ds = make_ds(tr_emb, tr_lab)
        self._val_ds = make_ds(va_emb, va_lab)
        self._test_ds = make_ds(te_emb, te_lab)

    def _make_loader(self, ds, *, shuffle: bool, sampler=None) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None and self._train_labels is not None
        sampler = (
            make_balanced_sampler(self._train_labels, seed=self.hparams.split_seed)
            if self.hparams.balanced_train_sampler
            else None
        )
        return self._make_loader(self._train_ds, shuffle=sampler is None, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None
        return self._make_loader(self._test_ds, shuffle=False)
