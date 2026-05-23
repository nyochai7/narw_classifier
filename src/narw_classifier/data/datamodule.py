"""LightningDataModule for the NARW upcall dataset.

Three-way split (train / val / test). Train loader uses a ``WeightedRandomSampler``
so each batch is class-balanced (addresses the ~8:1 imbalance). Val + test loaders
use the natural class distribution.

The held-out test set is the honest reportable number — never touched during
training / hyperparameter selection.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..utils.samplers import make_balanced_sampler
from .dataset import NARWAudioDataset
from .manifest import build_train_manifest
from .splits import day_stratified_split_3way, stratified_split_3way

_SPLITTERS_3WAY = {
    "day_stratified": day_stratified_split_3way,
    "random": stratified_split_3way,
}


class NARWDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        train_subdir: str = "train2",
        target_sample_rate: int = 2000,
        target_duration_s: float = 2.0,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        split_seed: int = 42,
        split_strategy: str = "day_stratified",
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
        self._test_ds: NARWAudioDataset | None = None
        self._train_labels: list[int] = []
        self.val_files: list[str] = []
        self.test_files: list[str] = []

    def prepare_data(self) -> None:
        if not self.train_dir.is_dir():
            raise FileNotFoundError(
                f"Train directory not found: {self.train_dir} "
                f"(set data.root / data.train_subdir in your Hydra config)"
            )

    def _make_dataset(self, files, labels) -> NARWAudioDataset:
        return NARWAudioDataset(
            files=files,
            labels=labels,
            data_dir=self.train_dir,
            target_sample_rate=self.hparams.target_sample_rate,
            target_duration_s=self.hparams.target_duration_s,
        )

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None and self._val_ds is not None and self._test_ds is not None:
            return
        files, labels = build_train_manifest(self.train_dir)
        if len(files) == 0:
            raise RuntimeError(f"No labeled .aif files found in {self.train_dir}")

        splitter = _SPLITTERS_3WAY.get(self.hparams.split_strategy)
        if splitter is None:
            raise ValueError(
                f"Unknown split_strategy: {self.hparams.split_strategy!r} "
                f"(expected one of {sorted(_SPLITTERS_3WAY)!r})"
            )
        tr_f, tr_y, va_f, va_y, te_f, te_y = splitter(
            files=files,
            labels=labels,
            val_fraction=self.hparams.val_fraction,
            test_fraction=self.hparams.test_fraction,
            seed=self.hparams.split_seed,
        )
        self._train_labels = tr_y
        self.val_files = list(va_f)
        self.test_files = list(te_f)
        self._train_ds = self._make_dataset(tr_f, tr_y)
        self._val_ds = self._make_dataset(va_f, va_y)
        self._test_ds = self._make_dataset(te_f, te_y)

    def _make_loader(
        self, ds, *, shuffle: bool, sampler=None, drop_last: bool = False
    ) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup() first"
        sampler = (
            make_balanced_sampler(self._train_labels, seed=self.hparams.split_seed)
            if self.hparams.balanced_train_sampler
            else None
        )
        return self._make_loader(
            self._train_ds, shuffle=sampler is None, sampler=sampler, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "Call setup() first"
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None, "Call setup() first"
        return self._make_loader(self._test_ds, shuffle=False)
