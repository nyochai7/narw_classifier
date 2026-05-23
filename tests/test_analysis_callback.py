"""Integration test for EvalArtifactsCallback.

Uses a 4-parameter LightningModule + a tiny in-memory DataModule so the test
runs in <1 s. Verifies the callback end-to-end: at end of training it writes
predictions.npz + the three PNGs + summary.json into the Hydra run dir, and
honors a DataModule's optional ``val_files`` attribute.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from narw_classifier.analysis.callback import EvalArtifactsCallback
from narw_classifier.analysis.predictions import load_predictions


class _TinyClassifier(pl.LightningModule):
    """Linear(2 → 1) + BCE-with-logits. Forward returns ``(B,)``."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(2, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y.float())
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y.float())
        self.log("val/loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


class _TinyDataModule(pl.LightningDataModule):
    """64-sample 2-D toy data, ~50/50 label split, deterministic."""

    def __init__(self, val_files: list[str] | None = None) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(0)
        n = 64
        x = torch.randn(n, 2, generator=rng)
        y = torch.randint(0, 2, (n,), generator=rng).long()
        self._ds = TensorDataset(x, y)
        self.val_files: list[str] = val_files if val_files is not None else []

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._ds, batch_size=16)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._ds, batch_size=16)


@pytest.fixture
def hydra_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Mock HydraConfig.get() so the callback writes to ``tmp_path``."""
    fake_cfg = types.SimpleNamespace(runtime=types.SimpleNamespace(output_dir=str(tmp_path)))
    monkeypatch.setattr(
        "narw_classifier.analysis.callback.HydraConfig.get",
        lambda: fake_cfg,
    )
    return tmp_path


def _make_trainer() -> pl.Trainer:
    return pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[EvalArtifactsCallback()],
    )


class TestEvalArtifactsCallback:
    def test_writes_all_expected_files_at_end_of_training(self, hydra_run_dir: Path):
        trainer = _make_trainer()
        trainer.fit(_TinyClassifier(), datamodule=_TinyDataModule())
        analysis = hydra_run_dir / "analysis"
        for name in (
            "predictions.npz",
            "pr_curve.png",
            "roc_curve.png",
            "confusion_matrix_t0.5.png",
            "summary.json",
        ):
            assert (analysis / name).exists(), f"missing {name}"

    def test_predictions_npz_matches_val_set_size(self, hydra_run_dir: Path):
        trainer = _make_trainer()
        trainer.fit(_TinyClassifier(), datamodule=_TinyDataModule())
        probs, labels, files = load_predictions(hydra_run_dir / "analysis" / "predictions.npz")
        assert probs.shape == (64,)
        assert labels.shape == (64,)
        assert files is None  # _TinyDataModule has val_files=[] (no filenames)

    def test_persists_val_files_when_datamodule_provides_them(self, hydra_run_dir: Path):
        val_files = [f"clip_{i:03d}.aif" for i in range(64)]
        trainer = _make_trainer()
        trainer.fit(_TinyClassifier(), datamodule=_TinyDataModule(val_files=val_files))
        _, _, files = load_predictions(hydra_run_dir / "analysis" / "predictions.npz")
        assert files == val_files
