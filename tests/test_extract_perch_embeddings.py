"""Tests for the embedding extraction loop. Uses a mock embedder so the test
doesn't depend on the 380 MB Perch ONNX file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from narw_classifier.data.perch_preprocess import PERCH_N_SAMPLES
from narw_classifier.training.extract_perch_embeddings import (
    _extract_embeddings,
    _PerchPreprocessDataset,
)


class _StubEmbedder:
    """Returns deterministic zero embeddings of the right shape. Used to exercise
    the extract loop end-to-end without the 380 MB Perch ONNX model."""

    def embed(self, batch: np.ndarray) -> np.ndarray:
        assert batch.ndim == 2 and batch.shape[1] == PERCH_N_SAMPLES
        return np.zeros((batch.shape[0], 1536), dtype=np.float32)


class TestPerchPreprocessDataset:
    def test_returns_correct_length_and_dtype(self, synthetic_train_dir: Path):
        files = sorted(p.name for p in synthetic_train_dir.iterdir())
        ds = _PerchPreprocessDataset(
            files=files,
            data_dir=synthetic_train_dir,
            pitch_shift_semitones=0.0,  # skip pitch_shift to keep the test fast
            sample_rate=32000,
            n_samples=PERCH_N_SAMPLES,
        )
        assert len(ds) == len(files)
        y = ds[0]
        assert y.shape == (PERCH_N_SAMPLES,)
        assert y.dtype == np.float32


class TestExtractEmbeddings:
    def _files(self, train_dir: Path) -> list[str]:
        return sorted(p.name for p in train_dir.iterdir())

    def test_serial_num_workers_zero(self, synthetic_train_dir: Path):
        files = self._files(synthetic_train_dir)
        emb = _extract_embeddings(
            _StubEmbedder(),
            data_dir=synthetic_train_dir,
            files=files,
            pitch_shift_semitones=0.0,
            sample_rate=32000,
            n_samples=PERCH_N_SAMPLES,
            batch_size=4,
            num_workers=0,
        )
        assert emb.shape == (len(files), 1536)
        assert emb.dtype == np.float32

    def test_parallel_num_workers_two_matches_serial(self, synthetic_train_dir: Path):
        files = self._files(synthetic_train_dir)
        kwargs = dict(
            data_dir=synthetic_train_dir,
            files=files,
            pitch_shift_semitones=0.0,
            sample_rate=32000,
            n_samples=PERCH_N_SAMPLES,
            batch_size=4,
        )
        # Same stub embedder behavior → same outputs regardless of worker count.
        # Just verifies the DataLoader-with-workers code path runs at all.
        emb_serial = _extract_embeddings(_StubEmbedder(), num_workers=0, **kwargs)
        emb_parallel = _extract_embeddings(_StubEmbedder(), num_workers=2, **kwargs)
        assert emb_serial.shape == emb_parallel.shape == (len(files), 1536)
        assert np.array_equal(emb_serial, emb_parallel)
