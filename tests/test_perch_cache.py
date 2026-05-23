from pathlib import Path

import numpy as np
import pytest

from narw_classifier.perch.cache import load_split, save_split


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_arrays_and_files(self, tmp_path: Path):
        emb = np.random.randn(5, 1536).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)
        files = [f"{i:08d}_x.aif" for i in range(5)]
        out = tmp_path / "subdir" / "train.npz"
        save_split(out, emb, labels, files)
        assert out.exists()
        emb_r, labels_r, files_r = load_split(out)
        assert np.allclose(emb_r, emb)
        assert np.array_equal(labels_r, labels)
        assert files_r == files

    def test_rejects_1d_embeddings(self, tmp_path: Path):
        with pytest.raises(ValueError, match="2-D"):
            save_split(
                tmp_path / "x.npz",
                embeddings=np.zeros(5, dtype=np.float32),
                labels=np.zeros(5, dtype=np.int64),
                files=["a.aif"] * 5,
            )

    def test_length_mismatch_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="length mismatch"):
            save_split(
                tmp_path / "x.npz",
                embeddings=np.zeros((5, 1536), dtype=np.float32),
                labels=np.zeros(4, dtype=np.int64),
                files=["a.aif"] * 5,
            )
