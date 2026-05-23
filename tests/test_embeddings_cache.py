from pathlib import Path

import numpy as np
import pytest

from narw_classifier.data.embeddings_cache import load_full_cache, load_split, save_split


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


class TestLoadFullCache:
    def test_legacy_two_file_layout_concatenates_in_order(self, tmp_path: Path):
        tr_emb = np.full((3, 4), 1.0, dtype=np.float32)
        va_emb = np.full((2, 4), 2.0, dtype=np.float32)
        save_split(tmp_path / "train.npz", tr_emb, np.array([1, 0, 1]), ["a.aif", "b.aif", "c.aif"])
        save_split(tmp_path / "val.npz", va_emb, np.array([0, 1]), ["d.aif", "e.aif"])
        emb, lab, files = load_full_cache(tmp_path)
        assert emb.shape == (5, 4)
        assert np.allclose(emb[:3], 1.0)
        assert np.allclose(emb[3:], 2.0)
        assert files == ["a.aif", "b.aif", "c.aif", "d.aif", "e.aif"]
        assert lab.tolist() == [1, 0, 1, 0, 1]

    def test_prefers_all_npz_when_present(self, tmp_path: Path):
        # all.npz with different content than the legacy files — loader should pick all.npz.
        all_emb = np.full((4, 2), 9.0, dtype=np.float32)
        save_split(tmp_path / "all.npz", all_emb, np.array([1, 1, 0, 0]), ["x.aif"] * 4)
        # add legacy files (should be ignored)
        save_split(
            tmp_path / "train.npz",
            np.zeros((2, 2), dtype=np.float32),
            np.array([0, 0]),
            ["legacy.aif"] * 2,
        )
        save_split(
            tmp_path / "val.npz",
            np.zeros((2, 2), dtype=np.float32),
            np.array([1, 1]),
            ["legacy.aif"] * 2,
        )
        emb, _, files = load_full_cache(tmp_path)
        assert emb.shape == (4, 2)
        assert np.allclose(emb, 9.0)
        assert files == ["x.aif"] * 4

    def test_missing_cache_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_full_cache(tmp_path)
