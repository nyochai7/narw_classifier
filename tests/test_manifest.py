from pathlib import Path

import pytest

from narw_classifier.dataset.manifest import build_train_manifest


class TestBuildTrainManifest:
    def test_returns_files_and_labels(self, synthetic_train_dir: Path):
        files, labels = build_train_manifest(synthetic_train_dir)
        assert len(files) == 20
        assert len(labels) == 20
        assert sum(labels) == 6  # 6 positives in the fixture
        assert all(name.endswith(".aif") for name in files)

    def test_skips_non_aif(self, synthetic_train_dir: Path):
        (synthetic_train_dir / "random.txt").write_text("nope")
        files, _ = build_train_manifest(synthetic_train_dir)
        assert "random.txt" not in files

    def test_skips_unlabeled_clips(self, synthetic_train_dir: Path):
        # Test-style filename without label suffix:
        (synthetic_train_dir / "20090404_000000_012s0ms_Test0.aif").write_bytes(
            (
                synthetic_train_dir / next(iter(p for p in synthetic_train_dir.iterdir()))
            ).read_bytes()
        )
        files, _ = build_train_manifest(synthetic_train_dir)
        assert not any(name.startswith("Test") or "_Test" in name for name in files)

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            build_train_manifest(tmp_path / "does_not_exist")
