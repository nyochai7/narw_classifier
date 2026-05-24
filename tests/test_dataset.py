from pathlib import Path

import pytest
import torch

from narw_classifier.dataset.dataset import NARWAudioDataset


class TestNARWAudioDataset:
    def test_returns_4000_sample_waveform_and_label(self, synthetic_clip: Path):
        ds = NARWAudioDataset(
            files=[synthetic_clip.name],
            labels=[1],
            data_dir=synthetic_clip.parent,
        )
        waveform, label = ds[0]
        assert waveform.shape == (4000,)
        assert waveform.dtype == torch.float32
        assert label == 1
        assert waveform.abs().sum() > 0

    def test_length_matches_files(self, synthetic_clip: Path):
        ds = NARWAudioDataset(
            files=[synthetic_clip.name, synthetic_clip.name],
            labels=[1, 0],
            data_dir=synthetic_clip.parent,
        )
        assert len(ds) == 2

    def test_mismatched_files_labels_raises(self, synthetic_clip: Path):
        with pytest.raises(ValueError):
            NARWAudioDataset(
                files=[synthetic_clip.name],
                labels=[1, 0],
                data_dir=synthetic_clip.parent,
            )
