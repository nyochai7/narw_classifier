from pathlib import Path

import pytest
import torch

from narw_classifier.data.dataset import NARWAudioDataset, pad_or_truncate


class TestPadOrTruncate:
    def test_pads_right_with_zeros(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        out = pad_or_truncate(x, target_length=5)
        assert out.shape == (5,)
        assert torch.equal(out[:3], x)
        assert torch.equal(out[3:], torch.zeros(2))

    def test_truncates(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = pad_or_truncate(x, target_length=2)
        assert torch.equal(out, torch.tensor([1.0, 2.0]))

    def test_passthrough_when_exact_length(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert pad_or_truncate(x, target_length=3) is x or torch.equal(
            pad_or_truncate(x, target_length=3), x
        )

    def test_rejects_multidim(self):
        with pytest.raises(ValueError):
            pad_or_truncate(torch.zeros(2, 3), target_length=5)


class TestNARWAudioDataset:
    def test_resamples_to_target_rate_and_pads(self, synthetic_clip: Path):
        # Synthetic clip is 2 kHz / 2 s → 4000 samples natively.
        ds = NARWAudioDataset(
            files=[synthetic_clip.name],
            labels=[1],
            data_dir=synthetic_clip.parent,
            target_sample_rate=32000,
            target_duration_s=5.0,
        )
        waveform, label = ds[0]
        assert waveform.shape == (32000 * 5,)
        assert waveform.dtype == torch.float32
        assert label == 1
        # After resampling 2 s of audio → 64000 samples; the remaining 96000 should be zero pad.
        assert torch.all(waveform[64000:] == 0.0)
        # And the signal-bearing region shouldn't be all zeros.
        assert waveform[:64000].abs().sum() > 0

    def test_length_matches_files(self, synthetic_clip: Path):
        ds = NARWAudioDataset(
            files=[synthetic_clip.name, synthetic_clip.name],
            labels=[1, 0],
            data_dir=synthetic_clip.parent,
            target_sample_rate=32000,
            target_duration_s=5.0,
        )
        assert len(ds) == 2

    def test_mismatched_files_labels_raises(self, synthetic_clip: Path):
        with pytest.raises(ValueError):
            NARWAudioDataset(
                files=[synthetic_clip.name],
                labels=[1, 0],
                data_dir=synthetic_clip.parent,
                target_sample_rate=32000,
                target_duration_s=5.0,
            )
