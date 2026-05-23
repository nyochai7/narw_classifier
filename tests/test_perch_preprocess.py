from pathlib import Path

import numpy as np
import pytest

from narw_classifier.perch.preprocess import (
    PERCH_N_SAMPLES,
    PERCH_SAMPLE_RATE,
    pad_or_center,
    preprocess_for_perch,
)


class TestPadOrCenter:
    def test_pads_symmetrically(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        out = pad_or_center(x, target_length=11)
        assert out.shape == (11,)
        # symmetric pad: 3 zeros on each side around the 5-sample signal
        assert np.array_equal(out, np.array([0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0], dtype=np.float32))

    def test_truncates_around_center(self):
        x = np.arange(11, dtype=np.float32)
        out = pad_or_center(x, target_length=5)
        # picks the middle 5 elements (indices 3..7)
        assert np.array_equal(out, np.array([3, 4, 5, 6, 7], dtype=np.float32))

    def test_passthrough_when_exact_length(self):
        x = np.arange(5, dtype=np.float32)
        out = pad_or_center(x, target_length=5)
        assert np.array_equal(out, x)

    def test_rejects_multidim(self):
        with pytest.raises(ValueError):
            pad_or_center(np.zeros((2, 3), dtype=np.float32), target_length=10)


class TestPreprocessForPerch:
    def test_output_length_and_dtype(self, synthetic_clip: Path):
        out = preprocess_for_perch(synthetic_clip)
        assert out.shape == (PERCH_N_SAMPLES,)
        assert out.dtype == np.float32

    def test_constants_match_perch_contract(self):
        # Perch v2 ONNX expects (B, 160_000) float32 at 32 kHz.
        assert PERCH_SAMPLE_RATE == 32_000
        assert PERCH_N_SAMPLES == 160_000

    def test_pitch_shift_zero_skips_phase_vocoder(self, synthetic_clip: Path):
        # With shift=0, the path is just resample + pad. Output should be non-zero
        # (the synthetic clip has a tone) and finite.
        out = preprocess_for_perch(synthetic_clip, pitch_shift_semitones=0)
        assert np.isfinite(out).all()
        assert np.abs(out).sum() > 0
