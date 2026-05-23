from datetime import datetime

import pytest
from omegaconf import OmegaConf

from narw_classifier.perch.train import default_run_name


def _cfg(pitch_shift_semitones=36, lr=1e-3, batch_size=256, max_epochs=10):
    return OmegaConf.create(
        {
            "preprocess": {"pitch_shift_semitones": pitch_shift_semitones},
            "model": {"lr": lr},
            "data": {"batch_size": batch_size},
            "trainer": {"max_epochs": max_epochs},
        }
    )


class TestPerchDefaultRunName:
    def test_includes_shift_tag_for_pitch_shifted(self):
        ts = datetime(2026, 5, 23, 12, 34, 56)
        name = default_run_name(_cfg(pitch_shift_semitones=36), now=ts)
        assert name == "perch_v2_linear_shift36_lr1e-3_bs256_e10_123456"

    def test_uses_noshift_tag_when_zero(self):
        ts = datetime(2026, 5, 23, 12, 34, 56)
        name = default_run_name(_cfg(pitch_shift_semitones=0), now=ts)
        assert "noshift" in name
        assert "shift0" not in name

    @pytest.mark.parametrize("shift", [12, 24, 48])
    def test_other_shift_values(self, shift):
        name = default_run_name(_cfg(pitch_shift_semitones=shift))
        assert f"shift{shift}" in name
