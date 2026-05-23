"""Tests for the unified training entrypoint.

Covers both pipeline branches of ``default_run_name`` (EfficientNet baseline +
Perch linear probe), since they produce different name formats.
"""

from datetime import datetime

import pytest
from omegaconf import OmegaConf

from narw_classifier.training.train import default_run_name


def _efficientnet_cfg(freeze_mode="linear_probe", lr=1e-3, batch_size=32, max_epochs=1):
    return OmegaConf.create(
        {
            "pipeline": "efficientnet",
            "model": {"freeze_mode": freeze_mode, "lr": lr},
            "data": {"batch_size": batch_size},
            "trainer": {"max_epochs": max_epochs},
        }
    )


def _perch_cfg(pitch_shift_semitones=36, lr=1e-3, batch_size=256, max_epochs=10):
    return OmegaConf.create(
        {
            "pipeline": "perch",
            "preprocess": {"pitch_shift_semitones": pitch_shift_semitones},
            "model": {"lr": lr},
            "data": {"batch_size": batch_size},
            "trainer": {"max_epochs": max_epochs},
        }
    )


class TestEfficientNetRunName:
    def test_includes_all_key_fields_and_timestamp(self):
        ts = datetime(2026, 5, 23, 10, 42, 17)
        name = default_run_name(_efficientnet_cfg(), now=ts)
        assert name == "linear_probe_lr1e-3_bs32_e1_104217"

    def test_reflects_freeze_mode_change(self):
        ts = datetime(2026, 5, 23, 10, 42, 17)
        name = default_run_name(_efficientnet_cfg(freeze_mode="full_finetune"), now=ts)
        assert name.startswith("full_finetune_")

    def test_reflects_lr_change(self):
        ts = datetime(2026, 5, 23, 10, 42, 17)
        name = default_run_name(_efficientnet_cfg(lr=3e-4, max_epochs=10), now=ts)
        assert "lr3e-4" in name
        assert "_e10_" in name


class TestPerchRunName:
    def test_includes_shift_tag_for_pitch_shifted(self):
        ts = datetime(2026, 5, 23, 12, 34, 56)
        name = default_run_name(_perch_cfg(pitch_shift_semitones=36), now=ts)
        assert name == "perch_v2_linear_shift36_lr1e-3_bs256_e10_123456"

    def test_uses_noshift_tag_when_zero(self):
        ts = datetime(2026, 5, 23, 12, 34, 56)
        name = default_run_name(_perch_cfg(pitch_shift_semitones=0), now=ts)
        assert "noshift" in name
        assert "shift0" not in name

    @pytest.mark.parametrize("shift", [12, 24, 48])
    def test_other_shift_values(self, shift):
        name = default_run_name(_perch_cfg(pitch_shift_semitones=shift))
        assert f"shift{shift}" in name


class TestUnknownPipeline:
    def test_raises_for_unknown_pipeline(self):
        cfg = OmegaConf.create(
            {
                "pipeline": "not_a_real_pipeline",
                "model": {"lr": 1e-3},
                "data": {"batch_size": 32},
                "trainer": {"max_epochs": 1},
            }
        )
        with pytest.raises(KeyError):
            default_run_name(cfg)
