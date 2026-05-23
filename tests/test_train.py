from datetime import datetime

from omegaconf import OmegaConf

from narw_classifier.training.train import default_run_name


class TestDefaultRunName:
    @staticmethod
    def _cfg(freeze_mode="linear_probe", lr=1e-3, batch_size=32, max_epochs=1):
        return OmegaConf.create(
            {
                "model": {"freeze_mode": freeze_mode, "lr": lr},
                "data": {"batch_size": batch_size},
                "trainer": {"max_epochs": max_epochs},
            }
        )

    def test_includes_all_key_fields_and_timestamp(self):
        ts = datetime(2026, 5, 23, 10, 42, 17)
        name = default_run_name(self._cfg(), now=ts)
        assert name == "linear_probe_lr1e-3_bs32_e1_104217"

    def test_reflects_freeze_mode_change(self):
        ts = datetime(2026, 5, 23, 10, 42, 17)
        name = default_run_name(self._cfg(freeze_mode="full_finetune"), now=ts)
        assert name.startswith("full_finetune_")

    def test_reflects_lr_change(self):
        ts = datetime(2026, 5, 23, 10, 42, 17)
        name = default_run_name(self._cfg(lr=3e-4, max_epochs=10), now=ts)
        assert "lr3e-4" in name
        assert "_e10_" in name
