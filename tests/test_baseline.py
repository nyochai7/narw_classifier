import torch

from narw_classifier.models.baseline import BaselineEfficientNet
from narw_classifier.models.preprocess import MelImagePreprocessor

# Production mel params (CLAUDE.md "Preprocessing pipelines").
_SAMPLE_RATE = 2000
_N_SAMPLES = int(_SAMPLE_RATE * 2.0)


def _make_model(freeze_mode: str) -> BaselineEfficientNet:
    pre = MelImagePreprocessor(
        sample_rate=_SAMPLE_RATE,
        n_fft=256,
        win_length=256,
        hop_length=64,
        n_mels=64,
        f_min=30.0,
        f_max=1000.0,
        image_size=64,  # tiny to keep tests fast
        top_db=80.0,
    )
    # pretrained=False to skip the ImageNet weights download.
    return BaselineEfficientNet(preprocessor=pre, freeze_mode=freeze_mode, pretrained=False)


class TestBaselineEfficientNet:
    def test_forward_produces_1d_logits(self):
        model = _make_model("linear_probe").eval()
        with torch.no_grad():
            logits = model(torch.randn(3, _N_SAMPLES))
        assert logits.shape == (3,)
        assert logits.dtype == torch.float32

    def test_linear_probe_freezes_backbone(self):
        model = _make_model("linear_probe")
        head_params = list(model.net.classifier.parameters())
        all_params = list(model.net.parameters())
        backbone_params = [p for p in all_params if not any(p is hp for hp in head_params)]
        assert all(not p.requires_grad for p in backbone_params)
        assert all(p.requires_grad for p in head_params)

    def test_full_finetune_unfreezes_everything(self):
        model = _make_model("full_finetune")
        assert all(p.requires_grad for p in model.net.parameters())

    def test_training_step_produces_loss(self):
        model = _make_model("linear_probe").train()
        waveform = torch.randn(2, _N_SAMPLES)
        label = torch.tensor([0, 1])
        loss = model.training_step((waveform, label), batch_idx=0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_configure_optimizers_only_includes_trainable(self):
        model = _make_model("linear_probe")
        opt = model.configure_optimizers()
        # configure_optimizers returns either an Optimizer or {"optimizer": ...}
        optimizer = opt if isinstance(opt, torch.optim.Optimizer) else opt["optimizer"]
        opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
        head_param_ids = {id(p) for p in model.net.classifier.parameters()}
        assert opt_params == head_param_ids
