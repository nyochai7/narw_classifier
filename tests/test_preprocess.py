import torch

from narw_classifier.models.preprocess import MelImagePreprocessor

# Production mel params (CLAUDE.md "Preprocessing pipelines"). image_size is small
# in tests to keep things fast; production size is 300.
_SAMPLE_RATE = 2000
_DURATION_S = 2.0
_N_SAMPLES = int(_SAMPLE_RATE * _DURATION_S)


def _make_preprocessor(image_size: int = 64) -> MelImagePreprocessor:
    return MelImagePreprocessor(
        sample_rate=_SAMPLE_RATE,
        n_fft=256,
        win_length=256,
        hop_length=64,
        n_mels=64,
        f_min=30.0,
        f_max=1000.0,
        image_size=image_size,
        top_db=80.0,
    )


class TestMelImagePreprocessor:
    def test_output_shape_matches_image_size_and_is_3_channel(self):
        pre = _make_preprocessor(image_size=64)
        batch = torch.randn(4, _N_SAMPLES)
        out = pre(batch)
        assert out.shape == (4, 3, 64, 64)

    def test_output_is_finite(self):
        pre = _make_preprocessor()
        out = pre(torch.randn(2, _N_SAMPLES))
        assert torch.isfinite(out).all()

    def test_imagenet_normalization_is_applied(self):
        # With silence input, log-mel is constant and per-clip normalize maps to 0 →
        # ImageNet mean/std shifts it away from 0 per channel. Cheap proxy: mean(|out|) > 0.
        pre = _make_preprocessor(image_size=32)
        out = pre(torch.zeros(1, _N_SAMPLES))
        assert out.abs().mean() > 0.0

    def test_batch_independence(self):
        pre = _make_preprocessor(image_size=32)
        a = torch.randn(1, _N_SAMPLES)
        b = torch.randn(1, _N_SAMPLES)
        out_pair = pre(torch.cat([a, b], dim=0))
        out_a = pre(a)
        out_b = pre(b)
        # Per-clip normalization: outputs must match when fed alone or in a batch.
        assert torch.allclose(out_pair[0:1], out_a, atol=1e-5)
        assert torch.allclose(out_pair[1:2], out_b, atol=1e-5)

    def test_rejects_wrong_input_rank(self):
        pre = _make_preprocessor()
        try:
            pre(torch.randn(_N_SAMPLES))  # 1-D, missing batch dim
        except ValueError:
            return
        raise AssertionError("expected ValueError for 1-D input")
