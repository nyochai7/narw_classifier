"""Embedder tests. Real ONNX inference is skipped if the model file isn't present
(don't trigger a 380 MB download during pytest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from narw_classifier.perch.embedder import (
    DEFAULT_ONNX_FILE,
    EMBEDDING_DIM,
    PerchEmbedder,
    _resolve_providers,
)

_MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
_ONNX_PATH = _MODELS_DIR / DEFAULT_ONNX_FILE
_HAVE_MODEL = _ONNX_PATH.exists()


class TestResolveProviders:
    def test_prefers_cuda_over_cpu_when_both_available(self):
        out = _resolve_providers(["CPUExecutionProvider", "CUDAExecutionProvider"])
        assert out[0] == "CUDAExecutionProvider"

    def test_prefers_coreml_over_cpu_on_mac(self):
        out = _resolve_providers(["CPUExecutionProvider", "CoreMLExecutionProvider"])
        assert out[0] == "CoreMLExecutionProvider"

    def test_cpu_only_returns_cpu(self):
        assert _resolve_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]

    def test_unknown_providers_trail_known_ones(self):
        out = _resolve_providers(["UnknownEP", "CPUExecutionProvider", "CUDAExecutionProvider"])
        assert out[0] == "CUDAExecutionProvider"
        assert out[1] == "CPUExecutionProvider"
        assert "UnknownEP" in out


@pytest.mark.skipif(not _HAVE_MODEL, reason=f"{_ONNX_PATH} not present; skip ONNX integration")
class TestPerchEmbedderIntegration:
    def test_embed_shape_and_dtype(self):
        embedder = PerchEmbedder(_ONNX_PATH)
        x = np.zeros((2, 160000), dtype=np.float32)
        out = embedder.embed(x)
        assert out.shape == (2, EMBEDDING_DIM)
        assert out.dtype == np.float32

    def test_rejects_wrong_input_rank(self):
        embedder = PerchEmbedder(_ONNX_PATH)
        with pytest.raises(ValueError, match="\\(B, 160000\\)"):
            embedder.embed(np.zeros(160000, dtype=np.float32))

    def test_rejects_wrong_n_samples(self):
        embedder = PerchEmbedder(_ONNX_PATH)
        with pytest.raises(ValueError, match="160000"):
            embedder.embed(np.zeros((1, 4000), dtype=np.float32))
