"""ONNX wrapper for Perch v2.

Downloads the model from HuggingFace once and runs batched inference to produce
1536-d embeddings from raw 32 kHz, 5-second, mono float32 waveforms.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

PERCH_HF_REPO = "justinchuby/Perch-onnx"
DEFAULT_ONNX_FILE = "perch_v2.onnx"
EMBEDDING_DIM = 1536


def download_perch_onnx(
    output_dir: Path,
    filename: str = DEFAULT_ONNX_FILE,
    repo: str = PERCH_HF_REPO,
) -> Path:
    """Download the ONNX model from HuggingFace if not already present locally."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    if target.exists():
        return target
    log.info("Downloading %s from %s...", filename, repo)
    p = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(output_dir))
    return Path(p)


_PREFERRED_PROVIDERS = (
    "CUDAExecutionProvider",  # CUDA GPU (Lightning AI etc.) — needs onnxruntime-gpu
    "CoreMLExecutionProvider",  # Apple Silicon (Mac)
    "CPUExecutionProvider",  # fallback, always available
)


def _resolve_providers(available: list[str]) -> list[str]:
    """Order ``available`` providers by ``_PREFERRED_PROVIDERS``; anything else trails."""
    preferred = [p for p in _PREFERRED_PROVIDERS if p in available]
    trailing = [p for p in available if p not in _PREFERRED_PROVIDERS]
    return preferred + trailing


class PerchEmbedder:
    """Loads Perch v2 ONNX and returns 1536-d embeddings for batches of 5-s 32-kHz clips.

    The session is built with ``CUDAExecutionProvider`` preferred when present (Lightning
    AI GPU instances with ``onnxruntime-gpu`` installed), falling back to CoreML on
    Apple Silicon and CPU otherwise. The active provider list is logged at construction
    time so you can confirm GPU is actually being used.
    """

    def __init__(self, onnx_path: Path, providers: list[str] | None = None) -> None:
        if providers is None:
            providers = _resolve_providers(ort.get_available_providers())
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        log.info("PerchEmbedder providers (in priority order): %s", self.session.get_providers())
        self._verify_io_contract()

    def _verify_io_contract(self) -> None:
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != "inputs":
            raise RuntimeError(
                f"Unexpected Perch ONNX inputs: {[(i.name, i.shape) for i in inputs]}"
            )
        out_names = [o.name for o in self.session.get_outputs()]
        if "embedding" not in out_names:
            raise RuntimeError(f"Perch ONNX missing 'embedding' output. Got: {out_names}")

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """``batch``: ``(B, 160000)`` float32. Returns ``(B, 1536)`` float32."""
        if batch.ndim != 2:
            raise ValueError(f"expected (B, 160000), got shape {batch.shape}")
        if batch.shape[1] != 160000:
            raise ValueError(f"expected 160000 samples per row, got {batch.shape[1]}")
        outputs = self.session.run(["embedding"], {"inputs": batch.astype(np.float32)})
        return outputs[0]
