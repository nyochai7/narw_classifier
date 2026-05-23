"""Save/load Perch embedding caches as compressed .npz archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_split(
    out_path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    files: list[str],
) -> None:
    """Save ``(embeddings (N, D), labels (N,), files [N])`` to a single .npz."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, D), got shape {embeddings.shape}")
    n = embeddings.shape[0]
    if len(labels) != n or len(files) != n:
        raise ValueError(
            f"length mismatch: embeddings={n}, labels={len(labels)}, files={len(files)}"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),
        labels=np.asarray(labels).astype(np.int64),
        files=np.array(files, dtype=object),
    )


def load_split(in_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load (embeddings, labels, files) from a .npz."""
    z = np.load(in_path, allow_pickle=True)
    return z["embeddings"], z["labels"], list(z["files"])
