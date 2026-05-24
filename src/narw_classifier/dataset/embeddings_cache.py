"""Save/load Perch embedding caches as compressed .npz archives.

Two on-disk layouts are supported:

- **all.npz** (preferred) — every clip's embedding in one file. Lets the
  DataModule re-split into train / val / test at load time without re-running
  the slow ONNX extraction.
- **train.npz + val.npz** (legacy) — the layout produced by older extraction
  runs (when the splitter ran before extraction). The loader concatenates the
  two transparently so old caches still work after the split-strategy refactor.
"""

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
    """Load ``(embeddings, labels, files)`` from a single .npz."""
    z = np.load(in_path, allow_pickle=True)
    return z["embeddings"], z["labels"], list(z["files"])


def load_full_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return all embeddings from ``cache_dir`` regardless of layout.

    Tries ``all.npz`` first; falls back to concatenating ``train.npz`` + ``val.npz``
    (the legacy two-file layout). Raises ``FileNotFoundError`` if neither exists.
    """
    cache_dir = Path(cache_dir)
    all_path = cache_dir / "all.npz"
    if all_path.exists():
        return load_split(all_path)
    train_path = cache_dir / "train.npz"
    val_path = cache_dir / "val.npz"
    if not (train_path.exists() and val_path.exists()):
        raise FileNotFoundError(
            f"No embedding cache at {cache_dir}: expected either all.npz or both "
            f"train.npz and val.npz."
        )
    tr_emb, tr_lab, tr_files = load_split(train_path)
    va_emb, va_lab, va_files = load_split(val_path)
    emb = np.concatenate([tr_emb, va_emb], axis=0)
    lab = np.concatenate([tr_lab, va_lab], axis=0)
    files = list(tr_files) + list(va_files)
    return emb, lab, files
