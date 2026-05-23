"""Extract Perch v2 embeddings for the NARW train set and cache them to disk.

Run::

    uv run python -m narw_classifier.perch.extract

One-time per (preprocess config, split config). Outputs:
    {embeddings.cache_dir}/{split_strategy}/train.npz
    {embeddings.cache_dir}/{split_strategy}/val.npz

Preprocessing (resample + librosa pitch_shift + pad) is the bottleneck; it runs
across `embeddings.num_workers` CPU processes via a PyTorch DataLoader, while
ONNX inference stays serial in the main process.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.manifest import build_train_manifest
from ..data.splits import day_stratified_split, stratified_split
from .cache import save_split
from .embedder import PerchEmbedder, download_perch_onnx
from .preprocess import preprocess_for_perch

log = logging.getLogger(__name__)


def _select_splitter(name: str):
    if name == "day_stratified":
        return day_stratified_split
    if name == "random":
        return stratified_split
    raise ValueError(f"Unknown split_strategy: {name!r}")


class _PerchPreprocessDataset(Dataset):
    """Returns the preprocessed 5-s 32-kHz waveform for file ``i``.

    Doing this in a Dataset (not inline) lets the DataLoader fan preprocessing
    out across worker processes — librosa's pitch_shift is single-threaded and
    is otherwise the wall-clock bottleneck.
    """

    def __init__(
        self,
        files: list[str],
        data_dir: Path,
        pitch_shift_semitones: float,
        sample_rate: int,
        n_samples: int,
    ) -> None:
        self.files = files
        self.data_dir = data_dir
        self.pitch_shift_semitones = pitch_shift_semitones
        self.sample_rate = sample_rate
        self.n_samples = n_samples

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> np.ndarray:
        return preprocess_for_perch(
            self.data_dir / self.files[i],
            pitch_shift_semitones=self.pitch_shift_semitones,
            target_sample_rate=self.sample_rate,
            target_n_samples=self.n_samples,
        )


def _extract_embeddings(
    embedder: PerchEmbedder,
    data_dir: Path,
    files: list[str],
    pitch_shift_semitones: float,
    sample_rate: int,
    n_samples: int,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    ds = _PerchPreprocessDataset(files, data_dir, pitch_shift_semitones, sample_rate, n_samples)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    n = len(files)
    out = np.zeros((n, 1536), dtype=np.float32)
    pos = 0
    for batch in tqdm(loader, desc="extract", total=len(loader)):
        # default_collate gives us a torch.Tensor (B, n_samples)
        arr = batch.numpy() if isinstance(batch, torch.Tensor) else np.stack(batch)
        out[pos : pos + len(arr)] = embedder.embed(arr)
        pos += len(arr)
    return out


@hydra.main(version_base=None, config_path="../../../conf", config_name="perch_config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    data_root = Path(to_absolute_path(cfg.data.root))
    train_dir = data_root / cfg.data.train_subdir
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    cache_dir = Path(to_absolute_path(cfg.embeddings.cache_dir)) / cfg.data.split_strategy
    onnx_dir = Path(to_absolute_path(cfg.embeddings.onnx_dir))

    files, labels = build_train_manifest(train_dir)
    if len(files) == 0:
        raise RuntimeError(f"No labeled .aif files found in {train_dir}")

    splitter = _select_splitter(cfg.data.split_strategy)
    tr_files, tr_labels, va_files, va_labels = splitter(
        files=files,
        labels=labels,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.data.split_seed,
    )
    log.info(
        "Split: train=%d (pos=%d), val=%d (pos=%d)",
        len(tr_files),
        sum(tr_labels),
        len(va_files),
        sum(va_labels),
    )

    onnx_path = download_perch_onnx(onnx_dir, filename=cfg.embeddings.onnx_file)
    log.info("Loading Perch ONNX from %s", onnx_path)
    embedder = PerchEmbedder(onnx_path)

    for split_name, split_files, split_labels in [
        ("train", tr_files, tr_labels),
        ("val", va_files, va_labels),
    ]:
        log.info("Extracting %s (%d clips)...", split_name, len(split_files))
        emb = _extract_embeddings(
            embedder,
            train_dir,
            split_files,
            pitch_shift_semitones=cfg.preprocess.pitch_shift_semitones,
            sample_rate=cfg.preprocess.sample_rate,
            n_samples=cfg.preprocess.n_samples,
            batch_size=cfg.embeddings.batch_size,
            num_workers=cfg.embeddings.num_workers,
        )
        out_file = cache_dir / f"{split_name}.npz"
        save_split(out_file, emb, np.array(split_labels), split_files)
        log.info("Wrote %s (shape=%s)", out_file, emb.shape)


if __name__ == "__main__":
    main()
