"""NARW audio Dataset for the EfficientNet baseline.

Each ``__getitem__`` returns ``(waveform, label)`` where ``waveform`` is a 1-D
float32 tensor of 4000 samples (2 s at the dataset's native 2 kHz). The
LightningModule turns the raw waveform into a log-mel image inside its
preprocessor; this Dataset only handles file I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class NARWAudioDataset(Dataset):
    """Train/val/test dataset over a list of ``.aif`` files with binary labels.

    Each ``__getitem__`` returns ``(waveform: float32 tensor [4000], label: int)``.
    The Kaggle clips are all already mono 2 kHz / 2 s; no resampling or padding
    is needed.
    """

    def __init__(
        self,
        files: Sequence[str],
        labels: Sequence[int],
        data_dir: Path,
    ) -> None:
        if len(files) != len(labels):
            raise ValueError(f"files / labels mismatch: {len(files)} vs {len(labels)}")
        self.files = list(files)
        self.labels = [int(x) for x in labels]
        self.data_dir = Path(data_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.data_dir / self.files[idx]
        audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return torch.from_numpy(audio).float(), self.labels[idx]
