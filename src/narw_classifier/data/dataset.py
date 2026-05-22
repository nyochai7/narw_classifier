"""NARW audio Dataset. Loads .aif clips, resamples to a target rate, pads/truncates to fixed length.

The Dataset returns *raw waveforms* (not spectrograms) so the same tensor can feed both the
EfficientNet baseline (which computes its own log-mel inside the model) and an audio
foundation model like Perch (which expects raw 32 kHz waveform).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset


def pad_or_truncate(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Right-pad with zeros or truncate ``waveform`` (1-D) to ``target_length`` samples."""
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be 1-D, got shape {tuple(waveform.shape)}")
    n = waveform.shape[0]
    if n == target_length:
        return waveform
    if n > target_length:
        return waveform[:target_length]
    out = waveform.new_zeros(target_length)
    out[:n] = waveform
    return out


class NARWAudioDataset(Dataset):
    """Train/val dataset over a list of ``.aif`` files with binary labels.

    Each ``__getitem__`` returns ``(waveform: float32 tensor [target_n_samples], label: int)``.
    """

    def __init__(
        self,
        files: Sequence[str],
        labels: Sequence[int],
        data_dir: Path,
        target_sample_rate: int,
        target_duration_s: float,
    ) -> None:
        if len(files) != len(labels):
            raise ValueError(f"files / labels mismatch: {len(files)} vs {len(labels)}")
        self.files = list(files)
        self.labels = [int(x) for x in labels]
        self.data_dir = Path(data_dir)
        self.target_sample_rate = int(target_sample_rate)
        self.target_n_samples = int(round(target_sample_rate * target_duration_s))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.data_dir / self.files[idx]
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        waveform = torch.from_numpy(audio).float()
        if sr != self.target_sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)
        waveform = pad_or_truncate(waveform, self.target_n_samples)
        return waveform, self.labels[idx]
