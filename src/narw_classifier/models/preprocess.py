"""Audio-to-image preprocessor.

Takes a batch of raw waveforms ``(B, n_samples)`` and produces a 3-channel image-like tensor
``(B, 3, image_size, image_size)`` suitable for an ImageNet-pretrained CNN:

1. Log-mel spectrogram (``torchaudio.transforms.MelSpectrogram`` + ``AmplitudeToDB``)
2. Per-clip min-max normalize to ``[0, 1]``
3. Replicate mono mel to 3 channels
4. Bilinear resize to ``(image_size, image_size)``
5. Apply ImageNet mean / std normalization (so we match the backbone's pretraining stats)

The module is fully differentiable and lives on the GPU once the LightningModule is moved.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TT

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_MIN_POWER = 1e-10  # numeric floor before log10


class MelImagePreprocessor(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        image_size: int,
        top_db: float = 80.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.image_size = image_size
        self.top_db = float(top_db)
        self.mel = TT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
        )
        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """``waveform``: ``(B, n_samples)`` float32. Returns ``(B, 3, image_size, image_size)``."""
        if waveform.ndim != 2:
            raise ValueError(f"expected (B, n_samples), got shape {tuple(waveform.shape)}")
        mel = self.mel(waveform)  # (B, n_mels, T), power spectrogram
        # Per-clip dB conversion with a per-clip top_db floor (batch-independent).
        log_mel = 10.0 * torch.log10(mel.clamp_min(_MIN_POWER))
        b = log_mel.size(0)
        per_clip_max = log_mel.reshape(b, -1).max(dim=1).values.view(b, 1, 1)
        log_mel = torch.maximum(log_mel, per_clip_max - self.top_db)
        # Per-clip min-max normalize to [0, 1].
        flat = log_mel.reshape(b, -1)
        mn = flat.min(dim=1).values.view(b, 1, 1)
        mx = flat.max(dim=1).values.view(b, 1, 1)
        norm = (log_mel - mn) / (mx - mn).clamp_min(1e-8)  # (B, n_mels, T)
        img = norm.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, n_mels, T)
        img = F.interpolate(
            img,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        img = (img - self.mean) / self.std
        return img
