"""Perch v2 audio preprocessing.

Pipeline (per CLAUDE.md "Preprocessing pipelines"):
    1. Load AIFF → mono float32 at native 2 kHz
    2. Resample to 32 kHz
    3. Pitch-shift up 3 octaves (36 semitones) — duration-preserving phase vocoder.
       Maps NARW upcalls (~50-250 Hz) into Perch's training band (~400-2000 Hz).
    4. Symmetric zero-pad / truncate to exactly 160 000 samples (5 s at 32 kHz)
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PERCH_SAMPLE_RATE = 32000
PERCH_DURATION_S = 5.0
PERCH_N_SAMPLES = int(PERCH_SAMPLE_RATE * PERCH_DURATION_S)
DEFAULT_PITCH_SHIFT_SEMITONES = 36  # +3 octaves


def pad_or_center(y: np.ndarray, target_length: int) -> np.ndarray:
    """Symmetrically zero-pad (or center-truncate) ``y`` (1-D) to ``target_length``."""
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    n = y.shape[0]
    if n == target_length:
        return y
    if n > target_length:
        start = (n - target_length) // 2
        return y[start : start + target_length]
    out = np.zeros(target_length, dtype=y.dtype)
    start = (target_length - n) // 2
    out[start : start + n] = y
    return out


def preprocess_for_perch(
    path: Path,
    pitch_shift_semitones: float = DEFAULT_PITCH_SHIFT_SEMITONES,
    target_sample_rate: int = PERCH_SAMPLE_RATE,
    target_n_samples: int = PERCH_N_SAMPLES,
) -> np.ndarray:
    """Load + resample + pitch-shift + pad. Returns 1-D float32 of length ``target_n_samples``."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=-1)
    y = y.astype(np.float32)
    if sr != target_sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate).astype(np.float32)
    if pitch_shift_semitones != 0:
        y = librosa.effects.pitch_shift(
            y, sr=target_sample_rate, n_steps=pitch_shift_semitones
        ).astype(np.float32)
    return pad_or_center(y, target_n_samples).astype(np.float32)
