"""Shared fixtures: synthetic .aif clips that look like the real NARW dataset (2 kHz, 2 s, mono)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SAMPLE_RATE = 2000
DURATION_S = 2.0


def _write_clip(path: Path, label: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * DURATION_S)
    # Mix tone (positives only) + noise so the two classes are distinguishable.
    noise = rng.standard_normal(n).astype(np.float32) * 0.05
    if label == 1:
        t = np.arange(n) / SAMPLE_RATE
        tone = 0.4 * np.sin(2 * np.pi * 150.0 * t).astype(np.float32)
        signal = noise + tone
    else:
        signal = noise
    sf.write(str(path), signal, SAMPLE_RATE, subtype="PCM_16", format="AIFF")


@pytest.fixture
def synthetic_train_dir(tmp_path: Path) -> Path:
    """Create a tiny train directory with 6 positive and 14 negative clips."""
    d = tmp_path / "train2"
    d.mkdir()
    seed = 0
    for i in range(6):
        name = f"20090401_000000_{i:03d}s0ms_TRAIN{i}_1.aif"
        _write_clip(d / name, label=1, seed=seed)
        seed += 1
    for i in range(14):
        name = f"20090401_000000_{i + 6:03d}s0ms_TRAIN{i + 6}_0.aif"
        _write_clip(d / name, label=0, seed=seed)
        seed += 1
    return d


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A single .aif clip at the real dataset's native (2 kHz, 2 s) format."""
    p = tmp_path / "20090401_000000_001s0ms_TRAIN0_1.aif"
    _write_clip(p, label=1, seed=42)
    return p
