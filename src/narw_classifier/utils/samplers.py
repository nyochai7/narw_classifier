"""Sampler factories shared across DataModules."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels: Sequence[int], seed: int) -> WeightedRandomSampler:
    """Return a sampler whose draws are uniform over classes (not over examples).

    Use on the train DataLoader to give each batch ~50/50 class balance, masking
    the underlying ~8:1 imbalance in the NARW upcall dataset.
    """
    labels_arr = np.asarray(labels)
    counts = np.bincount(labels_arr, minlength=2).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"Need both classes present; got counts={counts.tolist()}")
    class_weight = 1.0 / counts
    weights = torch.tensor([class_weight[y] for y in labels_arr], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True, generator=generator
    )
