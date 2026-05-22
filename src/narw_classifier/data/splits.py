"""Deterministic train/val split for the NARW dataset.

Stratified random split: same proportion of positives in train and val, reproducible via seed.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.model_selection import train_test_split


def stratified_split(
    files: Sequence[str],
    labels: Sequence[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Split ``(files, labels)`` into train and val sets stratified by label.

    Returns ``(train_files, train_labels, val_files, val_labels)``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if len(files) != len(labels):
        raise ValueError(f"files and labels length mismatch: {len(files)} vs {len(labels)}")

    train_files, val_files, train_labels, val_labels = train_test_split(
        list(files),
        list(labels),
        test_size=val_fraction,
        random_state=seed,
        stratify=list(labels),
        shuffle=True,
    )
    return train_files, train_labels, val_files, val_labels
