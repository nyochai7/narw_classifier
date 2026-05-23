"""Deterministic train/val splits for the NARW dataset.

Two strategies:
- ``stratified_split``: random, stratified by label. Simplest, but clips from the same
  recording date can land in both train and val, which inflates val metrics.
- ``day_stratified_split``: groups clips by date — every clip from a given date lands
  entirely in train OR val. Avoids the day-level leakage above.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..utils.filenames import parse_filename


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


def day_stratified_split(
    files: Sequence[str],
    labels: Sequence[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Split by recording date: every clip from a given date lands entirely in train OR val.

    Uses ``sklearn.model_selection.GroupShuffleSplit`` with the date prefix of each
    filename as the group. Returns ``(train_files, train_labels, val_files, val_labels)``.

    Raises ``RuntimeError`` if either side ends up missing a class (very unlikely on
    this dataset, but a safety check for tiny subsets).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if len(files) != len(labels):
        raise ValueError(f"files and labels length mismatch: {len(files)} vs {len(labels)}")

    files = list(files)
    labels = list(labels)
    groups: list[str] = []
    for f in files:
        parsed = parse_filename(f)
        if parsed is None:
            raise ValueError(f"Could not parse date from filename: {f}")
        groups.append(parsed.date)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(X=files, y=labels, groups=groups))

    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    if len(set(train_labels)) < 2:
        raise RuntimeError(
            "day_stratified_split: train side missing a class — try a different seed"
        )
    if len(set(val_labels)) < 2:
        raise RuntimeError("day_stratified_split: val side missing a class — try a different seed")

    return train_files, train_labels, val_files, val_labels
