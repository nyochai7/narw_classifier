"""Deterministic train/val/test splits for the NARW dataset.

Two strategies:

- ``stratified_split_3way``: random, stratified by label. Simplest, but clips
  from the same recording date can land in multiple splits, inflating the
  metrics on the held-out side.
- ``day_stratified_split_3way``: every clip from a given date lands in
  exactly one split. Avoids the day-level leakage above.

Both produce a (train, val, test) triple sized so ``val_fraction`` and
``test_fraction`` are fractions *of the original total*.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..utils.filenames import parse_filename


def _group_for(file: str) -> str:
    parsed = parse_filename(file)
    if parsed is None:
        raise ValueError(f"Could not parse date from filename: {file}")
    return parsed.date


_SplitTriple = tuple[
    list[str],
    list[int],
    list[str],
    list[int],
    list[str],
    list[int],
]


def _check_3way_fractions(val_fraction: float, test_fraction: float) -> None:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must be < 1; got "
            f"{val_fraction} + {test_fraction} = {val_fraction + test_fraction}"
        )


def stratified_split_3way(
    files: Sequence[str],
    labels: Sequence[int],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> _SplitTriple:
    """Stratified random 3-way split. Returns ``(tr_f, tr_y, va_f, va_y, te_f, te_y)``.

    Two stages: first carve off ``test_fraction``, then split the remainder into
    train and val such that val is ``val_fraction`` of the *original* total.
    """
    _check_3way_fractions(val_fraction, test_fraction)
    if len(files) != len(labels):
        raise ValueError(f"files and labels length mismatch: {len(files)} vs {len(labels)}")

    files = list(files)
    labels = list(labels)

    rest_files, test_files, rest_labels, test_labels = train_test_split(
        files,
        labels,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )
    val_fraction_of_rest = val_fraction / (1.0 - test_fraction)
    train_files, val_files, train_labels, val_labels = train_test_split(
        rest_files,
        rest_labels,
        test_size=val_fraction_of_rest,
        random_state=seed,
        stratify=rest_labels,
        shuffle=True,
    )
    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def day_stratified_split_3way(
    files: Sequence[str],
    labels: Sequence[int],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> _SplitTriple:
    """Day-grouped 3-way split. Every clip from a given date lands in exactly one
    of train / val / test. Returns ``(tr_f, tr_y, va_f, va_y, te_f, te_y)``."""
    _check_3way_fractions(val_fraction, test_fraction)
    if len(files) != len(labels):
        raise ValueError(f"files and labels length mismatch: {len(files)} vs {len(labels)}")

    files = list(files)
    labels = list(labels)
    groups = [_group_for(f) for f in files]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    rest_idx, test_idx = next(gss1.split(X=files, y=labels, groups=groups))

    rest_files = [files[i] for i in rest_idx]
    rest_labels = [labels[i] for i in rest_idx]
    rest_groups = [groups[i] for i in rest_idx]
    test_files = [files[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    val_fraction_of_rest = val_fraction / (1.0 - test_fraction)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_rest, random_state=seed)
    train_idx, val_idx = next(gss2.split(X=rest_files, y=rest_labels, groups=rest_groups))
    train_files = [rest_files[i] for i in train_idx]
    train_labels = [rest_labels[i] for i in train_idx]
    val_files = [rest_files[i] for i in val_idx]
    val_labels = [rest_labels[i] for i in val_idx]

    for name, ys in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        if len(set(ys)) < 2:
            raise RuntimeError(
                f"day_stratified_split_3way: {name} side missing a class — try a different seed"
            )

    return train_files, train_labels, val_files, val_labels, test_files, test_labels
