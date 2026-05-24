"""Scan the train directory and build a manifest of (filename, label) pairs."""

from __future__ import annotations

import os
from pathlib import Path

from ..utils.filenames import parse_filename


def build_train_manifest(train_dir: Path) -> tuple[list[str], list[int]]:
    """List ``.aif`` files in ``train_dir`` and parse the embedded label.

    Returns ``(filenames, labels)`` where ``labels`` are 0 or 1. Files with unparseable
    names or missing labels are skipped (and noted via a returned skip count).
    """
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {train_dir}")

    files: list[str] = []
    labels: list[int] = []
    for name in sorted(os.listdir(train_dir)):
        if not name.endswith(".aif"):
            continue
        parsed = parse_filename(name)
        if parsed is None or parsed.label is None:
            continue
        files.append(name)
        labels.append(parsed.label)
    return files, labels
