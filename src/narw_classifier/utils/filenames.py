"""Parse NARW dataset filenames.

Filenames look like ``20090328_000000_002s3ms_TRAIN0_0.aif``:
    - ``20090328``: date YYYYMMDD
    - ``000000``: time HHMMSS of the parent recording
    - ``002s3ms``: offset within the recording (``SsMms``)
    - ``TRAIN0`` / ``Test0``: split + intra-split index
    - ``_0`` / ``_1``: label (train only; absent for test)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERN = re.compile(
    r"^(?P<date>\d{8})"
    r"_(?P<time>\d{6})"
    r"_(?P<offset>\d+s\d+ms)"
    r"_(?P<split>TRAIN|Test)(?P<index>\d+)"
    r"(?:_(?P<label>\d))?"
    r"\.aif$"
)


@dataclass(frozen=True)
class ClipName:
    filename: str
    date: str  # YYYYMMDD
    time: str  # HHMMSS
    offset: str  # e.g. "002s3ms"
    split: str  # "TRAIN" or "Test"
    index: int
    label: int | None  # 0 or 1 for train; None for test


def parse_filename(filename: str) -> ClipName | None:
    """Parse a NARW clip filename. Returns ``None`` if the name doesn't match."""
    m = _PATTERN.match(filename)
    if m is None:
        return None
    raw = m.groupdict()
    return ClipName(
        filename=filename,
        date=raw["date"],
        time=raw["time"],
        offset=raw["offset"],
        split=raw["split"],
        index=int(raw["index"]),
        label=int(raw["label"]) if raw["label"] is not None else None,
    )
