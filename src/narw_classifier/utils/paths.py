"""Resolve filesystem paths so the same code runs locally and on Lightning AI."""

from __future__ import annotations

from pathlib import Path


def resolve_data_root(root: str | Path) -> Path:
    """Expand ``~`` and resolve to an absolute path. Raises if it doesn't exist."""
    p = Path(root).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Data root does not exist: {p}")
    return p
