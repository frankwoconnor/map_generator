"""Utility helpers (logging, small predicates) for Map Art Generator."""
from __future__ import annotations

from typing import Any


def log_progress(message: str) -> None:
    """Lightweight stdout logger with flush to surface progress to the UI."""
    print(message, flush=True)


def has_data(data: Any) -> bool:
    """Return True if a GeoDataFrame-like has rows; tolerant of None.

    Args:
        data: Any object; will check for 'empty' attr if present.
    """
    return data is not None and getattr(data, 'empty', False) is False
