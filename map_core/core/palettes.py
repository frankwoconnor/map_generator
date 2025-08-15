"""Palette loading utilities for Map Art Generator.

Provides a single source of truth for loading and caching color palettes.
"""
from __future__ import annotations

import json
from typing import Dict, List

PALETTES_FILE = 'palettes.json'
_PALETTES_CACHE: Dict[str, List[str]] | None = None


def load_palettes() -> Dict[str, List[str]]:
    """Load palettes from palettes.json with in-memory caching.

    Returns:
        Mapping of palette name to list of hex colors.
    Fallback:
        A small default dict if file is missing or invalid.
    """
    global _PALETTES_CACHE
    if _PALETTES_CACHE is not None:
        return _PALETTES_CACHE
    try:
        with open(PALETTES_FILE, 'r') as f:
            palettes = json.load(f)
            if isinstance(palettes, dict):
                _PALETTES_CACHE = palettes
                return _PALETTES_CACHE
    except Exception:
        pass
    # Fallback minimal palettes
    _PALETTES_CACHE = {
        'OrRd_3': ['#fee8c8', '#fdbb84', '#e34a33'],
        'YlGnBu_3': ['#edf8fb', '#b2e2e2', '#66c2a4'],
    }
    return _PALETTES_CACHE
