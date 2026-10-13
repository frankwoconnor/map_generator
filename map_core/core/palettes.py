"""Palette loading utilities for Map Art Generator with caching and API support."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# Default fallback palettes if file is missing or invalid
DEFAULT_PALETTES = {
    "OrRd_3": ["#fee8c8", "#fdbb84", "#e34a33"],
    "YlGnBu_3": ["#edf8fb", "#b2e2e2", "#66c2a4"],
    "YlGnBu_5": ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
    "Blues_5": ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
    "Greens_5": ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
    "Purples_5": ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"],
    "Reds_5": ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"],
    "Spectral_5": ["#9e0142", "#f46d43", "#ffffbf", "#66c2a5", "#5e4fa2"],
    "Set1_5": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"],
    "Set3_5": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"],
}

# Global cache for loaded palettes
_PALETTES_CACHE: Optional[Dict[str, List[str]]] = None
_PALETTES_FILE_PATH: Optional[str] = None
# Metadata keys (names starting with "_") preserved from the JSON file so that
# curated comments survive palette create/rename/delete writes.
_PALETTES_META: Dict[str, Any] = {}
# Original file key order (metadata + palettes) so saves reproduce the curated
# compact layout and remain byte-stable.
_PALETTES_KEY_ORDER: List[str] = []


def set_palettes_file_path(file_path: str) -> None:
    """Set the path to the palettes JSON file.

    Args:
        file_path: Absolute path to the palettes JSON file
    """
    global _PALETTES_FILE_PATH
    _PALETTES_FILE_PATH = file_path
    # Reset cache when file path changes
    reset_cache()


def get_palettes_file_path() -> str:
    """Get the current palettes file path.

    Returns:
        str: Current palettes file path
    """
    global _PALETTES_FILE_PATH
    if _PALETTES_FILE_PATH is None:
        # Default to config/palettes/palettes.json relative to project root
        _PALETTES_FILE_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "palettes",
            "palettes.json",
        )
    return _PALETTES_FILE_PATH


def reset_cache() -> None:
    """Reset the palettes cache, forcing reload on next access."""
    global _PALETTES_CACHE, _PALETTES_META, _PALETTES_KEY_ORDER
    _PALETTES_CACHE = None
    _PALETTES_META = {}
    _PALETTES_KEY_ORDER = []


def load_palettes() -> Dict[str, List[str]]:
    """Load palettes from the configured JSON file with caching.

    Returns:
        Dict[str, List[str]]: Mapping of palette name to list of hex color strings.
    """
    global _PALETTES_CACHE, _PALETTES_META, _PALETTES_KEY_ORDER

    if _PALETTES_CACHE is not None:
        return _PALETTES_CACHE

    palettes = _load_palettes_from_file()

    if palettes:
        _PALETTES_CACHE = palettes
        return _PALETTES_CACHE

    # Fallback to default palettes
    _PALETTES_META = {}
    _PALETTES_KEY_ORDER = []
    _PALETTES_CACHE = DEFAULT_PALETTES.copy()
    return _PALETTES_CACHE


def _load_palettes_from_file() -> Optional[Dict[str, List[str]]]:
    """Load palettes from the JSON file.

    Returns:
        Optional[Dict[str, List[str]]]: Loaded palettes or None if file doesn't exist/invalid
    """
    file_path = get_palettes_file_path()

    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and data:
                # Preserve curated metadata keys (names starting with "_") and
                # the original key order so saves reproduce the file layout.
                global _PALETTES_META, _PALETTES_KEY_ORDER
                _PALETTES_META = {k: v for k, v in data.items() if k.startswith("_")}
                _PALETTES_KEY_ORDER = list(data.keys())
                # Validate that all values are lists of strings
                validated_palettes = {}
                for name, colors in data.items():
                    if name.startswith("_"):
                        continue
                    if isinstance(colors, list) and all(
                        isinstance(color, str)
                        and color.startswith("#")
                        and len(color) == 7
                        for color in colors
                    ):
                        validated_palettes[name] = colors
                return validated_palettes
    except (json.JSONDecodeError, IOError):
        pass

    return None


def _format_palettes_document(
    meta: Dict[str, Any],
    key_order: List[str],
    palettes: Dict[str, List[str]],
) -> str:
    """Render the palettes JSON document mirroring the curated file layout.

    - Metadata comment keys render first in their original order, with a blank
      line separating each section.
    - Palettes render one compact list per line, in original key order, so
      existing entries keep their position.
    - Newly added palettes append at the end under a ``_custom`` section.

    Repeating the document through load/save is byte-stable.
    """
    lines = ["{"]
    first_section = True
    emitted = set()

    def section_gap() -> None:
        nonlocal first_section
        if not first_section:
            lines.append("")
        first_section = False

    for key in key_order:
        if key in meta:
            section_gap()
            lines.append(f"  {json.dumps(key)}: {json.dumps(meta[key])},")
        elif key in palettes:
            lines.append(f"  {json.dumps(key)}: {json.dumps(palettes[key])},")
            emitted.add(key)

    new_keys = [key for key in palettes if key not in emitted]
    if new_keys:
        section_gap()
        lines.append('  "_custom": "User-created palettes",')
        for key in new_keys:
            lines.append(f"  {json.dumps(key)}: {json.dumps(palettes[key])},")

    if len(lines) > 1:
        # Remove the trailing comma from the last emitted key line.
        for idx in range(len(lines) - 1, 0, -1):
            if lines[idx]:
                lines[idx] = lines[idx].rstrip(",")
                break
    lines.append("}")
    return "\n".join(lines) + "\n"


def save_palettes(palettes: Dict[str, List[str]]) -> bool:
    """Save palettes to the configured JSON file.

    Args:
        palettes: Dictionary of palette name to color list

    Returns:
        bool: True if saved successfully, False otherwise
    """
    file_path = get_palettes_file_path()

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Metadata keys are written back first so curated comments survive
        # saves, reproducing the hand-written compact layout byte-for-byte.
        doc = _format_palettes_document(_PALETTES_META, _PALETTES_KEY_ORDER, palettes)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc)

        # Update cache
        global _PALETTES_CACHE
        _PALETTES_CACHE = palettes.copy()
        return True
    except (IOError, TypeError):
        return False


def get_palette_names() -> List[str]:
    """Get list of available palette names.

    Returns:
        List[str]: Sorted list of palette names
    """
    palettes = load_palettes()
    return sorted(palettes.keys())


def get_palette(name: str) -> Optional[List[str]]:
    """Get a specific palette by name.

    Args:
        name: Name of the palette

    Returns:
        Optional[List[str]]: List of colors or None if not found
    """
    palettes = load_palettes()
    return palettes.get(name)


def add_palette(name: str, colors: List[str]) -> bool:
    """Add or update a palette.

    Args:
        name: Palette name
        colors: List of hex color strings

    Returns:
        bool: True if added successfully, False otherwise
    """
    palettes = load_palettes()
    palettes[name] = colors
    return save_palettes(palettes)


def remove_palette(name: str) -> bool:
    """Remove a palette.

    Args:
        name: Name of the palette to remove

    Returns:
        bool: True if removed successfully, False otherwise
    """
    palettes = load_palettes()
    if name not in palettes:
        return False

    del palettes[name]
    return save_palettes(palettes)


def validate_palette_colors(colors: List[str]) -> bool:
    """Validate that all colors are valid hex color strings.

    Args:
        colors: List of color strings to validate

    Returns:
        bool: True if all colors are valid hex colors
    """
    return isinstance(colors, list) and all(
        isinstance(color, str)
        and color.startswith("#")
        and len(color) == 7
        and all(c in "0123456789abcdefABCDEF" for c in color[1:])
        for color in colors
    )
