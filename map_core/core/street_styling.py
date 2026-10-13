"""Per-class street styling: palette-based or user-overridden color themes.

Streets are drawn per ``highway`` class (``motorway``, ``residential``, ...).
This module centralizes the class -> color mapping so the CLI engine and the
web studio render exactly the same colors. Line widths live in
``map_core.core.plot`` (``_highway_widths_for_style``), which also consumes the
per-category ``category_widths`` overrides defined here.

``By Road Class`` (``style_mode == "by_class"``) assigns each of the seven
road categories (Motorways ... Paths & Trails) a color. Resolution order per
category:

1. An explicit ``class_colors`` override (user tweaks a color picker)
2. The configured palette's color, with the darkest color on Motorways
3. The default grey importance ramp
4. The layer's ``edgecolor``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from map_core.core.palettes import load_palettes

# Ordered (id, label, highway values) list. Order matters: index 0 is the most
# important class (motorways), used for the darkest/thickest styling.
STREET_CATEGORIES: List[Any] = [
    (
        "motorways",
        "Motorways",
        ["motorway", "motorway_link", "trunk", "trunk_link"],
    ),
    ("primary", "Primary", ["primary", "primary_link"]),
    ("secondary", "Secondary", ["secondary", "secondary_link"]),
    ("tertiary", "Tertiary", ["tertiary", "tertiary_link"]),
    (
        "residential",
        "Residential",
        ["residential", "unclassified", "road", "living_street"],
    ),
    ("service", "Service", ["service"]),
    (
        "paths_trails",
        "Paths & Trails",
        ["pedestrian", "track", "footway", "path", "cycleway", "steps"],
    ),
]

# Default stroke color per category when by_class mode has neither a palette
# nor per-class overrides: a dark -> light grey importance ramp so motorways
# read as dark/thick and paths read as light/thin.
DEFAULT_STREET_CATEGORY_COLORS: Dict[str, str] = {
    "motorways": "#1a1a1a",
    "primary": "#2e2e2e",
    "secondary": "#4d4d4d",
    "tertiary": "#6b6b6b",
    "residential": "#8c8c8c",
    "service": "#a6a6a6",
    "paths_trails": "#bfbfbf",
}


def _palette_colors_darkest_first(palette_name: Optional[str]) -> List[str]:
    """Return a palette's colors with the darkest first, or [] if unusable.

    Matches the buildings convention (``buildings_styling._apply_auto_style``)
    where index 0 is the strongest color.
    """
    name = (palette_name or "").strip()
    if not name:
        return []
    try:
        palettes = load_palettes()
        colors = list(palettes.get(name) or [])
    except Exception:
        return []
    return colors[::-1]


def street_category_colors(
    edgecolor: str = "#000000", layer_style: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Effective per-category stroke colors for by_class mode.

    Returns an empty mapping when ``style_mode`` is not ``"by_class"``, which
    callers treat as "use the layer's edgecolor everywhere".
    """
    style = layer_style or {}
    if style.get("style_mode") != "by_class":
        return {}
    overrides = style.get("class_colors") or {}
    palette = _palette_colors_darkest_first(style.get("street_palette"))

    colors: Dict[str, str] = {}
    for idx, (cat_id, _label, _highways) in enumerate(STREET_CATEGORIES):
        color = overrides.get(cat_id)
        if not color and palette:
            # Categories past the palette's length keep the lightest color,
            # matching the "minor features are lighter" convention.
            color = palette[min(idx, len(palette) - 1)]
        if not color:
            color = DEFAULT_STREET_CATEGORY_COLORS.get(cat_id) or edgecolor
        colors[cat_id] = color
    return colors


def street_class_colors(
    edgecolor: str = "#000000", layer_style: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Expand per-category colors to every known highway value.

    Outside by_class mode (or when the style is unusable) an empty mapping is
    returned, meaning every edge keeps the layer's ``edgecolor``.
    """
    if (layer_style or {}).get("style_mode") != "by_class":
        return {}
    cat_colors = street_category_colors(edgecolor, layer_style)
    overrides = (layer_style or {}).get("class_colors") or {}

    result: Dict[str, str] = {}
    for cat_id, _label, highways in STREET_CATEGORIES:
        base = cat_colors.get(cat_id, edgecolor)
        for h in highways:
            # A per-highway override (keyed by the raw highway value) wins.
            result[h] = overrides.get(h, base)
    return result