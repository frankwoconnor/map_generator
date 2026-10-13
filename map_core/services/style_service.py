"""Shared style service: load, normalize, and persist the map style.

Extracted from ``app.py`` so both the web studio and the CLI generation
pipeline read/write/normalize ``style.json`` through one implementation
(single source of truth). Pure standard-library + ``map_core``; no Flask.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import map_core.core.config as cfg

# The active style file, resolved relative to the repo root / CWD.
STYLE_FILE = "style.json"


def load_style(style_path: Optional[str] = STYLE_FILE) -> Dict[str, Any]:
    """
    Load `style.json`, ensuring default values are present for missing keys.

    Returns:
        dict: The normalized style configuration dictionary suitable for the UI/backend.
    """
    raw = cfg.load_style(style_path) or {}
    style_data = cfg.normalize_style(raw)

    layers_settings = style_data.setdefault("layers", {})
    # Ensure generic layers exist with sensible defaults
    layers_settings.setdefault(
        "streets",
        {
            "enabled": True,
            "edgecolor": "#000000",
            "linewidth": 0.5,
            "alpha": 1.0,
            "hatch": None,
            "zorder": 3,
            "simplify_tolerance": 0.0,
            "min_size_threshold": 0.0,
            # Per-class street styling defaults (see street_styling.py).
            "style_mode": "uniform",
            "variable_widths": True,
            "street_palette": "",
            "class_colors": {},
            "category_widths": {},
            "linewidths": {},
        },
    )
    layers_settings.setdefault(
        "water",
        {
            "enabled": True,
            "facecolor": "#a6cee3",
            "edgecolor": "#a6cee3",
            "linewidth": 0.0,
            "alpha": 1.0,
            "hatch": None,
            "zorder": 1,
            "simplify_tolerance": 0.0,
            "min_size_threshold": 0.0,
        },
    )
    layers_settings.setdefault(
        "green",
        {
            "enabled": False,
            "facecolor": "#b2df8a",
            "edgecolor": "#33a02c",
            "linewidth": 0.0,
            "alpha": 1.0,
            "hatch": None,
            "zorder": 1,
            "simplify_tolerance": 0.0,
            "min_size_threshold": 0.0,
        },
    )
    buildings_settings = layers_settings.setdefault("buildings", {})

    buildings_settings.setdefault("size_categories", [])
    buildings_settings.setdefault("auto_style_mode", "manual")
    buildings_settings.setdefault("auto_size_palette", "")
    buildings_settings.setdefault("auto_distance_palette", "")
    # Manual color only stores facecolor
    buildings_settings.setdefault("manual_color_settings", {"facecolor": "#000000"})
    # Common building params (apply to all modes)
    buildings_settings.setdefault("edgecolor", "#000000")
    buildings_settings.setdefault("linewidth", 0.5)
    buildings_settings.setdefault("alpha", 1.0)
    buildings_settings.setdefault("hatch", None)
    buildings_settings.setdefault("zorder", 2)

    # Distance is stored and displayed in meters consistently
    location_settings = style_data.setdefault("location", {})
    # Ensure persistent UI-related defaults
    location_settings.setdefault("data_source", "remote")  # 'remote' or 'local'

    # Provide sensible offline defaults if missing: Cork city center and ~1km bbox (1km across)
    # Cork center approx
    default_lat = 51.8986
    default_lon = -8.4756
    # Half-size of 1 km box: 0.5 km
    # Degrees per km approx: lat ~ 1/111, lon ~ 1/(111*cos(lat))
    lat_deg_per_km = 1.0 / 111.0
    from math import cos, radians

    lon_deg_per_km = 1.0 / (111.32 * max(0.0001, cos(radians(default_lat))))
    dlat = 0.5 * lat_deg_per_km
    dlon = 0.5 * lon_deg_per_km
    default_bbox = [
        default_lon - dlon,
        default_lat - dlat,
        default_lon + dlon,
        default_lat + dlat,
    ]

    # If no query present, set coords for Cork center
    if not location_settings.get("query"):
        location_settings["query"] = f"{default_lat} {default_lon}"
    # Do not persist a manual bbox in the new UX; it will be computed from distance
    location_settings["bbox"] = None

    # Creative output controls (D4): crop mask, frame, title/coords text.
    output_settings = style_data.setdefault("output", {})
    output_settings.setdefault("mask", "none")  # none|circle|hexagon|rounded
    output_settings.setdefault(
        "mask_options",
        {"inset": 0.0, "rounded_radius": 0.15},
    )
    output_settings.setdefault("frame", {"width": 0.0, "color": "#000000"})
    output_settings.setdefault("title", "")
    output_settings.setdefault("subtitle", "")
    output_settings.setdefault("title_font_size", 16)
    output_settings.setdefault("subtitle_font_size", 11)
    output_settings.setdefault("title_color", "#000000")
    output_settings.setdefault("subtitle_color", "#444444")

    return style_data


def save_style_json(style_data: Dict[str, Any], style_path: str = STYLE_FILE) -> None:
    """Save the updated style data to `style.json`."""
    with open(style_path, "w") as f:
        json.dump(style_data, f, indent=2)
