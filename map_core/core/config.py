"""Configuration utilities for Map Art Generator.

Provides loading, normalization, and optional JSON Schema validation for style.json.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover - validation becomes a no-op if not installed
    jsonschema = None

STYLE_FILE = "style.json"
SCHEMA_PATH = os.path.join("maps2", "schemas", "style.schema.json")


def _layer_defaults() -> Dict[str, Any]:
    generic = {
        "enabled": True,
        "facecolor": "#000000",
        "edgecolor": "#000000",
        "linewidth": 0.5,
        "alpha": 1.0,
        "simplify_tolerance": 0.0,
        "min_size_threshold": 0.0,
        "hatch": None,
        "zorder": 1,
    }
    buildings = {
        **generic,
        # buildings-specific overrides
        "manual_color_settings": {"facecolor": "#000000"},
        "size_categories": [],
        "size_categories_enabled": False,
        "auto_style_mode": "manual",
        "auto_size_palette": "",
        "auto_distance_palette": "",
    }
    return {
        "streets": {**generic},
        "water": {**generic},
        "green": {**generic},
        "buildings": buildings,
    }


def _output_defaults() -> Dict[str, Any]:
    return {
        "separate_layers": True,
        "filename_prefix": "map",
        "figure_size": [10.0, 10.0],
        "background_color": "#FFFFFF",
        "transparent_background": False,
        "figure_dpi": 100,
        "margin": 0.05,
        "preview_type": "embedded",
    }


def _processing_defaults() -> Dict[str, Any]:
    return {
        "street_filter": [],
    }


def _location_defaults() -> Dict[str, Any]:
    return {
        "query": "",
        "distance": None,
        # Optional path to a local .pbf file to source OSM data from.
        # When provided, fetching should prefer this local file over remote queries.
        "pbf_path": None,
    }


def load_style(path: str = STYLE_FILE) -> Optional[Dict[str, Any]]:
    """Load style.json from disk.

    Returns None if file is missing or malformed.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def normalize_style(style: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized copy of style with defaults filled for all sections.

    Does not mutate the input object.
    """
    style = dict(style or {})

    # location
    loc = dict(_location_defaults())
    loc.update(style.get("location", {}))

    # layers
    defaults = _layer_defaults()
    layers = {}
    user_layers = style.get("layers", {}) or {}
    for k, dflt in defaults.items():
        merged = dict(dflt)
        merged.update(user_layers.get(k, {}) or {})
        layers[k] = merged

    # output
    out = dict(_output_defaults())
    out.update(style.get("output", {}) or {})

    # processing
    proc = dict(_processing_defaults())
    proc.update(style.get("processing", {}) or {})

    return {"location": loc, "layers": layers, "output": out, "processing": proc}


def _load_schema() -> Optional[Dict[str, Any]]:
    if not os.path.exists(SCHEMA_PATH):
        return None
    try:
        with open(SCHEMA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def validate_style(style: Dict[str, Any]) -> Optional[str]:
    """Validate style against the JSON Schema, if available.

    Returns an error message string if invalid, otherwise None.
    If jsonschema or schema file is not available, returns None (no-op).
    """
    if jsonschema is None:
        return None
    schema = _load_schema()
    if not schema:
        return None
    try:
        jsonschema.validate(instance=style, schema=schema)
        return None
    except Exception as e:  # keep message short for UI
        return str(e)
