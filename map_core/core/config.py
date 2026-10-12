"""Configuration utilities for Map Art Generator.

Provides loading, normalization, and optional JSON Schema validation for style.json.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from config.schemas.style_schema import StyleConfig

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover - validation becomes a no-op if not installed
    jsonschema = None

STYLE_FILE = "style.json"
SCHEMA_PATH = os.path.join("maps2", "schemas", "style.schema.json")


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

    Delegates to StyleConfig for a single source of truth. Does not mutate the
    input object.
    """
    return StyleConfig.from_dict(style or {}).to_dict()


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
