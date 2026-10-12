"""Dotted-path style overrides for the ``mapgen`` CLI.

``--set layers.buildings.facecolor=#ff0066`` applies a single value to the
loaded style dict, with light type coercion, so artists can iterate on a map
without hand-editing ``style.json``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

_INT_RE = re.compile(r"^-?\d+$")


def coerce(value: str) -> Any:
    """Turn a CLI string into bool/int/float/None/list/dict where obvious."""
    v = value.strip()
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    if v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    if _INT_RE.match(v):
        return int(v)
    try:
        return float(v)
    except ValueError:
        return value


def set_path(style: Dict[str, Any], path: str, value: Any) -> None:
    """Deep-set ``path`` (dot-separated) to ``value`` in ``style``.

    The path must already exist: creating new keys silently is how typos
    become "why is nothing happening" bugs, so an unknown key raises a
    ``KeyError`` listing the valid keys at that level.
    """
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise KeyError("Empty override path")
    cur: Any = style
    for i, part in enumerate(parts[:-1]):
        if not isinstance(cur, dict) or part not in cur:
            valid = sorted(cur.keys()) if isinstance(cur, dict) else []
            raise KeyError(
                f"Unknown key '{'.'.join(parts[: i + 1])}'. Valid keys here: {valid}"
            )
        cur = cur[part]
    last = parts[-1]
    if not isinstance(cur, dict) or last not in cur:
        valid = sorted(cur.keys()) if isinstance(cur, dict) else []
        raise KeyError(f"Unknown key '{path}'. Valid keys here: {valid}")
    cur[last] = value


def apply_overrides(
    style: Dict[str, Any], overrides: List[str], coerce_values: bool = True
) -> Dict[str, Any]:
    """Apply ``["a.b.c=value", ...]`` to ``style`` in order."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key=value")
        path, raw = item.split("=", 1)
        set_path(style, path.strip(), coerce(raw) if coerce_values else raw)
    return style
