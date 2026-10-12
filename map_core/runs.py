"""Discovery of generated runs in the output directory.

A "run" is one directory under ``paths.outputs_dir()`` containing the artifacts
of a single map generation (layer SVGs/PNGs, a combined SVG, ``config.json`` and
a server log). This module turns those directories into JSON/Jinja-friendly
records so the CLI, API and web UI all describe runs the same way.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from map_core import paths


def parse_run_timestamp(name: str) -> Optional[_dt.datetime]:
    """Parse a trailing ``_YYYYMMDD_HHMMSS`` stamp from a run directory name."""
    parts = name.split("_")
    if (
        len(parts) >= 2
        and len(parts[-2]) == 8
        and len(parts[-1]) == 6
        and parts[-2].isdigit()
        and parts[-1].isdigit()
    ):
        try:
            return _dt.datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def region_name(pbf_path: Optional[str]) -> Optional[str]:
    """Human-friendly region label from a PBF path (``cork.osm.pbf`` -> ``cork``)."""
    if not pbf_path:
        return None
    name = Path(pbf_path).name
    for suffix in (".osm.pbf", ".osm.bz2", ".osm", ".pbf", ".bz2"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name or None


def _rel(path: Optional[Path], base: Path) -> Optional[str]:
    """POSIX path relative to ``base``, for use in ``/output/<path>`` URLs."""
    if path is None:
        return None
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return None


def run_record(run_dir: Path, base: Path) -> Dict[str, Any]:
    """Summarise one run directory into a JSON/Jinja-friendly record."""
    files = [p for p in run_dir.iterdir() if p.is_file()]
    svgs = sorted(p for p in files if p.suffix.lower() == ".svg")
    pngs = sorted(p for p in files if p.suffix.lower() == ".png")

    combined_svg = next((p for p in svgs if p.stem.endswith("_combined")), None)
    combined_png = next((p for p in pngs if p.stem.endswith("_combined")), None)
    thumbnail = combined_png or (pngs[0] if pngs else None) or combined_svg

    run_id = run_dir.name
    layers: List[Dict[str, Any]] = []
    for svg in svgs:
        stem = svg.stem
        if (
            stem.endswith("_combined")
            or stem.endswith("_legend")
            or "_debug_" in stem
        ):
            continue
        layer = stem[len(run_id) + 1 :] if stem.startswith(run_id + "_") else stem
        png = svg.with_suffix(".png")
        layers.append(
            {
                "name": layer,
                "svg": _rel(svg, base),
                "png": _rel(png, base) if png.exists() else None,
            }
        )

    cfg: Dict[str, Any] = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            cfg = {}
    loc = cfg.get("location", {}) if isinstance(cfg, dict) else {}
    pbf_path = loc.get("pbf_path")
    timestamp = parse_run_timestamp(run_id)

    return {
        "id": run_id,
        "timestamp": timestamp.isoformat() if timestamp else None,
        "timestamp_display": timestamp.strftime("%Y-%m-%d %H:%M") if timestamp else "",
        "prefix": run_id.rsplit("_", 2)[0] if timestamp else run_id,
        "combined_svg": _rel(combined_svg, base),
        "combined_png": _rel(combined_png, base),
        "thumbnail": _rel(thumbnail, base),
        "layers": layers,
        "svg_count": len(svgs),
        "png_count": len(pngs),
        "location": loc.get("query"),
        "pbf_path": pbf_path,
        "region": region_name(pbf_path),
        "data_source": loc.get("data_source"),
    }


def discover_runs(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return run records from the output directory, newest first."""
    base = paths.outputs_dir()
    if not base.is_dir():
        return []

    records: List[Dict[str, Any]] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        try:
            record = run_record(entry, base)
        except OSError:
            continue
        # Ignore empty/unrelated directories that aren't real runs.
        if (
            record["timestamp"] is None
            and record["svg_count"] == 0
            and record["png_count"] == 0
        ):
            continue
        records.append(record)

    records.sort(key=lambda r: (r["timestamp"] or "", r["id"]), reverse=True)
    return records[:limit] if limit else records
