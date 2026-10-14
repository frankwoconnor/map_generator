"""Coastline helper supporting an optional 'coastline' map layer.

Coastlines come from the bundled OSM-derived shapefile
``coastlines-split-4326/lines.shp`` (WGS84 / EPSG:4326). Reading is
spatially filtered using the map bounding box so we pay only for the
features that actually touch the mapped area.

The layer is *optional*: when there is no coastline data for the bbox
(e.g. an entirely inland map) the reader returns None and callers
skip it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import geopandas as gpd

from map_core import paths

COASTLINE_DEFAULT_PATH = "coastlines-split-4326/lines.shp"


def resolve_coastline_path() -> Optional[str]:
    """Absolute path to the coastline shapefile, if present."""
    try:
        p = paths.resolve_data_path(COASTLINE_DEFAULT_PATH)
    except Exception:
        return None
    return str(p) if p and p.exists() else None


@lru_cache(maxsize=64)
def _read_cached(key: Tuple[float, float, float, float]) -> Optional[gpd.GeoDataFrame]:
    path = resolve_coastline_path()
    if path is None:
        return None
    west, south, east, north = key
    # geopandas/pyogrio bbox filtering takes (xmin, ymin, xmax, ymax)
    try:
        gdf = gpd.read_file(path, bbox=(west, south, east, north))
    except Exception:
        return None
    if gdf is None or len(gdf) == 0:
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    try:
        gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    except Exception:
        pass
    if len(gdf) == 0:
        return None
    return gdf


def _bbox_key(bbox, buf: float = 0.01) -> Tuple[float, float, float, float]:
    """Round the bbox plus a small buffer before reading so caching stays useful."""
    west, south, east, north = (float(x) for x in bbox)
    width = east - west
    height = north - south
    pad = max(width, height) * buf + 1e-6
    west -= pad
    south -= pad
    east += pad
    north += pad
    # round to ~11 m precision (4 decimals) for cache keys
    return (
        round(west, 4),
        round(south, 4),
        round(east, 4),
        round(north, 4),
    )


def load_coastlines(bbox) -> Optional[gpd.GeoDataFrame]:
    """Return coastlines intersecting ``bbox`` (west, south, east, north).

    Returns None when the shapefile is missing or there are no matching
    lines (inland map).
    """
    if bbox is None:
        return None
    try:
        west, south, east, north = (float(x) for x in bbox)
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        return None
    if east - west <= 0 or north - south <= 0:
        return None
    gdf = _read_cached(_bbox_key((west, south, east, north)))
    if gdf is None:
        return None
    return gdf.copy()


def has_coastlines(bbox) -> bool:
    return load_coastlines(bbox) is not None
