"""Utility helpers (logging, small predicates) for Map Art Generator."""

from __future__ import annotations

from typing import Any

import geopandas as gpd


def log_progress(message: str) -> None:
    """Lightweight stdout logger with flush to surface progress to the UI."""
    print(message, flush=True)


def drop_point_features(gdf: Any, layer_name: str = "<unknown>") -> Any:
    """Drop Point/MultiPoint features from a GeoDataFrame.

    OSM feature queries return node-tagged points alongside polygon/line
    features (for example ``natural``/``leisure`` nodes that slip into the
    water or green layers). When they reach the plotter they are rendered as
    small scatter markers (circles) in the SVG. This filter removes them
    before plotting while leaving lines and polygons untouched.

    Non-GeoDataFrame inputs (e.g. a NetworkX street graph) pass through
    unchanged.

    Args:
        gdf: Input GeoDataFrame (or a graph for streets).
        layer_name: Layer name used only for logging.

    Returns:
        The input GeoDataFrame without Point/MultiPoint geometries, or the
        original input unchanged when it is not a GeoDataFrame.
    """
    try:
        if (
            not has_data(gdf)
            or not isinstance(gdf, gpd.GeoDataFrame)
            or "geometry" not in gdf
        ):
            return gdf
        gdf = gdf[gdf.geometry.notnull()]
        try:
            gdf = gdf[~gdf.geometry.is_empty]
        except Exception:
            pass
        dropped = gdf[gdf.geom_type.isin(["Point", "MultiPoint"])]
        if len(dropped) > 0:
            log_progress(
                f"[filter] Dropped {len(dropped)} point feature(s) "
                f"(rendered as circles) from '{layer_name}'"
            )
        return gdf[~gdf.geom_type.isin(["Point", "MultiPoint"])]
    except Exception as e:
        log_progress(f"[filter] Skipping point-filter for '{layer_name}': {e}")
        return gdf


def has_data(data: Any) -> bool:
    """Return True if a GeoDataFrame-like has rows; tolerant of None.

    Args:
        data: Any object; will check for 'empty' attr if present.
    """
    return data is not None and getattr(data, "empty", False) is False
