"""Shared buildings styling logic for both separate layer and combined map outputs.

This module centralizes the buildings auto-coloring logic (auto_size, auto_distance,
manual_floorsize) to eliminate duplication between save_layer() and the combined
plot block in main.py.
"""

from __future__ import annotations

import io
import contextlib
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

from map_core.core.buildings import compute_metric as core_compute_buildings_metric
from map_core.core.palettes import load_palettes
from map_core.core.util import has_data, log_progress


def _get_building_common_params(layer_style: Dict[str, Any]) -> Dict[str, Any]:
    """Extract common building plotting params applied across all building modes."""
    return {
        "edgecolor": layer_style.get("edgecolor", "#000000"),
        "linewidth": layer_style.get("linewidth", 0.5),
        "alpha": layer_style.get("alpha", 1.0),
        "hatch": layer_style.get("hatch", None),
        "zorder": layer_style.get("zorder", 2),
    }


def _reproject_gdf_for_area_calc(
    gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, Optional[Any]]:
    """Project to local UTM for accurate area calculation.

    Returns:
        (projected_gdf, original_crs)
    """
    if not has_data(gdf):
        return gdf, None

    original_crs = gdf.crs
    if original_crs and original_crs.is_geographic:
        try:
            utm_crs = gdf.estimate_utm_crs()
            gdf_proj = gdf.to_crs(utm_crs)
        except Exception as e:
            log_progress(
                f"Warning: Could not reproject GeoDataFrame to UTM. Error: {e}. Using original CRS."
            )
            gdf_proj = gdf
    else:
        gdf_proj = gdf
    return gdf_proj, original_crs


def _get_center_proj(
    buildings_proj: gpd.GeoDataFrame,
    location_query: Optional[str] = None,
) -> Optional[Point]:
    """Get projected center point for distance calculations.

    Attempts to parse from location_query, falls back to bounds centroid.
    """
    center_proj = None
    if location_query:
        try:
            q = (location_query or "").replace(",", " ").split()
            if len(q) >= 2:
                lat_c = float(q[0])
                lon_c = float(q[1])
                center_wgs = gpd.GeoSeries([Point(lon_c, lat_c)], crs="EPSG:4326")
                center_proj = center_wgs.to_crs(buildings_proj.crs).iloc[0]
        except Exception:
            pass

    # Fallback to bounds centroid
    if center_proj is None:
        try:
            minx, miny, maxx, maxy = buildings_proj.total_bounds
            center_proj = Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)
        except Exception:
            center_proj = None

    return center_proj


def _apply_manual_style(
    ax: Any,
    data: gpd.GeoDataFrame,
    layer_style: Dict[str, Any],
    common_params: Dict[str, Any],
) -> None:
    """Apply manual (single color) style to buildings."""
    from map_core.core.plot import plot_map_layer

    manual = layer_style.get("manual_color_settings", {})
    face = manual.get("facecolor", "#000000")
    plot_map_layer(
        ax,
        "buildings",
        data,
        face,
        common_params["edgecolor"],
        common_params["linewidth"],
        common_params["alpha"],
        hatch=common_params["hatch"],
        zorder=common_params["zorder"],
    )


def _apply_manual_floorsize_style(
    ax: Any,
    data: gpd.GeoDataFrame,
    layer_style: Dict[str, Any],
    common_params: Dict[str, Any],
) -> None:
    """Apply manual_floorsize (category-based) style to buildings."""
    from map_core.core.plot import plot_map_layer

    categories = layer_style.get("size_categories", []) or []
    if not categories:
        _apply_manual_style(ax, data, layer_style, common_params)
        return

    try:
        buildings_proj, _ = _reproject_gdf_for_area_calc(data)
        if has_data(buildings_proj):
            areas = buildings_proj.geometry.area
        else:
            areas = pd.Series([0] * len(data), index=data.index)

        default_face = layer_style.get("manual_color_settings", {}).get("facecolor", "#000000")
        color_series = pd.Series(default_face, index=data.index)

        for cat in categories:
            facecolor = cat.get("facecolor", default_face)
            min_area = cat.get("min_area", None)
            max_area = cat.get("max_area", None)
            mask = pd.Series(True, index=data.index)
            if min_area is not None:
                mask &= areas >= float(min_area)
            if max_area is not None:
                mask &= areas <= float(max_area)
            color_series.loc[mask] = facecolor

        try:
            data.plot(
                ax=ax,
                color=color_series,
                edgecolor=common_params["edgecolor"],
                linewidth=common_params["linewidth"],
                alpha=common_params["alpha"],
                zorder=common_params["zorder"],
            )
        except Exception as e:
            log_progress(
                f"Warning: manual_floorsize vectorized plot failed: {e}. Falling back to per-feature plotting."
            )
            for idx, row in data.iterrows():
                single = gpd.GeoDataFrame(
                    geometry=[row.geometry], crs=getattr(data, "crs", None)
                )
                single.plot(
                    ax=ax,
                    fc=color_series.loc[idx],
                    ec=common_params["edgecolor"],
                    lw=common_params["linewidth"],
                    alpha=common_params["alpha"],
                    zorder=common_params["zorder"],
                )
    except Exception as e:
        log_progress(
            f"Warning: manual_floorsize failed: {e}. Falling back to manual plot."
        )
        _apply_manual_style(ax, data, layer_style, common_params)


def _apply_auto_style(
    ax: Any,
    data: gpd.GeoDataFrame,
    layer_style: Dict[str, Any],
    common_params: Dict[str, Any],
    mode: str,
    location_query: Optional[str] = None,
) -> None:
    """Apply auto_size or auto_distance style to buildings.

    Args:
        ax: Matplotlib axes
        data: Buildings GeoDataFrame
        layer_style: Layer style configuration
        common_params: Common plotting parameters
        mode: Either "auto_size" or "auto_distance"
        location_query: Optional location query string for distance calculations
    """
    from map_core.core.plot import plot_map_layer

    palette_key = "auto_size_palette" if mode == "auto_size" else "auto_distance_palette"
    palette_name = layer_style.get(palette_key)
    palettes = load_palettes()

    if not isinstance(palettes, dict):
        palettes = {}
    colors = palettes.get(palette_name) or []

    if not colors:
        log_progress(
            f"Warning: Palette '{palette_name}' not found or empty for '{mode}' mode. Falling back to manual color."
        )
        _apply_manual_style(ax, data, layer_style, common_params)
        return

    # Reverse palette so darker colors are first
    try:
        colors = list(colors)[::-1]
    except Exception:
        colors = list(colors)

    log_progress(f"Auto '{mode}' using palette '{palette_name}' (reversed) with {len(colors)} colors")

    # Compute metric
    if mode == "auto_size":
        metric = core_compute_buildings_metric(data, "area")
        if not hasattr(metric, "index") or len(metric) != len(data):
            metric = pd.Series([0] * len(data), index=data.index)
        data["metric"] = metric
    else:  # auto_distance
        buildings_proj, _ = _reproject_gdf_for_area_calc(data)
        if has_data(buildings_proj):
            center_proj = _get_center_proj(buildings_proj, location_query)
            if center_proj is not None:
                data["metric"] = buildings_proj.geometry.centroid.distance(center_proj)
            else:
                data["metric"] = 0
        else:
            data["metric"] = 0

    # Bin metric to palette colors
    num_classes = len(colors)
    metrics = data["metric"].fillna(data["metric"].median()) if has_data(data) else None
    bins = pd.qcut(metrics, q=num_classes, duplicates="drop") if metrics is not None else None

    if bins is not None and hasattr(bins, "cat"):
        data["__color__"] = [colors[i] if i >= 0 else colors[0] for i in bins.cat.codes]
    else:
        data["__color__"] = colors[0]

    # Plot vectorized colors; fallback to per-feature if needed
    try:
        data.plot(
            ax=ax,
            color=data["__color__"],
            edgecolor=common_params["edgecolor"],
            linewidth=common_params["linewidth"],
            alpha=common_params["alpha"],
            zorder=common_params["zorder"],
        )
    except Exception as e:
        log_progress(f"Warning: Auto vectorized plot failed: {e}. Falling back to per-feature plotting.")
        for _, row in data.iterrows():
            color_value = row.get("__color__", colors[0])
            single = gpd.GeoDataFrame(
                geometry=[row.geometry], crs=getattr(data, "crs", None)
            )
            single.plot(
                ax=ax,
                fc=color_value,
                ec=common_params["edgecolor"],
                lw=common_params["linewidth"],
                alpha=common_params["alpha"],
                zorder=common_params["zorder"],
            )


def style_buildings(
    ax: Any,
    data: gpd.GeoDataFrame,
    layer_style: Dict[str, Any],
    location_query: Optional[str] = None,
) -> None:
    """Apply buildings styling based on the configured mode.

    This is the main entry point for styling buildings, supporting:
    - manual: Single color from manual_color_settings.facecolor
    - manual_floorsize: Categories with min/max area and facecolor
    - auto_size: Quantile-based coloring by building area
    - auto_distance: Quantile-based coloring by distance from center

    Args:
        ax: Matplotlib axes to plot on
        data: Buildings GeoDataFrame
        layer_style: Layer style configuration dictionary
        location_query: Optional location query string ("lat lon") for distance calculations
    """
    if not has_data(data):
        return

    # Building feature queries can return node-tagged points which, when
    # plotted, render as stray circle markers in the SVG. Drop them here for
    # every styling mode (manual / manual_floorsize / auto_*).
    from map_core.core.util import drop_point_features

    data = drop_point_features(data, "buildings")
    if not has_data(data):
        return

    buildings_style_mode = layer_style.get("auto_style_mode", "manual")
    log_progress(f"Styling buildings with mode: {buildings_style_mode}")

    common_params = _get_building_common_params(layer_style)

    if buildings_style_mode == "manual":
        _apply_manual_style(ax, data, layer_style, common_params)

    elif buildings_style_mode == "manual_floorsize":
        _apply_manual_floorsize_style(ax, data, layer_style, common_params)

    elif buildings_style_mode in ["auto_size", "auto_distance"]:
        _apply_auto_style(ax, data, layer_style, common_params, buildings_style_mode, location_query)

    else:
        log_progress(f"Warning: Unknown buildings style mode '{buildings_style_mode}'. Using manual.")
        _apply_manual_style(ax, data, layer_style, common_params)