"""Per-layer plotting functions for Map Art Generator.

This module contains dedicated plotting functions for each layer type,
used by both separate layer saving and combined map plotting.
"""

from __future__ import annotations

from typing import Any, Dict

import geopandas as gpd
import networkx as nx
import osmnx as ox

from map_core.core.util import has_data, log_progress


def _get_plot_params(
    layer_style: Dict[str, Any], layer_name: str = None
) -> Dict[str, Any]:
    """Extract plotting parameters from a layer's style dictionary.

    Args:
        layer_style: Dictionary containing the layer's style parameters
        layer_name: Optional name of the layer (used for special cases like 'streets')
    """
    params = {
        "edgecolor": layer_style.get("edgecolor", "#000000"),
        "linewidth": layer_style.get("linewidth", 0.5),
        "alpha": layer_style.get("alpha", 1.0),
        "hatch": layer_style.get("hatch", None),
        "linestyle": layer_style.get("linestyle", "-"),
        "zorder": layer_style.get("zorder", 1),
    }
    # Only add facecolor if it's not the streets layer or if layer_name is not specified
    if layer_name is None or layer_name != "streets":
        params["facecolor"] = layer_style.get("facecolor", "#000000")
    return params


def plot_water_layer(
    ax: Any,
    data: gpd.GeoDataFrame,
    layer_style: Dict[str, Any],
) -> None:
    """Plot water layer features.

    Water-area polygons are first dissolved + seam-healed into one connected
    body (see :func:`map_core.core.water.clean_water_layer`) so the river reads
    as a single shape instead of many loose riverbank fragments. Linear
    waterways are kept separately for typed stroke rendering.
    """
    if not has_data(data):
        return

    cleaned = _clean_water_for_plot(data, layer_style)

    params = _get_plot_params(layer_style, "water")
    from map_core.core.plot import plot_map_layer
    plot_map_layer(
        ax,
        "water",
        cleaned,
        params["facecolor"],
        params["edgecolor"],
        params["linewidth"],
        params["alpha"],
        hatch=params["hatch"],
        zorder=params["zorder"],
    )


def _clean_water_for_plot(
    data: gpd.GeoDataFrame, layer_style: Dict[str, Any]
) -> gpd.GeoDataFrame:
    """Dissolve + heal water areas, then recombine with waterways for plotting.

    Falls back to the raw data if anything goes wrong (water is best-effort and
    must never crash a render).
    """
    import pandas as pd

    from map_core.core.water import clean_water_layer

    try:
        seam = float(layer_style.get("seam_heal", 0.15) or 0.0)
        min_area = float(layer_style.get("min_size_threshold", 0) or 0.0)
        min_w = float(layer_style.get("min_width", 0) or 0.0)
        areas, lines = clean_water_layer(
            data,
            seam_heal_m=seam,
            min_area_m2=min_area,
            min_width_m=min_w,
        )
        src_crs = getattr(data, "crs", None)
        parts = []
        crs = src_crs
        if areas is not None and len(areas):
            a = areas.copy()
            a["waterway"] = ""
            parts.append(a[["geometry", "waterway"]])
            crs = a.crs
        if lines is not None and len(lines):
            parts.append(lines[["geometry", "waterway"]])
            crs = lines.crs
        if not parts:
            return data
        return gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), crs=crs
        )
    except Exception as e:  # noqa: BLE001 - water is best-effort
        log_progress(f"[water] clean_water_layer failed, plotting raw: {e}")
        return data


def plot_streets_layer(
    ax: Any,
    data: Any,  # Can be GeoDataFrame or NetworkX graph
    layer_style: Dict[str, Any],
) -> None:
    """Plot streets layer (network graph or edges GeoDataFrame).

    Args:
        ax: Matplotlib axes
        data: Street network (NetworkX graph) or edges GeoDataFrame
        layer_style: Layer style configuration
    """
    if not has_data(data):
        return

    params = _get_plot_params(layer_style, "streets")
    from map_core.core.plot import plot_map_layer
    # Streets are lines, so we explicitly set facecolor to none.
    plot_map_layer(
        ax,
        "streets",
        data,
        facecolor="none",
        edgecolor=params["edgecolor"],
        linewidth=params["linewidth"],
        alpha=params["alpha"],
        hatch=params["hatch"],
        zorder=params["zorder"],
        layer_style=layer_style,
    )


def plot_green_layer(
    ax: Any,
    data: gpd.GeoDataFrame,
    layer_style: Dict[str, Any],
) -> None:
    """Plot green layer features.

    Args:
        ax: Matplotlib axes
        data: Green GeoDataFrame
        layer_style: Layer style configuration
    """
    if not has_data(data):
        return

    params = _get_plot_params(layer_style, "green")
    from map_core.core.plot import plot_map_layer
    plot_map_layer(
        ax,
        "green",
        data,
        params["facecolor"],
        params["edgecolor"],
        params["linewidth"],
        params["alpha"],
        hatch=params["hatch"],
        zorder=params["zorder"],
    )


def plot_generic_layer(
    ax: Any,
    data: gpd.GeoDataFrame,
    layer_name: str,
    layer_style: Dict[str, Any],
) -> None:
    """Plot a generic layer (for any layer not handled by specific functions).

    Args:
        ax: Matplotlib axes
        data: Layer GeoDataFrame
        layer_name: Name of the layer
        layer_style: Layer style configuration
    """
    if not has_data(data):
        return

    params = _get_plot_params(layer_style, layer_name)
    from map_core.core.plot import plot_map_layer
    plot_map_layer(
        ax,
        layer_name,
        data,
        params["facecolor"],
        params["edgecolor"],
        params["linewidth"],
        params["alpha"],
        hatch=params["hatch"],
        linestyle=params["linestyle"],
        zorder=params["zorder"],
    )


def plot_coastline_layer(
    ax: Any,
    data: gpd.GeoDataFrame,
    layer_style: Dict[str, Any],
) -> None:
    """Plot coastline lines as a bottom overlay.

    Coastlines are LineString features from an external shapefile
    (see :mod:`map_core.core.coastline`); there is nothing to
    dissolve, so we simply delegate to the generic line renderer
    using the layer style (which carries a low zorder by default).
    """
    plot_generic_layer(ax, data, "coastline", layer_style)
