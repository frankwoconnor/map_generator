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

    Args:
        ax: Matplotlib axes
        data: Water GeoDataFrame
        layer_style: Layer style configuration
    """
    if not has_data(data):
        return

    params = _get_plot_params(layer_style, "water")
    from map_core.core.plot import plot_map_layer
    plot_map_layer(
        ax,
        "water",
        data,
        params["facecolor"],
        params["edgecolor"],
        params["linewidth"],
        params["alpha"],
        hatch=params["hatch"],
        zorder=params["zorder"],
    )


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