"""Debug visualisation: per-layer, tag-color-coded diagnostic maps.

Rendered on demand (``--debug-legends`` / studio "Generate Debug Legends") to
inspect which features each layer contains. Extracted from ``plot.py`` because
it is a self-contained diagnostics subsystem; the production renderer
(:mod:`map_core.core.plot`) stays focused on final output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches  # noqa: F401  (used by plotters)
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401

from map_core.core.util import has_data, log_progress


def generate_debug_legend(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    figure_size: List[float] = None,
    figure_dpi: int = 150,
    background_color: str = "white",
) -> Tuple[Any, Any]:
    """Generate a standalone legend showing feature types for a layer.

    Args:
        layer_name: Name of the layer
        data: The layer data
        layer_style: Style configuration (not used, for consistency)
        figure_size: Figure size
        figure_dpi: DPI
        background_color: Background color

    Returns:
        (fig, ax) tuple with the legend figure
    """
    if figure_size is None:
        figure_size = [8, 6]

    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    ax.set_facecolor(background_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    if not has_data(data):
        ax.text(
            0.5,
            0.5,
            f"No data available for {layer_name}",
            ha="center",
            va="center",
            fontsize=12,
        )
        return fig, ax

    legend_handles = []
    legend_labels = []

    try:
        # Collect feature types without plotting
        if layer_name == "water":
            _collect_water_legend(data, legend_handles, legend_labels)
        elif layer_name == "streets":
            _collect_streets_legend(data, legend_handles, legend_labels)
        elif layer_name == "buildings":
            _collect_buildings_legend(data, legend_handles, legend_labels)
        elif layer_name == "green":
            _collect_green_legend(data, legend_handles, legend_labels)

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                title=f"{layer_name.capitalize()} Feature Types",
                loc="center",
                fontsize=10,
                title_fontsize=14,
                frameon=True,
                fancybox=True,
                shadow=True,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"No feature types found for {layer_name}",
                ha="center",
                va="center",
                fontsize=12,
            )

    except Exception as e:
        log_progress(f"Error generating debug legend for {layer_name}: {e}")
        ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha="center", va="center", fontsize=10)

    return fig, ax


def generate_debug_map(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    figure_size: List[float] = None,
    figure_dpi: int = 150,
    background_color: str = "white",
    margin: float = 0.05,
) -> Tuple[Any, Any]:
    """Generate a debug map showing features color-coded by their tags/types.

    This function creates a map where each unique feature type is rendered in a distinct color,
    making it easy to see what elements are present on each layer. This overrides user color
    preferences to ensure maximum visual distinction between feature types.

    Args:
        layer_name: Name of the layer (e.g., 'water', 'streets', 'buildings', 'green')
        data: The layer data (GeoDataFrame or NetworkX graph)
        layer_style: Style configuration for the layer (used for reference, colors overridden)
        bbox: Optional bounding box (west, south, east, north)
        figure_size: Figure size for the map
        figure_dpi: DPI for the map figure
        background_color: Background color for the map
        margin: Margin around the map

    Returns:
        (fig, ax) tuple with the debug map figure
    """
    if figure_size is None:
        figure_size = [10, 10]

    # Use the standard setup function like regular layers
    fig, ax = setup_figure_and_axes(
        figure_size, figure_dpi, background_color, margin, transparent=False
    )

    if not has_data(data):
        ax.text(
            0.5,
            0.5,
            f"No data available for {layer_name}",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        return fig, ax

    legend_handles = []
    legend_labels = []

    try:
        # Set bounding box before plotting
        if bbox:
            west, south, east, north = bbox
            # ax.set_xlim(west, east)
            # ax.set_ylim(south, north)

        # Plot features
        if layer_name == "water":
            _plot_water_debug(ax, data, legend_handles, legend_labels)
        elif layer_name == "streets":
            _plot_streets_debug(ax, data, legend_handles, legend_labels)
        elif layer_name == "buildings":
            _plot_buildings_debug(ax, data, legend_handles, legend_labels)
        elif layer_name == "green":
            _plot_green_debug(ax, data, legend_handles, legend_labels)

    except Exception as e:
        log_progress(f"Error generating debug map for {layer_name}: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error generating debug map:\n{str(e)}",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )

    return fig, ax


def _plot_water_debug(ax, data, legend_handles, legend_labels):
    """Plot water features with color-coded tags for debugging."""
    if not hasattr(data, "geometry"):
        return

    # Use tab20 colormap for distinct colors
    cmap = cm.get_cmap("tab20")
    feature_types = {}
    color_idx = 0

    # Collect all unique feature types
    for tag in ["natural", "waterway"]:
        if tag in data.columns:
            for val in data[tag].dropna().unique():
                feature_key = f"{tag}={val}"
                if feature_key not in feature_types:
                    feature_types[feature_key] = cmap(color_idx % 20)
                    color_idx += 1

    # Plot each feature type with its assigned color
    for feature_key, color in feature_types.items():
        tag, val = feature_key.split("=")
        mask = data[tag] == val
        subset = data[mask]

        if len(subset) > 0:
            # Determine if polygon or line
            geom_types = subset.geometry.geom_type.unique()
            is_polygon = any(gt in ["Polygon", "MultiPolygon"] for gt in geom_types)

            if is_polygon:
                subset.plot(
                    ax=ax, facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.7
                )
            else:
                subset.plot(ax=ax, color=color, linewidth=2, alpha=0.8)

            # Add to legend
            if is_polygon:
                legend_handles.append(
                    mpatches.Rectangle(
                        (0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5
                    )
                )
            else:
                legend_handles.append(mpatches.Patch(color=color))
            legend_labels.append(feature_key)


def _plot_streets_debug(ax, data, legend_handles, legend_labels):
    """Plot streets with color-coded highway types for debugging."""
    try:
        edges = None
        if isinstance(data, gpd.GeoDataFrame):
            edges = data
        elif isinstance(data, (nx.MultiDiGraph, nx.MultiGraph, nx.DiGraph, nx.Graph)):
            _, edges = ox.graph_to_gdfs(data)
        else:
            log_progress(
                f"Warning: Expected NetworkX graph or GeoDataFrame for streets debug, got {type(data)}"
            )
            return

        if not has_data(edges):
            log_progress("[debug] No street edges to plot.")
            return

        log_progress(f"[debug] Street edges columns: {edges.columns.tolist()}")
        if "highway" in edges.columns:
            log_progress(
                f"[debug] Unique highway values: {edges['highway'].dropna().unique().tolist()}"
            )

        if "highway" not in edges.columns or edges["highway"].dropna().empty:
            log_progress(
                "[debug] 'highway' column not found or empty in street edges. Plotting all streets in gray."
            )
            edges.plot(ax=ax, color="gray", linewidth=1, alpha=0.8)
            legend_handles.append(mpatches.Patch(color="gray"))
            legend_labels.append("Streets (no type info)")
            return

        # Use Set3 colormap for street types
        cmap = cm.get_cmap("Set3")
        highway_types = {}
        color_idx = 0

        # Collect all unique highway types
        for hw in edges["highway"].dropna().unique():
            if isinstance(hw, list):
                for h in hw:
                    if h not in highway_types:
                        highway_types[h] = cmap(color_idx % 12)
                        color_idx += 1
            else:
                if hw not in highway_types:
                    highway_types[hw] = cmap(color_idx % 12)
                    color_idx += 1

        # Plot each highway type with its assigned color
        for hw_type, color in sorted(highway_types.items()):
            # Filter edges that have this highway type
            mask = edges["highway"].apply(
                lambda x: hw_type in x if isinstance(x, list) else x == hw_type
            )
            subset = edges[mask]

            if len(subset) > 0:
                subset.plot(ax=ax, color=color, linewidth=1.5, alpha=0.8)
                legend_handles.append(mpatches.Patch(color=color))
                legend_labels.append(hw_type)

    except Exception as e:
        log_progress(f"Warning: Could not plot streets debug: {e}")


def _plot_buildings_debug(ax, data, legend_handles, legend_labels):
    """Plot buildings with color-coded types for debugging."""
    if not hasattr(data, "geometry"):
        return

    # Use Paired colormap for building types
    cmap = cm.get_cmap("Paired")
    building_types = {}
    color_idx = 0

    # Collect building types
    if "building" in data.columns:
        for val in data["building"].dropna().unique():
            if val not in building_types:
                building_types[val] = cmap(color_idx % 12)
                color_idx += 1

    # If no building column or all same, just use single color
    if not building_types:
        data.plot(
            ax=ax, facecolor="#ff7f0e", edgecolor="black", linewidth=0.3, alpha=0.7
        )
        legend_handles.append(
            mpatches.Rectangle((0, 0), 1, 1, facecolor="#ff7f0e", edgecolor="black")
        )
        legend_labels.append("buildings")
        return

    # Plot each building type with its assigned color
    for btype, color in sorted(building_types.items()):
        mask = data["building"] == btype
        subset = data[mask]

        if len(subset) > 0:
            subset.plot(
                ax=ax, facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.7
            )
            legend_handles.append(
                mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black")
            )
            legend_labels.append(f"building={btype}")


def _plot_green_debug(ax, data, legend_handles, legend_labels):
    """Plot green spaces with color-coded tags for debugging."""
    if not hasattr(data, "geometry"):
        return

    # Use Set2 colormap for green spaces
    cmap = cm.get_cmap("Set2")
    feature_types = {}
    color_idx = 0

    # Collect all unique feature types
    for tag in ["leisure", "landuse", "natural"]:
        if tag in data.columns:
            for val in data[tag].dropna().unique():
                feature_key = f"{tag}={val}"
                if feature_key not in feature_types:
                    feature_types[feature_key] = cmap(color_idx % 8)
                    color_idx += 1

    # If no tags, just plot everything
    if not feature_types:
        data.plot(
            ax=ax, facecolor="#90ee90", edgecolor="black", linewidth=0.3, alpha=0.7
        )
        legend_handles.append(
            mpatches.Rectangle((0, 0), 1, 1, facecolor="#90ee90", edgecolor="black")
        )
        legend_labels.append("green spaces")
        return

    # Plot each feature type with its assigned color
    for feature_key, color in feature_types.items():
        tag, val = feature_key.split("=")
        mask = data[tag] == val
        subset = data[mask]

        if len(subset) > 0:
            subset.plot(
                ax=ax, facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.7
            )
            legend_handles.append(
                mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black")
            )
            legend_labels.append(feature_key)
