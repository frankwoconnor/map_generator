from __future__ import annotations

import io
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib as mpl

# Non-interactive backend is required for threaded/background SVG generation on
# macOS and for headless environments.
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib import cm
from matplotlib.collections import LineCollection, PatchCollection
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from map_core.core.util import has_data, log_progress

# Creative output (crop mask / frame / title) lives in its own module; re-export
# here so existing ``from map_core.core.plot import apply_creative_output`` keeps
# working while the implementation stays focused.
from map_core.core.creative import (  # noqa: F401
    _format_coords_text,
    _rounded_rect_verts,
    _substitute_text,
    apply_creative_output,
)
# Debug diagnostic maps/legends live in their own module; re-export for the
# existing ``from map_core.core.plot import generate_debug_map`` call sites.
from map_core.core.debug import (  # noqa: F401
    generate_debug_legend,
    generate_debug_map,
)


def setup_figure_and_axes(
    figure_size: List[float],
    figure_dpi: int,
    background_color: str,
    margin: float,
    transparent: bool = False,
) -> Tuple[Any, Any]:
    """Set up matplotlib figure and axes for map plotting.

    Args:
        figure_size: List of [width, height] in inches
        figure_dpi: Dots per inch
        background_color: Figure and axes background color
        margin: Axes margin fraction
        transparent: If True, use transparent background

    Returns:
        (fig, ax) tuple
    """
    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    # Set background handling: when transparent, use 'none' facecolors and allow savefig(transparent=True)
    if transparent:
        # Ensure both figure and axes patches are transparent
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    else:
        ax.set_facecolor(background_color)
    ax.set_axis_off()
    ax.margins(margin)
    fig.tight_layout(pad=0)
    return fig, ax


def setup_figure_and_axes_isolated(
    figure_size: List[float],
    figure_dpi: int,
    background_color: str,
    margin: float,
    transparent: bool = False,
) -> Tuple[Any, Any]:
    """Create a figure WITHOUT pyplot's global figure registry.

    Uses the object-oriented API only: the figure is never registered as
    pyplot's "current" figure, so it is safe to create/draw/save concurrently
    from multiple threads (each thread gets a fully isolated figure).

    Args:
        figure_size: List of [width, height] in inches
        figure_dpi: Dots per inch
        background_color: Figure and axes background color
        margin: Axes margin fraction
        transparent: If True, use transparent background

    Returns:
        (fig, ax) tuple
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figure_size, dpi=figure_dpi)
    FigureCanvasAgg(fig)  # attach a canvas so Figure.savefig works
    ax = fig.add_subplot(111)
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    else:
        ax.set_facecolor(background_color)
    ax.set_axis_off()
    ax.margins(margin)
    try:
        fig.tight_layout(pad=0)
    except Exception:
        pass
    return fig, ax


def create_map_legend(
    style: Dict[str, Any],
    layer_data: Dict[str, Any],
    parent: bool = False,
    figure_size: List[float] = None,
    figure_dpi: int = 300,
    background_color: str = "white",
    margin: float = 0.05,
    transparent: bool = False,
) -> Tuple[Any, Any]:
    """Create a legend for the map based on the style configuration and layer data.

    Args:
        style: The style configuration dictionary
        layer_data: Dictionary containing the GeoDataFrames for each layer
        parent: If True, the legend will be created as a parent legend
        figure_size: List of [width, height] in inches
        figure_dpi: Dots per inch
        background_color: Figure and axes background color
        margin: Axes margin fraction
        transparent: If True, use transparent background

    Returns:
        (fig, ax)
    """
    if figure_size is None:
        figure_size = [10, 10]
    return setup_figure_and_axes(
        figure_size, figure_dpi, background_color, margin, transparent
    )


def _add_legend_item(
    handles: List,
    labels: List,
    color: str,
    edgecolor: str,
    linewidth: float,
    label: str,
    is_line: bool = False,
) -> None:
    """Helper to add a legend item to the handles and labels lists.

    Args:
        handles: List to append the patch/line to
        labels: List to append the label to
        color: Fill color for the patch
        edgecolor: Edge color for the patch
        linewidth: Line width
        label: Text label
        is_line: If True, create a line representation instead of patch
    """
    if is_line:
        item = mpatches.Rectangle(
            (0, 0), 1, 0.1, facecolor=color, edgecolor="none", label=label
        )
    else:
        item = mpatches.Patch(
            facecolor=color, edgecolor=edgecolor, linewidth=linewidth, label=label
        )
    handles.append(item)
    labels.append(label)


def generate_layer_legend(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    figure_size: List[float] = None,
    figure_dpi: int = 150,
    background_color: str = "white",
) -> Tuple[Any, Any]:
    """Generate a standalone legend showing unique features for a layer.

    Args:
        layer_name: Name of the layer (e.g., 'water', 'streets', 'buildings')
        data: The layer data (GeoDataFrame or NetworkX graph)
        layer_style: Style configuration for the layer
        figure_size: Figure size for the legend
        figure_dpi: DPI for the legend figure
        background_color: Background color for the legend

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

    legend_handles = []
    legend_labels = []

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

    try:
        if layer_name == "water":
            # Extract unique water feature types
            if hasattr(data, "geometry"):
                # Group by natural/waterway tags if available
                unique_features = {}
                if "natural" in data.columns:
                    for val in data["natural"].dropna().unique():
                        unique_features[f"Natural: {val}"] = layer_style.get(
                            "facecolor", "#a6cee3"
                        )
                if "waterway" in data.columns:
                    for val in data["waterway"].dropna().unique():
                        unique_features[f"Waterway: {val}"] = layer_style.get(
                            "edgecolor", "#a6cee3"
                        )

                # If no tags, just show generic water
                if not unique_features:
                    unique_features = {
                        "Water features": layer_style.get("facecolor", "#a6cee3")
                    }

                for label, color in unique_features.items():
                    _add_legend_item(
                        legend_handles,
                        legend_labels,
                        color,
                        layer_style.get("edgecolor", color),
                        layer_style.get("linewidth", 0.5),
                        label,
                    )

        elif layer_name == "streets":
            # Extract unique highway types from street network
            try:
                # Convert graph to edges GeoDataFrame
                if hasattr(data, "edges"):
                    nodes, edges = ox.graph_to_gdfs(data)
                    if "highway" in edges.columns:
                        unique_highways = edges["highway"].dropna().unique()
                        # Handle list values
                        highway_types = set()
                        for hw in unique_highways:
                            if isinstance(hw, list):
                                highway_types.update(hw)
                            else:
                                highway_types.add(hw)

                        for hw_type in sorted(highway_types):
                            _add_legend_item(
                                legend_handles,
                                legend_labels,
                                layer_style.get("edgecolor", "#000000"),
                                "none",
                                0.5,
                                hw_type.capitalize(),
                                is_line=True,
                            )
            except Exception as e:
                log_progress(f"Warning: Could not extract street types: {e}")
                _add_legend_item(
                    legend_handles,
                    legend_labels,
                    layer_style.get("edgecolor", "#000000"),
                    "none",
                    0.5,
                    "Streets",
                    is_line=True,
                )

        elif layer_name == "buildings":
            # Show building color scheme
            auto_mode = layer_style.get("auto_style_mode", "manual")
            if auto_mode == "manual":
                color = layer_style.get("manual_color_settings", {}).get(
                    "facecolor", "#000000"
                )
                _add_legend_item(
                    legend_handles,
                    legend_labels,
                    color,
                    layer_style.get("edgecolor", "#000000"),
                    layer_style.get("linewidth", 0.5),
                    "Buildings",
                )
            elif auto_mode in ["auto_size", "auto_distance"]:
                # Show the palette colors
                from map_core.core.palettes import load_palettes

                palette_key = (
                    "auto_size_palette"
                    if auto_mode == "auto_size"
                    else "auto_distance_palette"
                )
                palette_name = layer_style.get(palette_key, "")
                palettes = load_palettes()
                colors = (
                    palettes.get(palette_name, []) if isinstance(palettes, dict) else []
                )

                if colors:
                    colors_reversed = list(colors)[::-1]
                    labels = ["Smallest" if auto_mode == "auto_size" else "Farthest"]
                    labels.extend([""] * (len(colors_reversed) - 2))
                    labels.append("Largest" if auto_mode == "auto_size" else "Nearest")

                    for color, label in zip(colors_reversed, labels):
                        _add_legend_item(
                            legend_handles,
                            legend_labels,
                            color,
                            layer_style.get("edgecolor", "#000000"),
                            layer_style.get("linewidth", 0.5),
                            label if label else "",
                        )
            elif auto_mode == "manual_floorsize":
                categories = layer_style.get("size_categories", [])
                for cat in categories:
                    min_area = cat.get("min_area", 0)
                    max_area = cat.get("max_area", "∞")
                    label = f"{min_area}-{max_area} m²"
                    _add_legend_item(
                        legend_handles,
                        legend_labels,
                        cat.get("facecolor", "#000000"),
                        layer_style.get("edgecolor", "#000000"),
                        layer_style.get("linewidth", 0.5),
                        label,
                    )

        elif layer_name == "green":
            # Check for leisure/landuse/natural tags
            unique_features = {}
            if hasattr(data, "geometry"):
                for tag in ["leisure", "landuse", "natural"]:
                    if tag in data.columns:
                        for val in data[tag].dropna().unique():
                            unique_features[f"{tag.capitalize()}: {val}"] = (
                                layer_style.get("facecolor", "#b2df8a")
                            )

                if not unique_features:
                    unique_features = {
                        "Green spaces": layer_style.get("facecolor", "#b2df8a")
                    }

                for label, color in unique_features.items():
                    _add_legend_item(
                        legend_handles,
                        legend_labels,
                        color,
                        layer_style.get("edgecolor", color),
                        layer_style.get("linewidth", 0.5),
                        label,
                    )

        # Create the legend
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title=f"{layer_name.capitalize()} Legend",
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
                f"No legend items for {layer_name}",
                ha="center",
                va="center",
                fontsize=12,
            )

    except Exception as e:
        log_progress(f"Error generating legend for {layer_name}: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error generating legend:\n{str(e)}",
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
        )

    return fig, ax


def add_map_legend(ax: Any, layer_data: Dict[str, Any], style: Dict[str, Any]) -> None:
    """Add a legend to the map based on the style configuration and layer data.

    Args:
        ax: The matplotlib axes to add the legend to
        layer_data: Dictionary containing the GeoDataFrames for each layer
        style: The style configuration dictionary
    """
    if not style.get("legend", {}).get("enabled", True):
        return

    legend_config = style.get("legend", {})
    legend_handles = []

    # Add water features to legend
    if "water" in layer_data and has_data(layer_data["water"]):
        water_style = style.get("layers", {}).get("water", {})

        # Add water bodies (polygons)
        water_patches = []
        if "filters" in water_style and "natural" in water_style["filters"]:
            for water_type in water_style["filters"]["natural"]:
                if water_type in ["lake", "pond", "water", "reservoir"]:
                    patch = mpatches.Patch(
                        facecolor=water_style.get("facecolor", "#479711"),
                        edgecolor=water_style.get("edgecolor", "#f70202"),
                        linewidth=water_style.get("linewidth", 1.0),
                        label=f"{water_type.capitalize()}",
                    )
                    water_patches.append(patch)

        # Add waterways (lines)
        if "filters" in water_style and "waterway" in water_style["filters"]:
            for waterway_type in water_style["filters"]["waterway"]:
                if waterway_type in ["river", "stream", "canal"]:
                    line = mpatches.Rectangle(
                        (0, 0),
                        1,
                        0.2,  # width, height
                        facecolor=water_style.get("edgecolor", "#479711"),
                        edgecolor="none",
                        label=f"{waterway_type.capitalize()}",
                    )
                    water_patches.append(line)

        if water_patches:
            legend_handles.extend(water_patches)

    # Add green spaces to legend
    if "green" in layer_data and has_data(layer_data["green"]):
        green_style = style.get("layers", {}).get("green", {})
        if green_style.get("enabled", False):
            patch = mpatches.Patch(
                facecolor=green_style.get("facecolor", "#4CAF50"),
                edgecolor=green_style.get("edgecolor", "#000000"),
                linewidth=green_style.get("linewidth", 0.5),
                label="Green Space",
            )
            legend_handles.append(patch)

    # Add buildings to legend
    if "buildings" in layer_data and has_data(layer_data["buildings"]):
        building_style = style.get("layers", {}).get("buildings", {})
        if building_style.get("enabled", False):
            patch = mpatches.Patch(
                facecolor=building_style.get("facecolor", "#808080"),
                edgecolor=building_style.get("edgecolor", "#000000"),
                linewidth=building_style.get("linewidth", 0.5),
                label="Buildings",
            )
            legend_handles.append(patch)

    # Add streets to legend
    if "streets" in layer_data and has_data(layer_data["streets"]):
        street_style = style.get("layers", {}).get("streets", {})
        if street_style.get("enabled", False):
            line = mpatches.Rectangle(
                (0, 0),
                1,
                0.1,  # width, height
                facecolor=street_style.get("edgecolor", "#000000"),
                edgecolor="none",
                label="Streets",
            )
            legend_handles.append(line)

    # Only add legend if we have items to show
    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            title=legend_config.get("title", "Map Legend"),
            loc=legend_config.get("position", "lower right"),
            fontsize=legend_config.get("font_size", 10),
            title_fontsize=legend_config.get("title_size", 12),
            frameon=legend_config.get("fancybox", True),
            fancybox=legend_config.get("fancybox", True),
            shadow=legend_config.get("shadow", True),
            borderpad=legend_config.get("borderpad", 0.8),
            labelspacing=legend_config.get("labelspacing", 0.8),
            handlelength=legend_config.get("handlelength", 1.5),
            handletextpad=legend_config.get("handletextpad", 0.8),
            borderaxespad=legend_config.get("borderaxespad", 0.8),
        )

        # Set legend background color and transparency
        frame = legend.get_frame()
        frame.set_facecolor(legend_config.get("background_color", "#ffffff"))
        frame.set_edgecolor(legend_config.get("edge_color", "#000000"))
        frame.set_alpha(legend_config.get("alpha", 0.9))


# Default relative line widths per highway class (multipliers of the streets
# layer's base ``linewidth``). Override with an explicit ``linewidths`` dict in
# the streets layer style (absolute widths, interpreted like ``linewidth``).
DEFAULT_HIGHWAY_WIDTH_MULTIPLIERS: Dict[str, float] = {
    "motorway": 3.2,
    "motorway_link": 2.2,
    "trunk": 2.8,
    "trunk_link": 2.0,
    "primary": 2.4,
    "primary_link": 1.7,
    "secondary": 1.9,
    "secondary_link": 1.4,
    "tertiary": 1.5,
    "tertiary_link": 1.2,
    "residential": 1.1,
    "unclassified": 1.1,
    "road": 1.1,
    "living_street": 1.0,
    "service": 0.85,
    "pedestrian": 1.0,
    "track": 0.85,
    "footway": 0.7,
    "path": 0.7,
    "cycleway": 0.7,
    "steps": 0.6,
}

# Absolute minimum stroke width per class (points). These guarantee that major
# roads stay legible even when the layer's base ``linewidth`` is very small
# (e.g. 0.1) — the multiplier alone would shrink every road to a hairline.
MIN_HIGHWAY_LINEWIDTH: Dict[str, float] = {
    "motorway": 0.55,
    "motorway_link": 0.4,
    "trunk": 0.5,
    "trunk_link": 0.38,
    "primary": 0.45,
    "primary_link": 0.35,
    "secondary": 0.4,
    "secondary_link": 0.3,
    "tertiary": 0.35,
    "tertiary_link": 0.3,
    "residential": 0.22,
    "unclassified": 0.22,
    "road": 0.22,
    "living_street": 0.2,
    "service": 0.15,
    "footway": 0.15,
    "path": 0.15,
    "cycleway": 0.15,
    "track": 0.15,
    "steps": 0.12,
}


def _highway_widths_for_style(
    base_width: float, layer_style: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Return a ``highway`` value -> absolute line width map for streets.

    A layer style may provide an explicit ``linewidths`` dict (absolute values,
    like ``linewidth``). Without one, the default class multipliers are scaled
    by the layer's base ``linewidth`` (floored per class so major roads never
    shrink to invisibility). A ``category_widths`` dict (keyed by street
    category, see ``map_core.core.street_styling.STREET_CATEGORIES``) layers on
    top of either source and applies the same width to every highway value in
    the category.
    """
    overrides = (layer_style or {}).get("linewidths")
    explicit_overrides = isinstance(overrides, dict) and bool(overrides)
    if explicit_overrides:
        widths = {str(k): float(v) for k, v in overrides.items()}
    else:
        base = float(base_width) if base_width else 0.5
        widths = {}
        for highway_value, multiplier in DEFAULT_HIGHWAY_WIDTH_MULTIPLIERS.items():
            width = multiplier * base
            floor = MIN_HIGHWAY_LINEWIDTH.get(highway_value)
            if floor is not None:
                width = max(width, floor)
            widths[highway_value] = width

    category_widths = (layer_style or {}).get("category_widths")
    if isinstance(category_widths, dict) and category_widths:
        from map_core.core.street_styling import STREET_CATEGORIES

        for cat_id, _label, highways in STREET_CATEGORIES:
            category_width = category_widths.get(cat_id)
            if category_width is None:
                continue
            try:
                w = float(category_width)
            except (TypeError, ValueError):
                continue
            for h in highways:
                # A per-highway explicit override is more specific.
                if explicit_overrides and h in widths:
                    continue
                widths[h] = w
    return widths


def _plot_streets_with_class_widths(
    ax: Any,
    edges: Any,
    edgecolor: Any,
    linewidth: float,
    alpha: float,
    zorder: int,
    layer_style: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot street edges grouped by highway class so major roads stand out.

    Falls back to a single uniform draw when there is no usable ``highway``
    column, so unusual schemas still render. In ``by_class`` style mode each
    class is colored from the street theme (see ``map_core.core.street_styling``)
    even when ``variable_widths`` is off (uniform stroke width, distinct
    colors); otherwise every road uses the layer's ``edgecolor``. When both the
    color mode is ``uniform`` and ``variable_widths`` is explicitly False the
    layer draws as one uniform stroke (legacy behavior).
    """
    if not has_data(edges) or not hasattr(edges, "plot"):
        return

    style = layer_style or {}
    by_class = style.get("style_mode") == "by_class"
    variable_widths = style.get("variable_widths", True)

    # Color mode is uniform AND the class width hierarchy is disabled -> draw
    # everything as one uniform stroke (legacy behavior).
    if not by_class and not variable_widths:
        edges.plot(
            ax=ax, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder
        )
        return

    if "highway" not in edges.columns:
        edges.plot(
            ax=ax, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder
        )
        return

    # Per-class widths are only resolved when the hierarchy is enabled; in
    # by_class mode without variable widths every class keeps the base width.
    widths = _highway_widths_for_style(linewidth, style) if variable_widths else None

    # Class colors only apply in by_class mode; otherwise edgecolor is used
    # for every road.
    colors: Dict[str, str] = {}
    if by_class:
        try:
            from map_core.core.street_styling import street_class_colors

            colors = street_class_colors(edgecolor, style)
        except Exception as e:
            log_progress(
                f"Warning: street class colors unavailable ({e}); using uniform color."
            )

    # Each unique highway value may itself be a list (an edge can carry several).
    drawn = set()
    for value in edges["highway"].dropna().unique():
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for highway_value in values:
            if highway_value in drawn:
                continue
            drawn.add(highway_value)
            mask = edges["highway"].apply(
                lambda x, v=highway_value: v in x
                if isinstance(x, (list, tuple, set))
                else x == v
            )
            subset = edges[mask]
            if subset.empty:
                continue
            subset.plot(
                ax=ax,
                color=colors.get(highway_value, edgecolor),
                linewidth=widths.get(highway_value, linewidth) if widths else linewidth,
                alpha=alpha,
                zorder=zorder,
            )


def plot_map_layer(
    ax: Any,
    layer_name: str,
    data: Any,
    facecolor: Any,
    edgecolor: Any,
    linewidth: float,
    alpha: float,
    hatch: Optional[str] = None,
    linestyle: str = "-",
    zorder: int = 1,
    layer_style: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot a street network or generic GeoDataFrame on the given axis.

    Streets are drawn using the edges GeoDataFrame to avoid background overrides.
    Other layers are plotted as polygons/lines via GeoPandas.
    """
    from map_core.core.util import (drop_point_features,  # local import to avoid cycles
                                    has_data, log_progress)

    if not has_data(data):
        return

    # Never render Point/MultiPoint features: they show up as stray circles
    # (POI-like markers) in the SVG. Line/polygon layers pass through and
    # street graphs are untouched by the GeoDataFrame guard.
    data = drop_point_features(data, layer_name)
    if not has_data(data):
        return

    if layer_name == "streets":
        # If data is already an edges GeoDataFrame, plot directly (with a road
        # class hierarchy when the highway tag column is available).
        if hasattr(data, "geometry") and not hasattr(data, "graph"):
            try:
                _plot_streets_with_class_widths(
                    ax,
                    data,
                    edgecolor,
                    linewidth,
                    alpha,
                    zorder,
                    layer_style=layer_style,
                )
                return
            except Exception as e:
                log_progress(
                    f"Warning: failed to plot streets from GeoDataFrame directly: {e}"
                )
        # Otherwise, assume a NetworkX graph and convert to edges GDF
        try:
            nodes_gdf, gdf_edges = ox.graph_to_gdfs(data)
            if has_data(gdf_edges):
                gdf_edges.plot(
                    ax=ax,
                    color=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )
                return
        except Exception as e:
            log_progress(
                f"Warning: failed to plot streets via edges GDF ({e}); falling back to ox.plot_graph."
            )
            try:
                ox.plot_graph(
                    data,
                    ax=ax,
                    show=False,
                    close=False,
                    edge_color=edgecolor,
                    edge_linewidth=linewidth,
                    edge_alpha=alpha,
                    node_size=0,
                )
            except Exception as e2:
                log_progress(f"Warning: ox.plot_graph failed as well: {e2}")
            return
    else:
        # For water layers, handle different geometry types with smart width calculation
        if layer_name == "water" and has_data(data):
            _plot_water_features(
                ax, data, facecolor, edgecolor, linewidth, alpha, hatch, zorder
            )
        else:
            # Default plotting for other layers
            data.plot(
                ax=ax,
                fc=facecolor,
                ec=edgecolor,
                lw=linewidth,
                alpha=alpha,
                hatch=hatch,
                zorder=zorder,
            )


def _parse_width_tag(value: Any) -> Optional[float]:
    """Parse an OSM width tag (e.g. '5', '5m', '5 m') into meters."""
    if value is None or pd.isna(value):
        return None
    try:
        width_str = str(value).lower().replace("m", "").replace(" ", "").strip()
        return float(width_str)
    except (ValueError, AttributeError):
        return None


def _line_collection_for_waterways(
    gdf: Any,
    *,
    default_width_m: float,
    min_linewidth: float,
    meters_per_point: float,
    color: str,
    alpha: float,
    zorder: int,
) -> Optional[Any]:
    """Build a LineCollection for waterway lines with per-feature widths.

    Returns None if no valid line geometries are present.
    """
    import numpy as np

    segments: List[Any] = []
    linewidths: List[float] = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        width_m = _parse_width_tag(row.get("width")) or default_width_m
        lw = max(min_linewidth, width_m / meters_per_point)

        if geom.geom_type == "LineString":
            segments.append(np.array(geom.coords))
            linewidths.append(lw)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                if not part.is_empty:
                    segments.append(np.array(part.coords))
                    linewidths.append(lw)

    if not segments:
        return None

    lc = LineCollection(
        segments,
        colors=color,
        linewidths=linewidths,
        alpha=alpha,
        zorder=zorder,
        linestyle="-",
    )
    return lc


def _plot_water_features(
    ax: Any,
    data: Any,
    facecolor: Any,
    edgecolor: Any,
    linewidth: float,
    alpha: float,
    hatch: Optional[str],
    zorder: int,
    show_labels: bool = False,
) -> None:
    """Plot water features with intelligent width-based rendering.

    Handles:
    - Polygon water bodies (lakes, ponds, reservoirs, rivers)
    - Waterways with width tags (rivers, streams, canals)
    - Differentiated styling based on water type
    - Optional labels for named features

    Args:
        ax: Matplotlib axes
        data: Water features GeoDataFrame
        facecolor: Fill color for polygons
        edgecolor: Edge/line color
        linewidth: Base line width
        alpha: Transparency
        hatch: Hatch pattern
        zorder: Draw order
        show_labels: Whether to show labels for named features
    """
    from map_core.core.util import has_data, log_progress

    if not hasattr(data, "geometry"):
        log_progress("Warning: No geometry column found in water data")
        return

    # Make a copy to avoid modifying the original
    gdf = data.copy()

    # Store features with names for labeling
    named_features = []

    # Drop empty and invalid geometries
    gdf = gdf[~gdf.is_empty]
    if not all(gdf.geometry.is_valid):
        log_progress(
            f"Found {sum(~gdf.geometry.is_valid)} invalid geometries in water layer, fixing..."
        )
        gdf.geometry = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.is_valid]

    if gdf.empty:
        return

    # Define water colors based on type (use a nice blue palette)
    water_blue = facecolor if facecolor != "#000000" else "#88ccee"
    river_blue = "#5599cc"
    stream_blue = "#77bbee"

    # Separate polygons and lines
    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]

    # 1. Plot polygon water bodies (lakes, ponds, reservoirs, river polygons)
    if not polys.empty:
        # Differentiate between river polygons and other water bodies
        river_polys = (
            polys[polys.get("water", "") == "river"]
            if "water" in polys.columns
            else pd.DataFrame()
        )
        other_polys = (
            polys[polys.get("water", "") != "river"]
            if "water" in polys.columns
            else polys
        )

        # Plot regular water bodies (lakes, ponds, etc.)
        if not other_polys.empty:
            other_polys.plot(
                ax=ax,
                fc=water_blue,
                ec=edgecolor,
                lw=linewidth,
                alpha=alpha,
                hatch=hatch,
                zorder=zorder,
            )

        # Plot river polygons with slightly different color
        # Track river polygon bounds to avoid plotting overlapping centerlines
        river_poly_bounds = None
        if not river_polys.empty:
            river_polys.plot(
                ax=ax,
                fc=river_blue,
                ec=edgecolor,
                lw=linewidth,
                alpha=alpha,
                hatch=hatch,
                zorder=zorder,
            )
            # Collect bounds of river polygons to filter overlapping centerlines
            try:
                river_poly_bounds = river_polys.total_bounds
            except Exception:
                river_poly_bounds = None

    # 2. Plot waterway lines with width-based rendering
    if not lines.empty:
        # Calculate default scale: 1 point ≈ 5 meters (configurable)
        meters_per_point = 5.0
        default_river_width_m = 10.0
        default_stream_width_m = 3.0
        default_canal_width_m = 8.0

        # Separate by waterway type
        rivers = (
            lines[lines.get("waterway", "") == "river"]
            if "waterway" in lines.columns
            else pd.DataFrame()
        )
        streams = (
            lines[lines.get("waterway", "") == "stream"]
            if "waterway" in lines.columns
            else pd.DataFrame()
        )
        canals = (
            lines[lines.get("waterway", "") == "canal"]
            if "waterway" in lines.columns
            else pd.DataFrame()
        )
        other_waterways = lines[
            ~lines.index.isin(rivers.index)
            & ~lines.index.isin(streams.index)
            & ~lines.index.isin(canals.index)
        ]

        # If river polygons were plotted, filter out river centerlines that overlap
        # with river polygons to avoid misaligned double-rendering (polygon + line)
        if river_poly_bounds is not None and not rivers.empty:
            minx, miny, maxx, maxy = river_poly_bounds
            # Add small buffer to bounds for overlap check
            buf = 0.0001  # ~10m at equator
            rivers = rivers[
                ~rivers.geometry.apply(
                    lambda g: g.bounds[0] < maxx + buf and g.bounds[1] < maxy + buf
                    and g.bounds[2] > minx - buf and g.bounds[3] > miny - buf
                )
            ]

        # Plot rivers, streams, and canals with per-feature width via LineCollection
        line_collections = []

        if not rivers.empty:
            lc = _line_collection_for_waterways(
                rivers,
                default_width_m=default_river_width_m,
                min_linewidth=linewidth,
                meters_per_point=meters_per_point,
                color=river_blue,
                alpha=alpha,
                zorder=zorder,
            )
            if lc is not None:
                line_collections.append(lc)

        if not streams.empty:
            lc = _line_collection_for_waterways(
                streams,
                default_width_m=default_stream_width_m,
                min_linewidth=linewidth * 0.5,
                meters_per_point=meters_per_point,
                color=stream_blue,
                alpha=alpha,
                zorder=zorder,
            )
            if lc is not None:
                line_collections.append(lc)

        if not canals.empty:
            lc = _line_collection_for_waterways(
                canals,
                default_width_m=default_canal_width_m,
                min_linewidth=linewidth,
                meters_per_point=meters_per_point,
                color=river_blue,
                alpha=alpha,
                zorder=zorder,
            )
            if lc is not None:
                line_collections.append(lc)

        if not other_waterways.empty:
            lc = _line_collection_for_waterways(
                other_waterways,
                default_width_m=default_river_width_m,
                min_linewidth=linewidth * 2,
                meters_per_point=meters_per_point,
                color=river_blue,
                alpha=alpha,
                zorder=zorder,
            )
            if lc is not None:
                line_collections.append(lc)

        for lc in line_collections:
            ax.add_collection(lc)

    # 3. Add labels for named features if requested
    if show_labels and "name" in gdf.columns:
        for idx, row in gdf.iterrows():
            if pd.notna(row.get("name")) and row["name"] != "":
                try:
                    # Get centroid for label placement
                    centroid = row.geometry.centroid
                    if centroid and centroid.is_valid:
                        ax.annotate(
                            row["name"],
                            xy=(centroid.x, centroid.y),
                            xytext=(0, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color="#003366",
                            weight="normal",
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor="none",
                                alpha=0.7,
                            ),
                            zorder=zorder + 1,
                        )
                except Exception as e:
                    # Skip labels that fail
                    pass


# Legend-only collection functions (no plotting)
def _collect_water_legend(data, legend_handles, legend_labels):
    """Collect water feature types for legend without plotting."""
    if not hasattr(data, "geometry"):
        return

    cmap = cm.get_cmap("tab20")
    feature_types = {}
    color_idx = 0

    for tag in ["natural", "waterway"]:
        if tag in data.columns:
            for val in data[tag].dropna().unique():
                feature_key = f"{tag}={val}"
                if feature_key not in feature_types:
                    feature_types[feature_key] = cmap(color_idx % 20)
                    color_idx += 1

    for feature_key, color in feature_types.items():
        legend_handles.append(mpatches.Patch(color=color))
        legend_labels.append(feature_key)


def _collect_streets_legend(data, legend_handles, legend_labels):
    """Collect street highway types for legend without plotting."""
    try:
        edges = None
        if isinstance(data, gpd.GeoDataFrame):
            edges = data
        elif isinstance(data, (nx.MultiDiGraph, nx.MultiGraph, nx.DiGraph, nx.Graph)):
            _, edges = ox.graph_to_gdfs(data)
        else:
            log_progress(
                f"Warning: Expected NetworkX graph or GeoDataFrame for streets legend, got {type(data)}"
            )
            return

        if not has_data(edges):
            log_progress("[debug] No street edges for legend.")
            return

        if "highway" not in edges.columns or edges["highway"].dropna().empty:
            log_progress(
                "[debug] 'highway' column not found or empty in street edges. Adding generic legend entry."
            )
            legend_handles.append(mpatches.Patch(color="gray"))
            legend_labels.append("Streets (no type info)")
            return

        cmap = cm.get_cmap("Set3")
        highway_types = {}
        color_idx = 0

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

        for hw_type, color in sorted(highway_types.items()):
            legend_handles.append(mpatches.Patch(color=color))
            legend_labels.append(hw_type)

    except Exception as e:
        log_progress(f"Warning: Could not collect streets legend: {e}")


def _collect_buildings_legend(data, legend_handles, legend_labels):
    """Collect building types for legend without plotting."""
    if not hasattr(data, "geometry"):
        return

    cmap = cm.get_cmap("Paired")
    building_types = {}
    color_idx = 0

    if "building" in data.columns:
        for val in data["building"].dropna().unique():
            if val not in building_types:
                building_types[val] = cmap(color_idx % 12)
                color_idx += 1

    if building_types:
        for btype, color in sorted(building_types.items()):
            legend_handles.append(mpatches.Patch(color=color))
            legend_labels.append(f"building={btype}")
    else:
        legend_handles.append(mpatches.Patch(color="#ff7f0e"))
        legend_labels.append("buildings")


def _collect_green_legend(data, legend_handles, legend_labels):
    """Collect green space types for legend without plotting."""
    if not hasattr(data, "geometry"):
        return

    cmap = cm.get_cmap("Set2")
    feature_types = {}
    color_idx = 0

    for tag in ["leisure", "landuse", "natural"]:
        if tag in data.columns:
            for val in data[tag].dropna().unique():
                feature_key = f"{tag}={val}"
                if feature_key not in feature_types:
                    feature_types[feature_key] = cmap(color_idx % 8)
                    color_idx += 1

    if feature_types:
        for feature_key, color in feature_types.items():
            legend_handles.append(mpatches.Patch(color=color))
            legend_labels.append(feature_key)
    else:
        legend_handles.append(mpatches.Patch(color="#90ee90"))
        legend_labels.append("green spaces")

