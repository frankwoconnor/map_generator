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
# Legend generation lives in its own module; re-export for call sites.
from map_core.core.legend import (  # noqa: F401
    _add_legend_item,
    _collect_buildings_legend,
    _collect_green_legend,
    _collect_streets_legend,
    _collect_water_legend,
    add_map_legend,
    create_map_legend,
    generate_layer_legend,
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

    # One colour for all water: honour the user's configured facecolor.
    # (Previously river areas used a hard-coded blue that ignored the layer
    # colour and river/centerline were recoloured independently, so the water
    # layer looked wrong relative to the chosen palette.)
    water_blue = facecolor if facecolor != "#000000" else "#88ccee"

    # Polygon water bodies (lakes, ponds, reservoirs AND riverbank areas tagged
    # natural=water + water=river) vs waterway centrelines (river/stream/canal).
    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]

    # 1. Plot every water-area polygon with the configured water colour.
    water_area_union = None
    if not polys.empty:
        polys.plot(
            ax=ax,
            fc=water_blue,
            ec=edgecolor,
            lw=linewidth,
            alpha=alpha,
            hatch=hatch,
            zorder=zorder,
        )
        # A dissolved, slightly enlarged union of the water areas lets us drop
        # riverbank centrelines already shown as filled polygons (avoids a
        # doubled / misaligned look) without discarding the centreline of
        # rivers that are only mapped as lines.
        try:
            from shapely.ops import unary_union

            water_area_union = unary_union(list(polys.geometry.values)).buffer(
                max(1e-9, float(linewidth) * 1e-4)
            )
        except Exception:
            water_area_union = None

    # 2. Plot waterway centrelines with per-feature width, skipping any line
    #    already covered by a water-area polygon (precise length test).
    if not lines.empty:
        meters_per_point = 5.0
        # waterway::(default_width_m, linewidth scale relative to layer width)
        widths_by_type = {
            "river": (10.0, 1.0),
            "canal": (8.0, 1.0),
            "stream": (3.0, 0.5),
            "other": (10.0, 2.0),
        }
        wv = (
            lines["waterway"].fillna("")
            if "waterway" in lines.columns
            else pd.Series("", index=lines.index)
        )

        def _covered_by_area(geom) -> bool:
            if water_area_union is None:
                return False
            try:
                glen = geom.length
                if glen <= 0:
                    return False
                return geom.intersection(water_area_union).length >= 0.8 * glen
            except Exception:
                return False

        line_collections = []
        for wtype, (default_w, lw_scale) in widths_by_type.items():
            if wtype == "other":
                subset = lines[~wv.isin(["river", "canal", "stream"])]
            else:
                subset = lines[wv == wtype]
            if subset.empty:
                continue
            subset = subset[~subset.geometry.apply(_covered_by_area)]
            if subset.empty:
                continue
            lc = _line_collection_for_waterways(
                subset,
                default_width_m=default_w,
                min_linewidth=max(linewidth * lw_scale, 0.1),
                meters_per_point=meters_per_point,
                color=water_blue,
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

