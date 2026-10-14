import argparse
import contextlib
import io
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib

# Use a non-interactive backend before importing pyplot. This is required for
# threaded/background SVG generation on macOS (NSWindow must be created on the
# main thread) and for headless environments in general.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

STYLE_FILE = "style.json"

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork")

import map_core.core.config as cfg
from map_core import paths
from map_core.core.buildings import \
    compute_metric as core_compute_buildings_metric
from map_core.core.buildings_styling import style_buildings
from map_core.core.fetch import fetch_layer as core_fetch_layer
from map_core.core.layer_plot import (
    plot_water_layer,
    plot_streets_layer,
    plot_green_layer,
    plot_generic_layer,
    plot_coastline_layer,
)
# Coastline overlay (external OSM shapefile, bottom layer)
from map_core.core.coastline import load_coastlines
# Delegate common utilities to core modules for maintainability
from map_core.core.plot import \
    generate_layer_legend as core_generate_layer_legend
from map_core.core.plot import plot_map_layer as core_plot_map_layer
from map_core.core.plot import \
    setup_figure_and_axes as core_setup_figure_and_axes
from map_core.core.plot import apply_creative_output
from map_core.core.svg_post import (build_svg_opt_config, export_png_from_svg,
                                     load_optimize_config, optimize_svg_file,
                                     simplify_svg_shapes)
from map_core.core.util import has_data as core_has_data

from map_core.core.util import drop_point_features
from map_core.core.util import log_progress as core_log_progress


def log_progress(message: str) -> None:
    """Lightweight stdout logger proxying to core util."""
    core_log_progress(message)


# Configure OSMnx caching explicitly and surface in logs
OSMNX_CACHE_FOLDER = os.environ.get(
    "OSMNX_CACHE_FOLDER", os.path.join("..", "cache", "osmnx")
)
try:
    ox.settings.use_cache = True
    # Ensure absolute path for clarity
    cache_abs = os.path.abspath(OSMNX_CACHE_FOLDER)
    ox.settings.cache_folder = cache_abs
    log_progress(f"[osmnx] Cache enabled. Folder: {cache_abs}")
except Exception as e:
    log_progress(f"[osmnx] Warning: could not set cache settings: {e}")

# Global Matplotlib simplification for faster, smaller vector output
try:
    plt.rcParams["path.simplify"] = True
    # Tune threshold to trade off fidelity vs performance/file size
    plt.rcParams["path.simplify_threshold"] = 0.75
except Exception:
    pass

# Import the configuration manager
from config import config_manager, feature_visibility_manager
# Import palettes from core module instead of local implementation
from map_core.core.palettes import get_palette, load_palettes

# --- Helper Functions ---

# Use core utility directly
has_data = core_has_data


def _filter_geometry_for_layer(gdf: Any, layer_name: str) -> Any:
    """Light filter to remove point features (rendered as circles) only.

    Drops Point/MultiPoint geometries; keeps everything else unchanged.
    Applies to any layer to be minimally invasive.
    """
    return drop_point_features(gdf, layer_name=layer_name)


# Use core plotting function directly
plot_map_layer = core_plot_map_layer


def _get_combined_bbox(
    geodataframes: List[Any],
) -> Optional[Tuple[float, float, float, float]]:
    """Calculate a combined bounding box from a list of GeoDataFrames."""
    import networkx as nx

    all_bounds = []
    for gdf in geodataframes:
        if has_data(gdf):
            try:
                # For graphs, convert to GDF to get bounds
                if isinstance(gdf, nx.Graph):
                    nodes = ox.graph_to_gdfs(gdf, edges=False)
                    if has_data(nodes):
                        all_bounds.append(nodes.total_bounds)
                elif hasattr(gdf, "total_bounds"):
                    bounds = gdf.total_bounds
                    if len(bounds) == 4:
                        all_bounds.append(bounds)
            except Exception as e:
                log_progress(f"[bbox] Warning: could not get bounds from a layer: {e}")

    if not all_bounds:
        return None

    min_x, min_y, max_x, max_y = [], [], [], []
    for bounds in all_bounds:
        if len(bounds) == 4:
            min_x.append(bounds[0])
            min_y.append(bounds[1])
            max_x.append(bounds[2])
            max_y.append(bounds[3])

    if not min_x:
        return None

    return (min(min_x), min(min_y), max(max_x), max(max_y))


# Use core functions directly
_compute_buildings_metric = core_compute_buildings_metric
_setup_figure_and_axes = core_setup_figure_and_axes
# Thread-safe figure factory used by the parallel per-layer save path. It never
# touches pyplot's global figure registry, so concurrent layer exports cannot
# race and produce empty SVGs.
from map_core.core.plot import setup_figure_and_axes_isolated as core_setup_figure_isolated

_setup_figure_and_axes_isolated = core_setup_figure_isolated


def save_legend(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    output_directory: str,
    prefix: str,
    background_color: str,
    figure_dpi: int = 150,
) -> Optional[str]:
    """Save a legend for a layer showing unique features.

    Args:
        layer_name: Name of the layer
        data: Layer data
        layer_style: Style configuration for the layer
        output_directory: Directory to save legend
        prefix: Filename prefix
        background_color: Background color
        figure_dpi: DPI for the legend

    Returns:
        Path to saved legend SVG or None if not generated
    """
    if not has_data(data):
        return None

    try:
        fig, ax = core_generate_layer_legend(
            layer_name=layer_name,
            data=data,
            layer_style=layer_style,
            figure_size=[8, 6],
            figure_dpi=figure_dpi,
            background_color=background_color,
        )

        legend_path = os.path.join(
            output_directory, f"{prefix}_{layer_name}_legend.svg"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            plt.savefig(
                legend_path,
                format="svg",
                bbox_inches="tight",
                pad_inches=0.2,
                transparent=False,
            )
        plt.close(fig)
        return legend_path
    except Exception as e:
        log_progress(f"[legend] Error generating legend for {layer_name}: {e}")
        return None


def save_debug_legend(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    output_directory: str,
    prefix: str,
    background_color: str,
    figure_dpi: int = 150,
) -> Optional[str]:
    """Save a debug legend showing feature types for a layer.

    Args:
        layer_name: Name of the layer
        data: Layer data
        layer_style: Style configuration
        output_directory: Directory to save legend
        prefix: Filename prefix
        background_color: Background color
        figure_dpi: DPI for the legend

    Returns:
        Path to saved legend SVG or None if not generated
    """
    if not has_data(data):
        return None

    try:
        from map_core.core.plot import generate_debug_legend

        fig, ax = generate_debug_legend(
            layer_name=layer_name,
            data=data,
            layer_style=layer_style,
            figure_size=[8, 6],
            figure_dpi=figure_dpi,
            background_color=background_color,
        )

        legend_path = os.path.join(
            output_directory, f"{prefix}_{layer_name}_debug_legend.svg"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            plt.savefig(
                legend_path,
                format="svg",
                bbox_inches="tight",
                pad_inches=0.2,
                transparent=False,
            )
        plt.close(fig)
        return legend_path
    except Exception as e:
        log_progress(f"[debug] Error generating debug legend for {layer_name}: {e}")
        return None


def save_debug_map(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    output_directory: str,
    prefix: str,
    bbox: Optional[Tuple[float, float, float, float]],
    figure_size: List[float],
    background_color: str,
    figure_dpi: int,
    margin: float,
) -> Optional[str]:
    """Save a debug map for a layer showing features color-coded by type.

    Args:
        layer_name: Name of the layer
        data: Layer data
        layer_style: Style configuration for the layer (colors overridden)
        output_directory: Directory to save debug map
        prefix: Filename prefix
        bbox: Bounding box for the map
        figure_size: Figure size
        background_color: Background color
        figure_dpi: DPI for the map
        margin: Margin around the map

    Returns:
        Path to saved debug map SVG or None if not generated
    """
    if not has_data(data):
        return None

    try:
        from map_core.core.plot import generate_debug_map

        fig, ax = generate_debug_map(
            layer_name=layer_name,
            data=data,
            layer_style=layer_style,
            bbox=bbox,
            figure_size=figure_size,
            figure_dpi=figure_dpi,
            background_color=background_color,
            margin=margin,
        )

        debug_path = os.path.join(
            output_directory, f"{prefix}_{layer_name}_debug_map.svg"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            plt.savefig(
                debug_path,
                format="svg",
                bbox_inches="tight",
                pad_inches=0.2,
                transparent=False,
            )
        plt.close(fig)
        return debug_path
    except Exception as e:
        log_progress(f"[debug] Error generating debug map for {layer_name}: {e}")
        return None


def save_layer(
    layer_name: str,
    data: Any,
    layer_styles: Dict[str, Any],
    output_directory: str,
    prefix: str,
    figure_size: List[float],
    background_color: str,
    figure_dpi: int,
    margin: float,
    transparent: bool = False,
    suffix: str = "",
    plot_bbox: Optional[Tuple[float, float, float, float]] = None,
    location_query: Optional[str] = None,
) -> None:
    """Save a single map layer to an SVG file."""
    if not has_data(data):
        return

    # Normalize geometries to expected types (drops points from polygon layers, etc.)
    data = _filter_geometry_for_layer(data, layer_name)
    if not has_data(data):
        return

    fig, ax = _setup_figure_and_axes_isolated(
        figure_size, figure_dpi, background_color, margin, transparent=transparent
    )

    # Enforce a consistent bounding box for all layers if provided
    if plot_bbox:
        west, south, east, north = plot_bbox
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)

    # Buildings use shared styling logic
    if layer_name == "buildings":
        layer_style = layer_styles.get("buildings", {})
        style_buildings(ax, data, layer_style, location_query)
    elif layer_name == "streets":
        plot_streets_layer(ax, data, layer_styles[layer_name])
    elif layer_name == "water":
        plot_water_layer(ax, data, layer_styles[layer_name])
    elif layer_name == "green":
        plot_green_layer(ax, data, layer_styles[layer_name])
    elif layer_name == "coastline":
        plot_coastline_layer(ax, data, layer_styles[layer_name])
    else:
        plot_generic_layer(ax, data, layer_name, layer_styles[layer_name])

    out_svg_path = os.path.join(output_directory, f"{prefix}_{layer_name}{suffix}.svg")
    with contextlib.redirect_stdout(io.StringIO()):
        fig.savefig(
            out_svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
            transparent=transparent,
        )
    # The figure was never registered with pyplot (thread-safe isolated figure),
    # so clear it directly instead of plt.close(fig) which touches global state.
    fig.clear()
    return out_svg_path


def fetch_layer(
    query: str,
    dist: Optional[float],
    tags: Optional[Dict[str, Any]],
    is_graph: bool = False,
    custom_filter: Optional[str] = None,
    simplify_tolerance: Optional[float] = None,
    min_size_threshold: float = 0,
    min_width_m: float = 0,
    layer_name_for_debug: Optional[str] = None,
    pbf_path: Optional[str] = None,
    bbox_override: Optional[Tuple[float, float, float, float]] = None,
) -> Any:
    """Delegate fetching to core.fetch.fetch_layer for a single implementation."""
    # Clarify data source path to user: local PBF vs remote via OSMnx (with cache)
    source = "local PBF" if pbf_path else "remote (OSMnx)"
    cache_folder = getattr(ox.settings, "cache_folder", OSMNX_CACHE_FOLDER)
    if pbf_path:
        log_progress(
            f"[fetch] Layer '{layer_name_for_debug or ''}': using {source}: {pbf_path}"
        )
    else:
        log_progress(
            f"[fetch] Layer '{layer_name_for_debug or ''}': using {source}; will try cache first at: {cache_folder}"
        )
    return core_fetch_layer(
        query=query,
        dist=dist,
        tags=tags,
        is_graph=is_graph,
        custom_filter=custom_filter,
        simplify_tolerance=simplify_tolerance,
        min_size_threshold=min_size_threshold,
        min_width_m=min_width_m,
        layer_name_for_debug=layer_name_for_debug,
        pbf_path=pbf_path,
        bbox_override=bbox_override,
    )


def load_style(style_path: str = STYLE_FILE) -> Optional[Dict[str, Any]]:
    """Load, normalize, and (optionally) validate a style file."""
    raw = cfg.load_style(style_path)
    if raw is None:
        log_progress(f"Error: {style_path} not found or invalid.")
        return None
    style = cfg.normalize_style(raw)
    # Validate if a schema is available (no-op otherwise)
    err = cfg.validate_style(style)
    if err:
        log_progress(f"Schema validation warning: {err}")
    log_progress(f"Loaded style from: {style_path}")
    return style


def update_layer_tags(
    layer_name: str, gdf: Any, layer_tags_file: str = "config/layers/layer_tags.json"
):
    """Dynamically update the layer_tags.json file with new values found in the GeoDataFrame."""
    if not has_data(gdf):
        return

    try:
        with open(layer_tags_file, "r") as f:
            layer_tags = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_progress(
            f"[tags] Warning: Could not read {layer_tags_file}. Skipping dynamic tag update."
        )
        return

    if layer_name not in layer_tags:
        return

    layer_config = layer_tags[layer_name]
    if "tag_configs" not in layer_config:
        return

    made_changes = False
    for tag_key, tag_config in layer_config["tag_configs"].items():
        if tag_key in gdf.columns:
            # Manually find unique values to handle lists
            all_values = set()
            for item in gdf[tag_key].dropna():
                if isinstance(item, list):
                    for sub_item in item:
                        all_values.add(sub_item)
                else:
                    all_values.add(item)

            unique_values = sorted(list(all_values))

            for value in unique_values:
                if value not in tag_config["values"]:
                    tag_config["values"][value] = value
                    made_changes = True

    if made_changes:
        log_progress(
            f"[tags] Updating {layer_tags_file} with new tags for layer '{layer_name}'."
        )
        try:
            with open(layer_tags_file, "w") as f:
                json.dump(layer_tags, f, indent=2)
        except Exception as e:
            log_progress(f"[tags] Error writing to {layer_tags_file}: {e}")


def _generate_layer_task(args: Dict[str, Any]) -> Optional[str]:
    """Generate and optionally optimize a single layer SVG.

    This helper is designed to be executed concurrently via ThreadPoolExecutor.
    Each call creates its own matplotlib figure, so thread safety is preserved.

    Args:
        args: Dictionary containing:
            - layer_name: Name of the layer
            - data: Layer data (GeoDataFrame or graph)
            - layer_styles: Full layer styles dict
            - output_directory: Output directory
            - prefix: Filename prefix
            - figure_size: Figure size
            - background_color: Background color
            - figure_dpi: DPI
            - margin: Margin
            - transparent: Transparent background flag
            - svg_opt_config: Optional SVG optimization config
            - location_query: Optional location query (buildings only)
            - png_export: Optional PNG export settings

    Returns:
        Path to the generated SVG or None.
    """
    layer_name = args["layer_name"]
    data = args["data"]
    layer_styles = args["layer_styles"]
    output_directory = args["output_directory"]
    prefix = args["prefix"]
    figure_size = args["figure_size"]
    background_color = args["background_color"]
    figure_dpi = args["figure_dpi"]
    margin = args["margin"]
    transparent = args["transparent"]
    svg_opt_config = args.get("svg_opt_config")
    location_query = args.get("location_query")
    png_export = args.get("png_export")

    t0 = time.time()
    log_progress(f"[save] {layer_name.capitalize()} layer: start")
    svg_path = save_layer(
        layer_name,
        data,
        layer_styles,
        output_directory,
        prefix,
        figure_size,
        background_color,
        figure_dpi,
        margin,
        transparent=transparent,
        location_query=location_query,
    )
    log_progress(
        f"[save] {layer_name.capitalize()} layer: done in {time.time()-t0:.2f}s -> {svg_path}"
    )

    if svg_opt_config and svg_path:
        try:
            log_progress(f"[optimize] {layer_name.capitalize()} SVG: start")
            optimize_svg_file(svg_path, None, svg_opt_config)
            log_progress(f"[optimize] {layer_name.capitalize()} SVG: done")
        except Exception as e:
            log_progress(f"[optimize] {layer_name.capitalize()} SVG skipped due to error: {e}")

    # Shape simplification for this layer
    output_settings = args.get("output_settings") or {}
    if output_settings.get("svg_simplify_enabled", False) and svg_path:
        try:
            log_progress(f"[simplify] {layer_name.capitalize()} SVG: start")
            simplify_svg_shapes(
                svg_path,
                tolerance=output_settings.get("svg_simplify_tolerance", 0.1),
                preserve_topology=output_settings.get("svg_simplify_preserve_topology", True),
                min_segment_length=output_settings.get("svg_simplify_min_segment_length", 0.5),
                low_poly_mode=output_settings.get("svg_low_poly_mode", False),
                low_poly_grid=output_settings.get("svg_low_poly_grid", 1.0),
            )
            log_progress(f"[simplify] {layer_name.capitalize()} SVG: done")
        except Exception as e:
            log_progress(f"[simplify] {layer_name.capitalize()} SVG skipped due to error: {e}")

    # Export PNG if enabled
    if png_export and png_export.get("enabled") and svg_path:
        try:
            export_png_from_svg(
                svg_path,
                output_directory,
                prefix,
                png_export.get("dpi", 300),
                png_export.get("scale", 2.0),
                png_export.get("suffix", "_hq"),
                png_export.get("background"),
                png_export.get("source", "original"),
                svg_opt_config,
            )
        except Exception as e:
            log_progress(f"[png] {layer_name.capitalize()} PNG export failed: {e}")

    return svg_path


def _apply_cli_overrides(style: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply --pbf/--center/--dist/--out/--layers overrides to the loaded style."""
    loc = style.setdefault("location", {})
    out = style.setdefault("output", {})
    if args.pbf:
        loc["pbf_path"] = args.pbf
        loc["data_source"] = "local"
        log_progress(f"[cli] pbf_path <- {args.pbf}")
    if args.center:
        lat, lon = args.center
        loc["query"] = f"{lat} {lon}"
        loc["bbox"] = None
        log_progress(f"[cli] center <- {lat} {lon}")
    if args.dist is not None:
        loc["distance"] = float(args.dist)
        log_progress(f"[cli] distance <- {args.dist} m")
    if args.out:
        out["output_directory"] = args.out
        log_progress(f"[cli] output_directory <- {args.out}")
    if args.layers:
        wanted = {name.strip() for name in args.layers.split(",") if name.strip()}
        layers = style.setdefault("layers", {})
        unknown = wanted - set(layers)
        if unknown:
            log_progress(
                f"[cli] Warning: unknown layer(s) {sorted(unknown)}; known: {sorted(layers)}"
            )
        for name, cfg_layer in layers.items():
            if isinstance(cfg_layer, dict):
                cfg_layer["enabled"] = name in wanted
        log_progress(f"[cli] layers <- {sorted(wanted & set(layers))}")


def main() -> None:
    """Main function to generate the map art."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate map art from OpenStreetMap data."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="The timestamped filename prefix for the output files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=STYLE_FILE,
        help="Path to the style configuration file (e.g., config.json).",
    )
    parser.add_argument(
        "--pbf",
        type=str,
        help="Local .osm.pbf to read from (overrides location.pbf_path).",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        metavar=("LAT", "LON"),
        help="Map centre (overrides location.query).",
    )
    parser.add_argument(
        "--dist",
        type=float,
        metavar="METERS",
        help="Half-width of the map area in metres (overrides location.distance).",
    )
    parser.add_argument(
        "--out",
        type=str,
        metavar="DIR",
        help="Base output directory (overrides output.output_directory).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        metavar="a,b,c",
        help="Comma-separated layers to enable (others disabled), e.g. streets,buildings.",
    )
    args = parser.parse_args()

    # Load configuration from the specified style file
    try:
        style = load_style(args.config)
        if not style:
            sys.exit(1)
    except FileNotFoundError:
        log_progress(
            "Error: style.json not found. Please create one based on the instructions."
        )
        sys.exit(1)

    _apply_cli_overrides(style, args)

    # Make sure the managed data/cache/output directories exist.
    try:
        paths.ensure_dirs()
    except Exception as e:  # noqa: BLE001 - non-fatal; a later mkdir will surface it
        log_progress(f"[main] Warning: could not create data directories: {e}")

    # Load SVG optimization configuration from style.json (with fallback to config file)
    style_output = style.get("output") or {}
    svg_opt_config = build_svg_opt_config(style_output)
    # Also load from file as fallback for any missing values
    file_opt_config = load_optimize_config()
    if file_opt_config:
        svg_opt_config = {**file_opt_config, **svg_opt_config}

    # PNG export settings from style
    generate_png = style_output.get("generate_png", False)
    png_dpi = style_output.get("png_dpi", 300)
    png_scale = style_output.get("png_scale", 2.0)
    png_suffix = style_output.get("png_suffix", "_hq")
    png_background = style_output.get("png_background")
    png_source = style_output.get("png_source", "original")  # "original" | "optimized"

    # Resolve the output base directory (relative paths anchor to the repo root,
    # so "../output" means the sibling workspace dir regardless of CWD).
    output_base_directory = str(
        paths.resolve_output_dir(style["output"]["output_directory"])
    )

    # Use the prefix from the command line argument if provided, otherwise use the one from style.json
    filename_prefix = args.prefix if args.prefix else style["output"]["filename_prefix"]

    output_directory = os.path.join(output_base_directory, filename_prefix)
    os.makedirs(output_directory, exist_ok=True)

    # Save the current style configuration to the output directory for reproducibility
    config_output_path = os.path.join(output_directory, "config.json")
    with open(config_output_path, "w") as f:
        json.dump(style, f, indent=2)
    log_progress(f"Saved configuration to {config_output_path}")

    location_query = style["location"]["query"]
    location_distance = style["location"]["distance"]
    # Optional local PBF file path for offline data sourcing
    location_pbf_path = style.get("location", {}).get("pbf_path")
    # Compute bounding box from center point and distance
    location_bbox: Optional[Tuple[float, float, float, float]] = None
    raw_bbox = style.get("location", {}).get("bbox")

    # Manual bbox override if provided
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            location_bbox = tuple(float(x) for x in raw_bbox)
            log_progress(f"[main] Using manual bbox override: {location_bbox}")
        except Exception:
            location_bbox = None

    # Compute bbox from center + distance (OSMnx returns correct format)
    if location_bbox is None and location_distance is not None:
        try:
            q = (location_query or "").replace(",", " ").split()
            if len(q) >= 2:
                lat, lon = float(q[0]), float(q[1])
                # osmnx returns (west, south, east, north) == (minx, miny, maxx, maxy).
                location_bbox = ox.utils_geo.bbox_from_point(
                    (lat, lon), dist=float(location_distance)
                )
                log_progress(
                    f"[main] Computed bbox from center+distance: {location_bbox}"
                )
        except Exception as e:
            log_progress(f"[main] Warning: failed to compute bbox: {e}")
    # Resolve the PBF path against the data root (sources/, regions/), the repo
    # root, or CWD, so it works regardless of the process working directory.
    if location_pbf_path:
        resolved_pbf = str(paths.resolve_data_path(location_pbf_path))
        if resolved_pbf != location_pbf_path:
            log_progress(
                f"[main] Resolved PBF path '{location_pbf_path}' -> '{resolved_pbf}'"
            )
        else:
            log_progress(f"[main] Using PBF path '{resolved_pbf}'")
        exists = os.path.isfile(resolved_pbf)
        log_progress(f"[main] PBF exists: {exists}")
        location_pbf_path = resolved_pbf

        processing = style.get("processing", {}) or {}
        if exists and processing.get("auto_clip", True):
            from map_core.core.regions import auto_clip

            location_pbf_path = str(
                auto_clip(
                    location_pbf_path,
                    location_bbox,
                    threshold_mb=float(processing.get("auto_clip_threshold_mb", 20.0)),
                    margin=float(processing.get("auto_clip_margin", 0.2)),
                )
            )
    # If a local PBF is provided, harden OSMnx to cache-only to avoid any external calls
    if location_pbf_path:
        try:
            # Configuration summary for clarity (distance in meters)
            dist_m_dbg = (
                None if location_distance is None else int(round(location_distance))
            )
            log_progress("--- Run Configuration ---")
            log_progress(f"Location query: '{location_query}'")
            log_progress(f"Distance (m): {dist_m_dbg}")
            log_progress(f"Local PBF: {location_pbf_path or '(none)'}")
            log_progress(f"BBox override: {location_bbox or '(none)'}")
            enabled_layers = [
                k
                for k, v in style["layers"].items()
                if isinstance(v, dict) and v.get("enabled")
            ]
            log_progress(f"Enabled layers: {enabled_layers}")
            log_progress("-------------------------")

            # Enforce OSMnx cache_only when a local PBF is supplied to avoid network calls
            import osmnx as _ox  # use different alias to avoid shadowing top-level 'ox'

            _ox.settings.use_cache = True
            _ox.settings.cache_only = True
            log_progress(
                "[main] Local PBF provided -> OSMnx set to cache_only=True (no external requests)."
            )
        except Exception as e:
            log_progress(f"[main] Warning: failed to set OSMnx cache_only mode: {e}")

    # Get global plot settings
    figure_size = style["output"].get("figure_size", [10, 10])
    background_color = style["output"].get("background_color", "white")
    transparent_bg = style["output"].get("transparent_background", False)
    figure_dpi = style["output"].get("figure_dpi", 300)
    margin = style["output"].get("margin", 0.05)

    # Feature configurations are now managed by feature_visibility_manager

    # --- Fetching Data ---
    log_progress("Fetching data...")
    G = None
    buildings_gdf = None
    water_gdf = None
    green_gdf = None
    coastline_gdf = None

    # Step 1: Fetch streets first to establish a definitive bounding box.
    if style["layers"]["streets"]["enabled"]:
        t0 = time.time()
        log_progress(f"Output directory: {output_directory}")
        log_progress("[fetch] Streets: start")
        # Use feature_visibility_manager to get active street filters
        street_tags = feature_visibility_manager.get_active_osm_filters("streets")
        custom_street_filter = None
        if street_tags:
            # Convert dict of tags to OSMnx custom filter string if needed
            # For now, assuming simple highway filter as before
            if "highway" in street_tags and isinstance(street_tags["highway"], list):
                custom_street_filter = (
                    '["highway"~"' + "|".join(street_tags["highway"]) + '"]'
                )
            elif "highway" in street_tags and street_tags["highway"] is True:
                custom_street_filter = (
                    '["highway"~"[a-zA-Z0-9_]+"]'  # Match any highway
                )

        G = fetch_layer(
            location_query,
            location_distance,
            tags=street_tags,
            is_graph=True,
            custom_filter=custom_street_filter,
            pbf_path=location_pbf_path,
            bbox_override=location_bbox,
        )
        dt = time.time() - t0
        if G is None:
            log_progress(f"[fetch] Streets: done in {dt:.2f}s (no data)")
        elif hasattr(G, "edges") and callable(getattr(G, "edges")):
            log_progress(
                f"[fetch] Streets: done in {dt:.2f}s (nodes={len(G.nodes())}, edges={len(G.edges())})"
            )
        else:
            log_progress(f"[fetch] Streets: done in {dt:.2f}s (edges={len(G)})")

    # Step 2: Calculate a combined bounding box from all fetched layers.
    plot_bbox = None
    all_layers_for_bbox = []
    if G is not None:
        all_layers_for_bbox.append(G)
    if has_data(buildings_gdf):
        all_layers_for_bbox.append(buildings_gdf)
    if has_data(water_gdf):
        all_layers_for_bbox.append(water_gdf)
    if has_data(green_gdf):
        all_layers_for_bbox.append(green_gdf)

    if all_layers_for_bbox:
        log_progress("[main] Calculating combined bounding box for all layers...")
        combined_bbox = _get_combined_bbox(all_layers_for_bbox)
        if combined_bbox:
            location_bbox = combined_bbox
            plot_bbox = combined_bbox
            log_progress(
                f"[main] Using combined bounding box for all layers: {location_bbox}"
            )
        else:
            log_progress(
                "[main] Warning: could not determine a combined bounding box. Using street network bbox as fallback."
            )
            if G is not None:
                try:
                    # Check if G is a NetworkX graph
                    import networkx as nx

                    if isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiDiGraph)):
                        nodes = ox.graph_to_gdfs(G, edges=False)
                        west, south, east, north = nodes.unary_union.bounds
                        location_bbox = (west, south, east, north)
                        plot_bbox = location_bbox
                        log_progress(
                            f"[main] Bounding box derived from street network: {location_bbox}"
                        )
                    else:
                        # G is a GeoDataFrame (edges fallback)
                        west, south, east, north = G.total_bounds
                        location_bbox = (west, south, east, north)
                        plot_bbox = location_bbox
                        log_progress(
                            f"[main] Bounding box derived from street edges GeoDataFrame: {location_bbox}"
                        )
                except Exception as e:
                    log_progress(
                        f"[main] Warning: could not derive bbox from street graph/edges: {e}"
                    )
    else:
        log_progress("[main] No data fetched, cannot determine bounding box.")

    # Fetch buildings
    if style["layers"]["buildings"]["enabled"]:
        t0 = time.time()
        log_progress("[fetch] Buildings: start")
        buildings_simplify_tolerance = style["layers"]["buildings"].get(
            "simplify_tolerance", None
        )
        # Use feature_visibility_manager to get active building filters
        buildings_tags = feature_visibility_manager.get_active_osm_filters("buildings")
        log_progress(f"[config] Buildings tags: {json.dumps(buildings_tags)}")

        buildings_gdf = fetch_layer(
            location_query,
            location_distance,
            tags=buildings_tags,
            simplify_tolerance=buildings_simplify_tolerance,
            layer_name_for_debug="Buildings",
            pbf_path=location_pbf_path,
            bbox_override=location_bbox,
        )
        dt = time.time() - t0
        try:
            feat_count = 0 if buildings_gdf is None else len(buildings_gdf)
            log_progress(
                f"[fetch] Buildings: done in {dt:.2f}s (features={feat_count})"
            )
        except Exception:
            log_progress(f"[fetch] Buildings: done in {dt:.2f}s")

    # Fetch water
    if style["layers"]["water"]["enabled"]:
        t0 = time.time()
        log_progress("[fetch] Water: start")
        water_simplify_tolerance = style["layers"]["water"].get(
            "simplify_tolerance", None
        )
        water_min_size_threshold = style["layers"]["water"].get("min_size_threshold", 0)
        water_min_width = style["layers"]["water"].get("min_width", 0) or 0
        # Use feature_visibility_manager to get active water filters
        water_tags = feature_visibility_manager.get_active_osm_filters("water")
        log_progress(f"[config] Water tags: {json.dumps(water_tags)}")

        # Expand bbox slightly for water to avoid blunt clipping of river polygons at map edge
        water_bbox = location_bbox
        if water_bbox is not None:
            minx, miny, maxx, maxy = water_bbox
            width = maxx - minx
            height = maxy - miny
            buf = max(width, height) * 0.01  # 1% buffer
            water_bbox = (minx - buf, miny - buf, maxx + buf, maxy + buf)

        try:
            water_gdf = fetch_layer(
                location_query,
                location_distance,
                tags=water_tags,
                simplify_tolerance=water_simplify_tolerance,
                min_size_threshold=water_min_size_threshold,
                min_width_m=water_min_width,
                layer_name_for_debug="Water",
                pbf_path=location_pbf_path,
                bbox_override=water_bbox,
            )

            if water_gdf is None or water_gdf.empty:
                log_progress(
                    "[warning] No water features found with the current filters. The water layer will be empty."
                )
                water_gdf = None

        except Exception as e:
            if "No matching features" in str(e) or "No fallback PBF found" in str(e):
                log_progress(
                    f"[warning] No water features found with tags: {water_tags}. The water layer will be empty."
                )
                water_gdf = None
            else:
                log_progress(f"[error] Error fetching water features: {e}")
                raise

    # Fetch green (parkland/greenways)
    if style["layers"].get("green", {}).get("enabled"):
        t0 = time.time()
        log_progress("[fetch] Green: start")
        green_simplify_tolerance = style["layers"]["green"].get(
            "simplify_tolerance", None
        )
        green_min_size_threshold = style["layers"]["green"].get("min_size_threshold", 0)
        green_min_width = style["layers"]["green"].get("min_width", 0) or 0
        # Use feature_visibility_manager to get active green filters
        green_tags = feature_visibility_manager.get_active_osm_filters("green")
        log_progress(f"[config] Green tags: {json.dumps(green_tags)}")
        green_gdf = fetch_layer(
            location_query,
            location_distance,
            tags=green_tags,
            simplify_tolerance=green_simplify_tolerance,
            min_size_threshold=green_min_size_threshold,
            min_width_m=green_min_width,
            layer_name_for_debug="Green",
            pbf_path=location_pbf_path,
            bbox_override=location_bbox,
        )

    # Fetch coastline overlay (external OSM shapefile, bottom layer).
    # Only produces data when the bbox actually touches a coastal
    # region; inland maps get None and the layer is skipped.
    if style["layers"].get("coastline", {}).get("enabled"):
        t0 = time.time()
        log_progress("[fetch] Coastline: start")
        coast_bbox = location_bbox
        if coast_bbox is not None:
            minx, miny, maxx, maxy = coast_bbox
            width = maxx - minx
            height = maxy - miny
            buf = max(width, height) * 0.01  # 1% buffer
            coast_bbox = (minx - buf, miny - buf, maxx + buf, maxy + buf)
        try:
            coastline_gdf = load_coastlines(coast_bbox)
        except Exception as e:  # noqa: BLE001 - coastline is best-effort
            log_progress(f"[fetch] Coastline: failed ({e}); skipping")
            coastline_gdf = None
        dt = time.time() - t0
        n = 0 if coastline_gdf is None else len(coastline_gdf)
        log_progress(f"[fetch] Coastline: done in {dt:.2f}s (features={n})")

    # --- Debugging: Print Layer Information ---
    log_progress("--- Map Layer Information ---")

    if has_data(buildings_gdf):
        log_progress("Buildings (GeoDataFrame):")
        for geom_type, count in buildings_gdf.geometry.geom_type.value_counts().items():
            log_progress(f"  - {geom_type}: {count} features")
    else:
        log_progress("Buildings (GeoDataFrame): Not enabled or no data.")

    if water_gdf is not None and not water_gdf.empty:
        log_progress("Water (GeoDataFrame):")
        for geom_type, count in water_gdf.geometry.geom_type.value_counts().items():
            log_progress(f"  - {geom_type}: {count} features")
    else:
        log_progress(
            "Water (GeoDataFrame): No water features found with the current filters."
        )
        water_gdf = None  # Ensure it's None if empty

    if has_data(green_gdf):
        log_progress("Green (GeoDataFrame):")
        for geom_type, count in green_gdf.geometry.geom_type.value_counts().items():
            log_progress(f"  - {geom_type}: {count} features")
    else:
        log_progress("Green (GeoDataFrame): Not enabled or no data.")
    log_progress("---------------------------")

    # --- Generate Output ---
    log_progress("Generating layers...")
    if style["output"]["separate_layers"]:
        layer_tasks: List[Dict[str, Any]] = []

        # Pass output settings to layer tasks for PNG and simplification
        output_settings = style.get("output") or {}

        if style["layers"]["streets"]["enabled"]:
            layer_tasks.append(
                {
                    "layer_name": "streets",
                    "data": G,
                    "layer_styles": style["layers"],
                    "output_directory": output_directory,
                    "prefix": filename_prefix,
                    "figure_size": figure_size,
                    "background_color": background_color,
                    "figure_dpi": figure_dpi,
                    "margin": margin,
                    "transparent": transparent_bg,
                    "svg_opt_config": svg_opt_config,
                    "png_export": {
                        "enabled": generate_png,
                        "dpi": png_dpi,
                        "scale": png_scale,
                        "suffix": png_suffix,
                        "background": png_background,
                        "source": png_source,
                    },
                    "output_settings": output_settings,
                }
            )

        if style["layers"]["water"]["enabled"]:
            layer_tasks.append(
                {
                    "layer_name": "water",
                    "data": water_gdf,
                    "layer_styles": style["layers"],
                    "output_directory": output_directory,
                    "prefix": filename_prefix,
                    "figure_size": figure_size,
                    "background_color": background_color,
                    "figure_dpi": figure_dpi,
                    "margin": margin,
                    "transparent": transparent_bg,
                    "svg_opt_config": svg_opt_config,
                    "png_export": {
                        "enabled": generate_png,
                        "dpi": png_dpi,
                        "scale": png_scale,
                        "suffix": png_suffix,
                        "background": png_background,
                        "source": png_source,
                    },
                    "output_settings": output_settings,
                }
            )

        if style["layers"].get("green", {}).get("enabled"):
            layer_tasks.append(
                {
                    "layer_name": "green",
                    "data": green_gdf,
                    "layer_styles": style["layers"],
                    "output_directory": output_directory,
                    "prefix": filename_prefix,
                    "figure_size": figure_size,
                    "background_color": background_color,
                    "figure_dpi": figure_dpi,
                    "margin": margin,
                    "transparent": transparent_bg,
                    "svg_opt_config": svg_opt_config,
                    "png_export": {
                        "enabled": generate_png,
                        "dpi": png_dpi,
                        "scale": png_scale,
                        "suffix": png_suffix,
                        "background": png_background,
                        "source": png_source,
                    },
                    "output_settings": output_settings,
                }
            )

        if (
            style["layers"].get("coastline", {}).get("enabled")
            and coastline_gdf is not None
        ):
            layer_tasks.append(
                {
                    "layer_name": "coastline",
                    "data": coastline_gdf,
                    "layer_styles": style["layers"],
                    "output_directory": output_directory,
                    "prefix": filename_prefix,
                    "figure_size": figure_size,
                    "background_color": background_color,
                    "figure_dpi": figure_dpi,
                    "margin": margin,
                    "transparent": transparent_bg,
                    "svg_opt_config": svg_opt_config,
                    "png_export": {
                        "enabled": generate_png,
                        "dpi": png_dpi,
                        "scale": png_scale,
                        "suffix": png_suffix,
                        "background": png_background,
                        "source": png_source,
                    },
                    "output_settings": output_settings,
                }
            )

        if style["layers"]["buildings"]["enabled"] and has_data(buildings_gdf):
            layer_tasks.append(
                {
                    "layer_name": "buildings",
                    "data": buildings_gdf,
                    "layer_styles": style["layers"],
                    "output_directory": output_directory,
                    "prefix": filename_prefix,
                    "figure_size": figure_size,
                    "background_color": background_color,
                    "figure_dpi": figure_dpi,
                    "margin": margin,
                    "transparent": transparent_bg,
                    "svg_opt_config": svg_opt_config,
                    "location_query": location_query,
                    "png_export": {
                        "enabled": generate_png,
                        "dpi": png_dpi,
                        "scale": png_scale,
                        "suffix": png_suffix,
                        "background": png_background,
                        "source": png_source,
                    },
                    "output_settings": output_settings,
                }
            )

        if layer_tasks:
            t0_parallel = time.time()
            log_progress(
                f"[save] Generating {len(layer_tasks)} layer(s) in parallel..."
            )
            with ThreadPoolExecutor(max_workers=len(layer_tasks)) as executor:
                futures = {
                    executor.submit(_generate_layer_task, task): task["layer_name"]
                    for task in layer_tasks
                }
                for future in as_completed(futures):
                    layer_name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        log_progress(
                            f"[save] {layer_name.capitalize()} layer failed: {e}"
                        )
            log_progress(
                f"[save] Parallel layer generation done in {time.time()-t0_parallel:.2f}s"
            )

    # Generate debug maps if enabled (global flag)
    if style.get("output", {}).get("enable_debug_legends", False):
        log_progress("[debug] Debug maps enabled - generating color-coded feature maps")

        if style["layers"]["streets"]["enabled"] and G is not None:
            log_progress("[debug] Streets: generating legend and map")
            legend_path = save_debug_legend(
                "streets",
                G,
                style["layers"]["streets"],
                output_directory,
                filename_prefix,
                background_color,
                figure_dpi,
            )
            map_path = save_debug_map(
                "streets",
                G,
                style["layers"]["streets"],
                output_directory,
                filename_prefix,
                location_bbox,
                figure_size,
                background_color,
                figure_dpi,
                margin,
            )
            if legend_path and map_path:
                log_progress(
                    f"[debug] Streets: done -> legend:{legend_path}, map:{map_path}"
                )
            elif legend_path:
                log_progress(f"[debug] Streets: legend only -> {legend_path}")
            elif map_path:
                log_progress(f"[debug] Streets: map only -> {map_path}")

        if style["layers"]["water"]["enabled"] and has_data(water_gdf):
            log_progress("[debug] Water: generating legend and map")
            legend_path = save_debug_legend(
                "water",
                water_gdf,
                style["layers"]["water"],
                output_directory,
                filename_prefix,
                background_color,
                figure_dpi,
            )
            map_path = save_debug_map(
                "water",
                water_gdf,
                style["layers"]["water"],
                output_directory,
                filename_prefix,
                location_bbox,
                figure_size,
                background_color,
                figure_dpi,
                margin,
            )
            if legend_path and map_path:
                log_progress(
                    f"[debug] Water: done -> legend:{legend_path}, map:{map_path}"
                )
            elif legend_path:
                log_progress(f"[debug] Water: legend only -> {legend_path}")
            elif map_path:
                log_progress(f"[debug] Water: map only -> {map_path}")

        if style["layers"].get("green", {}).get("enabled") and has_data(green_gdf):
            log_progress("[debug] Green: generating legend and map")
            legend_path = save_debug_legend(
                "green",
                green_gdf,
                style["layers"]["green"],
                output_directory,
                filename_prefix,
                background_color,
                figure_dpi,
            )
            map_path = save_debug_map(
                "green",
                green_gdf,
                style["layers"]["green"],
                output_directory,
                filename_prefix,
                location_bbox,
                figure_size,
                background_color,
                figure_dpi,
                margin,
            )
            if legend_path and map_path:
                log_progress(
                    f"[debug] Green: done -> legend:{legend_path}, map:{map_path}"
                )
            elif legend_path:
                log_progress(f"[debug] Green: legend only -> {legend_path}")
            elif map_path:
                log_progress(f"[debug] Green: map only -> {map_path}")

        if style["layers"]["buildings"]["enabled"] and has_data(buildings_gdf):
            log_progress("[debug] Buildings: generating legend and map")
            legend_path = save_debug_legend(
                "buildings",
                buildings_gdf,
                style["layers"]["buildings"],
                output_directory,
                filename_prefix,
                background_color,
                figure_dpi,
            )
            map_path = save_debug_map(
                "buildings",
                buildings_gdf,
                style["layers"]["buildings"],
                output_directory,
                filename_prefix,
                location_bbox,
                figure_size,
                background_color,
                figure_dpi,
                margin,
            )
            if legend_path and map_path:
                log_progress(
                    f"[debug] Buildings: done -> legend:{legend_path}, map:{map_path}"
                )
            elif legend_path:
                log_progress(f"[debug] Buildings: legend only -> {legend_path}")
            elif map_path:
                log_progress(f"[debug] Buildings: map only -> {map_path}")

    # Combined output (always generated)
    t0_comb = time.time()
    log_progress("[save] Combined map: start")
    fig, ax = _setup_figure_and_axes(
        figure_size, figure_dpi, background_color, margin, transparent=transparent_bg
    )

    # Plot layers in a specific order (based on zorder from style.json)
    layers_to_plot = []

    # Coastline overlay (bottom layer; zorder 0 by default). Drawn
    # only when the bbox touched a coastal region.
    if (
        style["layers"].get("coastline", {}).get("enabled")
        and coastline_gdf is not None
    ):
        layers_to_plot.append(
            {
                "name": "coastline",
                "data": coastline_gdf,
                "style": style["layers"]["coastline"],
            }
        )

    # Prepare data for plotting, ensuring only enabled and existing layers are considered
    if style["layers"]["water"]["enabled"] and water_gdf is not None:
        layers_to_plot.append(
            {"name": "water", "data": water_gdf, "style": style["layers"]["water"]}
        )

    if style["layers"].get("green", {}).get("enabled") and green_gdf is not None:
        layers_to_plot.append(
            {"name": "green", "data": green_gdf, "style": style["layers"]["green"]}
        )

    if style["layers"]["streets"]["enabled"] and G is not None:
        layers_to_plot.append(
            {"name": "streets", "data": G, "style": style["layers"]["streets"]}
        )

    if style["layers"]["buildings"]["enabled"] and has_data(buildings_gdf):
        layers_to_plot.append(
            {
                "name": "buildings",
                "data": buildings_gdf,
                "style": style["layers"]["buildings"],
            }
        )

    # Sort layers by zorder (lowest zorder first, so higher zorder is drawn on top)
    layers_to_plot.sort(key=lambda x: x["style"].get("zorder", 1))

    for layer_info in layers_to_plot:
        layer_name = layer_info["name"]
        data = layer_info["data"]
        layer_style = layer_info["style"]

        if layer_name == "buildings":
            style_buildings(ax, data, layer_style, location_query)
        elif layer_name == "streets":
            plot_streets_layer(ax, data, layer_style)
        elif layer_name == "water":
            plot_water_layer(ax, data, layer_style)
        elif layer_name == "green":
            plot_green_layer(ax, data, layer_style)
        elif layer_name == "coastline":
            plot_coastline_layer(ax, data, layer_style)

    combined_svg_path = os.path.join(
        output_directory, f"{filename_prefix}_combined.svg"
    )
    # Creative output controls (D4): mask / frame / title baked into the map.
    apply_creative_output(fig, ax, style.get("output") or {}, location_query)
    with contextlib.redirect_stdout(io.StringIO()):
        plt.savefig(
            combined_svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
            transparent=transparent_bg,
        )
    plt.close(fig)
    log_progress(
        f"[save] Combined map: done in {time.time()-t0_comb:.2f}s -> {combined_svg_path}"
    )

    # Post-process combined SVG if configured
    if svg_opt_config and combined_svg_path:
        try:
            log_progress("[optimize] Combined SVG: start")
            optimize_svg_file(combined_svg_path, None, svg_opt_config)
            log_progress("[optimize] Combined SVG: done")
        except Exception as e:
            log_progress(f"[optimize] Combined SVG skipped due to error: {e}")

    # Shape simplification (low-poly / path reduction) if enabled
    style_output = style.get("output") or {}
    if style_output.get("svg_simplify_enabled", False) and combined_svg_path:
        try:
            log_progress("[simplify] Combined SVG: start")
            simplify_svg_shapes(
                combined_svg_path,
                tolerance=style_output.get("svg_simplify_tolerance", 0.1),
                preserve_topology=style_output.get("svg_simplify_preserve_topology", True),
                min_segment_length=style_output.get("svg_simplify_min_segment_length", 0.5),
                low_poly_mode=style_output.get("svg_low_poly_mode", False),
                low_poly_grid=style_output.get("svg_low_poly_grid", 1.0),
            )
            log_progress("[simplify] Combined SVG: done")
        except Exception as e:
            log_progress(f"[simplify] Combined SVG skipped due to error: {e}")

    # Export PNG if enabled
    if generate_png and combined_svg_path:
        try:
            export_png_from_svg(
                combined_svg_path,
                output_directory,
                filename_prefix,
                png_dpi,
                png_scale,
                png_suffix,
                png_background,
                png_source,
                svg_opt_config,
            )
        except Exception as e:
            log_progress(f"[png] Combined PNG export failed: {e}")


if __name__ == "__main__":
    main()
