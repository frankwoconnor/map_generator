import os
import json
from typing import Any, Dict, List, Optional, Tuple
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import multiprocessing
import time
import contextlib
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

STYLE_FILE = 'style.json'

if sys.platform == 'darwin':
    multiprocessing.set_start_method('fork')
    
# Delegate common utilities to core modules for maintainability
from map_core.core.plot import (
    setup_figure_and_axes as core_setup_figure_and_axes,
    plot_map_layer as core_plot_map_layer,
    generate_layer_legend as core_generate_layer_legend,
)
from map_core.core.svg_post import load_optimize_config, optimize_svg_file
from map_core.core.util import (
    log_progress as core_log_progress,
    has_data as core_has_data,
)
import map_core.core.config as cfg
from map_core.core.fetch import fetch_layer as core_fetch_layer
from map_core.core.buildings import compute_metric as core_compute_buildings_metric

def log_progress(message: str) -> None:
    """Lightweight stdout logger proxying to core util."""
    core_log_progress(message)

# Configure OSMnx caching explicitly and surface in logs
OSMNX_CACHE_FOLDER = os.environ.get('OSMNX_CACHE_FOLDER', os.path.join('..', 'cache', 'osmnx'))
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
    plt.rcParams['path.simplify'] = True
    # Tune threshold to trade off fidelity vs performance/file size
    plt.rcParams['path.simplify_threshold'] = 0.75
except Exception:
    pass

# Import the configuration manager
from config import config_manager, get_layer_tags, get_layer_tag_config

# Keep palettes for backward compatibility
# PALETTES_FILE = 'palettes.json'  # Remove this local constant
# _PALETTES_CACHE = None  # Remove this local cache

# Import palettes from core module instead of local implementation
from map_core.core.palettes import load_palettes, get_palette

def load_layer_tags() -> Dict[str, Any]:
    """Load per-layer OSM tag definitions using the config manager.

    Returns a dict mapping layer keys (e.g., 'buildings','water','green', etc.)
    to tag dictionaries compatible with fetch_layer.
    Falls back to sensible defaults if missing/invalid.
    """
    try:
        layer_tags = get_layer_tags()
        result = {}
        for layer_name, config in layer_tags.layers.items():
            # Convert the LayerTagConfig to the format expected by the rest of the code
            result[layer_name] = config.tags
        return result
    except Exception as e:
        log_progress(f"[config] Error loading layer tags from config: {e}; using built-in defaults")
        # Fallback to defaults if there's an error
        return {
            'buildings': { 'building': True },
            'water': { 'natural': ['water'], 'landuse': ['reservoir','basin'] },
            'waterways': { 'waterway': ['river','stream','canal','drain','ditch'] },
            'green': {
                'leisure': ['park','garden','pitch','recreation_ground'],
                'landuse': ['grass','meadow','recreation_ground'],
                'natural': ['grassland','heath'],
            },
            'aeroway': { 'aeroway': ['runway','taxiway','apron','terminal'] },
            'rail': { 'railway': True },
            'amenities': { 'amenity': True },
            'shops': { 'shop': True },
        }

# --- Helper Functions ---

# Use core utility directly
has_data = core_has_data

def _filter_geometry_for_layer(gdf: Any, layer_name: str) -> Any:
    """Light filter to remove point features (rendered as circles) only.

    Drops Point/MultiPoint geometries; keeps everything else unchanged.
    Applies to any layer to be minimally invasive.
    """
    try:
        if not has_data(gdf) or not isinstance(gdf, gpd.GeoDataFrame) or 'geometry' not in gdf:
            return gdf
        gdf = gdf[gdf.geometry.notnull()]
        try:
            gdf = gdf[~gdf.geometry.is_empty]
        except Exception:
            pass
        # Remove points only
        return gdf[~gdf.geom_type.isin(['Point', 'MultiPoint'])]
    except Exception as e:
        log_progress(f"[filter] Skipping point-filter for '{layer_name}': {e}")
        return gdf

def _reproject_gdf_for_area_calc(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Optional[Any]]:
    """Project to local UTM for accurate area calculation.

    Returns:
        (projected_gdf, original_crs)
    """
    if not has_data(gdf): # If no data, return as is
        return gdf, None # Return None for original_crs if no data

    original_crs = gdf.crs
    # Only attempt re-projection if CRS exists and is geographic
    if original_crs and original_crs.is_geographic:
        try:
            utm_crs = gdf.estimate_utm_crs()
            gdf_proj = gdf.to_crs(utm_crs)
        except Exception as e:
            log_progress(f"Warning: Could not reproject GeoDataFrame to UTM. Error: {e}. Using original CRS.")
            gdf_proj = gdf # Fallback to original if re-projection fails
    else:
        gdf_proj = gdf # If no geographic CRS, use as is
    return gdf_proj, original_crs

# Use core plotting function directly
plot_map_layer = core_plot_map_layer

def _get_plot_params(layer_style: Dict[str, Any], layer_name: str = None) -> Dict[str, Any]:
    """Extract plotting parameters from a layer's style dictionary.
    
    Args:
        layer_style: Dictionary containing the layer's style parameters
        layer_name: Optional name of the layer (used for special cases like 'streets')
    """
    params = {
        'edgecolor': layer_style.get('edgecolor', '#000000'),
        'linewidth': layer_style.get('linewidth', 0.5),
        'alpha': layer_style.get('alpha', 1.0),
        'hatch': layer_style.get('hatch', None),
        'linestyle': layer_style.get('linestyle', '-'),
        'zorder': layer_style.get('zorder', 1)
    }
    # Only add facecolor if it's not the streets layer or if layer_name is not specified
    if layer_name is None or layer_name != 'streets':
        params['facecolor'] = layer_style.get('facecolor', '#000000')
    return params

def _get_building_common_params(layer_style: Dict[str, Any]) -> Dict[str, Any]:
    """Extract common building plotting params applied across all building modes."""
    return {
        'edgecolor': layer_style.get('edgecolor', '#000000'),
        'linewidth': layer_style.get('linewidth', 0.5),
        'alpha': layer_style.get('alpha', 1.0),
        'hatch': layer_style.get('hatch', None),
        'zorder': layer_style.get('zorder', 2),
    }

# Use core functions directly
_compute_buildings_metric = core_compute_buildings_metric
_setup_figure_and_axes = core_setup_figure_and_axes

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

        legend_path = os.path.join(output_directory, f"{prefix}_{layer_name}_legend.svg")
        with contextlib.redirect_stdout(io.StringIO()):
            plt.savefig(
                legend_path,
                format='svg', bbox_inches='tight', pad_inches=0.2, transparent=False
            )
        plt.close(fig)
        return legend_path
    except Exception as e:
        log_progress(f"[legend] Error generating legend for {layer_name}: {e}")
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
) -> None:
    """Save a single map layer to an SVG file."""
    if not has_data(data):
        return

    # Normalize geometries to expected types (drops points from polygon layers, etc.)
    data = _filter_geometry_for_layer(data, layer_name)
    if not has_data(data):
        return

    fig, ax = _setup_figure_and_axes(figure_size, figure_dpi, background_color, margin, transparent=transparent)

    # Enforce a consistent bounding box for all layers if provided
    if plot_bbox:
        west, south, east, north = plot_bbox
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)

    # Buildings need special handling to support auto modes in separate output
    if layer_name == 'buildings':
        layer_style = layer_styles.get('buildings', {})
        buildings_style_mode = layer_style.get('auto_style_mode', 'manual')
        log_progress(f"Saving buildings layer with style mode: {buildings_style_mode}")

        # Common params from top-level buildings settings
        bparams = _get_building_common_params(layer_style)
        common_edge = bparams['edgecolor']
        common_linewidth = bparams['linewidth']
        common_alpha = bparams['alpha']
        common_hatch = bparams['hatch']
        common_z = bparams['zorder']

        if buildings_style_mode == 'manual':
            manual = layer_style.get('manual_color_settings', {})
            face = manual.get('facecolor', '#000000')
            plot_map_layer(ax, 'buildings', data, face, common_edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
        elif buildings_style_mode == 'manual_floorsize':
            categories = layer_style.get('size_categories', []) or []
            if not categories:
                manual = layer_style.get('manual_color_settings', {})
                face = manual.get('facecolor', '#000000')
                plot_map_layer(ax, 'buildings', data, face, common_edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
            else:
                try:
                    buildings_proj, _ = _reproject_gdf_for_area_calc(data)
                    if has_data(buildings_proj):
                        areas = buildings_proj.geometry.area
                    else:
                        areas = pd.Series([0] * len(data), index=data.index)
                    default_face = layer_style.get('manual_color_settings', {}).get('facecolor', '#000000')
                    color_series = pd.Series(default_face, index=data.index)
                    for cat in categories:
                        facecolor = cat.get('facecolor', default_face)
                        min_area = cat.get('min_area', None)
                        max_area = cat.get('max_area', None)
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
                            edgecolor=common_edge,
                            linewidth=common_linewidth,
                            alpha=common_alpha,
                            zorder=common_z
                        )
                    except Exception as e:
                        log_progress(f"Warning: Separate buildings manual_floorsize vectorized plot failed: {e}. Falling back to per-feature plotting.")
                        for idx, row in data.iterrows():
                            single = gpd.GeoDataFrame(geometry=[row.geometry], crs=getattr(data, 'crs', None))
                            single.plot(
                                ax=ax,
                                fc=color_series.loc[idx],
                                ec=common_edge,
                                lw=common_linewidth,
                                alpha=common_alpha,
                                zorder=common_z
                            )
                except Exception as e:
                    log_progress(f"Warning: Separate buildings manual_floorsize failed: {e}. Falling back to manual plot.")
                    manual = layer_style.get('manual_color_settings', {})
                    face = manual.get('facecolor', '#000000')
                    plot_map_layer(ax, 'buildings', data, face, common_edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
        elif buildings_style_mode in ['auto_size', 'auto_distance']:
            palette_key = 'auto_size_palette' if buildings_style_mode == 'auto_size' else 'auto_distance_palette'
            palette_name = layer_style.get(palette_key)
            palettes = load_palettes()
            # Defensive guard
            if not isinstance(palettes, dict):
                palettes = {}
            colors = (palettes.get(palette_name) or [])

            if not colors:
                log_progress(f"Warning: Palette '{palette_name}' not found or empty for '{buildings_style_mode}' mode in separate save. Falling back to manual color.")
                manual = layer_style.get('manual_color_settings', {})
                face = manual.get('facecolor', '#000000')
                plot_map_layer(ax, 'buildings', data, face, common_edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
            else:
                # Reverse palette so darker colors are first
                try:
                    colors = list(colors)[::-1]
                except Exception:
                    colors = list(colors)
                log_progress(f"Separate buildings auto '{buildings_style_mode}' using palette '{palette_name}' (reversed) with {len(colors)} colors")

                if buildings_style_mode == 'auto_size':
                    # Reproject to calculate area accurately
                    buildings_proj, _ = _reproject_gdf_for_area_calc(data)
                    if has_data(buildings_proj):
                        data['metric'] = buildings_proj.geometry.area
                    else:
                        data['metric'] = 0
                else: # auto_distance
                    # Compute distances from the map center (same logic as combined output)
                    buildings_proj, _ = _reproject_gdf_for_area_calc(data)
                    if has_data(buildings_proj):
                        center_proj = None
                        # Try to parse center from location_query ("lat lon")
                        try:
                            q = (location_query or '').replace(',', ' ').split()
                            if len(q) >= 2:
                                lat_c = float(q[0]); lon_c = float(q[1])
                                # Build WGS84 point then project to buildings_proj CRS
                                center_wgs = gpd.GeoSeries([Point(lon_c, lat_c)], crs='EPSG:4326')
                                center_proj_geom = center_wgs.to_crs(buildings_proj.crs).iloc[0]
                                center_proj = center_proj_geom
                        except Exception:
                            center_proj = None
                        # Fallback: use bounds centroid if parsing failed
                        if center_proj is None:
                            try:
                                minx, miny, maxx, maxy = buildings_proj.total_bounds
                                center_proj = Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)
                            except Exception:
                                center_proj = None
                        # Use centroid distances for polygons; if still no center, set zeros
                        if center_proj is not None:
                            data['metric'] = buildings_proj.geometry.centroid.distance(center_proj)
                        else:
                            data['metric'] = 0
                    else:
                        data['metric'] = 0

                    num_classes = len(colors)
                    metrics = data['metric'].fillna(data['metric'].median()) if has_data(data) else None
                    bins = pd.qcut(metrics, q=num_classes, duplicates='drop') if metrics is not None else None
                    if bins is not None and hasattr(bins, 'cat'):
                        data['__color__'] = [colors[i] if i >= 0 else colors[0] for i in bins.cat.codes]
                    else:
                        data['__color__'] = colors[-1]

                    # Plot vectorized colors; fallback to per-feature if needed
                    try:
                        data.plot(
                            ax=ax,
                            color=data['__color__'],
                            edgecolor=common_edge,
                            linewidth=common_linewidth,
                            alpha=common_alpha,
                            zorder=common_z
                        )
                    except Exception as e:
                        log_progress(f"Warning: Separate buildings vectorized plot failed: {e}. Falling back to per-feature plotting.")
                        for _, row in data.iterrows():
                            color_value = row.get('__color__', colors[0])
                            single = gpd.GeoDataFrame(geometry=[row.geometry], crs=getattr(data, 'crs', None))
                            single.plot(
                                ax=ax,
                                fc=color_value,
                                ec=common_edge,
                                lw=common_linewidth,
                                alpha=common_alpha,
                                zorder=common_z
                            )
                    except Exception as e:
                        log_progress(f"Warning: Separate buildings auto-coloring failed: {e}. Falling back to manual plot.")
                        manual = layer_style.get('manual_color_settings', {})
                        face = manual.get('facecolor', '#000000')
                        plot_map_layer(ax, 'buildings', data, face, common_edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
    elif layer_name == 'streets':
        params = _get_plot_params(layer_styles[layer_name], 'streets')
        # Streets are lines, so we explicitly set facecolor to none.
        plot_map_layer(ax, 'streets', data,
                       facecolor='none', edgecolor=params['edgecolor'], linewidth=params['linewidth'], alpha=params['alpha'],
                       hatch=params['hatch'], zorder=params['zorder'])
    elif layer_name == 'water':
        params = _get_plot_params(layer_styles[layer_name], 'water')
        plot_map_layer(ax, 'water', data,
                       params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                       hatch=params['hatch'], zorder=params['zorder'])
    else:
        # Get layer-specific plotting parameters
        params = _get_plot_params(layer_styles[layer_name], layer_name)
        plot_map_layer(ax, layer_name, data,
                       params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                       hatch=params['hatch'], linestyle=params['linestyle'], zorder=params['zorder'])

    out_svg_path = os.path.join(output_directory, f"{prefix}_{layer_name}{suffix}.svg")
    with contextlib.redirect_stdout(io.StringIO()):
        plt.savefig(
            out_svg_path,
            format='svg', bbox_inches='tight', pad_inches=0, transparent=transparent
        )
    plt.close(fig)
    return out_svg_path

def fetch_layer(
    query: str,
    dist: Optional[float],
    tags: Optional[Dict[str, Any]],
    is_graph: bool = False,
    custom_filter: Optional[str] = None,
    simplify_tolerance: Optional[float] = None,
    min_size_threshold: float = 0,
    layer_name_for_debug: Optional[str] = None,
    pbf_path: Optional[str] = None,
    bbox_override: Optional[Tuple[float, float, float, float]] = None,
) -> Any:
    """Delegate fetching to core.fetch.fetch_layer for a single implementation."""
    # Clarify data source path to user: local PBF vs remote via OSMnx (with cache)
    source = 'local PBF' if pbf_path else 'remote (OSMnx)'
    cache_folder = getattr(ox.settings, 'cache_folder', OSMNX_CACHE_FOLDER)
    if pbf_path:
        log_progress(f"[fetch] Layer '{layer_name_for_debug or ''}': using {source}: {pbf_path}")
    else:
        log_progress(f"[fetch] Layer '{layer_name_for_debug or ''}': using {source}; will try cache first at: {cache_folder}")
    return core_fetch_layer(
        query=query,
        dist=dist,
        tags=tags,
        is_graph=is_graph,
        custom_filter=custom_filter,
        simplify_tolerance=simplify_tolerance,
        min_size_threshold=min_size_threshold,
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

def main() -> None:
    """Main function to generate the map art."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate map art from OpenStreetMap data.')
    parser.add_argument('--prefix', type=str, help='The timestamped filename prefix for the output files.')
    parser.add_argument('--config', type=str, default=STYLE_FILE, help='Path to the style configuration file (e.g., config.json).')
    args = parser.parse_args()

    # Load configuration from the specified style file
    try:
        style = load_style(args.config)
        if not style:
            sys.exit(1)
    except FileNotFoundError:
        log_progress("Error: style.json not found. Please create one based on the instructions.")
        sys.exit(1)

    output_base_directory = style['output']['output_directory']

    # Use the prefix from the command line argument if provided, otherwise use the one from style.json
    filename_prefix = args.prefix if args.prefix else style['output']['filename_prefix']

    output_directory = os.path.join(output_base_directory, filename_prefix)
    os.makedirs(output_directory, exist_ok=True)

    # Save the current style configuration to the output directory for reproducibility
    config_output_path = os.path.join(output_directory, 'config.json')
    with open(config_output_path, 'w') as f:
        json.dump(style, f, indent=2)
    log_progress(f"Saved configuration to {config_output_path}")

    location_query = style['location']['query']
    location_distance = style['location']['distance']
    # Optional local PBF file path for offline data sourcing
    location_pbf_path = style.get('location', {}).get('pbf_path')
    # Compute bounding box from center point and distance
    location_bbox: Optional[Tuple[float, float, float, float]] = None
    raw_bbox = style.get('location', {}).get('bbox')

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
                north, south, east, west = ox.utils_geo.bbox_from_point((lat, lon), dist=float(location_distance))
                location_bbox = (west, south, east, north)
                log_progress(f"[main] Computed bbox from center+distance: {location_bbox}")
        except Exception as e:
            log_progress(f"[main] Warning: failed to compute bbox: {e}")
    # Resolve PBF path relative to the style.json location, to avoid CWD issues
    if location_pbf_path:
        style_dir = os.path.dirname(os.path.abspath(STYLE_FILE))
        if not os.path.isabs(location_pbf_path):
            resolved_pbf = os.path.abspath(os.path.join(style_dir, location_pbf_path))
        else:
            resolved_pbf = location_pbf_path
        if resolved_pbf != location_pbf_path:
            log_progress(f"[main] Resolved relative PBF path '{location_pbf_path}' -> '{resolved_pbf}'")
        else:
            log_progress(f"[main] Using absolute PBF path '{resolved_pbf}'")
        exists = os.path.isfile(resolved_pbf)
        log_progress(f"[main] PBF exists: {exists}")
        location_pbf_path = resolved_pbf
    # If a local PBF is provided, harden OSMnx to cache-only to avoid any external calls
    if location_pbf_path:
        try:
            # Configuration summary for clarity (distance in meters)
            dist_m_dbg = None if location_distance is None else int(round(location_distance))
            log_progress("--- Run Configuration ---")
            log_progress(f"Location query: '{location_query}'")
            log_progress(f"Distance (m): {dist_m_dbg}")
            log_progress(f"Local PBF: {location_pbf_path or '(none)'}")
            log_progress(f"BBox override: {location_bbox or '(none)'}")
            enabled_layers = [k for k, v in style['layers'].items() if isinstance(v, dict) and v.get('enabled')]
            log_progress(f"Enabled layers: {enabled_layers}")
            log_progress("-------------------------")

            # Enforce OSMnx cache_only when a local PBF is supplied to avoid network calls
            import osmnx as _ox  # use different alias to avoid shadowing top-level 'ox'
            _ox.settings.use_cache = True
            _ox.settings.cache_only = True
            log_progress("[main] Local PBF provided -> OSMnx set to cache_only=True (no external requests).")
        except Exception as e:
            log_progress(f"[main] Warning: failed to set OSMnx cache_only mode: {e}")


    # Get global plot settings
    figure_size = style['output'].get('figure_size', [10, 10])
    background_color = style['output'].get('background_color', 'white')
    transparent_bg = style['output'].get('transparent_background', False)
    figure_dpi = style['output'].get('figure_dpi', 300)
    margin = style['output'].get('margin', 0.05)

    # Load optional SVG optimization config once
    svg_opt_config = load_optimize_config()
    # Load per-layer tag configurations
    layer_tags_cfg = load_layer_tags()

    # --- Fetching Data ---
    log_progress("Fetching data...")
    G = None
    buildings_gdf = None
    water_gdf = None
    green_gdf = None

    # Step 1: Fetch streets first to establish a definitive bounding box.
    if style['layers']['streets']['enabled']:
        t0 = time.time()
        log_progress(f"Output directory: {output_directory}")
        log_progress("[fetch] Streets: start")
        street_filter_list = style['layers']['streets'].get('filter')
        custom_street_filter = None
        if street_filter_list:
            custom_street_filter = '["highway"~"' + '|'.join(street_filter_list) + '"]'
        G = fetch_layer(location_query, location_distance, tags=None, is_graph=True, custom_filter=custom_street_filter, pbf_path=location_pbf_path, bbox_override=location_bbox)
        dt = time.time() - t0
        try:
            edge_count = 0 if G is None else len(G.edges())
            node_count = 0 if G is None else len(G.nodes())
            log_progress(f"[fetch] Streets: done in {dt:.2f}s (nodes={node_count}, edges={edge_count})")
        except Exception:
            log_progress(f"[fetch] Streets: done in {dt:.2f}s")

    # Step 2: If a street graph was fetched, use its bbox for all other layers.
    plot_bbox = None
    if G is not None:
        try:
            nodes = ox.graph_to_gdfs(G, edges=False)
            west, south, east, north = nodes.unary_union.bounds
            location_bbox = (west, south, east, north)
            plot_bbox = location_bbox  # Use this for plotting all layers
            log_progress(f"[main] Bounding box derived from street network: {location_bbox}")
        except Exception as e:
            log_progress(f"[main] Warning: could not derive bbox from street graph: {e}")

    # Fetch buildings
    if style['layers']['buildings']['enabled']:
        t0 = time.time()
        log_progress("[fetch] Buildings: start")
        buildings_simplify_tolerance = style['layers']['buildings'].get('simplify_tolerance', None)
        # Tags from config with fallback
        buildings_tags = layer_tags_cfg.get('buildings', {'building': True})
        log_progress(f"[config] Buildings tags: {json.dumps(buildings_tags)}")
        
        # Apply layer-specific filter for buildings if present
        building_filter_list = style['layers']['buildings'].get('filter')
        if building_filter_list:
            buildings_tags['building'] = building_filter_list

        buildings_gdf = fetch_layer(location_query, location_distance, tags=buildings_tags,
                                       simplify_tolerance=buildings_simplify_tolerance,
                                       layer_name_for_debug='Buildings',
                                       pbf_path=location_pbf_path,
                                       bbox_override=location_bbox)
        dt = time.time() - t0
        try:
            feat_count = 0 if buildings_gdf is None else len(buildings_gdf)
            log_progress(f"[fetch] Buildings: done in {dt:.2f}s (features={feat_count})")
        except Exception:
            log_progress(f"[fetch] Buildings: done in {dt:.2f}s")

    # Fetch water
    if style['layers']['water']['enabled']:
        t0 = time.time()
        log_progress("[fetch] Water: start")
        water_simplify_tolerance = style['layers']['water'].get('simplify_tolerance', None)
        water_min_size_threshold = style['layers']['water'].get('min_size_threshold', 0)
        # OSM tags for water from config
        water_tags = layer_tags_cfg.get('water', {
            'natural': ['water']
        })
        log_progress(f"[config] Water tags: {json.dumps(water_tags)}")

        # Apply layer-specific filters for water if present (new format)
        water_filters = style['layers']['water'].get('filters', {})
        if water_filters:
            # Start with clean water tags and only add selected features
            water_tags = {}
            
            # Add natural water features if selected
            if 'natural' in water_filters and water_filters['natural']:
                water_tags['natural'] = water_filters['natural']
            
            # Add waterway features if selected
            if 'waterway' in water_filters and water_filters['waterway']:
                water_tags['waterway'] = water_filters['waterway']
            
            # If no features are selected, use minimal defaults
            if not water_tags:
                water_tags = {'natural': ['water']}  # Minimal fallback
                
            log_progress(f"[config] Applied water filters - Natural: {water_filters.get('natural', [])}, Waterway: {water_filters.get('waterway', [])}")
        else:
            # If no filters are set, use minimal defaults
            water_tags = {'natural': ['water']}
            log_progress("[config] No water filters set, using minimal natural water defaults only")

        log_progress(f"[config] Final water tags for fetching: {json.dumps(water_tags)}")

        try:
            water_gdf = fetch_layer(location_query, location_distance, tags=water_tags,
                                  simplify_tolerance=water_simplify_tolerance, min_size_threshold=water_min_size_threshold,
                                  layer_name_for_debug='Water',
                                  pbf_path=location_pbf_path,
                                  bbox_override=location_bbox)
            
            if water_gdf is None or water_gdf.empty:
                log_progress("[warning] No water features found with the current filters. The water layer will be empty.")
                water_gdf = None
                
        except Exception as e:
            if 'No matching features' in str(e) or 'No fallback PBF found' in str(e):
                log_progress(f"[warning] No water features found with tags: {water_tags}. The water layer will be empty.")
                water_gdf = None
            else:
                log_progress(f"[error] Error fetching water features: {e}")
                raise

    # Fetch green (parkland/greenways)
    if style['layers'].get('green', {}).get('enabled'):
        t0 = time.time()
        log_progress("[fetch] Green: start")
        green_simplify_tolerance = style['layers']['green'].get('simplify_tolerance', None)
        green_min_size_threshold = style['layers']['green'].get('min_size_threshold', 0)
        # OSM tags for green areas from config
        green_tags = layer_tags_cfg.get('green', {
            'leisure': ['park', 'garden', 'pitch', 'recreation_ground'],
            'landuse': ['grass', 'meadow', 'recreation_ground'],
            'natural': ['grassland', 'heath']
        })
        log_progress(f"[config] Green tags: {json.dumps(green_tags)}")
        green_gdf = fetch_layer(location_query, location_distance, tags=green_tags,
                                 simplify_tolerance=green_simplify_tolerance,
                                 min_size_threshold=green_min_size_threshold,
                                 layer_name_for_debug='Green',
                                 pbf_path=location_pbf_path,
                                 bbox_override=location_bbox)

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
        log_progress("Water (GeoDataFrame): No water features found with the current filters.")
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
    if style['output']['separate_layers']:
        if style['layers']['streets']['enabled']:
            t0 = time.time()
            log_progress("[save] Streets layer: start")
            streets_svg = save_layer('streets', G, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, transparent=transparent_bg)
            log_progress(f"[save] Streets layer: done in {time.time()-t0:.2f}s -> {streets_svg}")
            if svg_opt_config and streets_svg:
                try:
                    log_progress("[optimize] Streets SVG: start")
                    optimize_svg_file(streets_svg, None, svg_opt_config)
                    log_progress("[optimize] Streets SVG: done")
                except Exception as e:
                    log_progress(f"[optimize] Streets SVG skipped due to error: {e}")

            # Generate legend if enabled
            if style['layers']['streets'].get('legend_enabled', False):
                log_progress("[legend] Streets legend: start")
                legend_path = save_legend('streets', G, style['layers']['streets'], output_directory, filename_prefix, background_color, figure_dpi)
                if legend_path:
                    log_progress(f"[legend] Streets legend: done -> {legend_path}")
                else:
                    log_progress("[legend] Streets legend: skipped (no data or error)")

        if style['layers']['water']['enabled']:
            t0 = time.time()
            log_progress("[save] Water layer: start")
            water_svg = save_layer('water', water_gdf, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, transparent=transparent_bg)
            log_progress(f"[save] Water layer: done in {time.time()-t0:.2f}s -> {water_svg}")
            if svg_opt_config and water_svg:
                try:
                    log_progress("[optimize] Water SVG: start")
                    optimize_svg_file(water_svg, None, svg_opt_config)
                    log_progress("[optimize] Water SVG: done")
                except Exception as e:
                    log_progress(f"[optimize] Water SVG skipped due to error: {e}")

            # Generate legend if enabled
            if style['layers']['water'].get('legend_enabled', False):
                log_progress("[legend] Water legend: start")
                legend_path = save_legend('water', water_gdf, style['layers']['water'], output_directory, filename_prefix, background_color, figure_dpi)
                if legend_path:
                    log_progress(f"[legend] Water legend: done -> {legend_path}")
                else:
                    log_progress("[legend] Water legend: skipped (no data or error)")

        if style['layers'].get('green', {}).get('enabled'):
            t0 = time.time()
            log_progress("[save] Green layer: start")
            green_svg = save_layer('green', green_gdf, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, transparent=transparent_bg)
            log_progress(f"[save] Green layer: done in {time.time()-t0:.2f}s -> {green_svg}")
            if svg_opt_config and green_svg:
                try:
                    log_progress("[optimize] Green SVG: start")
                    optimize_svg_file(green_svg, None, svg_opt_config)
                    log_progress("[optimize] Green SVG: done")
                except Exception as e:
                    log_progress(f"[optimize] Green SVG skipped due to error: {e}")

            # Generate legend if enabled
            if style['layers']['green'].get('legend_enabled', False):
                log_progress("[legend] Green legend: start")
                legend_path = save_legend('green', green_gdf, style['layers']['green'], output_directory, filename_prefix, background_color, figure_dpi)
                if legend_path:
                    log_progress(f"[legend] Green legend: done -> {legend_path}")
                else:
                    log_progress("[legend] Green legend: skipped (no data or error)")

        if style['layers']['buildings']['enabled'] and has_data(buildings_gdf):
            t0 = time.time()
            log_progress("[save] Buildings layer: start")
            buildings_svg = save_layer('buildings', buildings_gdf, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, transparent=transparent_bg)
            log_progress(f"[save] Buildings layer: done in {time.time()-t0:.2f}s -> {buildings_svg}")
            if svg_opt_config and buildings_svg:
                try:
                    log_progress("[optimize] Buildings SVG: start")
                    optimize_svg_file(buildings_svg, None, svg_opt_config)
                    log_progress("[optimize] Buildings SVG: done")
                except Exception as e:
                    log_progress(f"[optimize] Buildings SVG skipped due to error: {e}")

            # Generate legend if enabled
            if style['layers']['buildings'].get('legend_enabled', False):
                log_progress("[legend] Buildings legend: start")
                legend_path = save_legend('buildings', buildings_gdf, style['layers']['buildings'], output_directory, filename_prefix, background_color, figure_dpi)
                if legend_path:
                    log_progress(f"[legend] Buildings legend: done -> {legend_path}")
                else:
                    log_progress("[legend] Buildings legend: skipped (no data or error)")

    # Combined output (always generated)
    t0_comb = time.time()
    log_progress("[save] Combined map: start")
    fig, ax = _setup_figure_and_axes(figure_size, figure_dpi, background_color, margin, transparent=transparent_bg)

    # Plot layers in a specific order (based on zorder from style.json)
    layers_to_plot = []
    
    # Prepare data for plotting, ensuring only enabled and existing layers are considered
    if style['layers']['water']['enabled'] and water_gdf is not None:
        layers_to_plot.append({
            'name': 'water',
            'data': water_gdf,
            'style': style['layers']['water']
        })
    
    if style['layers'].get('green', {}).get('enabled') and green_gdf is not None:
        layers_to_plot.append({
            'name': 'green',
            'data': green_gdf,
            'style': style['layers']['green']
        })

    if style['layers']['streets']['enabled'] and G is not None:
        layers_to_plot.append({
            'name': 'streets',
            'data': G,
            'style': style['layers']['streets']
        })

    if style['layers']['buildings']['enabled'] and has_data(buildings_gdf):
        layers_to_plot.append({
            'name': 'buildings',
            'data': buildings_gdf,
            'style': style['layers']['buildings']
        })

    # Sort layers by zorder (lowest zorder first, so higher zorder is drawn on top)
    layers_to_plot.sort(key=lambda x: x['style'].get('zorder', 1))

    for layer_info in layers_to_plot:
        layer_name = layer_info['name']
        data = layer_info['data']
        layer_style = layer_info['style']

        if layer_name == 'buildings':
            buildings_style_mode = layer_style.get('auto_style_mode', 'manual')
            log_progress(f"Plotting buildings with style mode: {buildings_style_mode}")
            # Common building params (apply to all modes)
            bparams = _get_building_common_params(layer_style)
            common_linewidth = bparams['linewidth']
            common_alpha = bparams['alpha']
            common_hatch = bparams['hatch']
            common_z = bparams['zorder']

            if buildings_style_mode == 'manual':
                manual = layer_style.get('manual_color_settings', {})
                face = manual.get('facecolor', '#000000')
                edge = bparams['edgecolor']
                plot_map_layer(ax, 'buildings', data, face, edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)

            elif buildings_style_mode == 'manual_floorsize':
                # Work on a copy to avoid mutating the original GDF
                data_local = data.copy()
                # Compute area metric via helper
                metric = _compute_buildings_metric(data_local, 'area')

                # Match against each category's min/max and color using category facecolor
                categories = layer_style.get('size_categories', [])
                if categories:
                    manual = layer_style.get('manual_color_settings', {})
                    default_face = manual.get('facecolor', '#000000')
                    color_series = pd.Series(default_face, index=data_local.index)
                    for category in categories:
                        min_area = category.get('min_area', 0)
                        max_area = category.get('max_area', float('inf'))
                        facecolor = category.get('facecolor', default_face)
                        # Inclusive bounds on both ends
                        mask = (metric >= float(min_area)) & (metric <= float(max_area))
                        color_series.loc[mask] = facecolor

                    # Plot by color using vectorized colors for reliability
                    try:
                        data_local.plot(
                            ax=ax,
                            color=color_series,
                            edgecolor=layer_style.get('edgecolor', '#000000'),
                            linewidth=common_linewidth,
                            alpha=common_alpha,
                            zorder=common_z
                        )
                    except Exception as e:
                        log_progress(f"Warning: Failed to plot grouped colors for buildings: {e}. Falling back to per-feature plotting.")
                        # Per-feature plotting fallback to ensure colors apply even if grouping fails
                        try:
                            for idx, row in data_local.iterrows():
                                single = gpd.GeoDataFrame(geometry=[row.geometry], crs=getattr(data_local, 'crs', None))
                                single.plot(
                                    ax=ax,
                                    fc=color_series.loc[idx],
                                    ec=layer_style.get('edgecolor', '#000000'),
                                    lw=common_linewidth,
                                    alpha=common_alpha,
                                    zorder=common_z
                                )
                        except Exception as e2:
                            log_progress(f"Warning: Per-feature plotting also failed: {e2}. Falling back to manual plot.")
                            manual = layer_style.get('manual_color_settings', {})
                            face = manual.get('facecolor', '#000000')
                            edge = layer_style.get('edgecolor', '#000000')
                            plot_map_layer(ax, 'buildings', data_local, face, edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)

            elif buildings_style_mode in ['auto_size', 'auto_distance']:
                # Correct palette key mapping
                palette_key = 'auto_size_palette' if buildings_style_mode == 'auto_size' else 'auto_distance_palette'
                palette_name = layer_style.get(palette_key)
                palettes = load_palettes()
                # Defensive: ensure dict and handle missing key gracefully
                if not isinstance(palettes, dict):
                    palettes = {}
                colors = (palettes.get(palette_name) or [])

                if not colors:
                    log_progress(f"Warning: Palette '{palette_name}' not found or empty for '{buildings_style_mode}' mode. Falling back to manual color.")
                    manual = layer_style.get('manual_color_settings', {})
                    face = manual.get('facecolor', '#000000')
                    edge = layer_style.get('edgecolor', '#000000')
                    plot_map_layer(ax, 'buildings', data, face, edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
                    continue

                # Reverse palette so we start with darker colors first
                try:
                    colors = list(colors)[::-1]
                except Exception:
                    colors = list(colors)
                log_progress(f"Auto mode '{buildings_style_mode}' using palette '{palette_name}' (reversed) with {len(colors)} colors")

                if buildings_style_mode == 'auto_size':
                    # Reproject to calculate area accurately
                    buildings_proj, _ = _reproject_gdf_for_area_calc(data)
                    if has_data(buildings_proj):
                        data['metric'] = buildings_proj.geometry.area
                    else:
                        data['metric'] = 0
                else: # auto_distance
                    # Compute distances from the map center in projected units for accuracy
                    buildings_proj, _ = _reproject_gdf_for_area_calc(data)
                    if has_data(buildings_proj):
                        center_proj = None
                        # Try to parse center from location_query ("lat lon")
                        try:
                            q = (location_query or '').replace(',', ' ').split()
                            if len(q) >= 2:
                                lat_c = float(q[0]); lon_c = float(q[1])
                                # Build WGS84 point then project to buildings_proj CRS
                                center_wgs = gpd.GeoSeries([Point(lon_c, lat_c)], crs='EPSG:4326')
                                center_proj_geom = center_wgs.to_crs(buildings_proj.crs).iloc[0]
                                center_proj = center_proj_geom
                        except Exception:
                            center_proj = None
                        # Fallback: use bounds centroid if parsing failed
                        if center_proj is None:
                            try:
                                minx, miny, maxx, maxy = buildings_proj.total_bounds
                                center_proj = Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)
                            except Exception:
                                center_proj = None
                        # Use centroid distances for polygons; if still no center, set zeros
                        if center_proj is not None:
                            data['metric'] = buildings_proj.geometry.centroid.distance(center_proj)
                        else:
                            data['metric'] = 0
                    else:
                        data['metric'] = 0

                # Map metric to palette colors using quantile bins
                try:
                    num_classes = len(colors)
                    # Use qcut to distribute features across classes
                    # Ensure metric has no NaNs
                    if has_data(data):
                        metrics = data['metric'].fillna(data['metric'].median())
                        bins = pd.qcut(metrics, q=num_classes, duplicates='drop')
                    else:
                        bins = None
                    if bins is not None and hasattr(bins, 'cat'):
                        data['__color__'] = [colors[i] if i >= 0 else colors[0] for i in bins.cat.codes]
                    else:
                        # Single-class fallback: darkest color
                        data['__color__'] = colors[0]
                except Exception as e:
                    log_progress(f"Warning: Separate buildings auto binning failed: {e}. Using darkest palette color.")
                    data['__color__'] = colors[0]

                # Plot by color using vectorized colors for reliability
                try:
                    if '__color__' in data:
                        # Debug: count classes
                        try:
                            class_counts = data['__color__'].value_counts().to_dict()
                            log_progress(f"Assigned colors/classes: {class_counts}")
                        except Exception:
                            pass
                        data.plot(
                            ax=ax,
                            color=data['__color__'],
                            edgecolor=layer_style.get('edgecolor', '#000000'),
                            linewidth=common_linewidth,
                            alpha=common_alpha,
                            zorder=common_z
                        )
                    else:
                        # Fallback single-color plot
                        manual = layer_style.get('manual_color_settings', {})
                        face = manual.get('facecolor', '#000000')
                        edge = layer_style.get('edgecolor', '#000000')
                        plot_map_layer(ax, 'buildings', data, face, edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
                except Exception as e:
                    log_progress(f"Warning: Failed to plot grouped colors for buildings: {e}. Falling back to per-feature plotting.")
                    # Per-feature plotting fallback to ensure colors apply even if grouping fails
                    try:
                        for idx, row in data.iterrows():
                            color_value = row.get('__color__', colors[-1])
                            single = gpd.GeoDataFrame(geometry=[row.geometry], crs=getattr(data, 'crs', None))
                            single.plot(
                                ax=ax,
                                fc=color_value,
                                ec=layer_style.get('edgecolor', '#000000'),
                                lw=common_linewidth,
                                alpha=common_alpha,
                                zorder=common_z
                            )
                    except Exception as e2:
                        log_progress(f"Warning: Per-feature plotting also failed: {e2}. Falling back to manual plot.")
                        manual = layer_style.get('manual_color_settings', {})
                        face = manual.get('facecolor', '#000000')
                        edge = layer_style.get('edgecolor', '#000000')
                        plot_map_layer(ax, 'buildings', data, face, edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
        elif layer_name == 'streets':
            params = _get_plot_params(layer_style)
            plot_map_layer(ax, 'streets', data,
                           params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                           hatch=params['hatch'], zorder=params['zorder'])
        elif layer_name == 'water':
            params = _get_plot_params(layer_style)
            plot_map_layer(ax, 'water', data,
                           params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                           hatch=params['hatch'], zorder=params['zorder'])
        elif layer_name == 'green':
            params = _get_plot_params(layer_style)
            plot_map_layer(ax, 'green', data,
                           params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                           hatch=params['hatch'], zorder=params['zorder'])

    combined_svg_path = os.path.join(output_directory, f"{filename_prefix}_combined.svg")
    with contextlib.redirect_stdout(io.StringIO()):
        plt.savefig(
            combined_svg_path,
            format='svg', bbox_inches='tight', pad_inches=0, transparent=transparent_bg
        )
    plt.close(fig)
    log_progress(f"[save] Combined map: done in {time.time()-t0_comb:.2f}s -> {combined_svg_path}")

    # Post-process combined SVG if configured
    if svg_opt_config and combined_svg_path:
        try:
            log_progress("[optimize] Combined SVG: start")
            optimize_svg_file(combined_svg_path, None, svg_opt_config)
            log_progress("[optimize] Combined SVG: done")
        except Exception as e:
            log_progress(f"[optimize] Combined SVG skipped due to error: {e}")

if __name__ == "__main__":
    main()
