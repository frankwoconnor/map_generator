import os
import json
from typing import Any, Dict, List, Optional, Tuple
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import multiprocessing
import contextlib
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

STYLE_FILE = 'style.json'

if sys.platform == 'darwin':
    multiprocessing.set_start_method('fork')
    
# Delegate common utilities to core modules for maintainability
from maps2.core.plot import (
    setup_figure_and_axes as core_setup_figure_and_axes,
    plot_map_layer as core_plot_map_layer,
)
from maps2.core.svg_post import load_optimize_config, optimize_svg_file
from maps2.core.util import (
    log_progress as core_log_progress,
    has_data as core_has_data,
)
import maps2.core.config as cfg
from maps2.core.fetch import fetch_layer as core_fetch_layer
from maps2.core.buildings import compute_metric as core_compute_buildings_metric

def log_progress(message: str) -> None:
    """Lightweight stdout logger proxying to core util."""
    core_log_progress(message)

PALETTES_FILE = 'palettes.json'
_PALETTES_CACHE = None

def load_palettes() -> Dict[str, List[str]]:
    """Load palettes from palettes.json.

    Returns:
        dict[str, list[str]]: Mapping of palette name to list of hex colors.
    Fallback:
        A small default dict if file is missing or invalid.
    """
    global _PALETTES_CACHE
    if _PALETTES_CACHE is not None:
        return _PALETTES_CACHE
    try:
        with open(PALETTES_FILE, 'r') as f:
            palettes = json.load(f)
            if isinstance(palettes, dict):
                _PALETTES_CACHE = palettes
                return _PALETTES_CACHE
    except Exception:
        pass
    # Fallback minimal palettes
    _PALETTES_CACHE = {
        'OrRd_3': ['#fee8c8', '#fdbb84', '#e34a33'],
        'YlGnBu_3': ['#edf8fb', '#b2e2e2', '#66c2a4']
    }
    return _PALETTES_CACHE

# --- Helper Functions ---

def has_data(data: Any) -> bool:
    """Proxy to core util has_data to keep a single source of truth."""
    return core_has_data(data)

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

def plot_map_layer(
    ax: Any,
    layer_name: str,
    data: Any,
    facecolor: Any,
    edgecolor: Any,
    linewidth: float,
    alpha: float,
    hatch: Optional[str] = None,
    linestyle: str = '-',
    zorder: int = 1,
) -> None:
    """Delegate plotting to core.plot to centralize behavior."""
    core_plot_map_layer(
        ax=ax,
        layer_name=layer_name,
        data=data,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        hatch=hatch,
        linestyle=linestyle,
        zorder=zorder,
    )

def _get_plot_params(layer_style: Dict[str, Any]) -> Dict[str, Any]:
    """Extract plotting parameters from a layer's style dictionary."""
    return {
        'facecolor': layer_style.get('facecolor', '#000000'),
        'edgecolor': layer_style.get('edgecolor', '#000000'),
        'linewidth': layer_style.get('linewidth', 0.5),
        'alpha': layer_style.get('alpha', 1.0),
        'hatch': layer_style.get('hatch', None),
        'linestyle': layer_style.get('linestyle', '-'),
        'zorder': layer_style.get('zorder', 1)
    }

def _get_building_common_params(layer_style: Dict[str, Any]) -> Dict[str, Any]:
    """Extract common building plotting params applied across all building modes."""
    return {
        'edgecolor': layer_style.get('edgecolor', '#000000'),
        'linewidth': layer_style.get('linewidth', 0.5),
        'alpha': layer_style.get('alpha', 1.0),
        'hatch': layer_style.get('hatch', None),
        'zorder': layer_style.get('zorder', 2),
    }

def _compute_buildings_metric(data: gpd.GeoDataFrame, metric: str) -> pd.Series:
    """Delegate to core.buildings.compute_metric for a single implementation."""
    return core_compute_buildings_metric(data, metric)

def _setup_figure_and_axes(
    figure_size: List[float],
    figure_dpi: int,
    background_color: str,
    margin: float,
    transparent: bool = False,
) -> Tuple[Any, Any]:
    """Delegate figure/axes setup to core.plot to keep one implementation."""
    return core_setup_figure_and_axes(
        figure_size=figure_size,
        figure_dpi=figure_dpi,
        background_color=background_color,
        margin=margin,
        transparent=transparent,
    )

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
) -> None:
    """Save a single map layer to an SVG file."""
    if not has_data(data):
        return

    fig, ax = _setup_figure_and_axes(figure_size, figure_dpi, background_color, margin, transparent=transparent)

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
            colors = palettes.get(palette_name)

            if not colors:
                log_progress(f"Warning: Palette '{palette_name}' not found or empty for '{buildings_style_mode}' mode in separate save. Falling back to manual color.")
                manual = layer_style.get('manual_color_settings', {})
                face = manual.get('facecolor', '#000000')
                plot_map_layer(ax, 'buildings', data, face, common_edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
            else:
                log_progress(f"Separate buildings auto '{buildings_style_mode}' using palette '{palette_name}' with {len(colors)} colors")
                try:
                    # Compute metric once via helper
                    metric_name = 'area' if buildings_style_mode == 'auto_size' else 'distance'
                    data = data.copy()
                    data['metric'] = _compute_buildings_metric(data, metric_name)

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
                            color_value = row.get('__color__', colors[-1])
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
        params = _get_plot_params(layer_styles[layer_name])
        plot_map_layer(ax, 'streets', data,
                       params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                       hatch=params['hatch'], zorder=params['zorder'])
    elif layer_name == 'water':
        params = _get_plot_params(layer_styles[layer_name])
        plot_map_layer(ax, 'water', data,
                       params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                       hatch=params['hatch'], zorder=params['zorder'])
    else:
        # Get layer-specific plotting parameters
        params = _get_plot_params(layer_styles[layer_name])
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
) -> Any:
    """Delegate fetching to core.fetch.fetch_layer for a single implementation."""
    return core_fetch_layer(
        query=query,
        dist=dist,
        tags=tags,
        is_graph=is_graph,
        custom_filter=custom_filter,
        simplify_tolerance=simplify_tolerance,
        min_size_threshold=min_size_threshold,
        layer_name_for_debug=layer_name_for_debug,
    )

def _compute_buildings_metric(data: gpd.GeoDataFrame, metric: str) -> pd.Series:
    """Delegate to core.buildings.compute_metric for a single implementation."""
    return core_compute_buildings_metric(data, metric)

def load_style() -> Dict[str, Any]:
    """Load, normalize, and (optionally) validate the style.json file."""
    raw = cfg.load_style(STYLE_FILE)
    if raw is None:
        log_progress("Error: style.json not found or invalid.")
        return None
    style = cfg.normalize_style(raw)
    # Validate if a schema is available (no-op otherwise)
    err = cfg.validate_style(style)
    if err:
        log_progress(f"Schema validation warning: {err}")
    print("Loaded style.json:", style)  # Debugging log surfaced to console
    return style

def main() -> None:
    """Main function to generate the map art."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate map art from OpenStreetMap data.')
    parser.add_argument('--prefix', type=str, help='The timestamped filename prefix for the output files.')
    args = parser.parse_args()

    # Load configuration from style.json
    try:
        style = load_style()
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

    street_filter_list = style.get('processing', {}).get('street_filter', None)

    # Construct custom filter string for streets
    custom_street_filter = None
    if street_filter_list:
        # OSMnx custom_filter expects a string like '["highway"~"motorway|trunk"]'.
        # We join the list elements with '|' to create the regex part.
        custom_street_filter = '["highway"~"' + '|'.join(street_filter_list) + '"]'

    # Get global plot settings
    figure_size = style['output'].get('figure_size', [10, 10])
    background_color = style['output'].get('background_color', 'white')
    transparent_bg = style['output'].get('transparent_background', False)
    figure_dpi = style['output'].get('figure_dpi', 300)
    margin = style['output'].get('margin', 0.05)

    # Load optional SVG optimization config once
    svg_opt_config = load_optimize_config()

    # --- Fetching Data ---
    log_progress("Fetching data...")
    G = None
    buildings_gdf = None
    water_gdf = None
    green_gdf = None

    # Fetch streets
    if style['layers']['streets']['enabled']:
        log_progress("Fetching street data...")
        G = fetch_layer(location_query, location_distance, tags=None, is_graph=True, custom_filter=custom_street_filter)

    # Fetch buildings
    if style['layers']['buildings']['enabled']:
        log_progress("Fetching building data...")
        buildings_simplify_tolerance = style['layers']['buildings'].get('simplify_tolerance', None)
        # min_size_threshold is not used directly here if size_categories are present
        buildings_gdf = fetch_layer(location_query, location_distance, tags={'building': True},
                                    simplify_tolerance=buildings_simplify_tolerance,
                                    layer_name_for_debug='Buildings')

    # Fetch water
    if style['layers']['water']['enabled']:
        log_progress("Fetching water data...")
        water_simplify_tolerance = style['layers']['water'].get('simplify_tolerance', None)
        water_min_size_threshold = style['layers']['water'].get('min_size_threshold', 0)
        water_gdf = fetch_layer(location_query, location_distance, tags={'natural': 'water'},
                                simplify_tolerance=water_simplify_tolerance, min_size_threshold=water_min_size_threshold,
                                layer_name_for_debug='Water')

    # Fetch green (parkland/greenways)
    if style['layers'].get('green', {}).get('enabled'):
        log_progress("Fetching green (parkland/greenways) data...")
        green_simplify_tolerance = style['layers']['green'].get('simplify_tolerance', None)
        green_min_size_threshold = style['layers']['green'].get('min_size_threshold', 0)
        # OSM tags for green areas
        green_tags = {
            'leisure': ['park', 'garden', 'pitch', 'recreation_ground'],
            'landuse': ['grass', 'meadow', 'recreation_ground'],
            'natural': ['grassland', 'heath']
        }
        green_gdf = fetch_layer(location_query, location_distance, tags=green_tags,
                                simplify_tolerance=green_simplify_tolerance,
                                min_size_threshold=green_min_size_threshold,
                                layer_name_for_debug='Green')

    # --- Debugging: Print Layer Information ---
    log_progress("--- Map Layer Information ---")

    if has_data(buildings_gdf):
        log_progress("Buildings (GeoDataFrame):")
        for geom_type, count in buildings_gdf.geometry.geom_type.value_counts().items():
            log_progress(f"  - {geom_type}: {count} features")
    else:
        log_progress("Buildings (GeoDataFrame): Not enabled or no data.")

    if has_data(water_gdf):
        log_progress("Water (GeoDataFrame):")
        for geom_type, count in water_gdf.geometry.geom_type.value_counts().items():
            log_progress(f"  - {geom_type}: {count} features")
    else:
        log_progress("Water (GeoDataFrame): Not enabled or no data.")
    
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
            log_progress("Saving streets layer...")
            streets_svg = save_layer('streets', G, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, transparent=transparent_bg)
            if svg_opt_config:
                optimize_svg_file(streets_svg, None, svg_opt_config)

        if style['layers']['water']['enabled']:
            log_progress("Saving water layer...")
            water_svg = save_layer('water', water_gdf, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, transparent=transparent_bg)
            if svg_opt_config:
                optimize_svg_file(water_svg, None, svg_opt_config)

        if style['layers'].get('green', {}).get('enabled'):
            log_progress("Saving green layer...")
            green_svg = save_layer('green', green_gdf, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, transparent=transparent_bg)
            if svg_opt_config:
                optimize_svg_file(green_svg, None, svg_opt_config)

        if style['layers']['buildings']['enabled'] and has_data(buildings_gdf):
            log_progress("Saving buildings layer...")
            buildings_svg = save_layer('buildings', buildings_gdf, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, transparent=transparent_bg)
            if svg_opt_config:
                optimize_svg_file(buildings_svg, None, svg_opt_config)

    # Combined output (always generated)
    log_progress("Saving combined map...")
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
                colors = palettes.get(palette_name)

                if not colors:
                    log_progress(f"Warning: Palette '{palette_name}' not found or empty for '{buildings_style_mode}' mode. Falling back to manual color.")
                    manual = layer_style.get('manual_color_settings', {})
                    face = manual.get('facecolor', '#000000')
                    edge = layer_style.get('edgecolor', '#000000')
                    plot_map_layer(ax, 'buildings', data, face, edge, common_linewidth, common_alpha, hatch=common_hatch, zorder=common_z)
                    continue

                log_progress(f"Auto mode '{buildings_style_mode}' using palette '{palette_name}' with {len(colors)} colors")

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
                        center_proj = buildings_proj.unary_union.centroid
                        # Use centroid distances for polygons
                        data['metric'] = buildings_proj.geometry.centroid.distance(center_proj)
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
                        # Single-class fallback: apply a uniform non-black color from the palette
                        data['__color__'] = colors[-1]
                except Exception as e:
                    log_progress(f"Warning: Failed to bin metrics for '{buildings_style_mode}': {e}. Using last palette color.")
                    data['__color__'] = colors[-1]

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
    log_progress(f"Combined map saved to {output_directory}")

    # Post-process combined SVG if configured
    if svg_opt_config:
        optimize_svg_file(combined_svg_path, None, svg_opt_config)

if __name__ == "__main__":
    main()
