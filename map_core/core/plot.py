from __future__ import annotations

import io
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
from matplotlib import cm
from matplotlib.collections import LineCollection, PatchCollection
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from map_core.core.util import has_data, log_progress

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
        ax.set_facecolor('none')
    else:
        ax.set_facecolor(background_color)
    ax.set_axis_off()
    ax.margins(margin)
    fig.tight_layout(pad=0)
    return fig, ax


def create_map_legend(
    style: Dict[str, Any],
    layer_data: Dict[str, Any],
    parent: bool = False,
    figure_size: List[float] = None,
    figure_dpi: int = 300,
    background_color: str = 'white',
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
    return setup_figure_and_axes(figure_size, figure_dpi, background_color, margin, transparent)


def _add_legend_item(handles: List, labels: List, color: str, edgecolor: str, linewidth: float, label: str, is_line: bool = False) -> None:
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
        item = mpatches.Rectangle((0, 0), 1, 0.1, facecolor=color, edgecolor='none', label=label)
    else:
        item = mpatches.Patch(facecolor=color, edgecolor=edgecolor, linewidth=linewidth, label=label)
    handles.append(item)
    labels.append(label)

def generate_layer_legend(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    figure_size: List[float] = None,
    figure_dpi: int = 150,
    background_color: str = 'white',
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
        ax.text(0.5, 0.5, f'No data available for {layer_name}',
                ha='center', va='center', fontsize=12)
        return fig, ax

    try:
        if layer_name == 'water':
            # Extract unique water feature types
            if hasattr(data, 'geometry'):
                # Group by natural/waterway tags if available
                unique_features = {}
                if 'natural' in data.columns:
                    for val in data['natural'].dropna().unique():
                        unique_features[f'Natural: {val}'] = layer_style.get('facecolor', '#a6cee3')
                if 'waterway' in data.columns:
                    for val in data['waterway'].dropna().unique():
                        unique_features[f'Waterway: {val}'] = layer_style.get('edgecolor', '#a6cee3')

                # If no tags, just show generic water
                if not unique_features:
                    unique_features = {'Water features': layer_style.get('facecolor', '#a6cee3')}

                for label, color in unique_features.items():
                    _add_legend_item(legend_handles, legend_labels, color, layer_style.get('edgecolor', color),
                                   layer_style.get('linewidth', 0.5), label)

        elif layer_name == 'streets':
            # Extract unique highway types from street network
            try:
                # Convert graph to edges GeoDataFrame
                if hasattr(data, 'edges'):
                    nodes, edges = ox.graph_to_gdfs(data)
                    if 'highway' in edges.columns:
                        unique_highways = edges['highway'].dropna().unique()
                        # Handle list values
                        highway_types = set()
                        for hw in unique_highways:
                            if isinstance(hw, list):
                                highway_types.update(hw)
                            else:
                                highway_types.add(hw)

                        for hw_type in sorted(highway_types):
                            _add_legend_item(legend_handles, legend_labels, layer_style.get('edgecolor', '#000000'),
                                           'none', 0.5, hw_type.capitalize(), is_line=True)
            except Exception as e:
                log_progress(f"Warning: Could not extract street types: {e}")
                _add_legend_item(legend_handles, legend_labels, layer_style.get('edgecolor', '#000000'),
                               'none', 0.5, 'Streets', is_line=True)

        elif layer_name == 'buildings':
            # Show building color scheme
            auto_mode = layer_style.get('auto_style_mode', 'manual')
            if auto_mode == 'manual':
                color = layer_style.get('manual_color_settings', {}).get('facecolor', '#000000')
                _add_legend_item(legend_handles, legend_labels, color, layer_style.get('edgecolor', '#000000'),
                               layer_style.get('linewidth', 0.5), 'Buildings')
            elif auto_mode in ['auto_size', 'auto_distance']:
                # Show the palette colors
                from map_core.core.palettes import load_palettes
                palette_key = 'auto_size_palette' if auto_mode == 'auto_size' else 'auto_distance_palette'
                palette_name = layer_style.get(palette_key, '')
                palettes = load_palettes()
                colors = palettes.get(palette_name, []) if isinstance(palettes, dict) else []

                if colors:
                    colors_reversed = list(colors)[::-1]
                    labels = ['Smallest' if auto_mode == 'auto_size' else 'Farthest']
                    labels.extend([''] * (len(colors_reversed) - 2))
                    labels.append('Largest' if auto_mode == 'auto_size' else 'Nearest')

                    for color, label in zip(colors_reversed, labels):
                        _add_legend_item(legend_handles, legend_labels, color, layer_style.get('edgecolor', '#000000'),
                                       layer_style.get('linewidth', 0.5), label if label else '')
            elif auto_mode == 'manual_floorsize':
                categories = layer_style.get('size_categories', [])
                for cat in categories:
                    min_area = cat.get('min_area', 0)
                    max_area = cat.get('max_area', '∞')
                    label = f"{min_area}-{max_area} m²"
                    _add_legend_item(legend_handles, legend_labels, cat.get('facecolor', '#000000'),
                                   layer_style.get('edgecolor', '#000000'), layer_style.get('linewidth', 0.5), label)

        elif layer_name == 'green':
            # Check for leisure/landuse/natural tags
            unique_features = {}
            if hasattr(data, 'geometry'):
                for tag in ['leisure', 'landuse', 'natural']:
                    if tag in data.columns:
                        for val in data[tag].dropna().unique():
                            unique_features[f'{tag.capitalize()}: {val}'] = layer_style.get('facecolor', '#b2df8a')

                if not unique_features:
                    unique_features = {'Green spaces': layer_style.get('facecolor', '#b2df8a')}

                for label, color in unique_features.items():
                    _add_legend_item(legend_handles, legend_labels, color, layer_style.get('edgecolor', color),
                                   layer_style.get('linewidth', 0.5), label)

        # Create the legend
        if legend_handles:
            ax.legend(handles=legend_handles,
                     title=f'{layer_name.capitalize()} Legend',
                     loc='center',
                     fontsize=10,
                     title_fontsize=14,
                     frameon=True,
                     fancybox=True,
                     shadow=True)
        else:
            ax.text(0.5, 0.5, f'No legend items for {layer_name}',
                   ha='center', va='center', fontsize=12)

    except Exception as e:
        log_progress(f"Error generating legend for {layer_name}: {e}")
        ax.text(0.5, 0.5, f'Error generating legend:\n{str(e)}',
               ha='center', va='center', fontsize=10, wrap=True)

    return fig, ax


def add_map_legend(ax: Any, layer_data: Dict[str, Any], style: Dict[str, Any]) -> None:
    """Add a legend to the map based on the style configuration and layer data.
    
    Args:
        ax: The matplotlib axes to add the legend to
        layer_data: Dictionary containing the GeoDataFrames for each layer
        style: The style configuration dictionary
    """
    if not style.get('legend', {}).get('enabled', True):
        return
        
    legend_config = style.get('legend', {})
    legend_handles = []
    
    # Add water features to legend
    if 'water' in layer_data and has_data(layer_data['water']):
        water_style = style.get('layers', {}).get('water', {})
        
        # Add water bodies (polygons)
        water_patches = []
        if 'filters' in water_style and 'natural' in water_style['filters']:
            for water_type in water_style['filters']['natural']:
                if water_type in ['lake', 'pond', 'water', 'reservoir']:
                    patch = mpatches.Patch(
                        facecolor=water_style.get('facecolor', '#479711'),
                        edgecolor=water_style.get('edgecolor', '#f70202'),
                        linewidth=water_style.get('linewidth', 1.0),
                        label=f"{water_type.capitalize()}"
                    )
                    water_patches.append(patch)
        
        # Add waterways (lines)
        if 'filters' in water_style and 'waterway' in water_style['filters']:
            for waterway_type in water_style['filters']['waterway']:
                if waterway_type in ['river', 'stream', 'canal']:
                    line = mpatches.Rectangle(
                        (0, 0), 1, 0.2,  # width, height
                        facecolor=water_style.get('edgecolor', '#479711'),
                        edgecolor='none',
                        label=f"{waterway_type.capitalize()}"
                    )
                    water_patches.append(line)
        
        if water_patches:
            legend_handles.extend(water_patches)
    
    # Add green spaces to legend
    if 'green' in layer_data and has_data(layer_data['green']):
        green_style = style.get('layers', {}).get('green', {})
        if green_style.get('enabled', False):
            patch = mpatches.Patch(
                facecolor=green_style.get('facecolor', '#4CAF50'),
                edgecolor=green_style.get('edgecolor', '#000000'),
                linewidth=green_style.get('linewidth', 0.5),
                label='Green Space'
            )
            legend_handles.append(patch)
    
    # Add buildings to legend
    if 'buildings' in layer_data and has_data(layer_data['buildings']):
        building_style = style.get('layers', {}).get('buildings', {})
        if building_style.get('enabled', False):
            patch = mpatches.Patch(
                facecolor=building_style.get('facecolor', '#808080'),
                edgecolor=building_style.get('edgecolor', '#000000'),
                linewidth=building_style.get('linewidth', 0.5),
                label='Buildings'
            )
            legend_handles.append(patch)
    
    # Add streets to legend
    if 'streets' in layer_data and has_data(layer_data['streets']):
        street_style = style.get('layers', {}).get('streets', {})
        if street_style.get('enabled', False):
            line = mpatches.Rectangle(
                (0, 0), 1, 0.1,  # width, height
                facecolor=street_style.get('edgecolor', '#000000'),
                edgecolor='none',
                label='Streets'
            )
            legend_handles.append(line)
    
    # Only add legend if we have items to show
    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            title=legend_config.get('title', 'Map Legend'),
            loc=legend_config.get('position', 'lower right'),
            fontsize=legend_config.get('font_size', 10),
            title_fontsize=legend_config.get('title_size', 12),
            frameon=legend_config.get('fancybox', True),
            fancybox=legend_config.get('fancybox', True),
            shadow=legend_config.get('shadow', True),
            borderpad=legend_config.get('borderpad', 0.8),
            labelspacing=legend_config.get('labelspacing', 0.8),
            handlelength=legend_config.get('handlelength', 1.5),
            handletextpad=legend_config.get('handletextpad', 0.8),
            borderaxespad=legend_config.get('borderaxespad', 0.8)
        )
        
        # Set legend background color and transparency
        frame = legend.get_frame()
        frame.set_facecolor(legend_config.get('background_color', '#ffffff'))
        frame.set_edgecolor(legend_config.get('edge_color', '#000000'))
        frame.set_alpha(legend_config.get('alpha', 0.9))


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
    """Plot a street network or generic GeoDataFrame on the given axis.

    Streets are drawn using the edges GeoDataFrame to avoid background overrides.
    Other layers are plotted as polygons/lines via GeoPandas.
    """
    from map_core.core.util import has_data, log_progress  # local import to avoid cycles

    if not has_data(data):
        return

    if layer_name == 'streets':
        # If data is already an edges GeoDataFrame, plot directly
        if hasattr(data, 'geometry') and not hasattr(data, 'graph'):
            try:
                data.plot(ax=ax, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
                return
            except Exception as e:
                log_progress(f"Warning: failed to plot streets from GeoDataFrame directly: {e}")
        # Otherwise, assume a NetworkX graph and convert to edges GDF
        try:
            nodes_gdf, gdf_edges = ox.graph_to_gdfs(data)
            if has_data(gdf_edges):
                gdf_edges.plot(ax=ax, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
                return
        except Exception as e:
            log_progress(f"Warning: failed to plot streets via edges GDF ({e}); falling back to ox.plot_graph.")
            try:
                ox.plot_graph(data, ax=ax, show=False, close=False,
                              edge_color=edgecolor, edge_linewidth=linewidth, edge_alpha=alpha, node_size=0)
            except Exception as e2:
                log_progress(f"Warning: ox.plot_graph failed as well: {e2}")
            return
    else:
        # For water layers, handle different geometry types with smart width calculation
        if layer_name == 'water' and has_data(data):
            _plot_water_features(ax, data, facecolor, edgecolor, linewidth, alpha, hatch, zorder)
        else:
            # Default plotting for other layers
            data.plot(ax=ax, fc=facecolor, ec=edgecolor, lw=linewidth,
                     alpha=alpha, hatch=hatch, zorder=zorder)


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

    if not hasattr(data, 'geometry'):
        log_progress("Warning: No geometry column found in water data")
        return

    # Make a copy to avoid modifying the original
    gdf = data.copy()

    # Store features with names for labeling
    named_features = []

    # Drop empty and invalid geometries
    gdf = gdf[~gdf.is_empty]
    if not all(gdf.geometry.is_valid):
        log_progress(f"Found {sum(~gdf.geometry.is_valid)} invalid geometries in water layer, fixing...")
        gdf.geometry = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.is_valid]

    if gdf.empty:
        return

    # Define water colors based on type (use a nice blue palette)
    water_blue = facecolor if facecolor != '#000000' else '#88ccee'
    river_blue = '#5599cc'
    stream_blue = '#77bbee'

    # Separate polygons and lines
    polys = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    lines = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]

    # 1. Plot polygon water bodies (lakes, ponds, reservoirs, river polygons)
    if not polys.empty:
        # Differentiate between river polygons and other water bodies
        river_polys = polys[polys.get('water', '') == 'river'] if 'water' in polys.columns else pd.DataFrame()
        other_polys = polys[polys.get('water', '') != 'river'] if 'water' in polys.columns else polys

        # Plot regular water bodies (lakes, ponds, etc.)
        if not other_polys.empty:
            other_polys.plot(ax=ax, fc=water_blue, ec=edgecolor, lw=linewidth,
                           alpha=alpha, hatch=hatch, zorder=zorder)

        # Plot river polygons with slightly different color
        if not river_polys.empty:
            river_polys.plot(ax=ax, fc=river_blue, ec=edgecolor, lw=linewidth,
                           alpha=alpha, hatch=hatch, zorder=zorder)

    # 2. Plot waterway lines with width-based rendering
    if not lines.empty:
        # Calculate default scale: 1 point ≈ 5 meters (configurable)
        meters_per_point = 5.0
        default_river_width_m = 10.0
        default_stream_width_m = 3.0
        default_canal_width_m = 8.0

        # Separate by waterway type
        rivers = lines[lines.get('waterway', '') == 'river'] if 'waterway' in lines.columns else pd.DataFrame()
        streams = lines[lines.get('waterway', '') == 'stream'] if 'waterway' in lines.columns else pd.DataFrame()
        canals = lines[lines.get('waterway', '') == 'canal'] if 'waterway' in lines.columns else pd.DataFrame()
        other_waterways = lines[~lines.index.isin(rivers.index) &
                                ~lines.index.isin(streams.index) &
                                ~lines.index.isin(canals.index)]

        # Plot rivers with width consideration
        if not rivers.empty:
            for idx, row in rivers.iterrows():
                # Try to get width from tags
                width_m = default_river_width_m
                if 'width' in rivers.columns and pd.notna(row.get('width')):
                    try:
                        # Width might be "5", "5 m", "5m", etc.
                        width_str = str(row['width']).lower().replace('m', '').replace(' ', '').strip()
                        width_m = float(width_str)
                    except (ValueError, AttributeError):
                        pass

                # Convert width to line width in points
                lw = max(linewidth, width_m / meters_per_point)

                # Plot individual river
                single_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=rivers.crs)
                single_gdf.plot(ax=ax, color=river_blue, linewidth=lw,
                              alpha=alpha, zorder=zorder, linestyle='-')

        # Plot streams (thinner)
        if not streams.empty:
            for idx, row in streams.iterrows():
                width_m = default_stream_width_m
                if 'width' in streams.columns and pd.notna(row.get('width')):
                    try:
                        width_str = str(row['width']).lower().replace('m', '').replace(' ', '').strip()
                        width_m = float(width_str)
                    except (ValueError, AttributeError):
                        pass

                lw = max(linewidth * 0.5, width_m / meters_per_point)
                single_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=streams.crs)
                single_gdf.plot(ax=ax, color=stream_blue, linewidth=lw,
                              alpha=alpha, zorder=zorder, linestyle='-')

        # Plot canals
        if not canals.empty:
            for idx, row in canals.iterrows():
                width_m = default_canal_width_m
                if 'width' in canals.columns and pd.notna(row.get('width')):
                    try:
                        width_str = str(row['width']).lower().replace('m', '').replace(' ', '').strip()
                        width_m = float(width_str)
                    except (ValueError, AttributeError):
                        pass

                lw = max(linewidth, width_m / meters_per_point)
                single_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=canals.crs)
                single_gdf.plot(ax=ax, color=river_blue, linewidth=lw,
                              alpha=alpha, zorder=zorder, linestyle='-')

        # Plot other waterways with default styling
        if not other_waterways.empty:
            other_waterways.plot(ax=ax, color=river_blue, linewidth=linewidth * 2,
                               alpha=alpha, zorder=zorder)

    # 3. Add labels for named features if requested
    if show_labels and 'name' in gdf.columns:
        for idx, row in gdf.iterrows():
            if pd.notna(row.get('name')) and row['name'] != '':
                try:
                    # Get centroid for label placement
                    centroid = row.geometry.centroid
                    if centroid and centroid.is_valid:
                        ax.annotate(
                            row['name'],
                            xy=(centroid.x, centroid.y),
                            xytext=(0, 0),
                            textcoords='offset points',
                            fontsize=8,
                            color='#003366',
                            weight='normal',
                            ha='center',
                            va='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor='none', alpha=0.7),
                            zorder=zorder + 1
                        )
                except Exception as e:
                    # Skip labels that fail
                    pass


def generate_debug_map(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    figure_size: List[float] = None,
    figure_dpi: int = 150,
    background_color: str = 'white',
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

    fig, ax = setup_figure_and_axes(figure_size, figure_dpi, background_color, margin, transparent=False)

    if not has_data(data):
        ax.text(0.5, 0.5, f'No data available for {layer_name}',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        return fig, ax

    legend_handles = []
    legend_labels = []

    try:
        if layer_name == 'water':
            _plot_water_debug(ax, data, legend_handles, legend_labels)
        elif layer_name == 'streets':
            _plot_streets_debug(ax, data, legend_handles, legend_labels)
        elif layer_name == 'buildings':
            _plot_buildings_debug(ax, data, legend_handles, legend_labels)
        elif layer_name == 'green':
            _plot_green_debug(ax, data, legend_handles, legend_labels)

        # Set bounds before legend (so legend doesn't affect bounds)
        if bbox:
            west, south, east, north = bbox
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
        else:
            # Auto-scale to fit the data
            ax.autoscale(enable=True)
            ax.autoscale_view(tight=True)

        ax.set_aspect('equal')
        ax.set_axis_off()

        # Add legend after setting bounds
        if legend_handles:
            ax.legend(handles=legend_handles, labels=legend_labels,
                     title=f'{layer_name.capitalize()} Features (Debug)',
                     loc='upper right',
                     fontsize=8,
                     title_fontsize=10,
                     frameon=True,
                     fancybox=True,
                     shadow=True,
                     bbox_to_anchor=(1.0, 1.0))

    except Exception as e:
        log_progress(f"Error generating debug map for {layer_name}: {e}")
        ax.text(0.5, 0.5, f'Error generating debug map:\n{str(e)}',
               ha='center', va='center', fontsize=10, transform=ax.transAxes)

    return fig, ax


def _plot_water_debug(ax, data, legend_handles, legend_labels):
    """Plot water features with color-coded tags for debugging."""
    if not hasattr(data, 'geometry'):
        return

    # Use tab20 colormap for distinct colors
    cmap = cm.get_cmap('tab20')
    feature_types = {}
    color_idx = 0

    # Collect all unique feature types
    for tag in ['natural', 'waterway']:
        if tag in data.columns:
            for val in data[tag].dropna().unique():
                feature_key = f'{tag}={val}'
                if feature_key not in feature_types:
                    feature_types[feature_key] = cmap(color_idx % 20)
                    color_idx += 1

    # Plot each feature type with its assigned color
    for feature_key, color in feature_types.items():
        tag, val = feature_key.split('=')
        mask = data[tag] == val
        subset = data[mask]

        if len(subset) > 0:
            # Determine if polygon or line
            geom_types = subset.geometry.geom_type.unique()
            is_polygon = any(gt in ['Polygon', 'MultiPolygon'] for gt in geom_types)

            if is_polygon:
                subset.plot(ax=ax, facecolor=color, edgecolor='black', linewidth=0.3, alpha=0.7)
            else:
                subset.plot(ax=ax, color=color, linewidth=2, alpha=0.8)

            # Add to legend
            if is_polygon:
                legend_handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5))
            else:
                legend_handles.append(mpatches.Patch(color=color))
            legend_labels.append(feature_key)


def _plot_streets_debug(ax, data, legend_handles, legend_labels):
    """Plot streets with color-coded highway types for debugging."""
    try:
        # Check if data is a NetworkX graph
        if isinstance(data, (nx.MultiDiGraph, nx.MultiGraph, nx.DiGraph, nx.Graph)):
            # Convert graph to edges GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(data)

            if 'highway' not in edges.columns:
                edges.plot(ax=ax, color='gray', linewidth=0.5)
                return

            # Use Set3 colormap for street types
            cmap = cm.get_cmap('Set3')
            highway_types = {}
            color_idx = 0

            # Collect all unique highway types
            for hw in edges['highway'].dropna().unique():
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
                mask = edges['highway'].apply(lambda x: hw_type in x if isinstance(x, list) else x == hw_type)
                subset = edges[mask]

                if len(subset) > 0:
                    subset.plot(ax=ax, color=color, linewidth=1.5, alpha=0.8)
                    legend_handles.append(mpatches.Patch(color=color))
                    legend_labels.append(hw_type)
        else:
            log_progress(f"Warning: Expected NetworkX graph for streets debug, got {type(data)}")

    except Exception as e:
        log_progress(f"Warning: Could not plot streets debug: {e}")


def _plot_buildings_debug(ax, data, legend_handles, legend_labels):
    """Plot buildings with color-coded types for debugging."""
    if not hasattr(data, 'geometry'):
        return

    # Use Paired colormap for building types
    cmap = cm.get_cmap('Paired')
    building_types = {}
    color_idx = 0

    # Collect building types
    if 'building' in data.columns:
        for val in data['building'].dropna().unique():
            if val not in building_types:
                building_types[val] = cmap(color_idx % 12)
                color_idx += 1

    # If no building column or all same, just use single color
    if not building_types:
        data.plot(ax=ax, facecolor='#ff7f0e', edgecolor='black', linewidth=0.3, alpha=0.7)
        legend_handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', edgecolor='black'))
        legend_labels.append('buildings')
        return

    # Plot each building type with its assigned color
    for btype, color in sorted(building_types.items()):
        mask = data['building'] == btype
        subset = data[mask]

        if len(subset) > 0:
            subset.plot(ax=ax, facecolor=color, edgecolor='black', linewidth=0.3, alpha=0.7)
            legend_handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black'))
            legend_labels.append(f'building={btype}')


def _plot_green_debug(ax, data, legend_handles, legend_labels):
    """Plot green spaces with color-coded tags for debugging."""
    if not hasattr(data, 'geometry'):
        return

    # Use Set2 colormap for green spaces
    cmap = cm.get_cmap('Set2')
    feature_types = {}
    color_idx = 0

    # Collect all unique feature types
    for tag in ['leisure', 'landuse', 'natural']:
        if tag in data.columns:
            for val in data[tag].dropna().unique():
                feature_key = f'{tag}={val}'
                if feature_key not in feature_types:
                    feature_types[feature_key] = cmap(color_idx % 8)
                    color_idx += 1

    # If no tags, just plot everything
    if not feature_types:
        data.plot(ax=ax, facecolor='#90ee90', edgecolor='black', linewidth=0.3, alpha=0.7)
        legend_handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor='#90ee90', edgecolor='black'))
        legend_labels.append('green spaces')
        return

    # Plot each feature type with its assigned color
    for feature_key, color in feature_types.items():
        tag, val = feature_key.split('=')
        mask = data[tag] == val
        subset = data[mask]

        if len(subset) > 0:
            subset.plot(ax=ax, facecolor=color, edgecolor='black', linewidth=0.3, alpha=0.7)
            legend_handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black'))
            legend_labels.append(feature_key)
