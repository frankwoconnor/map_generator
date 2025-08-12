import os
import json
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import multiprocessing
import contextlib
import io
import geopandas as gpd
from shapely.geometry import Point

if sys.platform == 'darwin':
    multiprocessing.set_start_method('fork')

def log_progress(message):
    print(message, flush=True)

# ColorBrewer Palettes (hardcoded for now, must match app.py)
COLORBREWER_PALETTES = {
    'YlGnBu_3': ['#edf8fb', '#b2e2e2', '#66c2a4'],
    'YlGnBu_5': ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0'],
    'OrRd_3': ['#fee8c8', '#fdbb84', '#e34a33'],
    'Greys_3': ['#f0f0f0', '#bdbdbd', '#636363'],
    'Set1_3': ['#e41a1c', '#377eb8', '#4daf4a'],
    'Set1_5': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'],
    'Paired_4': ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'],
    'Paired_6': ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c']
}

# --- Helper Functions ---

def has_data(data):
    """Checks if data is not None and not empty."""
    return data is not None and getattr(data, 'empty', False) is False

def _reproject_gdf_for_area_calc(gdf):
    """Reprojects a GeoDataFrame to a local UTM zone for accurate area calculation and returns the projected GDF and original CRS."""
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

def plot_map_layer(ax, layer_name, data, facecolor, edgecolor, linewidth, alpha, hatch=None, linestyle='-', zorder=1):
    """Plots either a street network or GeoDataFrame on the given axis."""
    if not has_data(data):
        return
    if layer_name == 'streets':
        # linestyle and zorder are not directly supported by ox.plot_graph
        ox.plot_graph(data, ax=ax, show=False, close=False, bgcolor='white',
                      edge_color=edgecolor, edge_linewidth=linewidth, edge_alpha=alpha, node_size=0)
    else:
        data.plot(ax=ax, fc=facecolor, ec=edgecolor, lw=linewidth, alpha=alpha, hatch=hatch, zorder=zorder)

def _get_plot_params(layer_style):
    """Extracts plotting parameters from a layer's style dictionary."""
    return {
        'facecolor': layer_style.get('facecolor', '#000000'),
        'edgecolor': layer_style.get('edgecolor', '#000000'),
        'linewidth': layer_style.get('linewidth', 0.5),
        'alpha': layer_style.get('alpha', 1.0),
        'hatch': layer_style.get('hatch', None),
        'linestyle': layer_style.get('linestyle', '-'),
        'zorder': layer_style.get('zorder', 1)
    }

def _setup_figure_and_axes(figure_size, figure_dpi, background_color, margin):
    """Sets up a matplotlib figure and axes with common settings."""
    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    ax.set_facecolor(background_color)
    ax.set_axis_off()
    ax.margins(margin)
    fig.tight_layout(pad=0)
    return fig, ax

def save_layer(layer_name, data, layer_styles, output_directory, prefix, figure_size, background_color, figure_dpi, margin, suffix=""):
    """Save a single map layer to an SVG file."""
    if not has_data(data):
        return

    fig, ax = _setup_figure_and_axes(figure_size, figure_dpi, background_color, margin)

    # Get layer-specific plotting parameters
    params = _get_plot_params(layer_styles[layer_name])

    plot_map_layer(ax, layer_name, data,
                   params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                   hatch=params['hatch'], linestyle=params['linestyle'], zorder=params['zorder'])

    with contextlib.redirect_stdout(io.StringIO()):
        plt.savefig(os.path.join(output_directory, f"{prefix}_{layer_name}{suffix}.svg"),
                    format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def fetch_layer(query, dist, tags, is_graph=False, custom_filter=None, simplify_tolerance=None, min_size_threshold=0, layer_name_for_debug=None):
    """Fetches geographic data (graph or GeoDataFrame) based on query and distance."""
    if is_graph:
        if dist:
            point = ox.geocode(query)
            G = ox.graph_from_point(point, dist=dist, network_type='all', custom_filter=custom_filter)
        else:
            G = ox.graph_from_place(query, network_type='all', custom_filter=custom_filter)
        return G
    else:
        if dist:
            point = ox.geocode(query)
            gdf = ox.features_from_point(point, dist=dist, tags=tags)
        else:
            gdf = ox.features_from_place(query, tags=tags)

        # Ensure gdf is always a GeoDataFrame and has a CRS attribute
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(geometry=[]) # Create empty GeoDataFrame if not already one
        if gdf.crs is None and not gdf.empty:
            # Attempt to set a default CRS if it has data but no CRS
            gdf = gdf.set_crs("EPSG:4326", allow_override=True) # Assuming WGS84 for OSM data

        # Filter to include only Polygon and MultiPolygon geometries for non-street layers
        if not gdf.empty: # Only filter if there's data
            gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
            # After filtering, if it becomes empty, ensure it's still a GeoDataFrame
            if gdf.empty and not isinstance(gdf, gpd.GeoDataFrame):
                gdf = gpd.GeoDataFrame(geometry=[], crs=gdf.crs if gdf.crs else None)

        # Reproject to a local UTM zone for accurate area calculation before simplification and filtering
        gdf_proj, original_crs = _reproject_gdf_for_area_calc(gdf)

        # Debugging: Print area statistics before filtering
        if layer_name_for_debug and has_data(gdf_proj):
            log_progress(f"--- {layer_name_for_debug} Area Statistics (before min_size_threshold) ---")
            log_progress(f"  Min Area: {gdf_proj.geometry.area.min():.6f} sq meters")
            log_progress(f"  Max Area: {gdf_proj.geometry.area.max():.6f} sq meters")
            log_progress(f"  Mean Area: {gdf_proj.geometry.area.mean():.6f} sq meters")
            log_progress(f"  Median Area: {gdf_proj.geometry.area.median():.6f} sq meters")
            log_progress(f"  Total Features: {len(gdf_proj)}")
            log_progress("------------------------------------------------------------------")

        # Apply simplification if tolerance is provided and data exists
        if simplify_tolerance is not None and has_data(gdf_proj):
            gdf_proj = gdf_proj.simplify(tolerance=simplify_tolerance)

        # Apply minimum size threshold for polygons (based on area) - only if size_categories is NOT used
        # This min_size_threshold is now primarily for water, or if buildings don't use categories
        if min_size_threshold > 0 and has_data(gdf_proj) and tags != {'building': True}: # Only apply if not buildings or if buildings don't have categories
            initial_count = len(gdf_proj)
            gdf_filtered = gdf_proj[gdf_proj.geometry.area >= min_size_threshold]
            # Ensure gdf_filtered remains a GeoDataFrame even if empty
            if not isinstance(gdf_filtered, gpd.GeoDataFrame):
                gdf_filtered = gpd.GeoDataFrame(gdf_filtered, geometry=gdf_filtered.geometry, crs=gdf_proj.crs)
            
            filtered_count = len(gdf_filtered)
            if initial_count != filtered_count:
                log_progress(f"--- {layer_name_for_debug} min_size_threshold Filtering ---")
                log_progress(f"  Threshold: {min_size_threshold} sq meters")
                log_progress(f"  Features before filtering: {initial_count}")
                log_progress(f"  Features after filtering: {filtered_count}")
                log_progress(f"  Features removed: {initial_count - filtered_count}")
                log_progress("--------------------------------------------------")
            gdf_proj = gdf_filtered

        # Reproject back to original CRS if it was projected
        if original_crs and original_crs.is_geographic and not gdf_proj.empty:
            return gdf_proj.to_crs(original_crs)
        else:
            return gdf_proj

def main():
    """Main function to generate the map art."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate map art from OpenStreetMap data.')
    parser.add_argument('--prefix', type=str, help='The timestamped filename prefix for the output files.')
    args = parser.parse_args()

    # Load configuration from style.json
    try:
        with open('style.json', 'r') as f:
            style = json.load(f)
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
    figure_dpi = style['output'].get('figure_dpi', 300)
    margin = style['output'].get('margin', 0.05)

    # --- Fetching Data ---
    log_progress("Fetching data...")
    G = None
    buildings_gdf = None
    water_gdf = None

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

    # --- Debugging: Print Layer Information ---
    log_progress("--- Map Layer Information ---")

    # Determine building styling mode
    buildings_style_mode = style['layers']['buildings'].get('auto_style_mode', 'manual')
    buildings_size_categories = [] # This will be populated either manually or automatically
    log_progress(f"Building styling mode: {buildings_style_mode}")

    if buildings_style_mode == 'manual':
        if style['layers']['buildings'].get('size_categories_enabled', False):
            buildings_size_categories = style['layers']['buildings'].get('size_categories', [])
            log_progress("Buildings (GeoDataFrame) - Using Manual Categories.")
        else:
            log_progress("Buildings (GeoDataFrame) - Manual Categories disabled, using general layer style.")
    elif buildings_style_mode == 'auto_size' and has_data(buildings_gdf):
        log_progress("Buildings (GeoDataFrame) - Auto-coloring by Size.")
        palette_name = style['layers']['buildings'].get('auto_size_palette')
        log_progress(f"  Selected auto_size_palette: {palette_name}")
        if palette_name and palette_name in COLORBREWER_PALETTES:
            colors = COLORBREWER_PALETTES[palette_name]
            num_classes = len(colors)
            log_progress(f"  Palette has {num_classes} classes.")

            # Reproject to UTM for accurate area calculation
            buildings_gdf_proj, original_crs = _reproject_gdf_for_area_calc(buildings_gdf)
            
            if not buildings_gdf_proj.empty:
                min_val = buildings_gdf_proj.geometry.area.min()
                max_val = buildings_gdf_proj.geometry.area.max()
                log_progress(f"  Building area range: {min_val:.2f} to {max_val:.2f} sq meters.")
                
                # Generate equal interval bins
                bin_edges = [min_val + (max_val - min_val) * i / num_classes for i in range(num_classes + 1)]
                log_progress(f"  Generated bin edges: {bin_edges}")

                for i in range(num_classes):
                    cat_name = f"Size_Class_{i+1}"
                    min_area = bin_edges[i]
                    max_area = bin_edges[i+1] if i < num_classes - 1 else None # Last bin has no upper limit

                    buildings_size_categories.append({
                        "name": cat_name,
                        "min_area": min_area,
                        "max_area": max_area,
                        "facecolor": colors[i],
                        "edgecolor": style['layers']['buildings'].get('edgecolor', '#000000'),
                        "linewidth": style['layers']['buildings'].get('linewidth', 0.0),
                        "alpha": style['layers']['buildings'].get('alpha', 1.0),
                        "hatch": style['layers']['buildings'].get('hatch', None),
                        "zorder": style['layers']['buildings'].get('zorder', 2)
                    })
            log_progress(f"  Final buildings_size_categories length: {len(buildings_size_categories)}")
        else:
            log_progress("  No valid ColorBrewer palette selected for auto-size coloring.")
    elif buildings_style_mode == 'auto_distance' and has_data(buildings_gdf):
        log_progress("Buildings (GeoDataFrame) - Auto-coloring by Distance.")
        palette_name = style['layers']['buildings'].get('auto_distance_palette')
        log_progress(f"  Selected auto_distance_palette: {palette_name}")
        if palette_name and palette_name in COLORBREWER_PALETTES:
            colors = COLORBREWER_PALETTES[palette_name]
            num_classes = len(colors)
            log_progress(f"  Palette has {num_classes} classes.")

            # Geocode the center point
            center_point = ox.geocode(location_query)
            center_gdf = gpd.GeoDataFrame([{'geometry': Point(center_point)}], crs='epsg:4326')

            # Reproject buildings to UTM for accurate distance calculation
            buildings_gdf_proj, original_crs = _reproject_gdf_for_area_calc(buildings_gdf)
            center_gdf_proj = center_gdf.to_crs(buildings_gdf_proj.crs)
            
            if not buildings_gdf_proj.empty:
                # Calculate distance from each building's centroid to the center point
                buildings_gdf_proj['distance'] = buildings_gdf_proj.geometry.centroid.distance(center_gdf_proj.geometry.iloc[0])
                
                min_val = buildings_gdf_proj['distance'].min()
                max_val = buildings_gdf_proj['distance'].max()
                log_progress(f"  Building distance range: {min_val:.2f} to {max_val:.2f} meters.")

                # Generate equal interval bins
                bin_edges = [min_val + (max_val - min_val) * i / num_classes for i in range(num_classes + 1)]
                log_progress(f"  Generated bin edges: {bin_edges}")

                for i in range(num_classes):
                    cat_name = f"Distance_Class_{i+1}"
                    min_dist = bin_edges[i]
                    max_dist = bin_edges[i+1] if i < num_classes - 1 else None # Last bin has no upper limit

                    buildings_size_categories.append({
                        "name": cat_name,
                        "min_distance": min_dist, # Store min_distance for filtering
                        "max_distance": max_dist, # Store max_distance for filtering
                        "facecolor": colors[i],
                        "edgecolor": style['layers']['buildings'].get('edgecolor', '#000000'),
                        "linewidth": style['layers']['buildings'].get('linewidth', 0.0),
                        "alpha": style['layers']['buildings'].get('alpha', 1.0),
                        "hatch": style['layers']['buildings'].get('hatch', None),
                        "zorder": style['layers']['buildings'].get('zorder', 2)
                    })
            log_progress(f"  Final buildings_size_categories length: {len(buildings_size_categories)}")
        else:
            log_progress("  No valid ColorBrewer palette selected for auto-distance coloring.")
    else:
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
    log_progress("---------------------------")

    # --- Generate Output ---
    log_progress("Generating layers...")
    if style['output']['separate_layers']:
        if style['layers']['streets']['enabled']:
            log_progress("Saving streets layer...")
            save_layer('streets', G, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin)

        # Buildings are saved within the size_categories loop if separate_layers is true

        if style['layers']['water']['enabled']:
            log_progress("Saving water layer...")
            save_layer('water', water_gdf, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin)

    # Combined output (always generated)
    log_progress("Saving combined map...")
    fig, ax = _setup_figure_and_axes(figure_size, figure_dpi, background_color, margin)

    # Plot layers in a specific order (based on zorder from style.json)
    layers_to_plot = []
    
    # Prepare data for plotting, ensuring only enabled and existing layers are considered
    if style['layers']['water']['enabled'] and water_gdf is not None:
        layers_to_plot.append({
            'name': 'water',
            'data': water_gdf,
            'style': style['layers']['water']
        })
    
    # Handle buildings based on generated categories or general layer style
    if style['layers']['buildings']['enabled'] and buildings_gdf is not None:
        if buildings_size_categories: # If categories are defined (manual or auto)
            # Determine which GeoDataFrame to use for plotting based on styling mode
            if buildings_style_mode == 'auto_distance':
                # For auto_distance, buildings_gdf_proj already has the 'distance' column
                data_for_plotting = buildings_gdf_proj
            else:
                # For other modes, use the original buildings_gdf
                data_for_plotting = buildings_gdf

            for category in buildings_size_categories:
                layers_to_plot.append({
                    'name': 'buildings', # Still 'buildings' layer
                    'data': data_for_plotting, # Pass the appropriate gdf for filtering
                    'style': category # Use the category's style
                })
        else: # No categories, use general building style
            layers_to_plot.append({
                'name': 'buildings',
                'data': buildings_gdf,
                'style': style['layers']['buildings']
            })

    if style['layers']['streets']['enabled'] and G is not None:
        layers_to_plot.append({
            'name': 'streets',
            'data': G,
            'style': style['layers']['streets']
        })

    # Sort layers by zorder (lowest zorder first, so higher zorder is drawn on top)
    layers_to_plot.sort(key=lambda x: x['style'].get('zorder', 1))

    for layer_info in layers_to_plot:
        layer_name = layer_info['name']
        data = layer_info['data'] # This is buildings_gdf for buildings, or G/water_gdf for others
        layer_style = layer_info['style'] # This is either style['layers'][layer_name] or a category dict

        params = _get_plot_params(layer_style)

        # Special handling for buildings: filter and plot based on category if layer_style is a category
        if layer_name == 'buildings' and 'name' in layer_style and ('min_area' in layer_style or 'min_distance' in layer_style): # Check if it's a category dict
            # Reproject to UTM for accurate area/distance calculation
            buildings_gdf_proj, original_crs = _reproject_gdf_for_area_calc(data) # Use 'data' which is buildings_gdf

            filtered_gdf = buildings_gdf_proj.copy() # Start with a copy to filter

            # Filter by area if min_area/max_area are present (auto-size or manual)
            if 'min_area' in layer_style and 'max_area' in layer_style:
                min_area = layer_style['min_area']
                max_area = layer_style['max_area']
                filtered_gdf = filtered_gdf[filtered_gdf.geometry.area >= min_area]
                if max_area is not None:
                    filtered_gdf = filtered_gdf[filtered_gdf.geometry.area < max_area]
            # Filter by distance if min_distance/max_distance are present (auto-distance)
            elif 'min_distance' in layer_style and 'max_distance' in layer_style:
                min_dist = layer_style['min_distance']
                max_dist = layer_style['max_distance']
                
                

                filtered_gdf = filtered_gdf[filtered_gdf['distance'] >= min_dist]
                if max_dist is not None:
                    filtered_gdf = filtered_gdf[filtered_gdf['distance'] < max_dist]
            
            # Reproject back to original CRS if it was projected
            if original_crs and original_crs.is_geographic and has_data(filtered_gdf):
                filtered_gdf = filtered_gdf.to_crs(original_crs)

            plot_map_layer(ax, layer_name, filtered_gdf, # Plot filtered_gdf
                           params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                           hatch=params['hatch'], linestyle=params['linestyle'], zorder=params['zorder'])
        else: # Not a buildings category, or buildings without categories
            plot_map_layer(ax, layer_name, data,
                           params['facecolor'], params['edgecolor'], params['linewidth'], params['alpha'],
                           hatch=params['hatch'], linestyle=params['linestyle'], zorder=params['zorder'])

    with contextlib.redirect_stdout(io.StringIO()):
        plt.savefig(os.path.join(output_directory, f"{filename_prefix}_combined.svg"), format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    log_progress(f"Map art generated successfully in the '{output_directory}' directory.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
