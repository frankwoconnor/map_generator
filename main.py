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

if sys.platform == 'darwin':
    multiprocessing.set_start_method('fork')

def log_progress(message):
    print(message, flush=True)

# --- Helper Functions ---

def has_data(data):
    """Checks if data is not None and not empty."""
    return data is not None and getattr(data, 'empty', False) is False

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

def save_layer(layer_name, data, layer_styles, output_directory, prefix, figure_size, background_color, figure_dpi, margin, suffix=""):
    """Save a single map layer to an SVG file."""
    if not has_data(data):
        return

    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    ax.set_facecolor(background_color)
    ax.set_axis_off()
    ax.margins(margin)
    fig.tight_layout(pad=0)

    # Get layer-specific plotting parameters
    facecolor = layer_styles[layer_name].get('facecolor', '#000000')
    edgecolor = layer_styles[layer_name].get('edgecolor', '#000000')
    linewidth = layer_styles[layer_name].get('linewidth', 0.5)
    alpha = layer_styles[layer_name].get('alpha', 1.0)
    hatch = layer_styles[layer_name].get('hatch', None)
    linestyle = layer_styles[layer_name].get('linestyle', '-') # Keep for other layers if needed
    zorder = layer_styles[layer_name].get('zorder', 1)

    plot_map_layer(ax, layer_name, data,
                   facecolor, edgecolor, linewidth, alpha,
                   hatch=hatch, linestyle=linestyle, zorder=zorder)

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
        
        # Filter to include only Polygon and MultiPolygon geometries for non-street layers
        if has_data(gdf):
            gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

        # Reproject to a local UTM zone for accurate area calculation before simplification and filtering
        original_crs = gdf.crs
        if original_crs and original_crs.is_geographic and not gdf.empty:
            # Estimate UTM CRS and reproject
            utm_crs = gdf.estimate_utm_crs()
            gdf_proj = gdf.to_crs(utm_crs)
        else:
            gdf_proj = gdf

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
        if original_crs and original_crs.is_geographic and not gdf.empty:
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

    if G:
        log_progress(f"Streets (Graph): Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
    else:
        log_progress("Streets (Graph): Not enabled or no data.")

    # Handle buildings based on size_categories or as a single layer
    buildings_size_categories = style['layers']['buildings'].get('size_categories', None)
    if buildings_size_categories and has_data(buildings_gdf):
        log_progress("Buildings (GeoDataFrame) - Split by Size Categories:")
        for category in buildings_size_categories:
            cat_name = category['name']
            min_area = category['min_area']
            max_area = category['max_area']

            # Reproject to UTM for accurate area calculation before filtering
            original_crs = buildings_gdf.crs
            if original_crs and original_crs.is_geographic and not buildings_gdf.empty:
                utm_crs = buildings_gdf.estimate_utm_crs()
                buildings_gdf_proj = buildings_gdf.to_crs(utm_crs)
            else:
                buildings_gdf_proj = buildings_gdf

            # Filter by area
            filtered_gdf = buildings_gdf_proj[buildings_gdf_proj.geometry.area >= min_area]
            if max_area is not None:
                filtered_gdf = filtered_gdf[filtered_gdf.geometry.area < max_area]
            
            # Reproject back to original CRS if it was projected
            if original_crs and original_crs.is_geographic and not buildings_gdf.empty:
                filtered_gdf = filtered_gdf.to_crs(original_crs)

            log_progress(f"  - {cat_name} ({min_area}-{max_area} sq m): {len(filtered_gdf)} features")

            # Save separate SVG for each category if separate_layers is true
            if style['output']['separate_layers']:
                log_progress(f"Saving {cat_name} buildings layer...")
                save_layer('buildings', filtered_gdf, style['layers'], output_directory, filename_prefix, figure_size, background_color, figure_dpi, margin, suffix=f"_{cat_name}")

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
    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    ax.set_facecolor(background_color)
    ax.set_axis_off()
    ax.margins(margin)
    fig.tight_layout(pad=0)

    # Plot layers in a specific order (based on zorder from style.json)
    layers_to_plot = []
    
    # Prepare data for plotting, ensuring only enabled and existing layers are considered
    if style['layers']['water']['enabled'] and water_gdf is not None:
        layers_to_plot.append({
            'name': 'water',
            'data': water_gdf,
            'style': style['layers']['water']
        })
    
    if style['layers']['buildings']['enabled'] and buildings_gdf is not None:
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
        data = layer_info['data']
        layer_style = layer_info['style']

        facecolor = layer_style.get('facecolor', '#000000')
        edgecolor = layer_style.get('edgecolor', '#000000')
        linewidth = layer_style.get('linewidth', 0.5)
        alpha = layer_style.get('alpha', 1.0)
        hatch = layer_style.get('hatch', None)
        linestyle = layer_style.get('linestyle', '-')
        zorder = layer_style.get('zorder', 1)

        # Special handling for buildings in combined output if size_categories are used
        if layer_name == 'buildings' and buildings_size_categories:
            # Plot all buildings fetched initially, before size category filtering
            plot_map_layer(ax, layer_name, buildings_gdf,
                           facecolor, edgecolor, linewidth, alpha,
                           hatch=hatch, linestyle=linestyle, zorder=zorder)
        else:
            plot_map_layer(ax, layer_name, data,
                           facecolor, edgecolor, linewidth, alpha,
                           hatch=hatch, linestyle=linestyle, zorder=zorder)

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
