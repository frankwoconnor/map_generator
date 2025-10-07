import os
import json
import mimetypes
from typing import Any, Dict, List, Optional, Mapping
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, jsonify, flash, get_flashed_messages, send_file
from flask_cors import CORS
import subprocess
import datetime
import glob
import map_core.core.config as cfg
from map_core.core.geocode import geocode_to_point
from map_core.core.palettes import load_palettes, get_palette_names, get_palette, add_palette, remove_palette, validate_palette_colors, set_palettes_file_path
from config.manager import get_layer_tags

# Initialize palettes with the correct path
palettes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config', 'palettes'))
set_palettes_file_path(os.path.join(palettes_dir, 'palettes.json'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')) # Directory where generated SVGs are saved
app.secret_key = os.urandom(24)

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

STYLE_FILE = 'style.json'
MAIN_SCRIPT = 'main.py'

# Master configuration for layer filters. Defines the OSM tags and values for the UI.
# The 'default' list will be used if no specific filter is saved in style.json
LAYER_FILTER_DEFINITIONS = {
    'streets': {
        'osm_key': 'highway',
        'options': [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "motorway_link", "trunk_link", "primary_link", "secondary_link",
            "residential", "unclassified", "road", "living_street", "service",
            "pedestrian", "track", "bus_guideway", "escape", "raceway", "busway",
            "footway", "bridleway", "steps", "corridor", "path", "cycleway"
        ],
        'default': [ # A sensible default set of roads
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "residential", "unclassified", "living_street", "road"
        ]
    },
    'water': {
        'osm_key': 'natural',
        'options': ["water", "coastline"],
        'default': ["water"]
    },
    'buildings': {
        'osm_key': 'building',
        'options': ["yes", "residential", "commercial", "industrial", "public"], # 'yes' is a generic value
        'default': ["yes"]
    }
}


HIGHWAY_TYPES = [
    # Major Roads
    "motorway", "trunk", "primary", "secondary", "tertiary",
    # Connecting Roads
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
    # Local Roads
    "residential", "unclassified", "road",
    # Special Road Types
    "living_street", "service", "pedestrian", "track", "bus_guideway", "escape", "raceway", "busway",
    # Paths
    "footway", "bridleway", "steps", "corridor", "path", "cycleway"
]

# --- Helper Functions ---

def load_style() -> Dict[str, Any]:
    """
    Load `style.json`, ensuring default values are present for missing keys.

    Returns:
        dict: The normalized style configuration dictionary suitable for the UI/backend.
    """
    raw = cfg.load_style(STYLE_FILE) or {}
    style_data = cfg.normalize_style(raw)

    layers_settings = style_data.setdefault('layers', {})
    # Ensure generic layers exist with sensible defaults
    layers_settings.setdefault('streets', {
        "enabled": True,
        "edgecolor": "#000000",
        "linewidth": 0.5,
        "alpha": 1.0,
        "hatch": None,
        "zorder": 3,
        "simplify_tolerance": 0.0,
        "min_size_threshold": 0.0,
    })
    layers_settings.setdefault('water', {
        "enabled": True,
        "facecolor": "#a6cee3",
        "edgecolor": "#a6cee3",
        "linewidth": 0.0,
        "alpha": 1.0,
        "hatch": None,
        "zorder": 1,
        "simplify_tolerance": 0.0,
        "min_size_threshold": 0.0,
    })
    layers_settings.setdefault('green', {
        "enabled": False,
        "facecolor": "#b2df8a",
        "edgecolor": "#33a02c",
        "linewidth": 0.0,
        "alpha": 1.0,
        "hatch": None,
        "zorder": 1,
        "simplify_tolerance": 0.0,
        "min_size_threshold": 0.0,
    })
    buildings_settings = layers_settings.setdefault('buildings', {})

    buildings_settings.setdefault('size_categories', [])
    buildings_settings.setdefault('auto_style_mode', 'manual')
    buildings_settings.setdefault('auto_size_palette', '')
    buildings_settings.setdefault('auto_distance_palette', '')
    # Manual color only stores facecolor
    buildings_settings.setdefault('manual_color_settings', {
        "facecolor": "#000000"
    })
    # Common building params (apply to all modes)
    buildings_settings.setdefault('edgecolor', '#000000')
    buildings_settings.setdefault('linewidth', 0.5)
    buildings_settings.setdefault('alpha', 1.0)
    buildings_settings.setdefault('hatch', None)
    buildings_settings.setdefault('zorder', 2)

    # Distance is stored and displayed in meters consistently
    location_settings = style_data.setdefault('location', {})
    # Ensure persistent UI-related defaults
    location_settings.setdefault('data_source', 'remote')  # 'remote' or 'local'

    # Provide sensible offline defaults if missing: Cork city center and ~1km bbox (1km across)
    # Cork center approx
    default_lat = 51.8986
    default_lon = -8.4756
    # Half-size of 1 km box: 0.5 km
    # Degrees per km approx: lat ~ 1/111, lon ~ 1/(111*cos(lat))
    lat_deg_per_km = 1.0 / 111.0
    from math import cos, radians
    lon_deg_per_km = 1.0 / (111.32 * max(0.0001, cos(radians(default_lat))))
    dlat = 0.5 * lat_deg_per_km
    dlon = 0.5 * lon_deg_per_km
    default_bbox = [default_lon - dlon, default_lat - dlat, default_lon + dlon, default_lat + dlat]

    # If no query present, set coords for Cork center
    if not location_settings.get('query'):
        location_settings['query'] = f"{default_lat} {default_lon}"
    # Do not persist a manual bbox in the new UX; it will be computed from distance
    location_settings['bbox'] = None

    return style_data

def get_available_pbf_files(pbf_folder: str) -> List[Dict[str, str]]:
    """Get list of available PBF files in the specified folder, avoiding duplicates.

    Args:
        pbf_folder: Path to the folder containing PBF files

    Returns:
        List of dictionaries with 'name' and 'path' keys for each PBF file
    """
    pbf_files = []
    found_paths = set()

    if not pbf_folder:
        return pbf_files

    try:
        # Resolve relative path to an absolute one for reliable checking
        if pbf_folder.startswith('../'):
            pbf_folder = os.path.abspath(os.path.join(os.getcwd(), pbf_folder))

        if os.path.exists(pbf_folder) and os.path.isdir(pbf_folder):
            # Find all .pbf and .osm.pbf files
            for pattern in ['*.pbf', '*.osm.pbf']:
                for filepath in glob.glob(os.path.join(pbf_folder, pattern)):
                    abs_path = os.path.abspath(filepath)
                    if abs_path not in found_paths:
                        found_paths.add(abs_path)
                        filename = os.path.basename(filepath)
                        # Store relative path for consistency in the UI/config
                        relative_path = os.path.relpath(filepath, os.getcwd())
                        pbf_files.append({
                            'name': filename,
                            'path': relative_path
                        })
    except Exception as e:
        print(f"Error scanning PBF folder '{pbf_folder}': {e}")

    # Sort by filename for consistent ordering
    pbf_files.sort(key=lambda x: x['name'].lower())
    return pbf_files

def _save_style_json(style_data: Dict[str, Any]) -> None:
    """Save the updated style data to `style.json`."""
    with open(STYLE_FILE, 'w') as f:
        json.dump(style_data, f, indent=2)

def _update_style_from_form(style: Dict[str, Any], form: Mapping[str, str]) -> Dict[str, Any]:
    """Orchestrate updating the style dictionary from form data.

    Args:
        style: Current style config (will be mutated in-place).
        form: Request form mapping with string keys/values.

    Returns:
        dict: The updated style config (same object as input).
    """
    _update_location_settings(style, form)
    _update_output_settings(style, form)
    _update_generic_layer_settings(style, form, 'streets')
    _update_generic_layer_settings(style, form, 'water')
    _update_generic_layer_settings(style, form, 'green')
    _update_buildings_settings(style, form)
    # The new generic layer settings function handles filters, so the specific processing one is removed.
    # _update_processing_settings(style, form)
    return style

def _update_location_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update location settings per new Local/Remote model (coords + distance required)."""
    loc = style.setdefault('location', {})

    # Data source selection
    data_source = form.get('location_data_source', loc.get('data_source', 'remote'))
    loc['data_source'] = 'local' if data_source == 'local' else 'remote'

    raw_query = (form.get('location_query') or loc.get('query') or '').strip()

    # Distance in meters (UI and engine use meters consistently)
    distance_m_raw = (form.get('location_distance') or '').strip()
    if distance_m_raw:
        try:
            m_int = max(0, int(float(distance_m_raw)))
            loc['distance'] = float(m_int)
        except ValueError:
            loc['distance'] = None
            flash("Invalid distance. Leave empty or provide an integer number of meters.", category='warning')
    else:
        loc['distance'] = None

    # PBF options: only applicable in local mode
    if loc['data_source'] == 'local':
        pbf_folder = (form.get('location_pbf_folder') or '').strip()
        if pbf_folder:
            loc['pbf_folder'] = pbf_folder
        elif 'pbf_folder' not in loc:
            loc['pbf_folder'] = '../osm-data/'  # Default PBF folder

        pbf_file_selection = (form.get('location_pbf_file_selection') or '').strip()
        manual_pbf_path = (form.get('location_pbf_path') or '').strip()
        if pbf_file_selection and pbf_file_selection != 'manual':
            loc['pbf_path'] = pbf_file_selection
        elif manual_pbf_path:
            loc['pbf_path'] = manual_pbf_path
        else:
            loc['pbf_path'] = None
            flash("Local mode selected: please choose a local .osm.pbf file.", category='warning')
    else:
        # Remote mode should not carry a PBF path
        loc['pbf_path'] = None

    # Manual bbox is no longer supported in the UI
    loc['bbox'] = None

    # Parse coordinates always (lat lon)
    q = raw_query.replace(',', ' ').split()
    if len(q) >= 2:
        try:
            lat = float(q[0]); lon = float(q[1])
            loc['query'] = f"{lat} {lon}"
        except Exception:
            loc['query'] = raw_query
            flash("Coordinates must be numeric: expected 'lat lon'.", category='warning')
    else:
        loc['query'] = raw_query
        flash("Provide coordinates as 'lat lon' (e.g., '51.8944 -8.4827').", category='warning')

    # If we have numeric coordinates but no distance, set a sensible default
    try:
        have_dist = loc.get('distance') is not None
        # Parse numeric coords from loc['query']
        q = (loc.get('query') or '').replace(',', ' ').split()
        numeric_coords = False
        if len(q) >= 2:
            float(q[0]); float(q[1])
            numeric_coords = True
        # Only set default if distance empty
        if numeric_coords and not have_dist:
            loc['distance'] = 1000.0
            flash("Distance not provided; defaulting to 1000 meters around the coordinate center.", category='info')
    except Exception:
        pass

def _update_output_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update output settings in the style dictionary."""
    output = style.setdefault('output', {})
    output['separate_layers'] = 'separate_layers' in form
    output['filename_prefix'] = form.get('filename_prefix', output.get('filename_prefix'))
    output['output_directory'] = form.get('output_directory', output.get('output_directory', '../output'))
    width_str = form.get('figure_size_width')
    height_str = form.get('figure_size_height')
    output.setdefault('figure_size', [10.0, 10.0])
    output['figure_size'][0] = float(width_str) if width_str else 10.0
    output['figure_size'][1] = float(height_str) if height_str else 10.0
    output['background_color'] = form.get('background_color', output.get('background_color'))
    # Transparent background flag overrides background color at save time
    output['transparent_background'] = 'transparent_background' in form
    dpi_str = form.get('figure_dpi')
    output['figure_dpi'] = int(dpi_str) if dpi_str else 300
    margin_str = form.get('margin')
    output['margin'] = float(margin_str) if margin_str else 0.05
    output['preview_type'] = form.get('preview_type', output.get('preview_type', 'embedded'))

def _get_tag_filter_values(form: Mapping[str, str], layer_name: str, tag_key: str) -> List[str]:
    """Get the selected values for a tag filter from the form data."""
    prefix = f"{layer_name}_tag_{tag_key}_"
    selected = []
    
    for key, value in form.items():
        if key.startswith(prefix) and value == 'on':
            selected.append(key[len(prefix):])
    
    return selected

def _update_layer_tag_filters(style: Dict[str, Any], form: Mapping[str, str], layer_name: str) -> None:
    """Update tag-based filters for a layer from form data."""
    from config.manager import get_layer_tags
    
    # Get the tag configuration for this layer
    layer = style.setdefault('layers', {}).get(layer_name, {})
    tag_configs = get_layer_tags().layers.get(layer_name, {}).tag_configs
    
    # Initialize filters if not present
    if 'filters' not in layer:
        layer['filters'] = {}
    
    # Update each tag filter
    for tag_key, tag_info in tag_configs.items():
        selected_values = _get_tag_filter_values(form, layer_name, tag_key)
        if selected_values:
            layer['filters'][tag_key] = selected_values
        elif tag_key in layer['filters']:
            del layer['filters'][tag_key]

def _update_generic_layer_settings(style: Dict[str, Any], form: Mapping[str, str], layer_name: str) -> None:
    """Update settings for generic layers like streets and water."""
    layers = style.setdefault('layers', {})
    if layer_name in layers:
        layer = layers[layer_name]
        layer['enabled'] = f'{layer_name}_enabled' in form
        if layer_name != 'streets':
            layer['facecolor'] = form.get(f'{layer_name}_facecolor', layer.get('facecolor', '#000000'))
        layer['edgecolor'] = form.get(f'{layer_name}_edgecolor', layer.get('edgecolor', '#000000'))
        linewidth_str = form.get(f'{layer_name}_linewidth')
        layer['linewidth'] = float(linewidth_str) if linewidth_str else 0.5
        alpha_str = form.get(f'{layer_name}_alpha')
        layer['alpha'] = float(alpha_str) if alpha_str else 1.0
        simplify_tolerance_str = form.get(f'{layer_name}_simplify_tolerance')
        layer['simplify_tolerance'] = float(simplify_tolerance_str) if simplify_tolerance_str else 0.0
        min_size_threshold_str = form.get(f'{layer_name}_min_size_threshold')
        layer['min_size_threshold'] = float(min_size_threshold_str) if min_size_threshold_str else 0.0
        hatch_value = form.get(f'{layer_name}_hatch')
        layer['hatch'] = None if hatch_value == 'null' else hatch_value
        zorder_str = form.get(f'{layer_name}_zorder')
        layer['zorder'] = int(zorder_str) if zorder_str else 1
        
        # Update tag-based filters
        _update_layer_tag_filters(style, form, layer_name)

def _update_buildings_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update the complex, multi-mode settings for the buildings layer."""
    buildings = style.setdefault('layers', {}).setdefault('buildings', {})
    buildings['enabled'] = 'buildings_enabled' in form
    buildings['simplify_tolerance'] = float(form.get('buildings_simplify_tolerance') or 0.0)
    buildings['min_size_threshold'] = float(form.get('buildings_min_size_threshold') or 0.0)

    # The form uses 'building_styling_mode' with values: 'manual', 'auto_distance', 'auto_size', 'manual_floorsize'
    buildings_style_mode = form.get('building_styling_mode', 'manual')
    print(f"DEBUG: Building styling mode received: {buildings_style_mode}")
    print(f"DEBUG: Form data: {dict(form)}")  # Debug: Print all form data
    
    # Set the auto_style_mode in the buildings settings
    buildings['auto_style_mode'] = buildings_style_mode

    # Reset all styling options before setting the active one
    buildings['size_categories'] = []
    buildings['size_categories_enabled'] = False
    
    # Initialize palette settings with empty strings
    buildings['auto_size_palette'] = ''
    buildings['auto_distance_palette'] = ''
    
    # Ensure manual settings exists (facecolor only)
    buildings.setdefault('manual_color_settings', {"facecolor": "#000000"})

    if buildings_style_mode == 'manual':
        manual_settings = buildings.setdefault('manual_color_settings', {})
        manual_settings['facecolor'] = form.get('buildings_manual_color_facecolor', 
                                             manual_settings.get('facecolor', '#000000'))
        print(f"DEBUG: Set manual color to {manual_settings['facecolor']}")

    elif buildings_style_mode == 'auto_size':
        palette = form.get('auto_size_palette', '')
        buildings['auto_size_palette'] = palette if palette else 'YlGnBu_5'
        print(f"DEBUG: Set auto_size_palette to {buildings['auto_size_palette']}")
        
    elif buildings_style_mode == 'auto_distance':
        palette = form.get('auto_distance_palette', '')
        buildings['auto_distance_palette'] = palette if palette else 'YlOrRd_5'
        print(f"DEBUG: Set auto_distance_palette to {buildings['auto_distance_palette']}")
    
    elif buildings_style_mode == 'manual_floorsize':
        # Parse dynamic categories. Inputs are named like:
        # buildings_size_category_{i}_name, _min_area, _max_area, _facecolor
        categories = []
        # Collect all indices present in the form by scanning keys
        indices = set()
        for key in form.keys():
            if key.startswith('buildings_size_category_'):
                try:
                    # Extract the index and field name
                    parts = key.split('_')
                    if len(parts) >= 5:  # buildings_size_category_0_name
                        idx = int(parts[3])
                        field = '_'.join(parts[4:])
                        indices.add(idx)
                except (ValueError, IndexError):
                    continue
        
        # For each found index, create a category
        for idx in sorted(indices):
            prefix = f'buildings_size_category_{idx}'
            name = form.get(f'{prefix}_name', f'Category {idx+1}')
            min_area = float(form.get(f'{prefix}_min_area', '0'))
            max_area = float(form.get(f'{prefix}_max_area', '0'))
            facecolor = form.get(f'{prefix}_facecolor', '#000000')
            
            if name:  # Only add if we have a name
                categories.append({
                    'name': name,
                    'min_area': min_area,
                    'max_area': max_area,
                    'facecolor': facecolor
                })
        
        if categories:
            buildings['size_categories'] = categories
            buildings['size_categories_enabled'] = True
    
    # Debug: Print the final buildings settings
    print(f"DEBUG: Final buildings settings: {json.dumps(buildings, indent=2, default=str)}")
    
    # No need to return anything as we're modifying the dictionary in-place

    # Update common params regardless of mode from dedicated fields
    edgecolor_val = form.get('buildings_edgecolor')
    linewidth_str = form.get('buildings_linewidth')
    alpha_str = form.get('buildings_alpha')
    hatch_value = form.get('buildings_hatch')
    zorder_str = form.get('buildings_zorder')

    if edgecolor_val is not None and edgecolor_val != "":
        buildings['edgecolor'] = edgecolor_val

    if linewidth_str is not None and linewidth_str != "":
        buildings['linewidth'] = float(linewidth_str)
    else:
        buildings.setdefault('linewidth', 0.5)

    if alpha_str is not None and alpha_str != "":
        buildings['alpha'] = float(alpha_str)
    else:
        buildings.setdefault('alpha', 1.0)

    if hatch_value is not None:
        buildings['hatch'] = None if hatch_value == 'null' else hatch_value
    else:
        buildings.setdefault('hatch', None)

    if zorder_str is not None and zorder_str != "":
        buildings['zorder'] = int(zorder_str)
    else:
        buildings.setdefault('zorder', 2)


def _validate_location_inputs(style: Dict[str, Any]) -> List[str]:
    """Validate location inputs after _update_style_from_form and normalization.

    - Coordinates must be 'lat lon' with lat in [-90, 90], lon in [-180, 180]
    - Distance must be between 10 and 20000 meters
    - If data_source == 'local', pbf_path must be present
    """
    errors: List[str] = []
    loc = style.get('location', {})
    query = (loc.get('query') or '').replace(',', ' ').split()
    if len(query) < 2:
        errors.append("Coordinates required as 'lat lon'.")
    else:
        try:
            lat = float(query[0]); lon = float(query[1])
            if not (-90.0 <= lat <= 90.0):
                errors.append("Latitude must be between -90 and 90.")
            if not (-180.0 <= lon <= 180.0):
                errors.append("Longitude must be between -180 and 180.")
        except Exception:
            errors.append("Coordinates must be numeric: expected 'lat lon'.")

    # Distance
    distance = loc.get('distance')
    try:
        if distance is None:
            errors.append("Distance is required (meters).")
        else:
            dval = float(distance)
            if not (10.0 <= dval <= 20000.0):
                errors.append("Distance must be between 10 and 20000 meters.")
    except Exception:
        errors.append("Distance must be a number (meters).")

    # Local PBF requirement
    data_source = loc.get('data_source')
    if data_source == 'local':
        pbf_path = loc.get('pbf_path')
        if not pbf_path:
            errors.append("Local mode selected: please choose a local .osm.pbf file.")

    return errors

@app.route('/scan-pbf-folder', methods=['POST'])
def scan_pbf_folder():
    """API endpoint to scan a folder for PBF files."""
    data = request.get_json()
    folder_path = data.get('folder_path') if data else None

    if not folder_path:
        return jsonify({'error': 'Folder path is required.'}), 400

    try:
        pbf_files = get_available_pbf_files(folder_path)
        return jsonify({'pbf_files': pbf_files})
    except Exception as e:
        return jsonify({'error': f'Failed to scan folder: {str(e)}'}), 500

@app.route('/api/palettes', methods=['GET'])
def api_get_palettes():
    """Get all available palettes."""
    try:
        palettes = load_palettes()
        return jsonify({
            'success': True,
            'palettes': palettes,
            'names': get_palette_names()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/palettes/<palette_name>', methods=['GET'])
def api_get_palette(palette_name):
    """Get a specific palette by name."""
    try:
        palette = get_palette(palette_name)
        if palette is None:
            return jsonify({'success': False, 'error': 'Palette not found'}), 404
        return jsonify({
            'success': True,
            'name': palette_name,
            'colors': palette
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/palettes', methods=['POST'])
def api_create_palette():
    """Create or update a palette."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        name = data.get('name')
        colors = data.get('colors')

        if not name or not colors:
            return jsonify({'success': False, 'error': 'Name and colors are required'}), 400

        if not validate_palette_colors(colors):
            return jsonify({'success': False, 'error': 'Invalid color format. Colors must be hex strings like #RRGGBB'}), 400

        if add_palette(name, colors):
            return jsonify({
                'success': True,
                'message': f'Palette "{name}" created/updated successfully',
                'palette': {'name': name, 'colors': colors}
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save palette'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/palettes/<palette_name>', methods=['DELETE'])
def api_delete_palette(palette_name):
    """Delete a palette."""
    try:
        if remove_palette(palette_name):
            return jsonify({
                'success': True,
                'message': f'Palette "{palette_name}" deleted successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Palette not found or could not be deleted'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/wall-art')
def wall_art():
    """Render the wall art generator page."""
    return render_template('wall_art.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    style = load_style()

    if request.method == 'POST' and request.form.get('action') == 'generate':
        print("==== Map Generation Request Received ====")
        print(f"HTTP Method: {request.method}")
        print(f"Action: {request.form.get('action')}")
        # Update style.json based on form data
        
        style = _update_style_from_form(style, request.form)

        # Normalize and validate using core config before persisting/generating
        normalized_style = cfg.normalize_style(style)
        validation_error = cfg.validate_style(normalized_style)
        if validation_error:
            flash(f"Schema validation warning: {validation_error}", category='warning')
        style = normalized_style

        # Quick-win: strict server-side validation
        input_errors = _validate_location_inputs(style)
        if input_errors:
            for e in input_errors:
                flash(e, category='error')
            # Re-render page with errors; do not run generation
            palettes = load_palettes()
            pbf_files = get_available_pbf_files(style.get('location', {}).get('pbf_folder', '../osm-data/'))
            warning_messages = get_flashed_messages(with_categories=True)
            return render_template(
                'index.html',
                style=style,
                palettes=palettes,
                pbf_files=pbf_files,
                warning_messages=warning_messages,
                layer_filters=LAYER_FILTER_DEFINITIONS,
            )
        # --- Pre-run logging ---
        loc = style.get('location', {})
        pbf_path = loc.get('pbf_path')
        data_source = 'LOCAL_PBF' if pbf_path else 'REMOTE_OSMNX'
        # Distance is stored in meters; show meters to avoid confusion for <1km
        distance_m = None
        try:
            distance_m = float(loc['distance']) if loc.get('distance') is not None else None
        except Exception:
            pass
        print("---- Run Parameters ----")
        print(f"Data Source: {data_source}")
        if pbf_path:
            print(f"PBF Path: {pbf_path}")
        print(f"Location Query: {loc.get('query')}")
        print(f"Distance (m): {int(distance_m) if distance_m is not None else 'None (admin area)'}")
        # Layers summary
        layers = style.get('layers', {})
        enabled_layers = [name for name, cfg_layer in layers.items() if cfg_layer.get('enabled')]
        print(f"Enabled Layers: {', '.join(enabled_layers) if enabled_layers else 'None'}")
        # Street filter summary
        # Street filter is now part of the layer, so this log is implicitly covered
        # Output settings
        out = style.get('output', {})
        print(f"Output Dir (base): {out.get('output_directory')}")
        print(f"Filename Prefix: {out.get('filename_prefix')}")
        print(f"Separate Layers: {out.get('separate_layers')}")

        # Generate timestamp for the run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_run_identifier = f"{style['output']['filename_prefix']}_{timestamp}"
        # Building layer settings are already handled by _update_style_from_form()
        # Remove redundant and conflicting overrides.

        # Processing settings

        # Save the updated style
        _save_style_json(style)
        expected_out_dir = os.path.join(out.get('output_directory', '../output'), timestamped_run_identifier)
        print(f"Expected Output Folder: {expected_out_dir}")

        print("Starting generation...")
        # Use virtual environment's Python to ensure all dependencies are available
        venv_python = os.path.join(os.path.dirname(__file__), '.venv', 'bin', 'python3')
        cmd = [venv_python, MAIN_SCRIPT, '--prefix', timestamped_run_identifier]
        print(f"Executing command: {cmd}")

        # Ensure output directory exists and set up a per-run server log file
        os.makedirs(expected_out_dir, exist_ok=True)
        log_path = os.path.join(expected_out_dir, f"{timestamped_run_identifier}_server.log")
        print(f"Streaming logs to: {log_path}")

        try:
            with open(log_path, 'a') as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                # Stream stdout in real-time to console and file
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end='')
                    lf.write(line)
                proc.wait()
                ret = proc.returncode

            print("---- Subprocess Complete ----")
            print(f"Return code: {ret}")
            print(f"Artifacts should be in: {expected_out_dir}")
            print("==== Map Generation Request Finished ====")

            if ret != 0:
                error_message = f"Map generation failed with exit code {ret}. See log: {log_path}"
                print(f"Error: {error_message}")
                palettes = load_palettes()
                pbf_files = get_available_pbf_files(style.get('location', {}).get('pbf_folder', '../osm-data/'))
                warning_messages = get_flashed_messages(with_categories=True)
                return render_template('index.html', style=style, palettes=palettes, pbf_files=pbf_files, error_message=error_message, warning_messages=warning_messages, layer_filters=LAYER_FILTER_DEFINITIONS)

            # Redirect to GET request to display the new map
            return redirect(url_for('index'))
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            print(f"Error: {error_message}")
            pbf_files = get_available_pbf_files(style.get('location', {}).get('pbf_folder', '../osm-data/'))
            warning_messages = get_flashed_messages(with_categories=True)
            return render_template('index.html', style=style, pbf_files=pbf_files, error_message=error_message, warning_messages=warning_messages, layer_filters=LAYER_FILTER_DEFINITIONS)

    else:
        # This block handles GET requests and POST requests that are not for generation
        # It will render the template with the current style and any previously generated content
        # (This part of the code is responsible for displaying the initial page and the final map)

        # Load the style again to ensure it's up-to-date after a potential save
        style = load_style()
        palettes = load_palettes()

        # Initialize defaults for template variables to avoid NameError
        generated_files = []
        combined_svg_path = None
        svg_content = None
        preview_layers = []
        error_message = None
        progress_log = ''

        # Try to find and load the last generated combined SVG
        # This logic is simplified for demonstration; in a real app, you might store
        # the last generated file path in a database or a more robust way.
        output_base_dir = app.config['UPLOAD_FOLDER']
        latest_run_dir = None
        latest_timestamp = None

        # List all subdirectories in the output folder (each represents a run)
        for entry in os.listdir(output_base_dir):
            entry_path = os.path.join(output_base_dir, entry)
            if os.path.isdir(entry_path):
                # Assuming subfolder name contains the timestamp in the format YYYYMMDD_HHMMSS
                parts = entry.split('_')
                if len(parts) > 1 and len(parts[-2]) == 8 and len(parts[-1]) == 6 and parts[-2].isdigit() and parts[-1].isdigit():
                    try:
                        current_timestamp = datetime.datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
                        if latest_timestamp is None or current_timestamp > latest_timestamp:
                            # Check for a combined SVG within this subfolder
                            latest_run_dir = entry_path
                            latest_timestamp = current_timestamp
                    except ValueError:
                        # Not a valid timestamped folder, ignore
                        pass

        if latest_run_dir:
            # Find all individual layer SVGs in the latest run directory
            for f in os.listdir(latest_run_dir):
                if f.endswith('.svg') and '_combined' not in f:
                    layer_name = f.split('_')[-1].replace('.svg', '')
                    preview_layers.append({
                        'name': layer_name,
                        'path': os.path.relpath(os.path.join(latest_run_dir, f), output_base_dir)
                    })
            # Sort layers by a sensible default order for display
            zorder_map = {layer.get('name'): style.get('layers', {}).get(layer.get('name'), {}).get('zorder', 99) for layer in preview_layers}
            preview_layers.sort(key=lambda x: zorder_map.get(x['name'], 99))

            # Also find the combined SVG for the main preview
            run_prefix = os.path.basename(latest_run_dir)
            combined_svg_in_folder = os.path.join(latest_run_dir, f"{run_prefix}_combined.svg")
            if os.path.exists(combined_svg_in_folder):
                combined_svg_path = os.path.relpath(combined_svg_in_folder, output_base_dir)
                if style['output'].get('preview_type') == 'embedded':
                    try:
                        with open(combined_svg_in_folder, 'r') as svg_file:
                            svg_content = svg_file.read()
                    except FileNotFoundError:
                        print(f"Error: Could not find SVG file at {combined_svg_in_folder}")
                        svg_content = None

        pbf_files = get_available_pbf_files(style.get('location', {}).get('pbf_folder', '../osm-data/'))
        warning_messages = get_flashed_messages(with_categories=True)
        
        # Ensure tag configs are in the style for the template
        layer_tags = get_layer_tags()
        for layer_name, layer_config in style.get('layers', {}).items():
            if layer_name in layer_tags.layers and 'tag_configs' not in layer_config:
                layer_config['tag_configs'] = layer_tags.layers[layer_name].tag_configs
        
        return render_template('index.html', 
                            style=style, 
                            palettes=palettes, 
                            pbf_files=pbf_files, 
                            generated_files=generated_files, 
                            combined_svg_path=combined_svg_path, 
                            svg_content=svg_content, 
                            error_message=error_message, 
                            progress_log=progress_log, 
                            warning_messages=warning_messages, 
                            layer_filters={},  # Empty for backward compatibility
                            preview_layers=preview_layers)

@app.route('/output/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', '5000')))