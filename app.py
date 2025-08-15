import os
import json
from typing import Any, Dict, List, Optional, Mapping
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, jsonify, flash, get_flashed_messages
import subprocess
import datetime
import glob
import map_core.core.config as cfg
from map_core.core.geocode import geocode_to_point

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')) # Directory where generated SVGs are saved
app.secret_key = os.urandom(24)

STYLE_FILE = 'style.json'
MAIN_SCRIPT = 'main.py'
PALETTES_FILE = 'palettes.json'

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
        "facecolor": "#000000",
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
    location_settings.setdefault('mode', 'address')  # 'address' or 'coords'
    location_settings.setdefault('allow_geocoding', False)

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
    # If no bbox present, set default 1km box
    if not isinstance(location_settings.get('bbox'), (list, tuple)) or len(location_settings.get('bbox') or []) != 4:
        location_settings['bbox'] = default_bbox

    return style_data

def load_palettes() -> Dict[str, List[str]]:
    """Load palettes from `palettes.json`.

    Returns:
        dict[str, list[str]]: Mapping of palette name to list of hex color strings.
    """
    try:
        with open(PALETTES_FILE, 'r') as f:
            palettes = json.load(f)
            if isinstance(palettes, dict):
                return palettes
    except Exception:
        pass
    return {}

def get_available_pbf_files(pbf_folder: str) -> List[Dict[str, str]]:
    """Get list of available PBF files in the specified folder.
    
    Args:
        pbf_folder: Path to the folder containing PBF files
        
    Returns:
        List of dictionaries with 'name' and 'path' keys for each PBF file
    """
    pbf_files = []
    
    if not pbf_folder:
        return pbf_files
        
    try:
        # Resolve relative path
        if pbf_folder.startswith('../'):
            pbf_folder = os.path.abspath(os.path.join(os.getcwd(), pbf_folder))
        
        if os.path.exists(pbf_folder) and os.path.isdir(pbf_folder):
            # Find all .pbf and .osm.pbf files
            for pattern in ['*.pbf', '*.osm.pbf']:
                for filepath in glob.glob(os.path.join(pbf_folder, pattern)):
                    filename = os.path.basename(filepath)
                    # Store relative path for consistency
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
    _update_processing_settings(style, form)
    return style

def _update_location_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update location settings in the style dictionary with optional geocoding."""
    loc = style.setdefault('location', {})

    # Mode: 'address' or 'coords' (default to address for backward compatibility)
    mode = form.get('location_mode', 'address')
    allow_geocode = 'allow_geocoding' in form
    # Persist selections to style for UI state retention
    loc['mode'] = mode
    loc['allow_geocoding'] = bool(allow_geocode)

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

    # PBF folder configuration
    pbf_folder = (form.get('location_pbf_folder') or '').strip()
    if pbf_folder:
        loc['pbf_folder'] = pbf_folder
    elif 'pbf_folder' not in loc:
        loc['pbf_folder'] = '../osm-data/'  # Default PBF folder
    
    # PBF file selection (either from dropdown or manual path)
    pbf_file_selection = (form.get('location_pbf_file_selection') or '').strip()
    manual_pbf_path = (form.get('location_pbf_path') or '').strip()
    
    if pbf_file_selection and pbf_file_selection != 'manual':
        # User selected a file from the dropdown
        loc['pbf_path'] = pbf_file_selection
    elif manual_pbf_path:
        # User entered a manual path
        loc['pbf_path'] = manual_pbf_path
    else:
        # No PBF file selected
        loc['pbf_path'] = None

    # Optional manual bbox override from form
    bx_fields = (
        form.get('location_bbox_minx'),
        form.get('location_bbox_miny'),
        form.get('location_bbox_maxx'),
        form.get('location_bbox_maxy'),
    )
    loc['bbox'] = None
    if all(v is not None and v.strip() != '' for v in bx_fields):
        try:
            bx = [float(b.strip()) for b in bx_fields]
            if len(bx) == 4 and bx[0] < bx[2] and bx[1] < bx[3]:
                loc['bbox'] = bx
            else:
                flash("Invalid bbox: ensure minx<maxx and miny<maxy.", category='warning')
        except Exception:
            flash("Invalid bbox values. Please enter numeric values.", category='warning')

    # Resolve query per mode
    if mode == 'coords':
        # Expect "lat lon" or "lat,lon"
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
    else:
        # Address mode: attempt to geocode; if not allowed, try cache-only
        latlon: Optional[tuple] = geocode_to_point(raw_query, allow_online=allow_geocode)
        if latlon is None:
            loc['query'] = raw_query
            if allow_geocode:
                flash("Could not geocode the address. Please try a different address or enter coordinates.", category='warning')
            else:
                flash("Geocoding disabled or cache miss. Enable 'Allow online geocoding' or enter coordinates.", category='warning')
        else:
            lat, lon = latlon
            loc['query'] = f"{lat} {lon}"
            # Helpful info for users
            flash(f"Resolved address to coordinates: {lat:.6f}, {lon:.6f}", category='info')

    # If using local PBF, and we have numeric coordinates but no distance and no manual bbox,
    # set a sensible default distance of 1000 meters to limit extent.
    try:
        have_bbox = isinstance(loc.get('bbox'), (list, tuple)) and len(loc.get('bbox') or []) == 4
        have_dist = loc.get('distance') is not None
        # Parse numeric coords from loc['query']
        q = (loc.get('query') or '').replace(',', ' ').split()
        numeric_coords = False
        if len(q) >= 2:
            float(q[0]); float(q[1])
            numeric_coords = True
        # Only set default if distance empty and no bbox
        if numeric_coords and not have_dist and not have_bbox:
            loc['distance'] = 1000.0
            flash("Distance not provided; defaulting to 1000 meters around the coordinate center.", category='info')
    except Exception:
        pass

def _update_output_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update output settings in the style dictionary."""
    output = style.setdefault('output', {})
    output['separate_layers'] = 'separate_layers' in form
    output['filename_prefix'] = form.get('filename_prefix', output.get('filename_prefix'))
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

def _update_generic_layer_settings(style: Dict[str, Any], form: Mapping[str, str], layer_name: str) -> None:
    """Update settings for generic layers like streets and water."""
    layers = style.setdefault('layers', {})
    if layer_name in layers:
        layer = layers[layer_name]
        layer['enabled'] = f'{layer_name}_enabled' in form
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

def _update_buildings_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update the complex, multi-mode settings for the buildings layer."""
    buildings = style.setdefault('layers', {}).setdefault('buildings', {})
    buildings['enabled'] = 'buildings_enabled' in form
    buildings['simplify_tolerance'] = float(form.get('buildings_simplify_tolerance') or 0.0)
    buildings['min_size_threshold'] = float(form.get('buildings_min_size_threshold') or 0.0)

    # The form uses 'building_styling_mode' with values: 'manual', 'auto_distance', 'auto_size', 'manual_floorsize'
    buildings_style_mode = form.get('building_styling_mode', 'manual')
    buildings['auto_style_mode'] = buildings_style_mode

    # Reset all styling options before setting the active one
    buildings['size_categories'] = []
    buildings['size_categories_enabled'] = False
    buildings['auto_size_palette'] = ''
    buildings['auto_distance_palette'] = ''
    # Ensure manual settings exists (facecolor only)
    buildings.setdefault('manual_color_settings', {"facecolor": "#000000"})

    if buildings_style_mode == 'manual':
        manual_settings = buildings.setdefault('manual_color_settings', {})
        manual_settings['facecolor'] = form.get('buildings_manual_color_facecolor', manual_settings.get('facecolor', '#000000'))

    elif buildings_style_mode == 'auto_size':
        palette = form.get('auto_size_palette', '')
        buildings['auto_size_palette'] = palette if palette else 'YlGnBu_5'

    elif buildings_style_mode == 'auto_distance':
        palette = form.get('auto_distance_palette', '')
        buildings['auto_distance_palette'] = palette if palette else 'OrRd_3'

    elif buildings_style_mode == 'manual_floorsize':
        # Parse dynamic categories. Inputs are named like:
        # buildings_size_category_{i}_name, _min_area, _max_area, _facecolor
        categories = []
        # Collect all indices present in the form by scanning keys
        indices = set()
        for key in form.keys():
            if key.startswith('buildings_size_category_'):
                try:
                    # Extract the index between the prefix and the next underscore
                    rest = key[len('buildings_size_category_'):]
                    idx_str = rest.split('_', 1)[0]
                    indices.add(int(idx_str))
                except Exception:
                    continue
        for idx in sorted(indices):
            name = form.get(f'buildings_size_category_{idx}_name', '').strip()
            min_area_str = form.get(f'buildings_size_category_{idx}_min_area', '').strip()
            max_area_str = form.get(f'buildings_size_category_{idx}_max_area', '').strip()
            face = form.get(f'buildings_size_category_{idx}_facecolor', '').strip()

            # Skip empty rows (must have at least a color or a range)
            if not (name or min_area_str or max_area_str or face):
                continue

            try:
                min_area = float(min_area_str) if min_area_str != '' else None
            except ValueError:
                min_area = None
            try:
                max_area = float(max_area_str) if max_area_str != '' else None
            except ValueError:
                max_area = None

            # Default color to black if missing
            facecolor = face if face else '#000000'

            # Normalize bounds: if both provided and min > max, swap
            if (min_area is not None) and (max_area is not None) and (min_area > max_area):
                min_area, max_area = max_area, min_area

            categories.append({
                'name': name or f'Category {idx+1}',
                'min_area': min_area,
                'max_area': max_area,
                'facecolor': facecolor
            })

        # Sort categories by min_area (None first), then max_area (None last)
        def _sort_key(cat):
            min_key = float('-inf') if cat.get('min_area') is None else float(cat['min_area'])
            max_key = float('inf') if cat.get('max_area') is None else float(cat['max_area'])
            return (min_key, max_key)

        categories_sorted = sorted(categories, key=_sort_key)

        # Validation: detect overlaps, zero-width, and fully-unbounded categories
        warnings: List[str] = []
        def as_num(v, low=False):
            if v is None:
                return float('-inf') if low else float('inf')
            try:
                return float(v)
            except Exception:
                return float('-inf') if low else float('inf')

        # Zero-width check and unbounded checks per category
        for i, cat in enumerate(categories_sorted):
            mn = cat.get('min_area')
            mx = cat.get('max_area')
            if (mn is not None) and (mx is not None) and float(mn) == float(mx):
                warnings.append(f"Category '{cat.get('name','')}' has zero width (min == max == {mn}).")
            if mn is None and mx is None:
                warnings.append(f"Category '{cat.get('name','')}' is fully unbounded (covers all areas).")

        # Overlap check: allow touching at boundary (prev_max == cur_min is OK)
        for i in range(1, len(categories_sorted)):
            prev = categories_sorted[i-1]
            cur = categories_sorted[i]
            prev_max = as_num(prev.get('max_area'), low=False)
            cur_min = as_num(cur.get('min_area'), low=True)
            if cur_min < prev_max:
                warnings.append(
                    f"Categories '{prev.get('name','')}' and '{cur.get('name','')}' overlap (prev max {prev.get('max_area')} > cur min {cur.get('min_area')})."
                )

        # Flash warnings (non-blocking); also attach to style for optional template use
        for w in warnings:
            flash(w, category='warning')
        buildings['validation_warnings'] = warnings

        buildings['size_categories'] = categories_sorted
        buildings['size_categories_enabled'] = len(categories) > 0

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

def _update_processing_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update processing settings in the style dictionary."""
    processing = style.setdefault('processing', {})
    street_filter_str = form.get('street_filter', '')
    processing['street_filter'] = [s.strip() for s in street_filter_str.split(',') if s.strip()]

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
        street_filter = style.get('processing', {}).get('street_filter', [])
        print(f"Street Filter: {street_filter if street_filter else 'Default/all'}")
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
        street_filter_str = request.form.get('street_filter', '')
        style['processing']['street_filter'] = [s.strip() for s in street_filter_str.split(',') if s.strip()]

        # Save the updated style
        _save_style_json(style)
        expected_out_dir = os.path.join(out.get('output_directory', '../output'), timestamped_run_identifier)
        print(f"Expected Output Folder: {expected_out_dir}")

        print("Starting generation...")
        cmd = ['python3', MAIN_SCRIPT, '--prefix', timestamped_run_identifier]
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
                return render_template('index.html', style=style, palettes=palettes, pbf_files=pbf_files, error_message=error_message, warning_messages=warning_messages)

            # Redirect to GET request to display the new map
            return redirect(url_for('index'))
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            print(f"Error: {error_message}")
            pbf_files = get_available_pbf_files(style.get('location', {}).get('pbf_folder', '../osm-data/'))
            warning_messages = get_flashed_messages(with_categories=True)
            return render_template('index.html', style=style, pbf_files=pbf_files, error_message=error_message, warning_messages=warning_messages)

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
        error_message = None
        progress_log = ''

        # Try to find and load the last generated combined SVG
        # This logic is simplified for demonstration; in a real app, you might store
        # the last generated file path in a database or a more robust way.
        output_base_dir = app.config['UPLOAD_FOLDER']
        latest_combined_svg = None
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
                            combined_svg_in_folder = os.path.join(entry_path, f"{entry}_combined.svg")
                            if os.path.exists(combined_svg_in_folder):
                                latest_combined_svg = os.path.relpath(combined_svg_in_folder, output_base_dir)
                                latest_timestamp = current_timestamp
                    except ValueError:
                        # Not a valid timestamped folder, ignore
                        pass

        if latest_combined_svg:
            combined_svg_path = latest_combined_svg
            if style['output'].get('preview_type') == 'embedded':
                try:
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], combined_svg_path)
                    with open(full_path, 'r') as svg_file:
                        svg_content = svg_file.read()
                except FileNotFoundError:
                    print(f"Error: Could not find SVG file at {full_path}")
                    svg_content = None

        pbf_files = get_available_pbf_files(style.get('location', {}).get('pbf_folder', '../osm-data/'))
        warning_messages = get_flashed_messages(with_categories=True)
        return render_template('index.html', style=style, palettes=palettes, pbf_files=pbf_files, generated_files=generated_files, combined_svg_path=combined_svg_path, svg_content=svg_content, error_message=error_message, progress_log=progress_log, warning_messages=warning_messages)

@app.route('/output/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', '5000')))