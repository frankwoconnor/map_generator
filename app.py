import os
import json
from typing import Any, Dict, List, Optional, Mapping
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, jsonify, flash, get_flashed_messages
import subprocess
import datetime
import glob
import maps2.core.config as cfg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'output' # Directory where generated SVGs are saved
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

    # Convert distance from meters to kilometers for UI display
    location_settings = style_data.setdefault('location', {})
    if location_settings.get('distance') is not None:
        location_settings['distance'] = location_settings['distance'] / 1000.0

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
    """Update location settings in the style dictionary."""
    location = style.setdefault('location', {})
    location['query'] = form.get('location_query', location.get('query'))
    distance_km = form.get('location_distance')
    location['distance'] = float(distance_km) * 1000 if distance_km else None

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
        print(f"Received POST request with action: {request.form.get('action')}")
        # Update style.json based on form data
        
        style = _update_style_from_form(style, request.form)

        # Normalize and validate using core config before persisting/generating
        normalized_style = cfg.normalize_style(style)
        validation_error = cfg.validate_style(normalized_style)
        if validation_error:
            flash(f"Schema validation warning: {validation_error}", category='warning')
        style = normalized_style

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

        print("Starting generation...")
        cmd = ['python3', MAIN_SCRIPT, '--prefix', timestamped_run_identifier]
        print(f"Executing command: {cmd}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Process finished with return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            # Redirect to GET request to display the new map
            return redirect(url_for('index'))
        except subprocess.CalledProcessError as e:
            error_message = f"Map generation failed: {e.stderr}"
            print(f"Error: {error_message}")
            palettes = load_palettes()
            warning_messages = get_flashed_messages(with_categories=True)
            return render_template('index.html', style=style, palettes=palettes, error_message=error_message, warning_messages=warning_messages)
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            print(f"Error: {error_message}")
            warning_messages = get_flashed_messages(with_categories=True)
            return render_template('index.html', style=style, error_message=error_message, warning_messages=warning_messages)

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

        warning_messages = get_flashed_messages(with_categories=True)
        return render_template('index.html', style=style, palettes=palettes, generated_files=generated_files, combined_svg_path=combined_svg_path, svg_content=svg_content, error_message=error_message, progress_log=progress_log, warning_messages=warning_messages)

@app.route('/output/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', '5000')))