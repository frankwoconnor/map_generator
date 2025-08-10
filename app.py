import os
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import subprocess
import datetime
import glob

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'output' # Directory where generated SVGs are saved
app.secret_key = os.urandom(24)

STYLE_FILE = 'style.json'
MAIN_SCRIPT = 'main.py'

# Helper to load style.json
def load_style():
    try:
        with open(STYLE_FILE, 'r') as f:
            style_data = json.load(f)
            # Ensure preview_type exists with a default value
            if 'output' not in style_data: style_data['output'] = {}
            if 'preview_type' not in style_data['output']: style_data['output']['preview_type'] = 'embedded'
            
            # Convert distance from meters to kilometers for UI display
            if 'location' in style_data and 'distance' in style_data['location'] and style_data['location']['distance'] is not None:
                style_data['location']['distance'] = style_data['location']['distance'] / 1000.0
            
            return style_data
    except FileNotFoundError:
        return {'output': {'preview_type': 'embedded'}} # Return default style if file not found

# Helper to save style.json
def save_style(style_data):
    with open(STYLE_FILE, 'w') as f:
        json.dump(style_data, f, indent=2)

@app.route('/', methods=['GET', 'POST'])
def index():
    style = load_style()
    generated_files = [] # Initialize as a list
    combined_svg_path = None
    svg_content = None
    error_message = None
    progress_log = ""

    if request.method == 'POST' and request.form.get('action') == 'generate':
        print(f"Received POST request with action: {request.form.get('action')}")
        # Update style.json based on form data

        # Location settings
        style['location']['query'] = request.form.get('location_query', style['location']['query'])
        distance_km = request.form.get('location_distance')
        style['location']['distance'] = float(distance_km) * 1000 if distance_km else None

        # Output settings
        style['output']['separate_layers'] = 'separate_layers' in request.form
        style['output']['filename_prefix'] = request.form.get('filename_prefix', style['output']['filename_prefix'])
        
        # Generate timestamp for the run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_run_identifier = f"{style['output']['filename_prefix']}_{timestamp}"
        
        # Save the updated style
        save_style(style)

        width_str = request.form.get('figure_size_width')
        style['output']['figure_size'][0] = float(width_str) if width_str else 10.0
        height_str = request.form.get('figure_size_height')
        style['output']['figure_size'][1] = float(height_str) if height_str else 10.0
        style['output']['background_color'] = request.form.get('background_color', style['output']['background_color'])
        dpi_str = request.form.get('figure_dpi')
        style['output']['figure_dpi'] = int(dpi_str) if dpi_str else 300
        margin_str = request.form.get('margin')
        style['output']['margin'] = float(margin_str) if margin_str else 0.05
        style['output']['preview_type'] = request.form.get('preview_type', style['output'].get('preview_type', 'embedded'))

        # Layers settings
        for layer_name in ['streets', 'buildings', 'water']:
            if layer_name in style['layers']:
                style['layers'][layer_name]['enabled'] = f'{layer_name}_enabled' in request.form
                style['layers'][layer_name]['facecolor'] = request.form.get(f'{layer_name}_facecolor', style['layers'][layer_name].get('facecolor', '#000000'))
                style['layers'][layer_name]['edgecolor'] = request.form.get(f'{layer_name}_edgecolor', style['layers'][layer_name].get('edgecolor', '#000000'))
                linewidth_str = request.form.get(f'{layer_name}_linewidth')
                style['layers'][layer_name]['linewidth'] = float(linewidth_str) if linewidth_str else 0.5
                alpha_str = request.form.get(f'{layer_name}_alpha')
                style['layers'][layer_name]['alpha'] = float(alpha_str) if alpha_str else 1.0
                simplify_tolerance_str = request.form.get(f'{layer_name}_simplify_tolerance')
                style['layers'][layer_name]['simplify_tolerance'] = float(simplify_tolerance_str) if simplify_tolerance_str else 0.0
                min_size_threshold_str = request.form.get(f'{layer_name}_min_size_threshold')
                style['layers'][layer_name]['min_size_threshold'] = float(min_size_threshold_str) if min_size_threshold_str else 0.0
                
                if layer_name != 'streets': # These apply to buildings and water
                    hatch_value = request.form.get(f'{layer_name}_hatch')
                    if hatch_value == 'null':
                        style['layers'][layer_name]['hatch'] = None
                    else:
                        style['layers'][layer_name]['hatch'] = hatch_value
                    zorder_str = request.form.get(f'{layer_name}_zorder')
                    style['layers'][layer_name]['zorder'] = int(zorder_str) if zorder_str else 1

        # Special handling for buildings size_categories
        if 'size_categories' in style['layers']['buildings']:
            new_size_categories = []
            # Assuming fixed number of categories for simplicity in UI
            for i in range(len(style['layers']['buildings']['size_categories'])):
                cat_name = request.form.get(f'buildings_size_category_{i}_name')
                min_area = float(request.form.get(f'buildings_size_category_{i}_min_area')) if request.form.get(f'buildings_size_category_{i}_min_area') else None
                max_area = float(request.form.get(f'buildings_size_category_{i}_max_area')) if request.form.get(f'buildings_size_category_{i}_max_area') else None
                if cat_name:
                    new_size_categories.append({
                        "name": cat_name,
                        "min_area": min_area,
                        "max_area": max_area
                    })
            style['layers']['buildings']['size_categories'] = new_size_categories

        # Processing settings
        street_filter_str = request.form.get('street_filter', '')
        style['processing']['street_filter'] = [s.strip() for s in street_filter_str.split(',') if s.strip()]

        # Save the updated style
        save_style(style)

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
            return render_template('index.html', style=style, error_message=error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            print(f"Error: {error_message}")
            return render_template('index.html', style=style, error_message=error_message)

    else:
        # This block handles GET requests and POST requests that are not for generation
        # It will render the template with the current style and any previously generated content
        # (This part of the code is responsible for displaying the initial page and the final map)

        # Load the style again to ensure it's up-to-date after a potential save
        style = load_style()

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

        return render_template('index.html', style=style, generated_files=generated_files, combined_svg_path=combined_svg_path, svg_content=svg_content, error_message=error_message, progress_log=progress_log)

@app.route('/output/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)