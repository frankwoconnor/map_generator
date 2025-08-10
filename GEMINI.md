# Project Specification: Map Art Generator

## 1. Project Name and Overview

**Project Name:** Map Art Generator

**Overview:**
The Map Art Generator is a web-based application that allows users to create customized SVG (Scalable Vector Graphics) maps from OpenStreetMap (OSM) data. Users can specify a location, define various styling parameters for different map layers (streets, buildings, water), and generate high-quality vector art maps. The application provides a simple web interface for configuration and uses Python for geospatial data fetching, processing, and rendering.

## 2. Core Functionality

The application performs the following key functions:

*   **User Configuration:** Provides a web form for users to input map generation parameters.
*   **Geospatial Data Fetching:** Retrieves OpenStreetMap data (street networks, building footprints, water bodies) for a specified geographical area using the `osmnx` library.
*   **Data Processing and Filtering:** Processes fetched data, including:
    *   Filtering geometries (e.g., only Polygons/MultiPolygons for buildings/water).
    *   Reprojecting data to a local UTM zone for accurate area calculations.
    *   Simplifying geometries based on a user-defined tolerance.
    *   Filtering features based on a minimum size threshold.
    *   Categorizing buildings by area for separate styling/output.
    *   Filtering streets by specific OpenStreetMap `highway` tags.
*   **Map Rendering:** Renders map layers into SVG format using `matplotlib`.
*   **Customizable Styling:** Allows extensive customization of each layer's appearance (face color, edge color, line width, transparency, hatch patterns, drawing order/z-index).
*   **Output Generation:** Generates SVG files, including individual layers (optional) and a combined map.
*   **Web Interface:** Serves as a user-friendly front-end for interacting with the map generation logic.

## 3. Architecture

The project follows a client-server architecture:

*   **Frontend (Client-side):** A simple HTML form (`index.html`) rendered by Flask, allowing users to input parameters.
*   **Backend (Server-side - Python Flask):**
    *   `app.py`: A Flask application that handles HTTP requests. It serves the web interface, processes form submissions, saves user configurations to `style.json`, and orchestrates the execution of the map generation script (`main.py`). It uses a POST-redirect-GET pattern for form submissions.
    *   `main.py`: A standalone Python script that performs the core map generation logic. It reads configuration from `style.json`, fetches data using `osmnx`, processes it, and renders SVG maps using `matplotlib`. It prints progress messages to `stdout` and errors to `stderr`.

## 4. Key Files and Their Roles

*   **`app.py`**:
    *   Flask application entry point.
    *   Manages web routes (`/`, `/output/<path:filename>`).
    *   Loads and saves `style.json`.
    *   Executes `main.py` as a subprocess.
    *   Renders `templates/index.html`.
    *   Serves generated SVG files from the `output/` directory.
*   **`main.py`**:
    *   Contains the core map generation logic.
    *   Parses command-line arguments (e.g., `--prefix` for output filenames).
    *   Reads map configuration from `style.json`.
    *   Uses `osmnx` to fetch geographic data.
    *   Processes and filters geospatial data.
    *   Uses `matplotlib` to plot and save SVG map layers.
    *   Includes helper functions for plotting and data fetching.
    *   Redirects `matplotlib`'s `stdout` during `savefig` calls to prevent unwanted output.
*   **`style.json`**:
    *   A JSON file that stores all user-configurable parameters for map generation.
    *   Acts as the central configuration file for `main.py`.
    *   Its structure is detailed in section 6.
*   **`requirements.txt`**:
    *   Lists Python package dependencies: `osmnx`, `matplotlib`, `flask`.
*   **`templates/index.html`**:
    *   The main HTML template for the web interface.
    *   Contains the form for map configuration.
    *   Displays the generated map preview and links to output files.
*   **`templates/generating.html` (Not currently used in the simplified version):**
    *   (Originally intended for displaying real-time progress, but removed for stability.)
*   **`cache/`**:
    *   Directory for `osmnx` to cache downloaded OpenStreetMap data, speeding up subsequent requests for the same area.
*   **`output/`**:
    *   Directory where all generated SVG map files are saved.
    *   Maps are organized into timestamped subdirectories (e.g., `output/my_map_20250809_123456/`).

## 5. Dependencies

The project requires the following Python packages:

*   `osmnx`: For downloading, constructing, analyzing, and visualizing street networks and other geospatial data from OpenStreetMap.
*   `matplotlib`: For creating static, animated, and interactive visualizations in Python, used here for rendering SVG maps.
*   `flask`: A micro web framework for Python, used for the web interface.
*   `geopandas`: (Implicitly required by `osmnx` for GeoDataFrames)
*   `shapely`: (Implicitly required by `osmnx` for geometric operations)

These dependencies should be listed in `requirements.txt`.

## 6. Configuration (`style.json` Details)

The `style.json` file is a JSON object with the following top-level keys:

*   **`location`**:
    *   `query` (string): The geographical query (e.g., "Cork, Ireland" or "Saint Fin Barre's Cathedral, Cork, Ireland").
    *   `distance` (float, optional): Radius in meters around the query point. If `null`, the administrative boundary of the `query` is used.
*   **`layers`**: (Object containing configuration for each map layer)
    *   **`streets`**:
        *   `enabled` (boolean): Whether to include streets.
        *   `facecolor` (string): Hex color code for street fill (though streets are lines, this might be a fallback or unused).
        *   `edgecolor` (string): Hex color code for street lines.
        *   `linewidth` (float): Thickness of street lines.
        *   `alpha` (float): Transparency (0.0-1.0).
        *   `simplify_tolerance` (float): Simplification tolerance for street geometries.
        *   `min_size_threshold` (float): Minimum size threshold (likely unused for lines).
    *   **`buildings`**:
        *   `enabled` (boolean): Whether to include buildings.
        *   `facecolor` (string): Hex color code for building fill.
        *   `edgecolor` (string): Hex color code for building outlines.
        *   `linewidth` (float): Thickness of building outlines.
        *   `alpha` (float): Transparency (0.0-1.0).
        *   `simplify_tolerance` (float): Simplification tolerance for building geometries.
        *   `hatch` (string or null): Hatch pattern (e.g., "/", "x", "o", or `null` for no hatch).
        *   `zorder` (integer): Drawing order (higher values draw on top).
        *   `size_categories` (array of objects, optional): Defines categories for buildings based on area. Each object has:
            *   `name` (string): Category name (e.g., "large").
            *   `min_area` (float): Minimum area in square meters (inclusive).
            *   `max_area` (float, optional): Maximum area in square meters (exclusive).
        *   `min_size_threshold` (float): Minimum area in square meters for a building to be rendered if `size_categories` are not used.
    *   **`water`**:
        *   `enabled` (boolean): Whether to include water bodies.
        *   `facecolor` (string): Hex color code for water fill.
        *   `edgecolor` (string): Hex color code for water outlines.
        *   `linewidth` (float): Thickness of water outlines.
        *   `alpha` (float): Transparency (0.0-1.0).
        *   `simplify_tolerance` (float): Simplification tolerance for water geometries.
        *   `hatch` (string or null): Hatch pattern.
        *   `zorder` (integer): Drawing order.
        *   `min_size_threshold` (float): Minimum area in square meters for a water body to be rendered.
*   **`output`**:
    *   `separate_layers` (boolean): If `true`, individual SVG files are generated for each layer.
    *   `filename_prefix` (string): Base name for output SVG files. A timestamp is appended automatically.
    *   `output_directory` (string): Path to the directory where SVG files are saved (e.g., "output").
    *   `figure_size` (array of two floats): `[width, height]` in inches for the output SVG.
    *   `background_color` (string): Hex color code for the map background.
    *   `figure_dpi` (integer): Dots Per Inch for the figure.
    *   `margin` (float): Margin around the map content (fraction of axis limits).
*   **`processing`**:
    *   `street_filter` (array of strings): List of OpenStreetMap `highway` tags to include (e.g., `["motorway", "residential"]`).

## 7. Input/Output

*   **Input:** User-defined parameters via the web form, saved to `style.json`.
*   **Output:**
    *   SVG files of generated maps (individual layers and/or combined map) saved in timestamped subdirectories within the `output/` folder.
    *   Error messages displayed on the web interface if map generation fails.

## 8. Usage Instructions

1.  **Clone the repository:** (Assumed to be done by the user/LLM)
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Run the Flask application:** `python3 app.py`
4.  **Access the web interface:** Open a web browser to `http://127.0.0.1:5000/` (or the address provided by Flask).
5.  **Configure map settings:** Adjust parameters in the web form.
6.  **Generate map:** Click the "Generate Map" button. The page will refresh after generation, displaying the new map.

## 9. Assumptions/Constraints

*   **Internet Connectivity:** Required for `osmnx` to download OpenStreetMap data.
*   **Python Environment:** Assumes a Python 3 environment.
*   **Operating System:** Compatible with Linux, macOS, and Windows (with `multiprocessing.set_start_method('fork')` for macOS).
*   **`style.json` existence:** The `app.py` expects `style.json` to exist. An empty dictionary is returned if not found, but `main.py` will error. A default `style.json` should be provided.
*   **Error Handling:** Basic error handling is in place, but comprehensive error logging and user feedback could be improved.
*   **Scalability:** Designed for single-user, on-demand map generation. Not optimized for high-volume concurrent requests.
*   **SVG Output:** The primary output format is SVG, suitable for vector graphics editing.
*   **No Real-time Progress Streaming:** The real-time progress streaming was removed for stability. Map generation is now a blocking operation from the user's perspective until completion.
