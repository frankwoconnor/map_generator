# Map Art Generator

Generate artistic maps from OpenStreetMap data with customizable layers and styles. This project provides a web-based interface for generating high-quality, customizable maps that can be used for artistic or practical purposes.

## Features

- 🗺️ Generate maps for any location worldwide using OpenStreetMap data
- 🎨 Customize map styles, colors, and layers
- 🏢 Support for buildings, streets, water, and green spaces
- 🖼️ Export as SVG (vector) or PNG (raster) formats
- ⚙️ Configurable through an intuitive web interface
- 🏗️ Modular architecture for easy extension

## Table of Contents
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/map-art-generator.git
   cd map-art-generator
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in your browser**
   Visit `http://localhost:5000` to access the web interface.

## Project Structure

```
Map_gen_v2/
├── app/                    # Flask application code
│   ├── routes/            # Route handlers
│   ├── services/          # Business logic
│   └── utils/             # Utility functions
├── config/                # Configuration files
│   ├── layers/           # Layer-specific configurations
│   ├── palettes/         # Color palettes
│   └── schemas/          # JSON schemas for validation
├── docs/                 # Documentation
├── map_core/             # Core map generation logic
│   └── core/             # Core functionality modules
├── static/               # Static assets (CSS, JS, images)
├── templates/            # HTML templates
├── tests/                # Test suite
│   ├── integration/     # Integration tests
│   ├── unit/            # Unit tests
│   └── test_data/       # Test fixtures
└── tools/                # Utility scripts
```

## Configuration

The application is highly configurable through various JSON files:

- `config/style.json` - Main style configuration
- `config/svg_optimize.json` - SVG post-processing settings
- `config/layers/layer_tags.json` - OSM tags for different layers
- `config/palettes/` - Color palettes

### Configuration Management

Configuration is managed through the `ConfigManager` class in `config/manager.py`, which provides:

- Loading and validation of configuration files
- Default values for missing settings
- Type conversion and normalization
- Caching for better performance

## Development

### Prerequisites

- Python 3.12+
- pip
- Git

### Setting Up for Development

1. Fork and clone the repository
2. Set up a virtual environment
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Running in Development Mode

```bash
FLASK_DEBUG=1 python app.py
```

This enables:
- Automatic reloading on code changes
- Debug mode for better error messages
- Development-specific configuration

## Testing

The test suite includes unit tests and integration tests:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest --cov=. tests/

# Run a specific test file
python -m pytest tests/unit/test_example.py -v
```

## Output Structure

Generated files are saved in the parent directory's `output/` folder with the following structure:

```
output/
└── YYYYMMDD_HHMMSS/         # Timestamped run directory
    ├── optimized/          # Optimized output files
    │   ├── map_opt.svg    # Optimized combined SVG
    │   ├── buildings_opt.svg
    │   └── ...
    ├── map.svg            # Original combined SVG
    ├── buildings.svg      # Individual layer SVGs
    ├── streets.svg
    ├── water.svg
    ├── green.svg
    └── map.png            # Raster export (if enabled)
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure all dependencies are installed with `pip install -r requirements.txt`
   - On macOS/Linux, you might need to install system libraries:
     ```bash
     # On Ubuntu/Debian
     sudo apt-get install python3-dev libcairo2-dev
     ```

2. **SVG Optimization Issues**
   - If you encounter issues with SVG optimization, check the logs for specific error messages
   - You can disable optimization by setting `"enabled": false` in `config/svg_optimize.json`

3. **Performance**
   - For large areas, generation might be slow. Consider reducing the map extent or simplifying the style
   - Enable caching of OSM data by setting up a local OSM database

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- Distance clipping (location.distance) is applied consistently across layers.
- Background color is respected in both SVG and PNG outputs.

## Running

### Local development
- Recommended: Python 3.10+ (works on 3.13 as well)
- Create a venv and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip && pip install -r requirements.txt`

### Flask server
- Default: `flask run` (or `python app.py` if supported). The app can be configured to run on port 5001.
- Generated outputs appear under `output/<timestamp_run>/`.

## Configuration

### `style.json`
Controls data source, styling, and output. Key sections:

- `location`: `query` or center + `distance` (meters) for bounding.
- `layers`:
  - `streets`, `buildings`, `water`, `green` each expose color, linewidth, alpha, z‑order, and optional `simplify_tolerance` (geometry simplification pre‑render).
- `output`:
  - `figure_size`: inches, e.g., `[10.0, 10.0]`
  - `background_color`: hex
  - `transparent_background`: bool (when true, override background to transparent on save)
  - `figure_dpi`: DPI for Matplotlib rasterization contexts
  - `preview_type`: embedded vs linked assets for quick UI previews

### `svg_optimize.json`
Drives the SVG post‑processing pipeline (`maps2/core/svg_post.py`). Key options:

- `enabled`: master switch
- `optimizer`: `"scour"` or `"none"`
- `scour`: conservative defaults for compatibility with Scour 0.38.x
  - `remove_metadata`, `enable_viewboxing`, `shorten_ids`, `digits` (e.g., 3), `indent_type: "none"`
- `css`:
  - `extract`: when true, move common style attributes into an external CSS file (`file: "styles.css"`)
  - `merge_rules`: deduplicate where possible
- `write_separate`: when true, keep original SVGs and write optimized copies
  - `optimized_suffix`: e.g., `"_opt"`
  - `optimized_dir`: e.g., `"optimized"` (relative to original)
- `png_export`:
  - `enabled`: render a high‑quality PNG
  - `dpi`, `scale`, `filename_suffix` (e.g., `"_hq"`)
  - `background`: `null` to preserve transparency; set a color to force background
  - `source`: `"original"` or `"optimized"` — choose which SVG to render from
- `experimental`:
  - `remove_clip_paths`: attempts to remove clipPaths and related attributes to reduce file size

## SVG Post‑Processing Pipeline
Implemented in `maps2/core/svg_post.py` and invoked by `main.py` after each SVG is saved.

Steps (config‑driven):

1) Optimize with Scour (if enabled)
   - Version‑safe flag mapping for Scour 0.38.x.
   - Precision via `--set-precision` from `scour.digits`.

2) Optional CSS extraction
   - Extracts common style attributes to `styles.css` and injects a `<style>@import ...</style>` into the optimized SVG for editor compatibility.

3) Write outputs
   - Original SVGs remain unchanged when `write_separate` is true.
   - Optimized SVGs go to `optimized/` with `*_opt.svg` naming.

4) Optional PNG export
   - Source selectable: original or optimized (`png_export.source`).
   - When exporting from optimized, the pipeline attempts to inline external CSS into a temporary buffer to avoid style loss (black PNGs).

## Output Structure

```
output/
  map_YYYYMMDD_HHMMSS/
    config.json
    <run>_buildings.svg
    <run>_streets.svg
    <run>_water.svg
    <run>_green.svg
    <run>_combined.svg
    <run>_combined_hq.png              # when png_export.source = original
    optimized/
      styles.css                        # when css.extract = true
      <run>_buildings_opt.svg
      <run>_streets_opt.svg
      <run>_water_opt.svg
      <run>_green_opt.svg
      <run>_combined_opt.svg
      <run>_combined_opt_hq.png        # when png_export.source = optimized
```

## Troubleshooting

- __Black PNGs__:
  - Set `png_export.source` to `"original"` (safe default), or
  - Keep `source: optimized` and ensure CSS extraction is ON; the pipeline will inline CSS for PNG export.
- __Scour errors (unknown flags)__:
  - We ship version‑safe mappings; if you pin a different Scour version, set `optimizer: "none"` temporarily or adjust `svg_optimize.json`.
- __Huge SVG files__:
  - Reduce `scour.digits` to 2–3.
  - Enable `css.extract` to deduplicate styles.
  - Consider `experimental.remove_clip_paths: true` (may alter visuals).
  - Increase layer‑level `simplify_tolerance` in `style.json` to reduce node counts pre‑render.
- __Transparent background__:
  - Use `style.json` → `output.transparent_background: true`; PNG transparency is preserved when `png_export.background` is `null`.

## Requirements Specification

### Runtime
- Python: 3.10+ (tested on 3.13)
- OS: macOS, Linux (Windows likely works; not primary target)

### Python Dependencies (pinned in `requirements.txt`)
- `osmnx>=2.0,<3.0` — OSM data access (uses `features_from_*` APIs)
- `matplotlib==3.10.5` — Rendering engine (matches SVG metadata)
- `scour==0.38.2` — SVG optimizer (version‑safe flags implemented)
- `cairosvg>=2.7.0,<3.0` — SVG → PNG conversion
- `lxml>=5.2,<6` — XML handling for CSS extraction and transforms
- `flask>=3.0,<4` — Web UI backend

Transitives pulled automatically (e.g., `shapely`, `networkx`, `pandas` for OSMnx; `tinycss2`, `cssselect2`, `cairocffi` for CairoSVG).

### Optional System Dependencies
- CairoSVG can use system Cairo via `cairocffi`. If PNG export fails at runtime, ensure libcairo is available in your environment.

### Configuration & Defaults
- See `style.schema.json` and `svg_optimize.json` for full option spaces and defaults.

## License
MIT (or project‑specific — update as needed).
