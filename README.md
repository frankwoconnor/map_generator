# Map Art Generator

Generate artistic maps from OpenStreetMap data with customizable layers and styles.

## Table of Contents
- Overview
- Architecture
- Installation
- Running
- Configuration
- SVG Post‑Processing Pipeline
- Output Structure
- Troubleshooting
- Requirements Specification
- License

## Components
- `main.py` — Map generation engine using OSMnx, GeoPandas, Matplotlib
- `app.py` — Flask UI for configuring parameters and running generation
- `style.json` — Configuration (location, layers, output, processing)
- `templates/` — HTML templates for the web UI
- `static/` — CSS/JS assets
- `output/` — Generated files

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the web app: `FLASK_APP=app.py flask run`
3. Configure options in the UI and click Generate.

## Architecture
The system has two primary entry points:

- `app.py` serves a Flask UI for interactive configuration and generation.
- `main.py` is the core engine that:
  - Loads `style.json` and validates against `maps2/schemas/style.schema.json`.
  - Fetches features via OSMnx (`features_from_place` / `features_from_point`).
  - Renders layers with Matplotlib and saves per‑layer SVGs plus a combined SVG.
  - Invokes a modular SVG post‑processing pipeline (`maps2/core/svg_post.py`).

Rendering helpers live in `maps2/core/` (e.g., `plot.py`, `fetch.py`, `buildings.py`).

## Tests
Run tests with:

```
pytest
```

## Notes
- Green (parkland/greenways), water, streets, and buildings are supported.
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
