# Project Overview

This project generates artistic maps from OpenStreetMap (OSM) data with a highly configurable styling pipeline and a modular SVG optimization/export stage.

## Key Features
- Fetch OSM data for streets, buildings, water, and green spaces.
- Multiple building color modes (manual, auto-size, auto-distance, manual floorsize categories).
- Combined and per-layer SVG outputs; optional optimized copies.
- High-quality PNG export from either original or optimized SVGs.
- Configurable SVG optimization (Scour) and optional CSS extraction.

## Core Architecture
- `app.py` (Flask): Web UI for configuration and running generations.
- `main.py`: Orchestrates data fetch, styling, rendering, and post-processing.
- `maps2/core/`:
  - `fetch.py`: OSM data retrieval and buffering/clipping.
  - `plot.py`: Plotting helpers for Matplotlib.
  - `buildings.py`: Building styling logic and category management.
  - `config.py`: Config normalization, defaults, and schema validation.
  - `svg_post.py`: SVG optimization, CSS extraction, and PNG export.

## Data Flow
1. UI collects parameters -> `style.json` updated/validated.
2. `main.py` fetches data with OSMnx according to `location`.
3. Layers are rendered to SVG (and combined SVG) via Matplotlib.
4. `svg_post.py` runs optimization pipeline; writes optimized copies and PNGs.

## Outputs
- Originals: `output/<run>/*`
- Optimized: `output/<run>/optimized/*_opt.svg` (+ `styles.css` if extracted)
- PNGs: next to chosen source (original or optimized) depending on config.
