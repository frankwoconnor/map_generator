# Map Art Generator

Generate stylised map artwork from OpenStreetMap data. Point the app at any
location, choose a style, and it renders an artistic map as a vector (SVG) or
raster (PNG) image — through a browser-based studio or the command line.

## Features
 
- 🌍 **Any location** — pick from a local OSM PBF region catalog or click on an
  interactive map (Leaflet) and set a radius.
- 🗺️ **Four layers** — streets, buildings, water, and green space, each
  independently styled and toggled.
- 🏙️ **Buildings auto-styling** — color buildings by floor size or by distance
  from the city centre using named palettes (quantile bins).
- 🛣️ **Streets styling** — per-road-class styling (category colours, variable
  widths by importance) plus a lightweight "greyscale importance ramp" default.
- 🎨 **Palette gallery & editor** — curate named colour palettes in the UI
  and apply them to streets or buildings; persisted to
  `config/palettes/palettes.json`.
- 🧩 **Presets** — save/load the whole style as a named preset; card gallery
  with in-place load, "Surprise me" random palette generator.
- ⚡ **Async generation** — generate in-place with a live progress log (SSE +
  polling fallback) and cancel support; manual "Refresh results" button;
  previous runs listed with PNG/SVG downloads.
- 👁️ **Live preview** — layer quick-toggle chips re-render a fast preview of
  the current style (cached features) so experiments never need a full run.
- ✂️ **Creative output** — circle/hexagon/rounded crop masks, a frame around
  the mask edge, and a title/subtitle with auto-coordinates (`{coords}`).
- 🖼️ **Export formats** — SVG (vector) and high-quality PNG; per-layer and
  combined; configurable DPI, scale, background, source (original/optimized).
- 🔧 **SVG optimization** — Scour-based cleanup + CSS extraction to external
  stylesheet; configurable rules, suffix, output directory; keep original + optimized.
- 🔷 **SVG shape simplification** — Douglas-Peucker path simplification with
  tolerance, topology preservation, short-segment removal; optional **low-poly
  mode** with grid snapping for faceted artistic style.
- 💡 **UI polish** — contextual tooltips on all advanced settings, palette
  gallery shows "in use by" hints, clearer palette section explanation.
- 🐛 **Bug fixes** — water layer artifacts fixed (geometry repair, polygon/
  centerline dedup, bbox buffering); point-feature (POI) circles dropped.
- 🏗️ **Modular core** — `map_core` is a plain Python library shared by the web
  studio, the `mapgen` CLI, and tests.

## Table of Contents

- [Quick Start](#quick-start)
- [Downloading OSM data](#downloading-osm-data)
- [Web Studio](#web-studio)
- [Command-Line Usage](#command-line-usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Data Layout](#data-layout)
- [Output Structure](#output-structure)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Prerequisites (macOS)

The geospatial stack needs the usual system libraries:

```bash
brew install geos proj spatialindex pkg-config
```

Python 3.11 is expected (the repo pins `.python-version`; [pyenv](https://github.com/pyenv/pyenv)
is recommended). Install the needed Earth-data tools if you want to prepare
your own OSM extracts (optional — a ready-made regional `.osm.pbf` is enough):

```bash
# osmium is used by `mapgen regions extract/bootstrap` to clip region PBFs
brew install osmium-tool
```

### 2. Get the code

```bash
git clone <your-remote>/map_generator.git
cd map_generator
```

### 3. Set up the environment

One command installs the pinned Python, creates `.venv`, and installs the
Python dependencies:

```bash
make setup          # or: bash scripts/setup_env.sh
```

To only install the Python dependencies into an existing venv, use
`make update` (`scripts/update_env.sh`).

### 4. Run the web studio

```bash
make run            # or: bash scripts/run.sh
```

Open <http://127.0.0.1:5000> in a browser. The studio is a Flask app
(`app.py`) that runs in development mode with auto-reload.

> **Data note** — the app looks for region PBFs under the *data root* (by
> default the folder **containing** the repo), i.e.
> `../data/regions/*.osm.pbf`. If you don't have a region PBF yet, see
> [Downloading OSM data](#downloading-osm-data). Without a PBF you can still
> use the map-picker, which fetches from the Overpass API.

## Downloading OSM data

Two ways to get data:

**Option A — any OSM extract (recommended for speed).** Put a regional
`.osm.pbf` (e.g. from [Geofabrik](https://download.geofabrik.de/)) in the
region directory and register it in the catalog:

```bash
# data root defaults to the directory containing the repo, so ../data/regions
mkdir -p ../data/regions
cp /path/to/ireland-and-northern-ireland-latest.osm.pbf ../data/regions/

# inspect it
.venv/bin/python -m mapgen info ../data/regions/ireland-and-northern-ireland-latest.osm.pbf

# clip a city-sized region and add it to the catalog
.venv/bin/python -m mapgen regions extract \
    --from ../data/regions/ireland-and-northern-ireland-latest.osm.pbf \
    --name cork --center 51.8986 -8.4756 --dist 8000
```

**Option B — preset city bootstrap.** `mapgen regions bootstrap` clips a
bundle of cities from one big extract and writes the catalog automatically:

```bash
.venv/bin/python -m mapgen regions bootstrap \
    --from ../data/regions/ireland-and-northern-ireland-latest.osm.pbf \
    --city cork --city dublin --city limerick
```

List what is available:

```bash
.venv/bin/python -m mapgen regions list
```

The default catalog is `<data root>/data/regions/regions.json` (override with
`--catalog` or the `MAPGEN_DATA_DIR` environment variable).

## Web Studio

The studio is organised as a left sidebar plus a preview/results panel.

### Sidebar sections

| Section | Purpose |
| --- | --- |
| **Pick a region** | Choose a catalogued region (dropdown). |
| **Pick on the map** | Click an interactive Leaflet map and set a radius (fetches from OSM live when no PBF covers the area). |
| **Presets** | Card gallery with colour swatches, enabled layers, region and palette badges; save the current style, load/delete in place — no page reload. A "Surprise me" button fills the form with a random palette, street widths and a random crop mask. |
| **Palettes** | Create, edit, rename and delete colour palettes; apply a palette to streets or buildings. |
| **Settings** | Global settings (output size, background, margin, DPI, SVG optimisation). |
| **Creative Output** | Crop mask (circle / hexagon / rounded rectangle with inset & corner radius), a frame around the mask edge (width + colour), and a title/subtitle with `{coords}` (e.g. *51.90° N / 8.47° W*) and `{region}` placeholders. |
| **Map Layers** | Enable/disable each layer and style it: streets (style mode, category colours, variable widths), buildings (auto-styling mode + palette, manual colour), water and green (colours, widths, alpha). |
| **Result** | Live generation log, cancel button, and downloads for the current run, plus **layer quick-toggle chips** that re-render a debounced live preview of the current style. |
| **Previous runs** | Gallery of prior output directories with PNG/SVG download links. |
| **Data & sources** | Human-readable summary of the PBF/data sources in use. |

### Generation flow

1. Configure layers/styles (or load a preset).
2. Click **Generate**. The job runs in a background process
   (`main.py`), and progress is streamed to the UI over Server-Sent Events
   (SSE) with a live log and a cancel button.
3. When it finishes, the combined map (and per-layer files, when enabled)
   appear under the results panel with download buttons.
4. Every run is saved to a timestamped directory under the output root
   (default `<data root>/output/region_<name>_<timestamp>`).

## Command-Line Usage

The project ships two CLI entry points: the `mapgen` command (nicer UX,
catalog-aware) and the low-level `python main.py` generator.

### `mapgen`

```bash
.venv/bin/python -m mapgen --help
.venv/bin/python -m mapgen info <file.osm.pbf> [--json]
.venv/bin/python -m mapgen regions list|info|extract|bootstrap|remove
.venv/bin/python -m mapgen generate ...     # see below
```

Generate using a catalogued region:

```bash
.venv/bin/python -m mapgen generate \
    --region cork \
    --out ../output \
    --prefix cork_blue
```

Generate from a raw PBF + centre point:

```bash
.venv/bin/python -m mapgen generate \
    --pbf ../data/regions/cork.osm.pbf \
    --center 51.8986 -8.4756 \
    --dist 1500 \
    --out ../output --prefix cork_blue \
    --layers streets,buildings \
    --set layers.buildings.facecolor=#ff0066
```

Options:

| Option | Description |
| --- | --- |
| `--region SLUG` | Region slug from the catalog (default catalog `<data root>/data/regions/regions.json`). |
| `--catalog FILE` | Alternative region catalog JSON. |
| `--pbf FILE` | Explicit `.osm.pbf` (overrides `location.pbf_path`). |
| `--center LAT LON` | Map centre (overrides `location.query`). |
| `--dist METRES` | Half-width of the map area in metres (overrides `location.distance`). |
| `--out DIR` | Output directory. |
| `--prefix STR` | Output filename prefix. |
| `--layers a,b,c` | Comma-separated layers to enable. |
| `--set KEY=VALUE` | Override any style value (repeatable; dotted paths like `layers.buildings.facecolor`). |

### Direct generator (`main.py`)

```bash
.venv/bin/python main.py \
    --config style.json \
    --pbf ../data/regions/cork.osm.pbf \
    --center 51.8986 -8.4756 \
    --dist 1500 \
    --out ../output --prefix cork_blue
```

`--prefix` is required; everything else falls back to `style.json` values.
Supported overrides: `--config`, `--pbf`, `--center`, `--dist`, `--out`,
`--prefix`.

## Configuration

Configuration is JSON, kept in the repo (`config/`) plus the working
`style.json` at the repo root:

| File | Purpose |
| --- | --- |
| `style.json` | The active style: `location`, `output`, `figure` (size, dpi, background, margin), `layers` (per-layer `enabled`, colours, widths, zorder), `svg_optimize`. This is the file the studio reads/writes. |
| `config/presets.json` | Named presets (whole styles you can save/apply). |
| `config/palettes/palettes.json` | Named colour palettes (curated entries plus user `_custom` palettes). |
| `config/layers/layer_tags.json` | The OSM tag sets fetched for each layer (`streets`, `buildings`, `water`, `green`) plus `exclude_tags`. |
| `config/regions/*.json` | Region catalog (generated by `mapgen regions ...`). |
| `config/schemas/style_schema.py` | Loading/validation logic for `style.json` defaults. |
| `config/svg_optimize.json` | Post-processing options for generated SVGs. |
| `.env.example` | Optional environment overrides (`MAPGEN_DATA_DIR`, Flask settings). Copy to `.env` to use. |

### Layers

- **streets** — `style_mode` (`manual` | `by_class`), `variable_widths`,
  `street_palette`, per-class `class_colors`/`category_widths`, plus
  `linewidth`/`edgecolor`/`alpha`.
- **buildings** — `auto_style_mode` (`manual` | `manual_floorsize` |
  `auto_size` | `auto_distance`), manual colour, or a palette + (for the auto
  modes) quantile bins by floor size / distance.
- **water / green** — `facecolor`, `edgecolor`, `linewidth`, `alpha`, `zorder`.

### Creative output (`output.*`)
 
- **`output.mask`** — `none` | `circle` | `hexagon` | `rounded`; `mask_options`
  control `inset` (fraction shaved off) and `rounded_radius` (for `rounded`).
- **`output.frame`** — stroked outline around the mask edge (`width` in points,
  `color`).
- **`output.title` / `output.subtitle`** — text above the map with
  `{coords}` / `{region}` placeholders plus font-size/colour keys. Only the
  combined map gets the creative treatment.

### Export formats (`output.*`)
 
- **`output.generate_png`** — `true`/`false` (default `false`). Also exposes
  `png_dpi` (default 300), `png_scale` (default 2.0 = 600 effective DPI),
  `png_suffix` (default `_hq`), `png_background`, `png_source` (`original` |
  `optimized`).
- **`output.svg_optimize_enabled`** — `true`/`false` (default `true`).
  `svg_optimizer` (`scour` | `none`), full Scour config (`svg_optimize_config`),
  CSS extraction (`svg_css_extract`, `svg_css_merge_rules`, `svg_css_attributes`),
  output (`svg_write_separate`, `svg_optimized_suffix`, `svg_optimized_dir`).
- **`output.svg_simplify_enabled`** — `true`/`false` (default `false`).
  `svg_simplify_tolerance` (Douglas-Peucker epsilon), `svg_simplify_preserve_topology`,
  `svg_simplify_min_segment_length`, `svg_low_poly_mode` + `svg_low_poly_grid`.

### Point features (POIs)

Feature queries can return node-tagged points that would otherwise render as
stray circle markers. The pipeline drops `Point`/`MultiPoint` geometries from
every layer before plotting (handled centrally in
`map_core.core.util.drop_point_features`).

## Project Structure

```
map_generator/
├── app.py                      # Flask web studio (routes, SSE jobs, style/preset/palette APIs)
├── main.py                     # Standalone CLI generator + per-layer SVG/PNG saving
├── style.json                  # Active style (studio + CLI default)
├── pyproject.toml              # Package metadata + console script (`mapgen` entry)
├── Makefile                    # setup / update / run / doctor / lock / clean
├── scripts/
│   ├── setup_env.sh            # pyenv + venv + pip deps
│   ├── update_env.sh           # re-install deps
│   ├── run.sh                  # start the Flask studio
│   └── doctor.sh               # environment health check
├── map_core/
│   ├── paths.py                # single source of truth for all data paths
│   └── core/
│       ├── fetch.py            # OSM fetch (local PBF via pyrosm, live via osmnx/Overpass)
│       ├── plot.py             # plot_map_layer + per-feature width logic + debug maps
│       ├── layer_plot.py       # per-layer plotting entry points
│       ├── street_styling.py   # road class categories, widths, palette colours
│       ├── buildings_styling.py# buildings auto/manual styling
│       ├── palettes.py         # palette load/save (byte-stable, metadata aware)
│       ├── regions.py          # region catalog + clipping (osmium)
│       ├── geocode.py          # place → lat/lon for "Pick on the map"
│       ├── svg_post.py         # SVG optimisation (CSS extraction, cleanup)
│       └── util.py             # logging, has_data, drop_point_features
├── mapgen/                     # `mapgen` CLI (info / regions / generate + overrides)
├── config/                     # palettes, presets, layer tags, schemas, svg opts
├── static/                     # studio JS/CSS
├── templates/                  # studio HTML
└── tests/
    ├── unit/                   # pytest unit tests
    └── integration/            # pytest integration tests
```

## Data Layout

All runtime data lives under a single *data root* (see
`map_core/paths.py`). The default root is the directory **containing** the
repo; override it with `MAPGEN_DATA_DIR` in `.env`:

```
Maps/                         <- MAPGEN_DATA_DIR (default: repo parent)
├── map_generator/            <- the git repository (code only)
├── data/
│   ├── sources/              raw PBF extracts (big, rarely read)
│   └── regions/              clipped area PBFs + regions.json
├── cache/
│   ├── clips/                auto-clip results
│   ├── features/             parquet feature cache
│   └── osmnx/                osmnx cache
├── output/                   generated maps (one timestamped dir per run)
└── logs/                     runtime logs
```

## Output Structure
 
Each run writes a timestamped directory under the output root:
 
```
output/region_cork_20261013_105133/
├── config.json                       # the style used for the run
├── *_combined.svg                    # all enabled layers composited (optimized copy in ./optimized)
├── *_combined_hq.png                 # high-quality raster composite (if PNG enabled)
├── *_combined_opt.svg                # optimized SVG (if svg_optimize_enabled + write_separate)
├── *_streets.svg / *_streets_hq.png  # per-layer files (if separate_layers=True)
├── *_buildings.svg
├── *_water.svg
├── *_green.svg
├── *_<layer>_debug_map.svg           # annotated diagnostics
├── *_<layer>_debug_legend.svg
├── optimized/                        # post-processed SVGs (CSS extracted into styles.css)
│   ├── *_combined_opt.svg
│   └── styles.css
├── *_server.log                      # full generation log (studio runs)
```

## Development

- **Run tests:** `make doctor` checks prerequisites, then `pytest`
  (config in `pytest.ini` collects `tests/unit` and `tests/integration`).
- **Lint/style:** the codebase follows PEP 8; keep changes consistent with
  surrounding code.
- **Adding a layer:** add the OSM tag set in `config/layers/layer_tags.json`,
  a plotting entry in `map_core/core/layer_plot.py` (or route through
  `plot_map_layer`), and a style section in `style.json` /
  `config/schemas/style_schema.py`.

## Testing

```bash
.venv/bin/python -m pytest -q            # all unit + integration tests
.venv/bin/python -m pytest tests/unit -q # unit tests only
```

Coverage reports are written to `coverage_report/` when `--cov` is enabled.
Integration tests exercise the full generate pipeline on a small fixture PBF.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError` for `pyrosm`/`geopandas` | Run `make update` / `scripts/update_env.sh` to reinstall the pinned deps into `.venv`. |
| GEOS/PROJ errors at import | `brew install geos proj spatialindex pkg-config` then re-create the venv. |
| No regions in the dropdown | Put a `.osm.pbf` in `<data root>/data/regions/` and run `mapgen regions extract/bootstrap` (see [Downloading OSM data](#downloading-osm-data)). |
| Studio generates but no output | Check the live log in the **Result** panel and the run's `*_server.log`; confirm the data root (`MAPGEN_DATA_DIR`) resolves where you expect. |
| Port 5000 already in use | `PORT=5001 .venv/bin/python app.py` (or set `FLASK_RUN_PORT`). |
| `mapgen` command not found | Install the package (`pip install -e .`) or use `python -m mapgen`. |