# Configuration

## style.json
Controls fetching, styling, and rendering.

- location
  - query: string (e.g., "Dublin, Ireland") or center+distance.
  - distance: meters radius used for circular clipping.
- layers
  - streets/water/green: facecolor, edgecolor, linewidth, alpha, zorder, simplify_tolerance.
  - buildings: common params above, plus building color mode and related palettes/categories.
- output
  - figure_size: [width, height] in inches
  - background_color: hex
  - transparent_background: boolean
  - figure_dpi: integer
  - preview_type: string (UI preview behavior)

## svg_optimize.json
Controls SVG post-processing.

- enabled: bool
- optimizer: "scour" | "none"
- scour:
  - digits: 2-4 recommended
  - remove_metadata, enable_viewboxing, shorten_ids, indent_type
- css:
  - extract: bool; when true writes styles.css and injects @import
  - merge_rules: bool
  - file: "styles.css"
- write_separate: bool; if true keep originals and write optimized copies
  - optimized_suffix: e.g., "_opt"
  - optimized_dir: e.g., "optimized"
- png_export:
  - enabled: bool
  - dpi: int (e.g., 300)
  - scale: float (e.g., 2.0)
  - filename_suffix: e.g., "_hq"
  - background: null or hex color
  - source: "original" | "optimized"
- experimental:
  - remove_clip_paths: bool

## Schemas
- JSON schema for style is in `maps2/schemas/style.schema.json` (validation optional but recommended).
