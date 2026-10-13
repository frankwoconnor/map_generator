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
  - **generate_png**: boolean (export PNG from SVG)
  - **png_dpi**: integer (base DPI, default 300)
  - **png_scale**: float (scale multiplier, default 2.0)
  - **png_suffix**: string (filename suffix, default "_hq")
  - **png_background**: hex or null
  - **png_source**: "original" | "optimized" (which SVG to rasterize)
  - **svg_optimize_enabled**: boolean (run Scour optimization)
  - **svg_optimizer**: "scour" | "none"
  - **svg_optimize_config**: Scour flags dict (digits, remove_metadata, etc.)
  - **svg_css_extract**: boolean (extract CSS to styles.css)
  - **svg_css_merge_rules**: boolean (merge CSS rules)
  - **svg_css_attributes**: list of attributes to extract
  - **svg_write_separate**: boolean (keep original + optimized)
  - **svg_optimized_suffix**: string (default "_opt")
  - **svg_optimized_dir**: string (default "optimized")
  - **svg_simplify_enabled**: boolean (Douglas-Peucker path simplification)
  - **svg_simplify_tolerance**: float (epsilon in SVG units, default 0.1)
  - **svg_simplify_preserve_topology**: boolean (default true)
  - **svg_simplify_min_segment_length**: float (remove short segments)
  - **svg_low_poly_mode**: boolean (aggressive faceted style)
  - **svg_low_poly_grid**: float (grid snap size, default 1.0)
 
## svg_optimize.json
Controls SVG post-processing (file-based fallback for style.json).
 
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
- JSON schema for style is in `config/schemas/style_schema.py` (validation optional but recommended).
