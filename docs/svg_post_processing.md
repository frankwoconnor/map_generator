# SVG Post-Processing Pipeline
 
Implemented in `map_core/core/svg_post.py` and driven by `config/svg_optimize.json` + `style.json` output section.
 
## Steps
1. Optimize (Scour)
   - Uses Scour when enabled; flags mapped for 0.38.x compatibility.
   - Precision via `digits` -> `--set-precision`.
2. CSS Extraction (optional)
   - Writes `styles.css` and injects `@import` into optimized SVG.
   - For PNG export from optimized SVG, CSS is inlined into a temporary buffer to prevent black PNGs.
3. **Shape Simplification** (optional, new)
   - Douglas-Peucker path simplification (`svg_simplify_tolerance`).
   - Topology preservation (`svg_simplify_preserve_topology`).
   - Short segment removal (`svg_simplify_min_segment_length`).
   - **Low-poly mode**: vertex grid snapping (`svg_low_poly_grid`) for faceted artistic style.
   - Runs on combined SVG and per-layer SVGs (when `separate_layers=true`).
4. Outputs
   - Originals unchanged (when `write_separate` true).
   - Optimized copies written to `optimized/` with `_opt` suffix.
5. PNG Export (optional)
   - `source: original` or `optimized`.
   - `background: null` keeps transparency; set a color to force background.
 
## Configuration (style.json `output` section)
 
| Key | Default | Description |
|-----|---------|-------------|
| `generate_png` | `false` | Export PNG from SVG |
| `png_dpi` | `300` | Base DPI |
| `png_scale` | `2.0` | Scale multiplier (2x = 600 DPI) |
| `png_suffix` | `_hq` | PNG filename suffix |
| `png_background` | `null` | Background color |
| `png_source` | `original` | `original` or `optimized` |
| `svg_optimize_enabled` | `true` | Enable Scour optimization |
| `svg_optimizer` | `scour` | `scour` or `none` |
| `svg_css_extract` | `true` | Extract CSS to styles.css |
| `svg_write_separate` | `true` | Keep original + optimized |
| `svg_simplify_enabled` | `false` | Enable shape simplification |
| `svg_simplify_tolerance` | `0.1` | Douglas-Peucker epsilon |
| `svg_low_poly_mode` | `false` | Aggressive low-poly style |
| `svg_low_poly_grid` | `1.0` | Grid snap size for low-poly |
 
## Tips
- If editors show correct colors but PNGs are black, set `png_export.source` to `original`.
- For smaller SVGs, try `digits: 2-3` and enable `css.extract`.
- `experimental.remove_clip_paths` can reduce size but may alter visuals.
- For low-poly art: `svg_simplify_enabled: true`, `svg_low_poly_mode: true`, `svg_simplify_tolerance: 0.3`, `svg_low_poly_grid: 1.0`.
