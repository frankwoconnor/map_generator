# SVG Post-Processing Pipeline

Implemented in `maps2/core/svg_post.py` and driven by `svg_optimize.json`.

## Steps
1. Optimize
   - Uses Scour when enabled; flags mapped for 0.38.x compatibility.
   - Precision via `digits` -> `--set-precision`.
2. CSS Extraction (optional)
   - Writes `styles.css` and injects `@import` into optimized SVG.
   - For PNG export from optimized SVG, CSS is inlined into a temporary buffer to prevent black PNGs.
3. Outputs
   - Originals unchanged (when `write_separate` true).
   - Optimized copies written to `optimized/` with `_opt` suffix.
4. PNG Export (optional)
   - `source: original` or `optimized`.
   - `background: null` keeps transparency; set a color to force background.

## Tips
- If editors show correct colors but PNGs are black, set `png_export.source` to `original`.
- For smaller SVGs, try `digits: 2-3` and enable `css.extract`.
- `experimental.remove_clip_paths` can reduce size but may alter visuals.
