# Troubleshooting

## Black PNGs
- Set `png_export.source` to `original`.
- Or keep `optimized` source and ensure `css.extract` is true; the exporter will inline CSS for PNG rendering.

## Large SVG size
- Lower `scour.digits` to 2–3.
- Enable `css.extract`.
- Increase `simplify_tolerance` per layer in `style.json`.
- Consider `experimental.remove_clip_paths: true`.

## Scour errors
- Set `optimizer: "none"` to isolate problems.
- Ensure Scour is `0.38.x` or adjust flags.

## Transparency issues
- Ensure `style.json` → `output.transparent_background: true`.
- Ensure `svg_optimize.json` → `png_export.background: null`.
