# Design Direction & Future Improvements

## Overview
This document captures the design decisions, explorations, and potential next steps for the Map Art Generator project.

---

## Completed Work (Phases C–E)

### Phase C: Core Generation & CLI
- **C1**: `mapgen` CLI with `info`, `regions`, `generate` commands
- **C2**: Region catalog with `bootstrap` and `extract` (osmium clipping)
- **C3**: PBF auto-clip with caching
- **C4**: Leaflet map picker with offline fallback

### Phase D: Web Studio (Main)
- **D1**: Two-pane studio layout (controls + sticky preview)
- **D2**: Preset gallery (card-based, in-place load, "Surprise me")
- **D3**: Layer quick-toggles + cheap live preview (SSE job tracking + polling fallback)
- **D4**: Creative output (mask, frame, title/subtitle with placeholders)
- **D5**: Wall art page (deferred)

### Phase E: Polish (Partial)
- **E1**: Dark mode (deferred)
- **E2**: Empty states + tooltips ✓ (tooltips implemented)
- **E3**: Frontend smoke tests (deferred)
- **E4**: Perf optimizations (feature cache, debounced preview)

### Additional Features Delivered
- **Output format controls**: PNG export (DPI, scale, source), SVG optimization (Scour + CSS extraction)
- **SVG shape simplification**: Douglas-Peucker path reduction, low-poly mode with grid snapping
- **Water layer artifact fixes**: geometry repair (`make_valid`), polygon/centerline dedup, bbox buffering
- **Point feature (POI) filtering**: central `drop_point_features` drops all Point/MultiPoint geometries
- **Job tracking robustness**: SSE + 2s polling fallback + 5s safety net + manual refresh button
- **UI tooltips**: `data-tooltip` CSS system on all advanced settings

---

## Design Decisions Explored

### 1. SVG Optimization: Scour vs SVGO
**Decision**: Scour (pure Python)
**Rationale**: Zero external deps, runs in-process, adequate for our use case.
**Trade-off**: SVGO has richer plugins (path merging, color conversion, unknown attr removal).
**Future**: Optional SVGO backend if Node.js available (`svg_optimizer: "svgo"`).

### 2. Shape Simplification: svgpathtools + lxml
**Decision**: Custom Douglas-Peucker implementation using svgpathtools for parsing.
**Rationale**: 
- Pure Python, no Node.js
- Works on path `d` attributes directly (no GeoJSON round-trip)
- Handles polygons, polylines, and paths
**Trade-off**: vpype has more mature simplification pipeline but is CLI-first.
**Future**: Consider vpype as library if it stabilizes Python API.

### 3. Water Layer Geometry Handling
**Problem**: River polygons (`water=river`) and centerlines (`waterway=river`) misaligned; clipping created blunt ends; `buffer(0)` repair created triangular wedges.
**Solution**:
- Use `shapely.make_valid()` instead of `buffer(0)` for polygons
- Filter centerlines overlapping river polygons (bbox-based)
- Add 1% bbox buffer for water layer fetches
**Trade-off**: Bbox filter may miss edge cases; could use spatial index for precision.

### 4. Job Tracking: SSE + Polling
**Problem**: SSE silently fails in dev server; UI stuck "Generating..."
**Solution**: 
- Primary: SSE (`EventSource`)
- Fallback: 2s polling after 5s safety net
- Manual: "Refresh results" button
**Trade-off**: Polling adds server load; acceptable for single-user studio.

### 5. Palette Gallery UX
**Problem**: "What are palettes? How do I use them?"
**Solution**: 
- Explanatory text in section header
- "In use by: streets (by class), buildings (by size)" badges
- Tooltips on Apply buttons
**Future**: Visual palette preview in layer selects (swatch strip exists but small).

---

## Next Steps (Prioritized)

### High Impact / Low Effort

1. **Dark Mode** (E1)
   - CSS custom properties already support theming
   - Add toggle in header, persist to localStorage

2. **Palette Preview in Layer Selects**
   - Expand `palette-preview` swatch strip in Streets/Buildings palette dropdowns
   - Show 5-color strip instead of single color

3. **VPype Integration for Advanced Simplification**
   - `vpype` has `linemerge`, `reloop`, `linesort` for pen-plotter optimization
   - Could replace/custom simplify step for better topology

4. **Export Presets**
   - Save/load output format configs (PNG+SVG+simplify) as named presets
   - Separate from style presets

### Medium Impact

5. **Batch Generation / CLI Enhancements**
   - `mapgen generate --region cork --styles preset1,preset2`
   - Parallel region processing
   - Output manifest JSON

6. **SVGO Optional Backend**
   - Detect `npx svgo` or `svgo` binary
   - Config: `svg_optimizer: "svgo"` + `svgo_config_file`

7. **Region Catalog UI Improvements**
   - Thumbnail preview in region picker
   - Drag-and-drop region ordering

8. **Layer Visibility Persistence**
   - Save layer chip toggle state to style.json
   - Restore on page load

### Advanced / Exploratory

9. **Real-time Collaboration**
   - Shared style.json via WebSocket
   - Multi-user studio (low priority)

10. **AI-Assisted Styling**
    - "Make it look like [map style]" → palette + widths
    - Embedding-based style search

11. **Print/Export Pipeline**
    - A4/A3/A2 page sizes with bleed
    - PDF via cairosvg
    - CMYK color profile support

12. **3D / Extrusion Export**
    - Building heights → STL/OBJ
    - Layered SVG for laser cutting

---

## Technical Debt / Refactoring

- **`main.py` too large** (~1800 lines) → split into `mapgen/commands/generate.py`, `mapgen/output.py`
- **`app.py` route sprawl** → move to `app/routes/` (partially done)
- **Style schema duplication** → generate JSON Schema from dataclasses
- **Test coverage gaps** → integration tests for new output formats, simplification
- **Type hints** → stricter mypy config

---

## Dependencies to Watch

| Package | Current | Notes |
|---------|---------|-------|
| `svgpathtools` | Added | For shape simplification |
| `scour` | Existing | SVG optimization |
| `cairosvg` | Existing | PNG export |
| `vpype` | Not yet | Candidate for advanced pipeline |
| `osmium` | System | Region clipping (required) |

---

## Migration Notes

### Style.json New Fields (auto-populated with defaults)
```json
{
  "output": {
    "generate_png": false,
    "png_dpi": 300,
    "png_scale": 2.0,
    "png_suffix": "_hq",
    "png_background": null,
    "png_source": "original",
    "svg_optimize_enabled": true,
    "svg_optimizer": "scour",
    "svg_css_extract": true,
    "svg_css_merge_rules": true,
    "svg_write_separate": true,
    "svg_optimized_suffix": "_opt",
    "svg_optimized_dir": "optimized",
    "svg_simplify_enabled": false,
    "svg_simplify_tolerance": 0.1,
    "svg_simplify_preserve_topology": true,
    "svg_simplify_min_segment_length": 0.5,
    "svg_low_poly_mode": false,
    "svg_low_poly_grid": 1.0
  }
}
```

Run `make setup` to install new `svgpathtools` dependency.

---

*Last updated: 2026-10-13*