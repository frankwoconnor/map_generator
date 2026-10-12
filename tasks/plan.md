# Implementation Plan: Map Art Generator v2 — Regions, UX, CLI & Creative Features

## Overview

Turn the generator into a tool artists can use fluidly: pick a place, pick a style, get a map — in seconds, not minutes. The three pain points today are (1) the Ireland-wide PBF is too big to read per run, (2) the web UI blocks and gives no feedback, and (3) the CLI is just `main.py --prefix`. This plan introduces a **Region Library** (pre-clipped, catalogued city/area PBFs built with `osmium`), a **non-blocking job model** shared by UI/CLI/API, a real **`mapgen` CLI**, and a set of **creative/backend features** aimed at stylised artwork.

Design goal: *a tool for artists and creatives to make interesting, stylised artwork from map data.*

## Architecture Decisions

- **Clipping with `osmium-tool`, not pyrosm.** pyrosm must parse the whole file even with a `bounding_box`; `osmium extract` streams and is 10–50× faster. Wrap it in `map_core/core/regions.py` with a subprocess call, and keep a slow pure-Python fallback (pyrosm `bounding_box`) only for environments without osmium. On the Pi: `apt install osmium-tool`.
- **Region Library as the primary data source.** `../data/regions/<slug>.osm.pbf` + `../data/regions/regions.json` (name, slug, bbox, source file, size, feature counts, created). The UI/CLI pick a *region*, not a raw file path. The giant file becomes an input to the extractor, not to the renderer.
- **Transparent auto-clip.** If a user still points at a big PBF with a centre+distance, the fetch layer first extracts a cached bbox slice (with margin) into `../cache/clips/<hash>.osm.pbf` and reads from that. Zero UX change, massive speed-up on repeat runs.
- **One generation entry point.** Refactor `main.py:main()` into `map_core/generate.py:generate_map(style: dict, run_id: str, progress_cb) -> RunResult`. UI, CLI and HTTP API all call this. The subprocess shim in `app.py` is replaced by a background job runner.
- **Async jobs + progress stream.** `POST /api/generate` returns a `job_id`; `GET /api/jobs/<id>/events` streams progress (SSE). The UI stops blocking. A simple in-process thread pool + JSON job records on disk is enough (single-user tool); no Celery/Redis.
- **CLI with `argparse` subcommands** (no new dependency), exposed as `python -m mapgen` and a `mapgen` console script. `--set key.path=value` overrides any style field, so artists can iterate without editing JSON.
- **Leaflet map picker** for centre/bbox selection (CDN, no build step). Falls back to lat/lon inputs when offline.
- **Layered, editable SVG output.** Each layer rendered into its own `<g id="layer-buildings" inkscape:label="Buildings">` group so output opens cleanly in Illustrator/Inkscape/Affinity for further work.

## Dependency Graph

```
Phase 0  bug fixes + osmium prerequisite
   │
Phase 1  regions.py (extract/info/catalog)  ──► CLI `mapgen regions …`
   │                                         └─► API /api/regions, /api/pbf/info
   │
Phase 2  generate.py (single entry point, progress callback)
   │        ├─► CLI `mapgen generate`
   │        └─► job runner + /api/generate + SSE
   │                 └─► UI: non-blocking generate, progress panel
Phase 3  UI: map picker, region picker, PBF info panel, run gallery
Phase 4  Creative features (layers, styling, output formats)
Phase 5  Polish, docs, Pi deploy
```

## Task List

### Phase 0: Prerequisites & Quick Fixes
- [ ] Task 0.1: Repair osmium-tool install and document prerequisite
- [ ] Task 0.2: Fix streets graph construction from PBF (`gdf_nodes` x/y) and `ox.MultiDiGraph` bbox warning
- [ ] Task 0.3: Add `--pbf`, `--center`, `--dist`, `--out` to current `main.py` as a stop-gap

### Checkpoint 0
- [ ] `osmium fileinfo -e` works on the Ireland PBF
- [ ] Small-PBF generation runs clean with no warnings in the log
- [ ] Tests pass

### Phase 1: Region Library (big-file workaround)
- [ ] Task 1.1: `map_core/core/regions.py` — `pbf_info()`, `extract_bbox()`, `extract_polygon()`
- [ ] Task 1.2: Region catalog (`regions.json`) — add/list/remove/refresh, feature counts per layer
- [ ] Task 1.3: Ireland city presets (`config/regions/ireland_cities.json`) + `mapgen regions bootstrap`
- [ ] Task 1.4: Transparent auto-clip cache in `fetch_from_pbf()`
- [ ] Task 1.5: Region thumbnail (streets-only, 400px PNG) generated on extract

### Checkpoint 1
- [ ] `mapgen regions extract --from ../data/ireland…pbf --city cork` produces `../data/regions/cork.osm.pbf` in < 3 min
- [ ] Generating Cork from the region file takes < 30 s
- [ ] Pointing at the big file with centre+dist auto-clips and the second run is fast

### Phase 2: Single Entry Point, Jobs, CLI, API
- [ ] Task 2.1: Extract `generate_map()` into `map_core/generate.py` with `progress_cb` and `RunResult`
- [ ] Task 2.2: `mapgen` CLI package: `generate`, `regions`, `presets`, `info`, `render`
- [ ] Task 2.3: `--set` style overrides and `--preset` loading in the CLI
- [ ] Task 2.4: Job runner (`map_core/jobs.py`): thread pool, on-disk job records, cancel
- [ ] Task 2.5: HTTP API: `POST /api/generate`, `GET /api/jobs/<id>`, `GET /api/jobs/<id>/events` (SSE), `GET /api/runs`
- [ ] Task 2.6: UI switches to async generate with a live progress/log panel

### Checkpoint 2
- [ ] `mapgen generate --region cork --preset blueprint --set layers.buildings.facecolor=#ffffff` works end-to-end
- [ ] UI generate no longer blocks; progress visible; result appears without reload
- [ ] `app.py` no longer shells out to `main.py`

### Phase 3: Web UI / UX
- [ ] Task 3.1: Leaflet map picker — click to set centre, drag rectangle/radius for distance, syncs with inputs
- [ ] Task 3.2: Region picker — dropdown of catalogued regions with bbox overlay on the map, "Extract new region…" dialog
- [ ] Task 3.3: PBF/Region info panel — size, bbox, feature counts per layer, last updated, thumbnail
- [ ] Task 3.4: Run gallery — grid of previous runs (thumbnail, timestamp, region, preset), click to reload its config
- [ ] Task 3.5: Preset gallery with thumbnails; "Save current as preset"; "Surprise me" (random palette + style)
- [ ] Task 3.6: Preview improvements — layer toggles, zoom/pan, download buttons per artifact

### Checkpoint 3
- [ ] A new user can go from opening the page to a finished Cork map without typing coordinates or paths
- [ ] Works on the Pi with no build step

### Phase 4: Creative & Backend Features
- [ ] Task 4.1: Layered SVG output — one `<g>` per layer with Inkscape labels; optional `--flatten`
- [ ] Task 4.2: New layers: railways, coastline/sea fill, paths/cycleways, landuse (industrial/residential), POIs as glyphs
- [ ] Task 4.3: Street styling by class — per-highway-type linewidth/colour, casing (outline) lines
- [ ] Task 4.4: Crop masks — circle, hexagon, rounded-rect, custom SVG path; plus border/frame and title/coords text
- [ ] Task 4.5: Print-ready output — page sizes (A4/A3/A2/custom), DPI, bleed, PDF export
- [ ] Task 4.6: Batch & variation — `mapgen generate --presets a,b,c`, `--seed` random styles, contact-sheet PNG
- [ ] Task 4.7: Local-PBF feature cache (parquet) keyed by region+tags, matching the existing remote cache

### Checkpoint 4
- [ ] Open a generated SVG in Inkscape: layers are separate and named
- [ ] `mapgen generate --region cork --presets all --contact-sheet` produces one overview image

### Phase 5: Polish & Deploy
- [ ] Task 5.1: Docs — `docs/cli.md`, `docs/regions.md`, `docs/api.md`; README quick-start rewritten around regions
- [ ] Task 5.2: Pi deploy — osmium-tool install step in `deploy-local`, regions dir on the Pi, smoke test
- [ ] Task 5.3: Tests — regions (mocked osmium), CLI arg parsing + `--set`, jobs lifecycle, API contract

### Checkpoint 5
- [ ] Fresh clone → `make setup` → `mapgen regions bootstrap --city cork` → `mapgen generate --region cork` works
- [ ] Deployed and verified on 192.168.1.5

## Task Details

### Task 0.1: Repair osmium-tool install and document prerequisite
**Description:** `osmium` currently fails with a Boost symbol error. Reinstall and add a `make doctor` target that checks osmium, pyrosm, cairo.
**Acceptance criteria:**
- [ ] `osmium fileinfo -e ../data/ireland-and-northern-ireland-latest.osm.pbf` prints bbox and counts
- [ ] `make doctor` reports status of osmium / pyrosm / cairosvg
**Verification:** run both commands.
**Dependencies:** None. **Files:** `Makefile`, `scripts/doctor.sh`, `docs/usage.md`. **Scope:** S

### Task 0.2: Fix streets graph construction from PBF
**Description:** `fetch_from_pbf()` logs `gdf_nodes must contain 'x' and 'y' columns` and falls back to edges; `_combined_bbox` logs `osmnx has no attribute 'MultiDiGraph'`. Add `x`/`y` from node geometry before `graph_from_gdfs`; use `networkx.MultiDiGraph` for the isinstance check.
**Acceptance criteria:**
- [ ] No warning in log for the small Cork PBF; streets rendered from graph path
- [ ] bbox computed from all layers (no "fallback" message)
**Dependencies:** None. **Files:** `map_core/core/fetch.py`, `main.py`. **Scope:** S

### Task 0.3: Stop-gap CLI flags on `main.py`
**Description:** Add `--pbf`, `--center LAT LON`, `--dist M`, `--out DIR`, `--layers a,b` overriding `style.json` so quick CLI runs are possible before Phase 2 lands.
**Acceptance criteria:**
- [ ] `python main.py --pbf ../new-map/data/clipped_…pbf --center 51.8986 -8.4756 --dist 500 --prefix quick` works without touching `style.json`
**Dependencies:** None. **Files:** `main.py`. **Scope:** XS

### Task 1.1: `regions.py` core
**Description:** `pbf_info(path) -> PbfInfo` (wraps `osmium fileinfo -e -j`), `extract_bbox(src, bbox, dst, strategy="smart")`, `extract_polygon(src, poly_path, dst)`. Pure-Python fallback via pyrosm `bounding_box` + `osmium` python bindings if CLI missing (log a warning about speed).
**Acceptance criteria:**
- [ ] `pbf_info` returns bbox, node/way/relation counts, timestamp, size
- [ ] `extract_bbox` on the Ireland file for Cork (≈ -8.60,51.82,-8.35,51.95) completes and output opens in pyrosm
- [ ] Raises a clear error when osmium is missing and fallback disabled
**Dependencies:** 0.1. **Files:** `map_core/core/regions.py`, `tests/unit/test_regions.py`. **Scope:** M

### Task 1.2: Region catalog
**Description:** `RegionCatalog` reading/writing `../data/regions/regions.json`; each entry: slug, name, bbox, source, path, size_bytes, created, counts per layer (buildings/streets/water/green via a quick pyrosm pass). `add_region(name, bbox, src)`, `list()`, `get(slug)`, `remove(slug)`, `refresh(slug)`.
**Acceptance criteria:**
- [ ] Extracting a region registers it with counts
- [ ] Catalog survives restart; missing files flagged as `stale`
**Dependencies:** 1.1. **Files:** `map_core/core/regions.py`, `config/schemas/region_schema.py`. **Scope:** M

### Task 1.3: Ireland city presets + bootstrap
**Description:** `config/regions/ireland_cities.json` with bboxes for Dublin, Cork, Galway, Limerick, Waterford, Belfast, Derry, Kilkenny, Sligo, Drogheda, Dundalk, Wexford, Athlone, Killarney, Letterkenny. `mapgen regions bootstrap --from <big.pbf> [--city cork,dublin|--all]`.
**Acceptance criteria:**
- [ ] One command produces per-city PBFs and catalog entries
- [ ] Each city file < 60 MB and generates a map in < 30 s
**Dependencies:** 1.2, 2.2 (CLI shell; can land as `scripts/bootstrap_regions.py` first). **Files:** `config/regions/ireland_cities.json`, `mapgen/cmd_regions.py`. **Scope:** S

### Task 1.4: Transparent auto-clip cache
**Description:** In `fetch_from_pbf()`, if the PBF is larger than a threshold (default 50 MB) and a bbox is known, extract `bbox + 20 % margin` to `../cache/clips/<sha1(src,bbox)>.osm.pbf` (once), then read from the clip. Expose `processing.auto_clip` (bool) and `processing.auto_clip_threshold_mb` in the style schema.
**Acceptance criteria:**
- [ ] First run against the Ireland file with 1 km distance completes (clip time + render)
- [ ] Second run reuses the clip; fetch stage < 5 s
- [ ] Disabled via config → old behaviour
**Dependencies:** 1.1. **Files:** `map_core/core/fetch.py`, `config/schemas/style_schema.py`, `docs/configuration.md`. **Scope:** M

### Task 1.5: Region thumbnails
**Description:** After extract, render a 400 px streets-only PNG next to the region file for use in the UI picker and run gallery.
**Acceptance criteria:**
- [ ] `../data/regions/<slug>.png` exists after extraction
**Dependencies:** 1.2, 2.1 (uses `generate_map` with a minimal style). **Files:** `map_core/core/regions.py`. **Scope:** S

### Task 2.1: `generate_map()` single entry point
**Description:** Move the body of `main.py:main()` into `map_core/generate.py`: `generate_map(style, run_id, output_dir, progress_cb=None) -> RunResult` where `RunResult` lists artifacts, timings and warnings. `log_progress` routes through `progress_cb`. `main.py` becomes a thin wrapper. Matplotlib Agg guard lives here.
**Acceptance criteria:**
- [ ] `main.py` behaviour byte-identical for the small Cork PBF (compare SVGs)
- [ ] `generate_map` callable from a test with a temp output dir
**Dependencies:** 0.2. **Files:** `map_core/generate.py`, `main.py`, `tests/integration/test_generate.py`. **Scope:** L → split: 2.1a move code, 2.1b add `RunResult`/`progress_cb`

### Task 2.2: `mapgen` CLI package
**Description:** `mapgen/__main__.py` with argparse subcommands: `generate`, `regions {list,info,extract,bootstrap,remove}`, `presets {list,show}`, `info <pbf>`, `render <svg> --png --dpi`. Console script via `pyproject.toml` (`mapgen = mapgen.__main__:main`). Rich-free; plain text + `--json` output flag.
**Acceptance criteria:**
- [ ] `mapgen generate --region cork --out ../output/cork1` produces artifacts
- [ ] `mapgen regions list --json` is machine-readable
- [ ] `mapgen --help` documents every subcommand
**Dependencies:** 2.1, 1.2. **Files:** `mapgen/__main__.py`, `mapgen/cmd_generate.py`, `mapgen/cmd_regions.py`, `mapgen/cmd_presets.py`, `pyproject.toml`. **Scope:** M

### Task 2.3: `--set` overrides and `--preset`
**Description:** `--preset NAME` loads from `config/presets.json`; `--set a.b.c=value` (repeatable) deep-sets with type coercion (bool/int/float/hex/list). `--style FILE` loads an explicit style JSON. Precedence: style file < preset < --set.
**Acceptance criteria:**
- [ ] `--set layers.buildings.facecolor=#ff0066 --set layers.water.enabled=false` reflected in `config.json` of the run
- [ ] Bad path or type gives a clear error listing valid keys at that level
**Dependencies:** 2.2. **Files:** `mapgen/overrides.py`, `tests/unit/test_overrides.py`. **Scope:** S

### Task 2.4: Job runner
**Description:** `map_core/jobs.py`: `JobManager.submit(style, run_id) -> job_id`, `status(job_id)`, `events(job_id)` (generator), `cancel(job_id)`. One worker thread (generation is CPU-bound and matplotlib is not thread-safe across figures in parallel beyond what `generate_map` already does). Records persisted to `../output/<run_id>/job.json`.
**Acceptance criteria:**
- [ ] Submit/poll/complete lifecycle works in a unit test with a stub generator
- [ ] Cancel marks the job cancelled and stops at the next progress tick
**Dependencies:** 2.1. **Files:** `map_core/jobs.py`, `tests/unit/test_jobs.py`. **Scope:** M

### Task 2.5: HTTP API
**Description:** `POST /api/generate` (body = style overrides + region/pbf), `GET /api/jobs/<id>`, `GET /api/jobs/<id>/events` (SSE), `GET /api/runs` (list past runs with artifacts), `GET /api/regions`, `POST /api/regions/extract`, `GET /api/pbf/info?path=`. Remove the `subprocess` call from `index()`.
**Acceptance criteria:**
- [ ] `curl -X POST /api/generate -d '{"region":"cork","preset":"blueprint"}'` returns a job id; SSE streams `[fetch]`/`[save]` lines; final event carries artifact URLs
- [ ] Existing form POST still works (it now submits a job and redirects to the job page)
**Dependencies:** 2.4, 1.2. **Files:** `app.py` (or new `app/routes/api.py`), `docs/api.md`. **Scope:** M

### Task 2.6: UI async generate
**Description:** Generate button calls `/api/generate`, opens SSE, shows a progress panel (stage, elapsed, log tail), then swaps in the preview without page reload. Disable the button while running; show cancel.
**Acceptance criteria:**
- [ ] No full-page block during generation
- [ ] Errors from the job appear inline with the log tail
**Dependencies:** 2.5. **Files:** `static/app.js`, `templates/index.html`, `static/styles.css`. **Scope:** M

### Task 3.1: Leaflet map picker
**Description:** Add a Leaflet map (CDN) to the Location section. Click sets centre; a draggable circle shows `distance`; inputs and map stay in sync. Optional rectangle draw for explicit bbox. Graceful if tiles are unreachable (map hidden, inputs still work).
**Acceptance criteria:**
- [ ] Clicking the map updates lat/lon fields and vice-versa
- [ ] Changing distance resizes the circle
**Dependencies:** None (independent of backend). **Files:** `templates/index.html`, `static/map_picker.js`, `static/styles.css`. **Scope:** M

### Task 3.2: Region picker
**Description:** Replace "PBF Folder / Scan / File" with "Region" dropdown fed by `/api/regions` (plus "Browse file…" for manual). Selecting a region pans the map to its bbox and draws it. "Extract new region…" opens a dialog: name, draw/accept bbox, source file; submits to `/api/regions/extract` with progress.
**Acceptance criteria:**
- [ ] A fresh user can extract "Galway" from the Ireland file entirely in the UI
**Dependencies:** 2.5, 3.1. **Files:** `templates/index.html`, `static/app.js`. **Scope:** M

### Task 3.3: PBF / region info panel
**Description:** Panel under the picker: file size, bbox, node/way counts, per-layer feature counts, timestamp, thumbnail, and hints ("no water features found in this region" before you generate).
**Acceptance criteria:**
- [ ] Panel populates within 1 s for a catalogued region (from `regions.json`), lazily for raw files
**Dependencies:** 1.2, 2.5. **Files:** `templates/index.html`, `static/app.js`. **Scope:** S

### Task 3.4: Run gallery
**Description:** `/runs` page and a sidebar strip: thumbnail (combined PNG), run id, region, preset, elapsed; click → preview + "Load this config" + download links. Backed by `GET /api/runs`.
**Acceptance criteria:**
- [ ] All previous runs under `../output` listed with thumbnails
**Dependencies:** 2.5. **Files:** `templates/runs.html`, `static/app.js`, `app.py`. **Scope:** M

### Task 3.5: Preset gallery + "Surprise me"
**Description:** Presets shown as cards with a thumbnail (rendered once against a bundled tiny PBF). "Save current as preset" with name. "Surprise me" picks a random palette + line widths + crop mask within sane ranges and fills the form.
**Acceptance criteria:**
- [ ] Clicking a preset card applies it; thumbnails regenerate via `mapgen presets thumbnails`
**Dependencies:** 2.2, 2.3. **Files:** `templates/index.html`, `static/app.js`, `mapgen/cmd_presets.py`. **Scope:** M

### Task 3.6: Preview improvements
**Description:** Layer visibility toggles on the embedded SVG (uses the `<g id="layer-…">` groups from 4.1), pan/zoom, per-artifact download buttons, "open in new tab".
**Dependencies:** 4.1. **Files:** `static/preview.js`, `templates/index.html`. **Scope:** S

### Task 4.1: Layered SVG output
**Description:** Render each layer into a named group (`<g id="layer-buildings" inkscape:groupmode="layer" inkscape:label="Buildings">`) in the combined SVG; keep separate-layer files. Implement via matplotlib `gid` on collections + a post-pass in `svg_post.py` that wraps by gid.
**Acceptance criteria:**
- [ ] Inkscape shows Buildings/Streets/Water/Green as layers
- [ ] File size change < 5 %
**Dependencies:** 2.1. **Files:** `map_core/core/layer_plot.py`, `map_core/core/svg_post.py`. **Scope:** M

### Task 4.2: New layers
**Description:** railways, coastline + sea fill (from `natural=coastline` closed via bbox), paths/cycleways, landuse categories, POI glyphs (configurable marker per amenity). Each is a standard generic layer in the schema with tags in `config/layers/`.
**Dependencies:** 2.1. **Files:** `config/layers/layer_tags.json`, `config/schemas/style_schema.py`, `map_core/core/layer_plot.py`, `templates/index.html`. **Scope:** L → one task per layer

### Task 4.3: Street styling by class + casing
**Description:** `layers.streets.by_class: {motorway: {linewidth, color}, primary: …}` with a default ladder; optional casing (wider darker line beneath). Vectorised with `LineCollection` per class.
**Dependencies:** 0.2. **Files:** `map_core/core/layer_plot.py`, `config/schemas/style_schema.py`. **Scope:** M

### Task 4.4: Crop masks, frame, text
**Description:** `output.mask: circle|hexagon|rounded|path:<svg d>`; `output.frame` (width, colour); `output.title` and `output.subtitle` (default: place name + "51.90° N / 8.47° W") with font family/size/position. Implemented as matplotlib clip path + text artists.
**Dependencies:** 2.1. **Files:** `map_core/core/plot.py`, `config/schemas/style_schema.py`, `templates/index.html`. **Scope:** M

### Task 4.5: Print-ready output
**Description:** `output.page: A4|A3|A2|custom WxH mm`, `output.bleed_mm`, DPI; PDF export via cairosvg; PNG at exact pixel size.
**Dependencies:** 2.1. **Files:** `map_core/core/svg_post.py`, `config/schemas/style_schema.py`. **Scope:** S

### Task 4.6: Batch & variation
**Description:** `mapgen generate --presets a,b,c` (or `all`), `--seed N` deterministic random style, `--contact-sheet` composes all PNGs into one grid image.
**Dependencies:** 2.2, 3.5 (random style generator). **Files:** `mapgen/cmd_generate.py`, `map_core/core/variation.py`. **Scope:** M

### Task 4.7: Local-PBF feature cache
**Description:** Extend the parquet cache in `fetch.py` to `LOCAL_PBF` keyed by (file sha1 prefix+mtime, bbox, tags, simplify, threshold).
**Dependencies:** 1.4. **Files:** `map_core/core/fetch.py`. **Scope:** S

### Tasks 5.1–5.3
Docs, Pi deployment (osmium-tool apt install in `deploy-local`, regions dir sync), tests. **Scope:** S each.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| osmium-tool unavailable/broken on a target machine | High (blocks regions) | `make doctor`; pyrosm fallback with warning; ship pre-clipped city files alongside the big one |
| Ireland city bboxes wrong/too large | Med | Start with 5 cities, verify sizes; bbox table editable in JSON |
| Refactor of `main.py` into `generate_map` changes output | High | Golden-file test: compare SVGs for the small Cork PBF before/after |
| Matplotlib thread-safety in job runner | Med | Single worker thread; Agg backend already enforced |
| Leaflet tiles need internet (Pi may be offline) | Low | Map is optional; inputs always work; consider local tile cache later |
| SSE through a reverse proxy on the Pi | Low | Fall back to polling `GET /api/jobs/<id>` |
| Scope creep in Phase 4 | Med | Each creative feature is its own task; ship Phases 0–3 first |

## Decisions (2026-10-12)

- Bootstrap all 15 Irish cities in the region table; verify Cork first.
- Leaflet via CDN is acceptable; degrade gracefully offline.
- Add `pyproject.toml` with a `mapgen` console script; keep `requirements.txt`.
- Start with Phase 0 + Phase 1.

## Open Questions

- Should region files live in `../data/regions/` (outside repo, synced to the Pi) or be downloadable per city from a release/asset bucket?
- Keep the existing `wall_art.html` flow as-is, or fold it into the run gallery ("use this run in a wall-art template") in Phase 3/4?
