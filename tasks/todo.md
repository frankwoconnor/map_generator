# Task List: Map Art Generator v2 — Regions, UX, CLI & Creative Features

See `tasks/plan.md` for details, acceptance criteria and architecture decisions.

## Phase 0: Prerequisites & Quick Fixes

- [x] **Task 0.1**: Repair osmium-tool install; add `make doctor`
  - Files: `Makefile`, `scripts/doctor.sh`, `docs/usage.md` — Scope: S — Deps: none
  - Status: DONE — `brew reinstall osmium-tool`; `osmium fileinfo -e` on Ireland PBF in <1 s
- [x] **Task 0.2**: Fix PBF streets graph (`gdf_nodes` x/y) and `ox.MultiDiGraph` bbox warning
  - Files: `map_core/core/fetch.py`, `main.py` — Scope: S — Deps: none
  - Status: DONE — local streets now return filtered+clipped edges GDF; highway filter was being ignored for local PBFs and is now applied
- [x] **Task 0.3**: Stop-gap CLI flags on `main.py` (`--pbf --center --dist --out --layers`)
  - Files: `main.py` — Scope: XS — Deps: none
  - Status: DONE

- [x] **Checkpoint 0**: osmium works on the Ireland PBF; small-PBF run has no warnings; tests pass

## Phase 1: Region Library

- [ ] **Task 1.1**: `map_core/core/regions.py` — `pbf_info`, `extract_bbox`, `extract_polygon` (+ fallback)
  - Scope: M — Deps: 0.1
- [ ] **Task 1.2**: Region catalog `regions.json` (add/list/get/remove/refresh, per-layer counts)
  - Scope: M — Deps: 1.1
- [ ] **Task 1.3**: Ireland city presets + `regions bootstrap`
  - Scope: S — Deps: 1.2
- [x] **Task 1.4**: Transparent auto-clip cache in `fetch_from_pbf()`
  - Scope: M — Deps: 1.1
  - Status: DONE — `regions.auto_clip()` + `processing.auto_clip*` style fields. Verified on the 354 MB Ireland PBF: 1 km area clips to 1.8 MB in ~15 s, second run is a cache hit; whole run ~22–26 s. Guarded against scrambled/oversized bboxes.
- [ ] **Task 1.5**: Region thumbnails on extract
  - Scope: S — Deps: 1.2, 2.1

- [ ] **Checkpoint 1**: Cork extract < 3 min; Cork generate < 30 s; auto-clip second run fast

## Phase 2: Single Entry Point, Jobs, CLI, API

- [ ] **Task 2.1a**: Move `main()` body into `map_core/generate.py:generate_map()` (golden-file test)
  - Scope: M — Deps: 0.2
- [ ] **Task 2.1b**: Add `RunResult` + `progress_cb` to `generate_map()`
  - Scope: S — Deps: 2.1a
- [ ] **Task 2.2**: `mapgen` CLI package (`generate`, `regions`, `presets`, `info`, `render`) + console script
  - Scope: M — Deps: 2.1b, 1.2
- [ ] **Task 2.3**: `--set` overrides, `--preset`, `--style`
  - Scope: S — Deps: 2.2
- [ ] **Task 2.4**: Job runner `map_core/jobs.py` (submit/status/events/cancel, on-disk records)
  - Scope: M — Deps: 2.1b
- [ ] **Task 2.5**: HTTP API (`/api/generate`, `/api/jobs/*`, SSE, `/api/runs`, `/api/regions`, `/api/pbf/info`)
  - Scope: M — Deps: 2.4, 1.2
- [ ] **Task 2.6**: UI async generate with progress panel
  - Scope: M — Deps: 2.5

- [ ] **Checkpoint 2**: CLI end-to-end with `--set`; UI non-blocking; no subprocess in `app.py`

## Phase 3: Web UI / UX

- [ ] **Task 3.1**: Leaflet map picker (centre click, distance circle, optional bbox)
  - Scope: M — Deps: none
- [ ] **Task 3.2**: Region picker + "Extract new region…" dialog
  - Scope: M — Deps: 2.5, 3.1
- [ ] **Task 3.3**: PBF / region info panel
  - Scope: S — Deps: 1.2, 2.5
- [ ] **Task 3.4**: Run gallery
  - Scope: M — Deps: 2.5
- [ ] **Task 3.5**: Preset gallery + save-as-preset + "Surprise me"
  - Scope: M — Deps: 2.2, 2.3
- [ ] **Task 3.6**: Preview: layer toggles, pan/zoom, downloads
  - Scope: S — Deps: 4.1

- [ ] **Checkpoint 3**: New user → finished Cork map without typing coords/paths; runs on the Pi

## Phase 4: Creative & Backend Features

- [ ] **Task 4.1**: Layered SVG output (named `<g>` per layer, Inkscape labels)
  - Scope: M — Deps: 2.1
- [ ] **Task 4.2a**: Railways layer
- [ ] **Task 4.2b**: Coastline + sea fill layer
- [ ] **Task 4.2c**: Paths/cycleways layer
- [ ] **Task 4.2d**: Landuse categories layer
- [ ] **Task 4.2e**: POI glyph layer
  - Scope: S each — Deps: 2.1
- [ ] **Task 4.3**: Street styling by class + casing
  - Scope: M — Deps: 0.2
- [ ] **Task 4.4**: Crop masks, frame, title/coords text
  - Scope: M — Deps: 2.1
- [ ] **Task 4.5**: Print-ready output (page sizes, bleed, PDF)
  - Scope: S — Deps: 2.1
- [ ] **Task 4.6**: Batch & variation (`--presets`, `--seed`, `--contact-sheet`)
  - Scope: M — Deps: 2.2, 3.5
- [ ] **Task 4.7**: Local-PBF parquet feature cache
  - Scope: S — Deps: 1.4

- [ ] **Checkpoint 4**: Inkscape shows named layers; contact sheet from `--presets all`

## Phase 5: Polish & Deploy

- [ ] **Task 5.1**: Docs (`docs/cli.md`, `docs/regions.md`, `docs/api.md`, README quick-start)
- [ ] **Task 5.2**: Pi deploy (osmium-tool install, regions sync, smoke test on 192.168.1.5)
- [ ] **Task 5.3**: Tests (regions w/ mocked osmium, CLI/`--set`, jobs lifecycle, API contract)

- [ ] **Checkpoint 5**: Fresh clone → setup → bootstrap → generate works; deployed to the Pi
