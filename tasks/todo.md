# Task List: Map Art Generator Code Quality Improvements

## Phase 1: Critical Bug Fixes

- [x] **Task 1**: Fix undefined `location_query` in buildings auto_distance mode
  - Files: `main.py`
  - Scope: Small
  - Dependencies: None

- [x] **Task 2**: Fix command injection vulnerability in `app.py`
  - Files: `app.py`
  - Scope: Small
  - Dependencies: None

## Phase 2: Required Refactoring - Extract Shared Logic

- [x] **Task 3**: Create `map_core/core/buildings_styling.py` with shared auto-coloring logic
  - Files: `map_core/core/buildings_styling.py` (new), `main.py`
  - Scope: Medium
  - Dependencies: Task 1
  - Status: DONE - Module created with shared functions for manual, manual_floorsize, auto_size, and auto_distance modes. Both `save_layer()` and combined plot block now use `style_buildings()`.

- [x] **Task 4**: Create `map_core/core/layer_plot.py` with per-layer plot functions
  - Files: `map_core/core/layer_plot.py` (new), `map_core/core/plot.py`, `main.py`
  - Scope: Medium
  - Dependencies: Task 3

- [x] **Task 5a**: Extract `fetch_from_pbf()` from `fetch_layer()`
  - Files: `map_core/core/fetch.py`
  - Scope: Medium
  - Dependencies: None
  - Status: DONE - Function created handling local PBF fetching with graph support and post-processing integration

- [x] **Task 5b**: Extract `fetch_from_remote()` from `fetch_layer()`
  - Files: `map_core/core/fetch.py`
  - Scope: Medium
  - Dependencies: Task 5a
  - Status: DONE - Function created handling remote OSMnx fetching with cache support and fallback to local PBF

- [x] **Task 5c**: Extract `post_process_features()` shared logic
  - Files: `map_core/core/fetch.py`
  - Scope: Medium
  - Dependencies: Task 5a, 5b
  - Status: DONE - Shared post-processing function created with helpers for geometry repair, bbox clipping, simplification, area filtering, and column selection. Both fetch functions use it.

## Phase 3: Configuration & Schema Completeness

- [x] **Task 6**: Complete `StyleConfig` schema to cover all style.json sections
  - Files: `config/schemas/style_schema.py`
  - Scope: Medium
  - Dependencies: None
  - Status: DONE - Added LocationConfig, StreetsConfig, WaterConfig, GreenConfig, OutputConfig, ProcessingConfig; fixed BuildingsConfig.size_categories type.

- [x] **Task 7**: Move config normalization into `StyleConfig` schema
  - Files: `config/schemas/style_schema.py`, `map_core/core/config.py`
  - Scope: Medium
  - Dependencies: Task 6
  - Status: DONE - `normalize_style()` now delegates to `StyleConfig.from_dict().to_dict()`; defaults live in the schema.

## Phase 4: Dead Code Removal & Cleanup

- [x] **Task 8**: Remove deprecated code from `config/manager.py`
  - Files: `config/manager.py`, `config/schemas/layer_tags.py` (potentially)
  - Scope: Small
  - Dependencies: Task 7
  - Status: DONE - Removed deprecated layer tag methods and public exports; kept layer_tags.py schema available for direct import if needed.

- [x] **Task 9**: Remove commented-out code from `main.py`
  - Files: `main.py`
  - Scope: Small
  - Dependencies: None
  - Status: DONE - Removed commented `load_layer_tags()` and palette cache constants.

- [x] **Task 10**: Consolidate OSMnx cache configuration
  - Files: `map_core/core/fetch.py`, `main.py`
  - Scope: Small
  - Dependencies: None
  - Status: DONE - Removed duplicate OSMnx cache setup from fetch.py; single source of truth in main.py.

## Phase 5: Optional Optimizations

- [x] **Task 11**: Vectorize water line plotting with LineCollection
  - Files: `map_core/core/plot.py`
  - Scope: Medium
  - Dependencies: Task 4
  - Status: DONE - Added `_line_collection_for_waterways()` and replaced per-feature loops for rivers/streams/canals/other with batched LineCollection rendering.

- [x] **Task 12**: Parallelize separate layer SVG generation
  - Files: `main.py`
  - Scope: Small
  - Dependencies: Task 4
  - Status: DONE - Added `_generate_layer_task()` and ThreadPoolExecutor-based parallel generation for separate layer SVGs.

## Phase 6: Nit Fixes & Polish

- [x] **Task 13**: Fix palette reversal inconsistency
  - Files: `main.py` (combined plot block)
  - Scope: XS
  - Dependencies: Task 3
  - Status: DONE - Palette handling consolidated in `map_core/core/buildings_styling.py`; reversed palette is used consistently with `colors[0]` fallback.

- [x] **Task 14**: Add timeout to cached geocode requests
  - Files: `map_core/core/geocode.py`
  - Scope: XS
  - Dependencies: None
  - Status: DONE - Added bounded read timeout to geocode cache load via ThreadPoolExecutor.

- [x] **Task 15**: Standardize on pathlib.Path
  - Files: Multiple
  - Scope: Small
  - Dependencies: None
  - Status: DONE - Replaced os.path usage in app.py, fetch.py, svg_post.py with pathlib.Path.

## Checkpoints

- [x] **Checkpoint 1**: Critical Bugs Fixed (After Tasks 1-2)
- [x] **Checkpoint 2**: Core Refactoring Complete (After Tasks 3-5c) - Buildings styling extracted, layer plotting extracted, fetch_layer decomposed
- [x] **Checkpoint 3**: Config Schema Complete (After Tasks 6-7)
- [x] **Checkpoint 4**: Cleanup Complete (After Tasks 8-10)
- [x] **Checkpoint 5**: Optimizations Done (After Tasks 11-12)
- [x] **Checkpoint 6**: Polish Complete (After Tasks 13-15)
