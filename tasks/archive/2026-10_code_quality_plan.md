# Implementation Plan: Map Art Generator Code Quality Improvements

## Overview
Address all issues identified in the code review across five axes: Critical bugs, Required refactoring, Architecture improvements, Optional optimizations, and Nit fixes. Tasks are ordered by dependency and risk (critical bugs first, then foundation refactoring, then polish).

## Architecture Decisions
- **Extract shared buildings styling** into `map_core/core/buildings_styling.py` to eliminate duplication between `save_layer()` and combined plot
- **Decompose `fetch_layer()`** into separate functions per data source (PBF, remote) with shared post-processing
- **Complete `StyleConfig` schema** to cover all style.json sections (streets, green, processing)
- **Move config normalization** from `map_core/core/config.py` into `config/schemas/style_schema.py` to eliminate cross-layer import
- **Remove deprecated code** from `config/manager.py` and commented code from `main.py`

## Task List

### Phase 1: Critical Bug Fixes (Must complete before any refactoring)

#### Task 1: Fix undefined `location_query` in buildings auto_distance mode
**Description:** The `location_query` variable is referenced in `main.py` lines 614 and 1656 but not in scope for `save_layer()` function. Need to pass it as parameter or use closure.
**Acceptance criteria:**
- [ ] `save_layer()` accepts `location_query` parameter
- [ ] Combined plot block passes `location_query` when calling buildings auto_distance logic
- [ ] Auto-distance mode works correctly in both separate and combined outputs
**Verification:**
- [ ] Run generation with `auto_distance` mode enabled
- [ ] Verify buildings colored by distance from center
**Dependencies:** None
**Files:** `main.py` (save_layer, combined plot block)
**Scope:** Small

#### Task 2: Fix command injection vulnerability in `app.py`
**Description:** `subprocess.Popen` uses user-controlled `filename_prefix` without validation. Sanitize to alphanumeric + underscore only.
**Acceptance criteria:**
- [ ] `filename_prefix` validated with regex `^[a-zA-Z0-9_]+$`
- [ ] Invalid prefix raises clear error before subprocess call
- [ ] Existing valid prefixes still work
**Verification:**
- [ ] Test with malicious input (e.g., `prefix; rm -rf /`)
- [ ] Test with valid prefix
**Dependencies:** None
**Files:** `app.py` (index route, lines 847-865)
**Scope:** Small

### Phase 2: Required Refactoring - Extract Shared Logic

#### Task 3: Create `map_core/core/buildings_styling.py` with shared auto-coloring logic
**Description:** Extract the identical buildings auto_size/auto_distance/manual_floorsize logic from `save_layer()` (lines 560-708) and combined plot (lines 1599-1781) into a single module.
**Acceptance criteria:**
- [ ] New module exports `style_buildings_auto()`, `style_buildings_manual_floorsize()`, `style_buildings_manual()`
- [ ] Both `save_layer()` and combined plot use the shared functions
- [ ] No behavioral change in output SVGs
**Verification:**
- [ ] Run generation with all building modes
- [ ] Compare SVGs before/after (visual diff)
- [ ] Tests pass
**Dependencies:** Task 1 (needs location_query fix)
**Files:** `map_core/core/buildings_styling.py` (new), `main.py`
**Scope:** Medium

#### Task 4: Create `map_core/core/layer_plot.py` with per-layer plot functions
**Description:** Extract water, streets, green plotting logic from `plot_map_layer()` and combined plot into dedicated functions to eliminate duplication.
**Acceptance criteria:**
- [ ] `plot_water_layer()`, `plot_streets_layer()`, `plot_green_layer()` exported
- [ ] `plot_map_layer()` delegates to these
- [ ] Combined plot uses same functions
**Verification:**
- [ ] All layer outputs identical before/after
- [ ] Tests pass
**Dependencies:** Task 3
**Files:** `map_core/core/layer_plot.py` (new), `map_core/core/plot.py`, `main.py`
**Scope:** Medium

#### Task 5: Decompose `fetch_layer()` into source-specific functions
**Description:** Split the 442-line `fetch_layer()` into `fetch_from_pbf()`, `fetch_from_remote()`, and shared `post_process_features()`.
**Acceptance criteria:**
- [ ] Three new functions with clear single responsibilities
- [ ] `fetch_layer()` becomes thin dispatcher
- [ ] Cache logic preserved in remote path
- [ ] Geometry repair, clip, simplify, filter in shared post-process
**Verification:**
- [ ] Fetch from local PBF works
- [ ] Fetch from remote works
- [ ] Cache hit/miss behavior preserved
- [ ] Tests pass
**Dependencies:** None
**Files:** `map_core/core/fetch.py`
**Scope:** Large (consider splitting into 5a/5b/5c)

### Phase 3: Configuration & Schema Completeness

#### Task 6: Complete `StyleConfig` schema to cover all style.json sections
**Description:** Current `StyleConfig` only models buildings/water/output. Add streets, green, processing sections with all fields from style.json.
**Acceptance criteria:**
- [ ] `StyleConfig` has `streets`, `green`, `processing` fields
- [ ] All layer fields from style.json represented (filter, filters, legend_enabled, etc.)
- [ ] `from_dict()`/`to_dict()` round-trip preserves all data
**Verification:**
- [ ] Load style.json, convert to StyleConfig, back to dict - verify no data loss
- [ ] Validation works with full config
**Dependencies:** None
**Files:** `config/schemas/style_schema.py`
**Scope:** Medium

#### Task 7: Move config normalization into `StyleConfig` schema
**Description:** Move `normalize_style()` logic from `map_core/core/config.py` into `StyleConfig.from_dict()` to eliminate cross-layer import and centralize normalization.
**Acceptance criteria:**
- [ ] `StyleConfig.from_dict()` applies all defaults from `_layer_defaults()`, `_output_defaults()`, etc.
- [ ] `map_core/core/config.py` delegates to `StyleConfig.from_dict()`
- [ ] No behavioral change
**Verification:**
- [ ] All existing tests pass
- [ ] Config loading works end-to-end
**Dependencies:** Task 6
**Files:** `config/schemas/style_schema.py`, `map_core/core/config.py`
**Scope:** Medium

### Phase 4: Dead Code Removal & Cleanup

#### Task 8: Remove deprecated code from `config/manager.py`
**Description:** Remove `get_layer_tags()`, `save_layer_tags()`, `get_layer_tag_config()`, `update_layer_tag_config()` methods marked "deprecated, will be removed".
**Acceptance criteria:**
- [ ] Deprecated methods removed
- [ ] No remaining references to them in codebase
- [ ] `LayerTags` and `LayerTagConfig` classes can be removed if unused
**Verification:**
- [ ] grep for deprecated method names returns nothing
- [ ] Tests pass
**Dependencies:** Task 7 (ensure new feature config system is complete)
**Files:** `config/manager.py`, `config/schemas/layer_tags.py` (potentially)
**Scope:** Small

#### Task 9: Remove commented-out code from `main.py`
**Description:** Remove the commented `load_layer_tags()` function (lines 74-107), commented `PALETTES_FILE`/`_PALETTES_CACHE` (lines 110-111), and any other dead code.
**Acceptance criteria:**
- [ ] No commented-out functions or constants
- [ ] Code is clean
**Verification:**
- [ ] Visual inspection
- [ ] Tests pass
**Dependencies:** None
**Files:** `main.py`
**Scope:** Small

#### Task 10: Consolidate OSMnx cache configuration
**Description:** Remove duplicate OSMnx cache setup in `fetch.py` (lines 94-107) since it's already configured in `main.py` (lines 44-62).
**Acceptance criteria:**
- [ ] Single source of truth for OSMnx cache settings
- [ ] Cache behavior unchanged
**Verification:**
- [ ] Run with local PBF - verify cache_only mode works
- [ ] Run remote - verify cache used
**Dependencies:** None
**Files:** `map_core/core/fetch.py`, `main.py`
**Scope:** Small

### Phase 5: Optional Optimizations

#### Task 11: Vectorize water line plotting with LineCollection
**Description:** Replace per-feature iteration in `_plot_water_features()` (lines 690-722) with `LineCollection` for batch rendering with per-segment linewidths.
**Acceptance criteria:**
- [ ] River/stream/canal lines rendered via single `LineCollection.add_collection()`
- [ ] Linewidths vary by width tag
- [ ] Visual output identical
- [ ] Performance improvement measurable
**Verification:**
- [ ] Time water layer rendering before/after
- [ ] Visual comparison of SVGs
**Dependencies:** Task 4 (layer_plot module)
**Files:** `map_core/core/layer_plot.py` (or plot.py)
**Scope:** Medium

#### Task 12: Parallelize separate layer SVG generation
**Description:** Use `concurrent.futures.ThreadPoolExecutor` to generate separate layer SVGs concurrently in `main.py` (lines 1225-1328).
**Acceptance criteria:**
- [ ] Separate layers generated in parallel
- [ ] Output files identical
- [ ] Total generation time reduced
**Verification:**
- [ ] Time full generation before/after
- [ ] Verify all SVGs produced correctly
**Dependencies:** Task 4 (clean layer plot functions)
**Files:** `main.py`
**Scope:** Small

### Phase 6: Nit Fixes & Polish

#### Task 13: Fix palette reversal inconsistency
**Description:** In combined plot, palette reversed at line 1635 but fallback uses `colors[-1]` (line 1693) instead of `colors[0]`. Standardize on one convention.
**Acceptance criteria:**
- [ ] Consistent palette indexing across all code paths
- [ ] Auto-coloring produces expected dark-to-light or light-to-dark progression
**Verification:**
- [ ] Visual check of auto-colored buildings
**Dependencies:** Task 3
**Files:** `main.py` (combined plot block)
**Scope:** XS

#### Task 14: Add timeout to cached geocode requests
**Description:** `geocode_to_point()` in `geocode.py` has timeout only for online requests. Add timeout for cache file I/O.
**Acceptance criteria:**
- [ ] File read has reasonable timeout
- [ ] No hanging on corrupted cache
**Verification:**
- [ ] Test with large/malformed cache file
**Dependencies:** None
**Files:** `map_core/core/geocode.py`
**Scope:** XS

#### Task 15: Standardize on pathlib.Path
**Description:** Replace remaining `os.path` calls with `pathlib.Path` for consistency (mixed usage in fetch.py, svg_post.py, app.py).
**Acceptance criteria:**
- [ ] All path operations use `Path`
- [ ] No `os.path` imports unless needed for specific functions
**Verification:**
- [ ] grep for `os.path.` returns minimal results
- [ ] Tests pass
**Dependencies:** None
**Files:** Multiple
**Scope:** Small

## Checkpoints

### Checkpoint 1: Critical Bugs Fixed (After Tasks 1-2)
- [ ] `location_query` bug fixed - auto_distance works
- [ ] Command injection fixed - input validated
- [ ] All existing tests pass

### Checkpoint 2: Core Refactoring Complete (After Tasks 3-5)
- [ ] Buildings styling extracted and shared
- [ ] Layer plotting extracted and shared
- [ ] fetch_layer decomposed
- [ ] No behavioral changes
- [ ] All tests pass

### Checkpoint 3: Config Schema Complete (After Tasks 6-7)
- [ ] StyleConfig covers all style.json sections
- [ ] Normalization in schema, not map_core
- [ ] Cross-layer import eliminated
- [ ] All tests pass

### Checkpoint 4: Cleanup Complete (After Tasks 8-10)
- [ ] No deprecated code
- [ ] No commented-out code
- [ ] No duplicate cache config
- [ ] All tests pass

### Checkpoint 5: Optimizations Done (After Tasks 11-12)
- [ ] Water plotting vectorized
- [ ] Parallel layer generation
- [ ] Performance improved
- [ ] All tests pass

### Checkpoint 6: Polish Complete (After Tasks 13-15)
- [ ] Palette consistency fixed
- [ ] Geocode timeout added
- [ ] Pathlib standardized
- [ ] All tests pass

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Refactoring breaks visual output | High | Visual diff comparison at each checkpoint; run full generation before/after |
| fetch_layer decomposition breaks caching | High | Preserve cache key logic exactly; add cache hit/miss test |
| StyleConfig schema missing fields | Medium | Round-trip test with actual style.json |
| Parallel SVG generation causes matplotlib issues | Medium | Use thread pool (not process pool); matplotlib is thread-safe for separate figures |

## Open Questions
- Should `config/schemas/layer_tags.py` be deleted entirely after Task 8? (Depends on whether any external code uses it)
- Is there a need to maintain backward compatibility for old style.json formats without `streets`/`green` sections?
- Should we add a `pyproject.toml` for modern Python packaging?