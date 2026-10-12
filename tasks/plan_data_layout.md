# Plan: Map data, cache & project file structure

Status: proposal (not implemented). Companion to `tasks/plan.md`.
Goal: make it obvious where data lives, guarantee the app works from a fresh
clone, and let the UI drive data selection (sources → regions → generate)
instead of typing filesystem paths.

---

## 1. How it works today

All runtime data currently lives **outside the repo**, as siblings under `Maps/`:

```
Maps/                         <- multi-project workspace (also holds new-map/, maptoposter/)
├── map_generator/            <- the git repo (code)
├── data/                     <- 367 MB: ireland PBF (354 MB) + regions/ (13 MB)
│   └── regions/{regions.json, cork.osm.pbf}
├── cache/                    <- 5.5 MB: clips/, features/, osmnx/
└── output/                   <- 95 MB: one dir per run (svgs, pngs, config.json)
```

But the code does **not agree** on that location. Paths are hand-written as
relative strings in six different styles:

| Reference | Where | Reality |
|-----------|-------|---------|
| `../data/` | root `style.json`, `scripts/doctor.sh`, CLI, `config/regions` hint | ✅ exists |
| `../osm-data/` | `app.py` default `pbf_folder` (×5), `fetch.py` fallback, `scripts/extract_and_verify_lough.sh` | ❌ **does not exist** |
| `../../data/osm/` | `config/style.json` | ❌ does not exist |
| `../cache/features` | `fetch.py::_cache_paths` (hard-coded) | ✅ exists |
| `../cache/clips` | `regions.py` default | ✅ exists |
| `../output` vs `../../output` | root vs `config/style.json` | both "work" only from a specific CWD |

### Concrete symptoms

1. **The UI's default scan finds nothing.** `app.py` defaults `pbf_folder` to
   `../osm-data/`, which does not exist. A new user opens the page, hits
   *Scan*, and sees an empty list.
2. **Relative paths depend on the current working directory.** `../../data/osm/`
   only resolves if CWD is the repo root; run from anywhere else and it breaks.
   Nothing anchors paths to the repo or a data root.
3. **Two `style.json` files** with different data paths and different
   `output_directory` (root `../output`, `config/../../output`). Only the root
   one is live (`main.py`/`app.py` both define `STYLE_FILE = "style.json"`);
   `config/style.json` is read solely by `config/ConfigManager.get_style_config()`,
   which **nothing calls**. It is stale and misleading.
4. **Cache locations are hard-coded** inside library functions
   (`Path("../cache/features")`, `Path("../cache/clips")`) — not configurable,
   and wrong if the process CWD differs.
5. **`OSMNX_CACHE_FOLDER`** is exported by `scripts/run.sh` and documented in
   `.env.example`, but **never read by any Python code**. OSMnx caching is
   therefore whatever `ox.settings.cache_folder` defaults to (outside our
   control), not `../cache/osmnx`.
6. **No notion of "sources" vs "regions".** Both are just `*.pbf` in a folder;
   the UI can't distinguish a 354 MB country extract from a 12 MB city clip,
   and can't show size/feature counts before generating.

---

## 2. Principle

> The repo is **code only**. Every byte of runtime data lives under a single,
> configurable **data root**, resolved by one module. No library code builds
> relative paths by hand.

One root, one resolver (`map_core/paths.py`), documented subdirectories.

---

## 3. Proposed layout

Default data root = **`<repo>/data`** (self-contained, gitignored). Override with
`MAPGEN_DATA_DIR=/big/volume/mapgen` for large setups or the Pi.

```
map_generator/
├── data/                     # <DATA_ROOT>  (gitignored, relocatable)
│   ├── sources/              # raw PBF extracts as downloaded
│   │   └── ireland-and-northern-ireland-latest.osm.pbf
│   ├── regions/              # clipped, named areas + catalog
│   │   ├── regions.json
│   │   └── cork.osm.pbf
│   ├── cache/                # everything regenerable
│   │   ├── clips/            # auto-clip results
│   │   ├── features/         # parquet feature cache
│   │   └── osmnx/            # OSMnx HTTP cache
│   └── output/               # run artifacts (svgs, pngs, config.json)
│
├── config/                   # STATIC, committed: layers, palettes, schemas,
│   │                         #   regions/ireland_cities.json
│   └── (drop style.json — see below)
├── map_core/                 # library (code)
├── mapgen/                   # CLI (code)
├── app.py  main.py           # entry points
└── style.json                # the ONE live style (committed default)
```

Why this shape:

- **`sources/` vs `regions/`** makes the domain explicit: `sources` are big and
  rarely read; `regions` are small and read every run. The UI can label them.
- **`cache/` is one dir** so a "clear cache" button is a single delete and its
  size is one `du`.
- **`data/` is a single gitignore line** and a single thing to copy to the Pi
  (or to exclude when backing up code).
- **`output/` stays browsable** and is the run gallery's source of truth.

`.gitignore` becomes: `data/` (plus keep `__pycache__`, `.venv`, `*.egg-info`, `.DS_Store`).

---

## 4. `map_core/paths.py` — the single resolver

```python
REPO_ROOT = Path(__file__).resolve().parents[2]

def data_root() -> Path:            # MAPGEN_DATA_DIR or REPO_ROOT/"data"
def sources_dir() -> Path:          # <root>/sources
def regions_dir() -> Path:          # <root>/regions
def region_catalog_path() -> Path:  # <root>/regions/regions.json
def cache_dir() -> Path:            # <root>/cache
def clips_dir() -> Path:            # <root>/cache/clips
def features_dir() -> Path:         # <root>/cache/features
def osmnx_cache_dir() -> Path:      # <root>/cache/osmnx
def outputs_dir() -> Path:          # <root>/output
def ensure_dirs() -> None:          # mkdir -p, called at startup
def describe() -> dict:             # for /api/env: paths + free space + counts
```

- Reads `MAPGEN_DATA_DIR` (and loads `.env` if `python-dotenv` is present, as
  `config/__init__.py` already attempts).
- All getters return **absolute** paths and `mkdir(parents=True, exist_ok=True)`
  on first use, so callers never check existence.
- A tiny `free_space_bytes()` helper for the UI ("14 GB free").

Then replace every hard-coded path:

| File | Change |
|------|--------|
| `map_core/core/fetch.py` | `../cache/features` → `paths.features_dir()`; `../osm-data/{…}` fallback → `paths.sources_dir()`; set `ox.settings.cache_folder = paths.osmnx_cache_dir()` and `use_cache=True` (honour an opt-out) |
| `map_core/core/regions.py` | `DEFAULT_CATALOG` → `paths.region_catalog_path()`; `auto_clip` cache default → `paths.clips_dir()` |
| `config/schemas/style_schema.py` | `output_directory` default stays a *string* but is resolved through `paths.outputs_dir()` in `main.py` |
| `main.py` | resolve `pbf_path` against `paths.sources_dir()` (not `dirname(STYLE_FILE)`); resolve `output_directory` via `paths`; when `--out` is relative, anchor to repo root |
| `app.py` | default `pbf_folder` → `paths.sources_dir()`; region scan → catalog |
| `scripts/run.sh` | drop the inert `OSMNX_CACHE_FOLDER` export (handled in Python now) |
| `scripts/doctor.sh` | report `paths.data_root()` and per-dir status |
| `.env.example` | document `MAPGEN_DATA_DIR`; remove `OSMNX_CACHE_FOLDER` |

### The two `style.json` files

Keep the single **root `style.json`** (runtime config, committed default, one
source of truth). Delete `config/style.json`, and point `ConfigManager` at the
root file if it's ever used (`config/loader.py` currently resolves relative to
`config/`). If `ConfigManager` is genuinely dead, schedule it for removal —
but verify callers first (there are none today).

---

## 5. What's needed to make the project work

"Working" = a fresh clone can produce a map. The minimum:

1. **Tooling** — `.venv` with `requirements.txt`; `osmium-tool` (clipping);
   `cairosvg` (PNG/PDF). Already covered by `make setup` + `make doctor`.
2. **One source of map data**, either:
   - a local PBF dropped in `data/sources/` (recommended; works offline), or
   - remote fetch (Overpass/OSMnx), which needs network + a warm cache.
3. **Writable dirs** — `data/{sources,regions,cache,output}`. `ensure_dirs()`
   creates them; `make doctor` reports free space.
4. **A named region** — either pre-built (`mapgen regions bootstrap`) or
   auto-clipped on first run. This is the main UX gap today.

`make doctor` should print the resolved data root and each dir's size/count, and
say plainly if no sources exist yet, with the Geofabrik download hint.

---

## 6. Tying it into the UI

The web UI is where this pays off. Today it shows a *PBF folder* box, a *Scan*
button and a file list — developer-shaped, and defaulting to a missing dir.
Replace it with a **Data** section driven by the new paths + catalog.

### New read-only endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/env` | `data_root`, each dir + size, disk free, `osmium_available`, counts (sources, regions, runs) |
| `GET /api/sources` | source PBFs (name, size, bbox, timestamp) — `pbf_info`, cached |
| `GET /api/regions` | catalog entries (name, slug, size, bbox, counts, status) |
| `GET /api/pbf/info?path=` | `pbf_info` for any file |
| `GET /api/runs` | output dirs: id, timestamp, prefix/region, artifacts, thumbnail |
| `POST /api/regions/extract` | `{name, source, center, dist}` → clip (job) |

### UI behaviour

- **Location → "Data"**:
  - Read-only line: *Data root: `…/map_generator/data` (14 GB free)*.
  - **Region** dropdown (from `/api/regions`) — the primary choice. Picking one
    sets `pbf_path`, and fills centre/distance from its bbox (already implemented
    in the CLI). Shows size, bbox and object counts beneath.
  - **Source** dropdown (from `/api/sources`) + *"New region from this source…"*
    for raw extracts; picking a source shows its size/bbox and warns it's large.
  - *Browse file…* stays as an escape hatch.
- **Info panel** — the "what's in here" view you asked about earlier: size, bbox,
  node/way/relation counts, last updated, and *N* features are optional
  per-layer counts (from `pbf_info`, or a one-off pyrosm pass). Hints like
  "no water features here" *before* generating.
- **New region dialog** — name + centre/distance (or, later, draw on the map
  picker); posts to `/api/regions/extract`; shows the same progress pattern as
  generate.
- **Cache panel** — total size, per-category (clips/features/osmnx), *Clear*.
- **Run gallery** — grid of `data/output/*` runs (thumbnail, region, preset,
  timestamp), click to reload config or re-run.

Dependencies: extraction takes ~15 s, so tying it to the UI properly wants the
**job runner + SSE** from `tasks/plan.md` (Phase 2). Until then, use a blocking
call with a spinner.

---

## 7. Migration & task list

Ordered so nothing breaks mid-way.

### Phase A — paths plumbing (no behaviour change)
- [ ] A1: add `map_core/paths.py` (`MAPGEN_DATA_DIR`, getters, `ensure_dirs`, `describe`)
- [ ] A2: route `regions.py` catalog + clip cache through `paths`
- [ ] A3: route `fetch.py` features cache + osmnx cache through `paths`; drop the `../osm-data` fallback in favour of `paths.sources_dir()`
- [ ] A4: `main.py` resolves pbf/output via `paths`
- [ ] A5: `app.py` default folder → `paths.sources_dir()`
- [ ] A6: `doctor.sh` reports the resolved layout; `.env.example` documents `MAPGEN_DATA_DIR`
- [ ] A7: tests for `paths` (env override, default, ensure_dirs) with `tmp_path`

- **Checkpoint A**: `MAPGEN_DATA_DIR=/tmp/x python -c "from map_core import paths; paths.ensure_dirs(); print(paths.describe())"` shows the layout; full suite green; a map still generates.

### Phase B — migrate the files
- [ ] B1: `mkdir data/{sources,regions,cache,output}`; `mv` the Ireland PBF → `sources/`, `regions/` → `regions/`, caches → `cache/`, runs → `output/`
- [ ] B2: update `style.json` (`pbf_path`, `pbf_folder`, `output_directory`) to resolved/relative-to-root values
- [ ] B3: delete stale `config/style.json`; repoint or remove `ConfigManager`
- [ ] B4: `.gitignore` → `data/`

- **Checkpoint B**: `mapgen regions list` and `mapgen generate --region cork` work with the new layout; old `../data` etc. gone.

### Phase C — UI integration
- [ ] C1: `/api/env`, `/api/sources`, `/api/regions`, `/api/runs`
- [ ] C2: Data section (region + source dropdowns, info panel)
- [ ] C3: region thumbnails (streets-only PNG per region) for the picker/gallery
- [ ] C4: cache panel
- [ ] C5: `/api/regions/extract` (blocking first, job-based after Phase 2 of `plan.md`)

- **Checkpoint C**: open the page → pick *Cork* from a dropdown → see its size/counts → generate, without typing a path.

---

## 8. Decisions needed

1. **Data root default**: repo-local `./data` (self-contained; recommended) vs
   keep the sibling `Maps/` workspace (`MAPGEN_DATA_DIR=..` achieves it).
2. **Reorganise sources/regions** (move the Ireland PBF into `sources/`) or keep
   flat `data/` with just `regions/` added.
3. **Delete `config/style.json`** now, or leave it and just document which file is live.
4. **Scope of the UI work**: data + region picker only (C1–C2), or go all the way
   to thumbnails/cache/gallery (C1–C5).
