# Implementation Plan: Web UI — a studio, not a form

Status: proposal (not implemented). Expands **Phase 3 of `tasks/plan.md`** into
buildable tasks; does not replace it. Task list for this plan lives here (the
existing `tasks/todo.md` tracks the v2 plan).

Progress:
- **Phase A shipped** (commit `897bae4` + `230fdf6`): two-pane studio, result
  panel, run gallery + `/api/runs`, `map_core/runs.py`, path plumbing, dead
  template removed, data-layout Phase B (PBF moved to `data/sources/`).
- **Phase B shipped** (commit `93e6f71`): `map_core/jobs.py` job runner,
  `/api/generate`, `/api/jobs/<id>` + SSE events + cancel, in-page progress
  panel, client-side validation, in-place result swap. The no-JS form POST
  still runs synchronously.
- **Phase C shipped** (commits `e8bc199`, `8e0bdd1`, `17ade24`): `/api/env`,
  `/api/sources`, `/api/regions`, `/api/pbf/info`, `/api/regions/extract`; region
  quick-pick, data/sources panel, "New region…" extraction job, and the Leaflet
  map picker with offline fallback.
- **Next**: D (creative richness) and E (ship).

Goal: make the web UI a place an artist actually *wants* to spend time — see
data, pick an area, iterate visually, watch it render, and export wall art.
Priority order: **richness** (more to see and do), **usability** (fewer wrong
turns, faster loops), **features** (things the CLI can do that the UI can't).

---

## 1. What exists today (findings)

The UI is one page: `templates/index.html` (419 lines) + `static/app.js`
(511) + `static/styles.css` (706). It is a single long `<form>` of collapsible
`<details>` sections (Presets, Settings, Map Layers) that POSTs to `/`.

**The biggest problem: results are never shown.** `app.py::index()` (GET branch,
lines 1009–1067) finds the latest run, builds `combined_svg_path`,
`svg_content` and `preview_layers`, and passes them to the template — but
`index.html` renders **none** of them (grep for `preview`/`svg` finds only the
palette previews). After a successful generate the page simply reloads to the
form. The work product is invisible.

Other concrete findings:

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Generation **blocks the HTTP request**; subprocess stdout goes to the server console and a log file only — no browser progress | `app.py:894–915` |
| 2 | No result/preview/download UI; `/output/<path>` route exists but nothing links to it | `index.html` (absent), `app.py:1070` |
| 3 | No run history/gallery; the latest run is re-derived on every GET by scanning a timestamp pattern | `app.py:980–1030` |
| 4 | `templates/generating.html` is a finished progress page that **nothing references** | `grep render_template` |
| 5 | `templates/base.html` references `static/css/styles.css` and `static/js/main.js`, **neither of which exists**; no template extends it | `base.html:7,18` |
| 6 | PBF selection is developer-shaped: folder textbox + *Scan* + file dropdown + manual path, and the template still hard-codes `../osm-data/` as the fallback/placeholder | `index.html:120–153` |
| 7 | No region concept, no map/area picker, no "what's in this data" info (size/bbox/counts) | — |
| 8 | Palette CRUD API exists (create/get/delete) but the UI only *reads* palettes to colour buildings | `app.js:263–323`, `app.py:647–712` |
| 9 | Presets work but every action calls `alert()` then `window.location.reload()` | `app.js:348–431` |
| 10 | `wall_art.html` (948 lines) is a separate, client-side compositor at `/wall-art` — upload image + text + template + download — **not connected** to generated output | `app.py:780–783` |
| 11 | The action-bar status chip is a hard-coded "Ready"; no dirty-state, no validation feedback until submit | `index.html:65`, `app.js` (unused) |
| 12 | Path handling in the UI is still CWD-relative (`get_available_pbf_files` resolves `../` against `Path.cwd()`; `expected_out_dir` is relative) | `app.py:190–191, 876–878` |
| 13 | No dark mode, no keyboard affordances, minimal ARIA | `styles.css` |
| 14 | Layer edits are text/number inputs; no visual palette/layer preview until a full render | `index.html:277–309` |

What's *good* and worth keeping: a clean token-based design system
(`--space-*`, `--radius-*`, `--shadow-*`), collapsible sections, a sticky action
bar, feature-category checkboxes, building colour modes with palette swatches.

---

## 2. Design direction

1. **Two-pane studio.** Left: controls (scrollable). Right: a **sticky live
   preview** (the current map, per-layer toggle, zoom). The action bar sits with
   the preview, not above the form. On narrow screens the preview stacks on top.
2. **Progressive disclosure.** A first-run user should get a good map from
   "pick a region → pick a palette → Generate", with everything else behind
   *Advanced*.
3. **Show, don't tell.** Thumbnails for regions and past runs, live palette
   swatches, a bbox drawn on a map, feature counts before you commit to a render.
4. **Never block.** Async job + server-sent progress (reuse `generating.html`'s
   idea); the browser stays responsive and the result appears without a reload.
5. **One design system.** Delete the dead templates, keep the CSS variables, add
   dark-mode tokens.
6. **Degrade gracefully offline.** Leaflet from CDN, but the form still works
   with no network (coordinates + region dropdown are enough).

---

## 3. Architecture decisions

- **Async job runner is the backbone.** Tasks B\* depend on `map_core/jobs.py`
  + the HTTP API (v2 plan Tasks 2.4–2.5). Phase A can ship without them.
- **Server-rendered shell, JSON-hydrated client.** Keep Jinja for the initial
  form (low risk, works without JS), but move app.js to hydrate from a single
  `/api/state` payload instead of scattered `initial-data` + hard-coded values.
  One serializer becomes the source of truth.
- **Cheap live preview.** Use a **streets-only** render (a few seconds, the CLI
  already supports `--layers`) as the interactive preview; the *Generate* button
  does the full multi-layer render. Debounce, and cache by a hash of the relevant
  settings.
- **Data endpoints come from the data-layout work**: `/api/env`, `/api/sources`,
  `/api/regions`, `/api/pbf/info` (see `tasks/plan_data_layout.md`, Phase C).
- **Reuse, don't duplicate.** Run discovery, output serving and preset/palette
  logic already exist server-side; expose them as JSON and render client-side.
- **Fix UI paths via `map_core/paths.py`** as part of this work (finding 12).

---

## 4. Task list

### Phase A — Result view & studio shell (fix the glaring gap)

**Task A1: Results panel — show the generated map**
Description: Render the latest run's combined SVG inline, list per-layer SVGs as
thumbnails with visibility toggles, and provide download links (SVG per layer,
combined PNG/SVG). Uses the data `app.py::index()` already computes.
Acceptance:
- [x] After a successful generate, the combined map is visible without any extra clicks
- [x] Each layer artifact is listed with a working download link
- [x] Empty state shown when there is no run yet
Verification: `pytest tests/ -q`; manual: generate Cork, confirm image + links.
Dependencies: none. Files: `templates/index.html`, `static/app.js`,
`static/styles.css`, `app.py::index()`. Scope: M.

**Task A2: Two-pane responsive layout**
Description: Restructure `index.html` into controls + sticky preview column; move
the action bar to the preview column; stack on narrow screens.
Acceptance:
- [x] ≥1024px: controls left, preview sticky right
- [x] <768px: preview appears above controls, single column
- [x] No horizontal scroll at 360px
Verification: manual at 1440/768/360px. Deps: A1.
Files: `templates/index.html`, `static/styles.css`. Scope: S.

**Task A3: Run gallery**
Description: `GET /api/runs` returns runs (id, timestamp, prefix, region,
artifacts, thumbnail). UI shows a grid; clicking a run loads its preview and a
"re-run with these settings" action; "load config into form" reloads inputs.
Acceptance:
- [x] Gallery lists all runs newest-first with thumbnails
- [x] Selecting a run shows its map and its config
- [ ] "Re-run" starts a job with that run's settings (deferred to Phase C/D)
Verification: manual against `../output`; unit test for the runs dict.
Deps: A1. Files: `app.py`, `static/app.js`, `static/styles.css`. Scope: M.

**Task A4: Remove dead templates / consolidate**
Description: Delete `templates/base.html` (broken, unused). Repurpose
`generating.html` into the progress component used by B2, or delete if B2 renders
its own. Ensure only reachable templates remain.
Acceptance:
- [x] No template references a non-existent asset
- [x] Every file in `templates/` is reachable from a route
Verification: `grep render_template`; manual page load (no 404s in console).
Deps: none. Files: `templates/`. Scope: XS.
NOTE: `generating.html` was kept (retired) because B2 renders its own inline
progress panel; it can be deleted when convenient.

**Task A5: Route UI paths through `map_core/paths.py`**
Description: Replace `get_available_pbf_files`' CWD-resolution and the
`../osm-data/` template fallback; resolve `expected_out_dir` via
`paths.outputs_dir()`; show the resolved data root in the UI footer.
Acceptance:
- [x] UI works when Flask's CWD is not the repo root
- [x] No `../osm-data/` string remains in templates/app.py
- [x] Footer shows data root + free space
Verification: run `app.py` from `/tmp`; confirm scan + output resolve.
Deps: `tasks/plan_data_layout.md` Phase A. Files: `app.py`, `templates/index.html`. Scope: S.

### Checkpoint A
- [ ] Open page → region intact → Generate → map appears in the pane
- [ ] Run gallery lists and re-runs
- [ ] `pytest tests/ -q` green
- [ ] Review with the user before Phase B

### Phase B — Async generation & live progress *(depends on v2 Tasks 2.4–2.5)*

**Task B1: API** — `POST /api/generate` → `{job_id}`; `GET /api/jobs/<id>`;
`GET /api/jobs/<id>/events` (SSE, text/event-stream); `POST /api/jobs/<id>/cancel`.
Acceptance: SSE streams log lines + a terminal `done`/`error` event; a second
concurrent job is allowed or explicitly rejected. Verification: `curl -N` the SSE
endpoint. Deps: 2.4. Files: `app.py`, `map_core/jobs.py`. Scope: M.
NOTE: implemented as a generic thread/subprocess job runner in `map_core/jobs.py`
(it still shells out to `main.py`; swapping in `generate_map()` later is a drop-in).

**Task B2: Progress UI** — submit → job card with live log, progress bar
(indeterminate is fine), cancel; on `done`, load the result into the pane
without reload. Acceptance: no full-page reload; cancel stops the process; errors
render inline with a link to the run log. Verification: manual + devtools network.
Deps: B1, A1. Files: `static/app.js`, `templates/index.html`, `static/styles.css`. Scope: M.

**Task B3: Inline validation & dirty-state** — validate on the client (coords,
distance range, local-mode PBF required) before submit; show field errors; status
chip reflects editing/saving/ready. Acceptance: no submit-needed to see
coordinate/distance errors; status chip is truthful. Verification: manual.
Deps: none. Files: `static/app.js`, `static/styles.css`. Scope: S.

### Checkpoint B
- [ ] Generate no longer blocks; progress visible; result appears in place
- [ ] Cancel works; errors are actionable
- [ ] `app.py` no longer shells out for generation (uses `generate_map()`)

### Phase C — Data & area selection *(depends on data-layout Phase C endpoints)*

**Task C1: Data panel** — read-only data root, free space, and counts of
sources/regions/runs from `/api/env`, with a "clear cache" action.
Acceptance: shows resolved root + the three counts; clears `cache/` after confirm.
Verification: toggle `MAPGEN_DATA_DIR`, reload. Deps: path endpoints.
Files: `templates/index.html`, `static/app.js`. Scope: S.

**Task C2: Region picker + info panel** — dropdown from `/api/regions`; selecting
sets `pbf_path` and fills centre/distance from the bbox; beneath it show size,
bbox, per-layer feature counts and a thumbnail.
Acceptance: choosing *Cork* fills all fields; counts shown before generating.
Verification: manual. Deps: C1. Files: `app.py`, `templates/index.html`, `static/app.js`. Scope: M.

**Task C3: New region dialog** — pick a source + centre + half-width (+ name) →
`POST /api/regions/extract`; blocks with a spinner until 2.5 lands, then becomes a job.
Acceptance: extracting a new city produces a catalogued region that appears in C2.
Verification: extract a test city, list it. Deps: C2. Files: `app.py`, `static/app.js`. Scope: M.

**Task C4: Map picker (Leaflet via CDN, graceful fallback)** — click to set the
centre, drag a radius/rectangle to set distance, draw the region bbox; syncs with
the lat/lon/distance inputs both ways. Offline: falls back to a plain
coordinates + distance form with an explanatory note.
Acceptance:
- [x] pan/zoom/click updates inputs
- [x] inputs update the map
- [x] works offline with reduced features
- [ ] drag-a-rectangle mode (deferred; radius is set with a draggable circle
  handle + slider instead)
Verification: manual online + block the CDN. Deps: C2.
Files: `templates/index.html`, `static/app.js`, `static/styles.css`. Scope: L.

### Checkpoint C
- [x] A new user goes from opening the page to a finished Cork map without typing coordinates or a path
- [x] Region info (size/bbox/counts) is visible before generating
- [x] Works on the Pi with no build step and no network (degraded picker)

### Phase D — Creative richness

**Task D0: Per-class street styling (done)** — streets gained a `style_mode`
(uniform / by-class), variable per-class line widths as a UI toggle,
palette-based colour themes with per-class colour overrides, and per-category
width overrides. Modeled on the buildings auto-styling machinery; colors
resolved in `map_core/core/street_styling.py` so CLI and studio agree. See
commit `2c779a8`. Remaining: expose the preset gallery / masks / wall art.

**Task D1: Palette gallery + editor (done)** — a "Palettes" studio section with a
gallery of cards (swatch strip, in-use badge), inline create form, and per-card
Apply to Streets (switches streets to by-class + re-fills class colours from the
theme) / Apply to Buildings, plus Edit / Rename / Delete. New palettes appear in
every palette select immediately and persist in `config/palettes/palettes.json`;
palette writes preserve the file's curated metadata and compact layout
byte-for-byte. Sample-map live preview is deferred to D3's cheap-preview task.
Files: `static/app.js`, `templates/index.html`, `static/styles.css`,
`map_core/core/palettes.py`. Scope: M.

**Task D2: Preset gallery (done)** — replaced the dropdown + `alert()`/reload with a
card gallery: swatch strip (derived server-side from the preset style, no map
thumbnail needed), enabled-layers chips, region/distance and palette badges;
"Load" applies in place by re-hydrating every form control from the parsed
style (`populateFormFromStyle`) with no reload; Delete removes without leaving
the page; "Surprise me" picks a random palette + random street widths + a
random crop mask. `GET /api/presets` returns per-card metadata;
`PUT /api/presets/<name>` returns the normalized style for hydration; new
`GET /api/style`. Commit `71a59ad`. Files: `static/app.js`, `templates/index.html`,
`app.py`. Scope: M.

**Task D3: Layer quick-toggles + live preview (done)** — a compact legend of layer
chips (on/off, colour dot) over the Result panel; toggling syncs the form's
enabled checkboxes and re-renders a debounced cheap preview that reflects the
full current style (including masks/titles). Implemented as `POST /api/preview`:
re-uses `_update_style_from_form` + `fetch_layer` with an in-process feature
cache (first render a few seconds, subsequent toggles near-instant), a small
Agg figure, the point-feature filter, and `apply_creative_output`. Streets are
drawn uniformly so the preview stays cheap. Commit `71a59ad`. Files:
`static/app.js`, `templates/index.html`, `app.py`. Scope: M.

**Task D4: Crop mask / frame / title (done)** — creative output controls exposed in a
new "Creative Output" studio section and implemented in the render core
(`map_core/core/plot.py::apply_creative_output`): `output.mask` =
circle|hexagon|rounded (clip-path applied to every artist), `mask_options`
(inset, corner radius), `output.frame` (width/color stroked around the mask
edge), and `output.title`/`subtitle` with `{coords}`/`{region}` placeholders.
Schema (`OutputConfig`) extended accordingly so normalization round-trips;
applied in `main.py`'s combined save and in the live preview. Verified
end-to-end on Cork. Commit `71a59ad`. Scope: M.

**Task D5: Integrate wall art** — "Open in wall art" from a finished run, passing
the PNG/SVG into the compositor; or fold `wall_art.html` into the studio as a
final step. Deps: A1. Files: `templates/wall_art.html`, `static/app.js`. Scope: M.

### Checkpoint D
- [x] Palettes and presets are first-class (create, preview, apply, delete) without leaving the page
- [x] Layer toggles update a preview in a couple of seconds
- [ ] A finished map can be turned into wall art without a file round-trip (D5 - deferred)

### Phase E — Polish

**Task E1: Dark mode + accessibility** — dark tokens, `prefers-color-scheme`,
labelled controls, focus rings, keyboard navigation for sections/toggles.
Acceptance: axe/lighthouse has no critical issues; tab-through reaches every control.
Files: `static/styles.css`, `templates/index.html`. Scope: M.

**Task E2: Empty states & first-run** — no sources → a Geofabrik download hint;
no regions → "bootstrap cities"; no runs → a friendly prompt. Deps: C1.
Files: `templates/index.html`, `static/app.js`. Scope: S.

**Task E3: Frontend tests** — a small Playwright (or requests+BeautifulSoup)
smoke suite: page loads, region select fills fields, generate returns a job,
result pane renders. Files: `tests/integration/`. Scope: M.

**Task E4: Perf** — debounce preview renders, cache thumbnails, lazy-load
palettes, virtualize long run lists. Deps: D3, A3. Scope: S.

**Task E5: Tooltips & contextual help (done)** — `data-tooltip` CSS system on all
advanced settings (Simplify Tolerance, Min Size, Hatch Pattern, Variable widths,
Color modes, Palette gallery), water layer section. Commit `adad89b`. Scope: S.

**Task E6: Output format controls (done)** — "Output Formats" section in Settings:
Generate PNG (DPI, scale, suffix, background, source), SVG optimization
(Scour/None, CSS extraction, merge rules, suffix/dir), Shape simplification
(tolerance, topology, min segment, low-poly mode + grid snap). Commit `adad89b`.
Scope: M.

**Task E7: SVG shape simplification / low-poly (done)** — Douglas-Peucker path
reduction via `svgpathtools`; `svg_simplify_enabled`, `svg_simplify_tolerance`,
`svg_low_poly_mode` + `svg_low_poly_grid` for faceted artistic style. Runs on
combined + per-layer SVGs. Commit `adad89b`. Scope: M.

**Task E8: Water layer artifact fixes (done)** — geometry repair via
`shapely.make_valid()` (no `buffer(0)` wedges), river centerline dedup (bbox
overlap filter), 1% bbox buffer for water fetches. Commit `771da5b`. Scope: S.

**Task E9: Job tracking robustness (done)** — SSE + 2s polling fallback + 5s
safety net + manual "Refresh results" button in job panel. Commit `771da5b`.
Scope: S.

### Checkpoint E
- [ ] Dark mode, keyboard, and empty states work
- [ ] Smoke suite green
- [ ] Ready for review

---

## 5. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Job runner (v2 2.4/2.5) not built yet — B blocked | High | Ship Phase A now; stub `/api/generate` with a background thread if needed |
| matplotlib/SSE threading issues under Flask dev server | Med | Headless `Agg` backend; run jobs in a worker thread, never touch the request context |
| Leaflet CDN unavailable offline | Med | Coordinate/region form is the primary path; map is an enhancement |
| Rewriting the form breaks server-side parsing | Med | Keep Jinja as source of truth; hydrate from one `/api/state`; add smoke tests first |
| Cheap preview vs full render diverge visually | Med | Preview uses the same `generate_map()` code path with fewer layers |
| Large SVG inlined into the page is slow | Med | Serve via `/output/...` and use `<img>`/`<object>` + thumbnails; inline only the small ones |

## 6. Open questions

1. **Priority:** ship the results view + gallery first (Phase A), or go straight
   for async + progress (Phase B)?
2. **Preview:** is a cheap *streets-only* live preview worth the complexity, or
   render on demand only?
3. **Map picker:** inline Leaflet (CDN, needs network) vs. an offline
   wireframe/bbox editor vs. skip it and rely on the region dropdown?
4. **Shell:** incremental enhancement of the current form, or a clean single-page
   studio rewrite (more churn, better result)?
5. **Scope of Phase D:** which of palettes / presets / masks / wall-art matter most?
