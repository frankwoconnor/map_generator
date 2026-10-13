import datetime
import glob
import json
import mimetypes
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from flask import (Flask, Response, flash, get_flashed_messages, jsonify,
                   redirect, render_template, request, send_file,
                   send_from_directory, stream_with_context, url_for)
from flask_cors import CORS

import map_core.core.config as cfg
from config.manager import feature_visibility_manager
from config.schemas.feature_config import (FeatureCategory, FeatureLayerConfig,
                                           FeatureVisibilityManager)
from map_core import paths
from map_core.jobs import TERMINAL_STATUSES, JobManager
from map_core.core.geocode import geocode_to_point
from map_core.core.regions import RegionCatalog, pbf_info, slugify
from map_core.runs import discover_runs
from map_core.core.palettes import (add_palette, get_palette,
                                    get_palette_names, load_palettes,
                                    remove_palette, set_palettes_file_path,
                                    validate_palette_colors)
from map_core.core.street_styling import (STREET_CATEGORIES,
                                          street_category_colors)
from map_core.services.style_service import (load_style,
                                             save_style_json)
from map_core.services.style_form_service import (
    _street_class_table,
    _update_style_from_form,
    _validate_location_inputs,
)
# Backwards-compatible alias for in-file call sites.
_save_style_json = save_style_json

# Initialize palettes with the correct path
palettes_dir = Path(__file__).parent / "config" / "palettes"
set_palettes_file_path(str(palettes_dir / "palettes.json"))

# Make sure the managed data/cache/output directories exist on startup.
try:
    paths.ensure_dirs()
except Exception as _e:  # noqa: BLE001
    print(f"[app] Warning: could not create data directories: {_e}")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(paths.outputs_dir())
app.secret_key = os.urandom(24)

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

STYLE_FILE = "style.json"
PRESETS_FILE = "config/presets.json"
MAIN_SCRIPT = "main.py"
DEFAULT_PBF_FOLDER = str(paths.sources_dir())

# Background job runner for async generation (see map_core/jobs.py).
job_manager = JobManager()


# --- Preset Helper Functions ---


def _load_presets() -> Dict[str, Any]:
    """Load presets from presets.json."""
    try:
        with open(PRESETS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_presets(presets: Dict[str, Any]) -> None:
    """Save presets to presets.json."""
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def _preset_card_meta(name: str, preset: Any) -> Dict[str, Any]:
    """Derive gallery-card metadata from a preset style (no thumbnail render).

    Collects the colour swatches, enabled layers, referenced palettes and the
    region/query so the UI can render a small preview card without a map.
    """
    preset = preset if isinstance(preset, dict) else {}
    layers = preset.get("layers", {})
    safe_layers = {
        k: (v if isinstance(v, dict) else {})
        for k, v in layers.items()
        if isinstance(layers, dict)
    }
    enabled = [k for k in ("streets", "buildings", "water", "green")
               if safe_layers.get(k, {}).get("enabled")]

    streets = safe_layers.get("streets", {})
    buildings = safe_layers.get("buildings", {})

    palettes = load_palettes() or {}
    used_palettes: List[str] = []
    colors: List[str] = []

    def _note_palette(value: Any) -> None:
        if value and value not in used_palettes:
            used_palettes.append(value)

    _note_palette(streets.get("street_palette"))
    _note_palette(buildings.get("auto_size_palette"))
    _note_palette(buildings.get("auto_distance_palette"))

    # Swatches: by-class street colors, building palette, water/green fills.
    class_colors = streets.get("class_colors") or {}
    if isinstance(class_colors, dict) and class_colors:
        colors = [c for c in class_colors.values() if c]
    else:
        street_color = streets.get("edgecolor")
        if street_color:
            colors.append(street_color)

    bmode = buildings.get("auto_style_mode", "manual")
    bpalette = buildings.get("auto_size_palette") or buildings.get(
        "auto_distance_palette"
    )
    if bmode != "manual" and bpalette and bpalette in palettes:
        for c in palettes[bpalette]:
            if c not in colors:
                colors.append(c)
    elif bmode == "manual":
        manual = buildings.get("manual_color_settings") or {}
        fc = manual.get("facecolor")
        if fc and fc not in colors:
            colors.append(fc)

    for lname in ("water", "green"):
        if not safe_layers.get(lname, {}).get("enabled"):
            continue
        fc = safe_layers.get(lname, {}).get("facecolor")
        if fc and fc not in colors:
            colors.append(fc)

    location = preset.get("location", {})
    return {
        "name": name,
        "enabled": enabled,
        "colors": [c for c in colors if c][:14],
        "palettes": used_palettes,
        "region": (location.get("query") or "").strip(),
        "distance": location.get("distance"),
        "street_mode": streets.get("style_mode", "uniform"),
        "buildings_mode": bmode,
    }


def get_available_pbf_files(pbf_folder: str) -> List[Dict[str, str]]:
    """Get list of available PBF files in the specified folder, avoiding duplicates.

    Args:
        pbf_folder: Path to the folder containing PBF files

    Returns:
        List of dictionaries with 'name' and 'path' keys for each PBF file
    """
    pbf_files = []
    found_paths = set()

    if not pbf_folder:
        return pbf_files

    try:
        folder = paths.resolve_data_path(pbf_folder)

        if folder.exists() and folder.is_dir():
            # Find all .pbf and .osm.pbf files
            for pattern in ["*.pbf", "*.osm.pbf"]:
                for filepath in folder.glob(pattern):
                    abs_path = str(filepath.resolve())
                    if abs_path not in found_paths:
                        found_paths.add(abs_path)
                        filename = filepath.name
                        # Store the absolute path; main.py resolves it via
                        # map_core.paths, and it may live outside the CWD.
                        pbf_files.append({"name": filename, "path": abs_path})
    except Exception as e:
        print(f"Error scanning PBF folder '{pbf_folder}': {e}")

    # Sort by filename for consistent ordering
    pbf_files.sort(key=lambda x: x["name"].lower())
    return pbf_files


def _index_context(style: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """Template context for the studio page - one source of truth for all renders."""
    runs = discover_runs(limit=12)
    context: Dict[str, Any] = {
        "style": style,
        "palettes": load_palettes(),
        "pbf_files": get_available_pbf_files(
            style.get("location", {}).get("pbf_folder", DEFAULT_PBF_FOLDER)
        ),
        "feature_manager": feature_visibility_manager,
        "street_classes": _street_class_table(style),
        "default_pbf_folder": DEFAULT_PBF_FOLDER,
        "data_root": str(paths.data_root()),
        "runs": runs,
        "latest_run": runs[0] if runs else None,
        "regions": _regions_for_form(),
        "layer_filters": {},
    }
    context.update(extra)
    return context


def _dir_stats(d: Path) -> Dict[str, Any]:
    """File count / byte total for one data directory (recursive)."""
    if not d.is_dir():
        return {"files": 0, "size_bytes": 0}
    files = 0
    total = 0
    for root, _dirs, names in os.walk(d):
        for n in names:
            p = Path(root) / n
            try:
                total += p.stat().st_size
                files += 1
            except OSError:
                pass
    return {"files": files, "size_bytes": total}


def _rel_to_repo(p: Path) -> str:
    try:
        return os.path.relpath(p, paths.REPO_ROOT)
    except ValueError:
        return str(p)


def _regions_for_form() -> List[Dict[str, Any]]:
    """Catalog regions, each annotated with form-ready values for quick pick."""
    cat = RegionCatalog()
    out: List[Dict[str, Any]] = []
    for r in cat.list():
        if r.status(cat.dir) != "ok":
            continue
        item = r.to_dict()
        item["form_path"] = _rel_to_repo(r.resolved_path(cat.dir))
        if r.bbox:
            w, s, e, n = r.bbox
            item["centre"] = [round((s + n) / 2, 5), round((w + e) / 2, 5)]
            # Half the bbox height in metres (1 deg lat ~= 111320 m).
            item["radius_m"] = int(round((n - s) / 2.0 * 111320.0))
        out.append(item)
    out.sort(key=lambda r: r["name"].lower())
    return out


def _new_run_prefix(style: Dict[str, Any]) -> str:
    """Build the timestamped run identifier, validating the prefix."""
    filename_prefix = style.get("output", {}).get("filename_prefix") or "map"
    if not re.match(r"^[a-zA-Z0-9_]+$", filename_prefix):
        raise ValueError(
            "Invalid filename_prefix: must contain only letters, numbers and underscores"
        )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{filename_prefix}_{timestamp}"


def _submit_generation(style: Dict[str, Any], prefix: str):
    """Start a background generation job and return it (non-blocking)."""
    out = style.get("output", {})
    out_dir = paths.resolve_output_dir(out.get("output_directory", "../output")) / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / f"{prefix}_server.log")
    cmd = job_manager.generation_command(prefix)
    return job_manager.submit(
        cmd,
        cwd=str(paths.REPO_ROOT),
        title=prefix,
        log_path=log_path,
        out_dir=str(out_dir),
    )


@app.route("/scan-pbf-folder", methods=["POST"])
def scan_pbf_folder():
    """API endpoint to scan a folder for PBF files."""
    data = request.get_json()
    folder_path = data.get("folder_path") if data else None

    if not folder_path:
        return jsonify({"error": "Folder path is required."}), 400

    try:
        pbf_files = get_available_pbf_files(folder_path)
        return jsonify({"pbf_files": pbf_files})
    except Exception as e:
        return jsonify({"error": f"Failed to scan folder: {str(e)}"}), 500


@app.route("/api/palettes", methods=["GET"])
def api_get_palettes():
    """Get all available palettes."""
    try:
        palettes = load_palettes()
        return jsonify(
            {"success": True, "palettes": palettes, "names": get_palette_names()}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/palettes/<palette_name>", methods=["GET"])
def api_get_palette(palette_name):
    """Get a specific palette by name."""
    try:
        palette = get_palette(palette_name)
        if palette is None:
            return jsonify({"success": False, "error": "Palette not found"}), 404
        return jsonify({"success": True, "name": palette_name, "colors": palette})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/palettes", methods=["POST"])
def api_create_palette():
    """Create or update a palette."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        name = data.get("name")
        colors = data.get("colors")

        if not name or not colors:
            return (
                jsonify({"success": False, "error": "Name and colors are required"}),
                400,
            )

        if not validate_palette_colors(colors):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid color format. Colors must be hex strings like #RRGGBB",
                    }
                ),
                400,
            )

        if add_palette(name, colors):
            return jsonify(
                {
                    "success": True,
                    "message": f'Palette "{name}" created/updated successfully',
                    "palette": {"name": name, "colors": colors},
                }
            )
        else:
            return jsonify({"success": False, "error": "Failed to save palette"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/palettes/<palette_name>", methods=["DELETE"])
def api_delete_palette(palette_name):
    """Delete a palette."""
    try:
        if remove_palette(palette_name):
            return jsonify(
                {
                    "success": True,
                    "message": f'Palette "{palette_name}" deleted successfully',
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Palette not found or could not be deleted",
                    }
                ),
                404,
            )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/presets", methods=["GET"])
def api_get_presets():
    """Get all presets with gallery-card metadata (swatches, layers, region)."""
    presets = _load_presets()
    cards = [
        _preset_card_meta(name, style_data)
        for name, style_data in presets.items()
    ]
    cards.sort(key=lambda c: c["name"].lower())
    return jsonify({"success": True, "presets": cards})


@app.route("/api/style", methods=["GET"])
def api_get_style():
    """Return the parsed, normalized current style (used by in-place hydration)."""
    return jsonify({"success": True, "style": load_style()})


# --- Cheap live preview (D3) ------------------------------------------------


_PREVIEW_CACHE: Dict[tuple, Any] = {}
_PREVIEW_LAYER_ORDER = ["water", "green", "buildings", "streets"]


def _preview_layer_tags() -> Dict[str, Any]:
    """Load the per-layer OSM tag config used for preview fetches."""
    path = Path(__file__).parent / "config" / "layers" / "layer_tags.json"
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return {}


def _preview_cache_key(
    pbf: Optional[str], layer: str, query: str, dist: float, tags: Dict[str, Any]
) -> tuple:
    """Stable in-process cache key for one layer's preview fetch."""
    return (
        pbf,
        layer,
        query,
        round(float(dist or 0)),
        json.dumps(tags, sort_keys=True),
    )


def render_preview_svg(style: Dict[str, Any]) -> Dict[str, Any]:
    """Render a small, fast SVG preview of the given style.

    Layer features are fetched once and cached in-process (keyed by
    pbf/query/dist/tags) so toggling layers afterwards is near-instant.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from map_core.core.buildings_styling import style_buildings
    from map_core.core.fetch import fetch_layer
    from map_core.core.layer_plot import (plot_green_layer,
                                          plot_streets_layer, plot_water_layer)
    from map_core.core.util import has_data as core_has_data
    from map_core.core.util import drop_point_features

    layers = style.get("layers", {})
    loc = style.get("location", {})
    query = (loc.get("query") or "").strip()
    dist: float = loc.get("distance") or 1500.0
    pbf = loc.get("pbf_path")
    if not query:
        return {"error": "Set a location first."}

    output = style.get("output", {})
    bg = output.get("background_color", "#ffffff")
    if output.get("transparent_background"):
        bg = "none"

    enabled = [
        n for n in _PREVIEW_LAYER_ORDER if layers.get(n, {}).get("enabled")
    ]
    if not enabled:
        return {"error": "Enable at least one layer."}

    tags_all = _preview_layer_tags()
    fig = plt.figure(figsize=(6.0, 6.0), dpi=72, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    for layer_name in _PREVIEW_LAYER_ORDER:
        if not layers.get(layer_name, {}).get("enabled"):
            continue
        cfg_l = layers[layer_name]
        tags = (tags_all.get(layer_name) or {}).get("tags") or {}
        key = _preview_cache_key(pbf, layer_name, query, dist, tags)
        data = _PREVIEW_CACHE.get(key)
        if data is None:
            try:
                data = fetch_layer(
                    query=query,
                    dist=dist,
                    tags=tags,
                    is_graph=(layer_name == "streets"),
                    simplify_tolerance=cfg_l.get("simplify_tolerance"),
                    min_size_threshold=cfg_l.get("min_size_threshold", 0.0),
                    layer_name_for_debug=f"preview_{layer_name}",
                    pbf_path=pbf or None,
                )
            except Exception as e:  # noqa: BLE001 - preview is best-effort
                print(f"[preview] {layer_name} fetch failed: {e}")
                continue
            if not core_has_data(data):
                continue
            _PREVIEW_CACHE[key] = data
            if len(_PREVIEW_CACHE) > 40:  # simple upper bound, drop oldest
                _PREVIEW_CACHE.pop(next(iter(_PREVIEW_CACHE)))

        data = drop_point_features(data, layer_name)
        if not core_has_data(data):
            continue

        try:
            if layer_name == "streets":
                # Keep the preview cheap: uniform colour, no per-class widths.
                quick = dict(cfg_l)
                quick["style_mode"] = "uniform"
                quick["variable_widths"] = False
                plot_streets_layer(ax, data, quick)
            elif layer_name == "water":
                plot_water_layer(ax, data, cfg_l)
            elif layer_name == "green":
                plot_green_layer(ax, data, cfg_l)
            elif layer_name == "buildings":
                style_buildings(ax, data, cfg_l, query)
        except Exception as e:  # noqa: BLE001 - preview is best-effort
            print(f"[preview] {layer_name} plot failed: {e}")

    # Reflect the crop mask / frame / title in the preview so what you see
    # matches what Generate produces.
    try:
        from map_core.core.plot import apply_creative_output

        apply_creative_output(fig, ax, output, query)
    except Exception as e:  # noqa: BLE001 - best-effort
        print(f"[preview] creative output failed: {e}")

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return {"svg": buf.getvalue()}


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Render a cheap preview SVG from the current form state."""
    try:
        style = load_style()
        _update_style_from_form(style, request.form, warn=flash)
    except Exception as e:  # noqa: BLE001
        return (
            jsonify({"success": False, "error": f"Could not parse form: {e}"}),
            400,
        )
    result = render_preview_svg(style)
    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400
    return jsonify({"success": True, "svg": result["svg"]})


@app.route("/api/presets", methods=["POST"])
def api_save_preset():
    """Save a new preset."""
    data = request.get_json()
    if not data or "name" not in data or "style" not in data:
        return (
            jsonify({"success": False, "error": "Preset name and style are required."}),
            400,
        )

    preset_name = data["name"]
    style_data = data["style"]

    presets = _load_presets()
    presets[preset_name] = style_data
    _save_presets(presets)

    return jsonify(
        {"success": True, "message": f'Preset "{preset_name}" saved successfully.'}
    )


@app.route("/api/presets/<preset_name>", methods=["GET"])
def api_get_preset(preset_name):
    """Get a specific preset by name."""
    presets = _load_presets()
    if preset_name not in presets:
        return jsonify({"success": False, "error": "Preset not found."}), 404
    return jsonify({"success": True, "preset": presets[preset_name]})


@app.route("/api/presets/<preset_name>", methods=["PUT"])
def api_load_preset(preset_name):
    """Load a preset and save it as the current style.

    Also returns the parsed style so the client can re-hydrate the form in
    place without a page reload.
    """
    presets = _load_presets()
    if preset_name not in presets:
        return jsonify({"success": False, "error": "Preset not found."}), 404

    raw_style = presets[preset_name]
    _save_style_json(raw_style)
    try:
        parsed = load_style()
    except Exception as e:  # noqa: BLE001 - best-effort parse for hydration
        parsed = raw_style
        print(f"[preset] Warning: could not normalize '{preset_name}' style: {e}")
    return jsonify(
        {
            "success": True,
            "message": f'Preset "{preset_name}" loaded successfully.',
            "style": parsed,
        }
    )


@app.route("/api/presets/<preset_name>", methods=["DELETE"])
def api_delete_preset(preset_name):
    """Delete a preset by name."""
    presets = _load_presets()
    if preset_name not in presets:
        return jsonify({"success": False, "error": "Preset not found."}), 404

    del presets[preset_name]
    _save_presets(presets)

    return jsonify(
        {"success": True, "message": f'Preset "{preset_name}" deleted successfully.'}
    )


@app.route("/api/runs", methods=["GET"])
def api_get_runs():
    """List generated runs (newest first) for the gallery."""
    try:
        limit = request.args.get("limit", type=int)
        return jsonify({"success": True, "runs": discover_runs(limit=limit)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/env")
def api_env():
    """Read-only summary of the resolved data layout and disk health."""
    try:
        env = {
            "data_root": str(paths.data_root()),
            "repo_root": str(paths.REPO_ROOT),
            "free_space_bytes": paths.free_space_bytes(),
            "osmium_available": paths.osmium_available(),
            "dirs": {
                "sources": _dir_stats(paths.sources_dir()),
                "regions": _dir_stats(paths.regions_dir()),
                "clips": _dir_stats(paths.clips_dir()),
                "features": _dir_stats(paths.features_dir()),
                "output": _dir_stats(paths.outputs_dir()),
            },
        }
        return jsonify({"success": True, "env": env})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sources")
def api_sources():
    """Raw source PBF files available for extraction."""
    d = paths.sources_dir()
    sources = []
    if d.is_dir():
        suffixes = (".osm.pbf", ".osm", ".pbf", ".osm.bz2")
        for p in sorted(d.iterdir()):
            if p.is_file() and p.name.lower().endswith(suffixes):
                size = p.stat().st_size
                sources.append(
                    {
                        "name": p.name,
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 1),
                        "path": str(p),
                    }
                )
    return jsonify({"success": True, "sources": sources})


@app.route("/api/regions/extract", methods=["POST"])
def api_regions_extract():
    """Clip a new region from a source PBF as a background job."""
    source = (request.form.get("source") or "").strip()
    name = (request.form.get("name") or "").strip()
    strategy = (request.form.get("strategy") or "smart").strip() or "smart"

    errors: List[str] = []
    lat = lon = dist_m = None
    try:
        lat = float(request.form.get("lat"))
        lon = float(request.form.get("lon"))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError
    except (TypeError, ValueError):
        errors.append("Latitude and longitude must be numbers (lat between -90..90, lon -180..180).")
    try:
        dist_m = float(request.form.get("dist_m"))
        if not (50 <= dist_m <= 100_000):
            raise ValueError
    except (TypeError, ValueError):
        errors.append("Half-width (metres) must be between 50 and 100000.")
    if not source:
        errors.append("A source PBF is required.")
    if not name or not re.match(r"^[a-zA-Z0-9_\- ]+$", name):
        errors.append("A region name is required (letters, numbers, spaces, dashes, underscores).")
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    resolved_source = paths.resolve_data_path(source)
    if not resolved_source.is_file():
        return jsonify(
            {"success": False, "errors": [f"Source PBF not found: {resolved_source}"]}
        ), 400

    slug = slugify(name)
    cat = RegionCatalog()
    if cat.get(slug) is not None:
        return jsonify(
            {
                "success": False,
                "errors": [
                    f"A region named '{slug}' already exists. Pick a different name or "
                    "remove the old region first."
                ],
            }
        ), 400

    try:
        cmd = job_manager.extract_region_command(
            name=name,
            source=str(resolved_source),
            lat=lat,
            lon=lon,
            dist_m=dist_m,
            margin=0.0,
            strategy=strategy,
        )
        job = job_manager.submit(
            cmd,
            cwd=str(paths.REPO_ROOT),
            title=f"Extract {name}",
            out_dir=str(paths.regions_dir()),
        )
    except Exception as e:  # noqa: BLE001 - surface start-up failure to the client
        return jsonify({"success": False, "errors": [str(e)]}), 500
    return jsonify({"success": True, "job_id": job.id, "slug": slug, "status": job.status})


@app.route("/api/regions")
def api_regions():
    """Region catalog entries annotated for the quick-pick UI."""
    try:
        cat = RegionCatalog()
        regions = []
        for r in cat.list():
            item = r.to_dict()
            resolved = r.resolved_path(cat.dir)
            item["status"] = "ok" if r.status(cat.dir) == "ok" else "missing"
            item["size_mb"] = (
                round(r.size_bytes / (1024 * 1024), 1) if r.size_bytes else None
            )
            item["form_path"] = _rel_to_repo(resolved)
            regions.append(item)
        regions.sort(key=lambda r: r["name"].lower())
        return jsonify({"success": True, "regions": regions})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pbf/info")
def api_pbf_info():
    """osmium fileinfo for any resolved data path (size/bbox/object counts)."""
    p = request.args.get("path", "").strip()
    if not p:
        return jsonify({"success": False, "error": "Missing 'path' query parameter"}), 400
    resolved = paths.resolve_data_path(p)
    if not resolved.is_file():
        return jsonify({"success": False, "error": f"File not found: {resolved}"}), 404
    try:
        info = pbf_info(resolved, extended=True)
        return jsonify({"success": True, "info": info.to_dict()})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Start a background generation from the studio form; never blocks."""
    style = _update_style_from_form(load_style(), request.form, warn=flash)
    style = cfg.normalize_style(style)
    validation_error = cfg.validate_style(style)
    if validation_error:
        flash(f"Schema validation warning: {validation_error}", category="warning")

    input_errors = _validate_location_inputs(style)
    if input_errors:
        return jsonify({"success": False, "errors": input_errors}), 400

    try:
        prefix = _new_run_prefix(style)
        _save_style_json(style)
        job = _submit_generation(style, prefix)
    except Exception as e:  # noqa: BLE001 - report start-up failure to the client
        return jsonify({"success": False, "errors": [str(e)]}), 500
    return jsonify({"success": True, "job_id": job.id, "status": job.status})


@app.route("/api/jobs/<job_id>", methods=["GET"])
def api_job_get(job_id):
    """Snapshot of a job (status + bounded log tail)."""
    job = job_manager.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Unknown job"}), 404
    return jsonify({"success": True, "job": job.snapshot()})


@app.route("/api/jobs/<job_id>/events")
def api_job_events(job_id):
    """Server-sent events: `data:` log lines then a terminal `terminal` event."""
    job = job_manager.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Unknown job"}), 404

    def generate():
        # Already finished before the client connected? Send the verdict at once.
        if job.status in TERMINAL_STATUSES:
            yield (
                f"event: terminal\ndata: "
                f"{json.dumps({'job_id': job_id, 'status': job.status, 'error': job.error})}"
                f"\n\n"
            )
            return
        while True:
            if job.status in TERMINAL_STATUSES:
                break
            ev = job.next_event(timeout=10.0)
            if ev is None:
                continue
            if ev["type"] == "terminal":
                yield (
                    f"event: terminal\ndata: "
                    f"{json.dumps({'job_id': job_id, 'status': ev['status'], 'error': ev.get('error')})}"
                    f"\n\n"
                )
                return
            yield f"data: {json.dumps(ev)}\n\n"
        yield (
            f"event: terminal\ndata: "
            f"{json.dumps({'job_id': job_id, 'status': job.status, 'error': job.error})}"
            f"\n\n"
        )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_job_cancel(job_id):
    """Request cancellation of a running job."""
    if job_manager.cancel(job_id):
        return jsonify({"success": True, "status": "cancelling"})
    return jsonify({"success": False, "error": "No running job with that id"}), 404


@app.route("/wall-art")
def wall_art():
    """Render the wall art generator page."""
    return render_template("wall_art.html")


@app.route("/", methods=["GET", "POST"])
def index():
    style = load_style()
    presets = _load_presets()

    if request.method == "POST" and request.form.get("action") == "generate":
        print("==== Map Generation Request Received ====")
        print(f"HTTP Method: {request.method}")
        print(f"Action: {request.form.get('action')}")
        # Update style.json based on form data

        style = _update_style_from_form(style, request.form, warn=flash)

        # Normalize and validate using core config before persisting/generating
        normalized_style = cfg.normalize_style(style)
        validation_error = cfg.validate_style(normalized_style)
        if validation_error:
            flash(f"Schema validation warning: {validation_error}", category="warning")
        style = normalized_style

        # Quick-win: strict server-side validation
        input_errors = _validate_location_inputs(style)
        if input_errors:
            for e in input_errors:
                flash(e, category="error")
            # Re-render page with errors; do not run generation
            warning_messages = get_flashed_messages(with_categories=True)
            return render_template(
                "index.html",
                warning_messages=warning_messages,
                **_index_context(style),
            )
        # --- Pre-run logging ---
        loc = style.get("location", {})
        pbf_path = loc.get("pbf_path")
        data_source = "LOCAL_PBF" if pbf_path else "REMOTE_OSMNX"
        # Distance is stored in meters; show meters to avoid confusion for <1km
        distance_m = None
        try:
            distance_m = (
                float(loc["distance"]) if loc.get("distance") is not None else None
            )
        except Exception:
            pass
        print("---- Run Parameters ----")
        print(f"Data Source: {data_source}")
        if pbf_path:
            print(f"PBF Path: {pbf_path}")
        print(f"Location Query: {loc.get('query')}")
        print(
            f"Distance (m): {int(distance_m) if distance_m is not None else 'None (admin area)'}"
        )
        # Layers summary
        layers = style.get("layers", {})
        enabled_layers = [
            name for name, cfg_layer in layers.items() if cfg_layer.get("enabled")
        ]
        print(
            f"Enabled Layers: {', '.join(enabled_layers) if enabled_layers else 'None'}"
        )
        # Street filter summary
        # Street filter is now part of the layer, so this log is implicitly covered
        # Output settings
        out = style.get("output", {})
        print(f"Output Dir (base): {out.get('output_directory')}")
        print(f"Filename Prefix: {out.get('filename_prefix')}")
        print(f"Separate Layers: {out.get('separate_layers')}")

        # Generate timestamp for the run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Validate filename_prefix for security (no shell injection)
        filename_prefix = style['output']['filename_prefix']
        if not re.match(r'^[a-zA-Z0-9_]+$', filename_prefix):
            raise ValueError("Invalid filename_prefix: must contain only letters, numbers, and underscores")
        timestamped_run_identifier = f"{filename_prefix}_{timestamp}"
        # Building layer settings are already handled by _update_style_from_form()
        # Remove redundant and conflicting overrides.

        # Processing settings

        # Save the updated style
        _save_style_json(style)
        # Resolve via map_core.paths so it matches where main.py writes.
        expected_out_dir = str(
            paths.resolve_output_dir(out.get("output_directory", "../output"))
            / timestamped_run_identifier
        )
        print(f"Expected Output Folder: {expected_out_dir}")

        print("Starting generation...")
        # Use virtual environment's Python to ensure all dependencies are available
        venv_python = str(Path(__file__).parent / ".venv" / "bin" / "python3")
        cmd = [venv_python, MAIN_SCRIPT, "--prefix", timestamped_run_identifier]
        print(f"Executing command: {cmd}")

        # Ensure output directory exists and set up a per-run server log file
        Path(expected_out_dir).mkdir(parents=True, exist_ok=True)
        log_path = str(
            Path(expected_out_dir) / f"{timestamped_run_identifier}_server.log"
        )
        print(f"Streaming logs to: {log_path}")

        try:
            with open(log_path, "a") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                # Stream stdout in real-time to console and file
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    lf.write(line)
                proc.wait()
                ret = proc.returncode

            print("---- Subprocess Complete ----")
            print(f"Return code: {ret}")
            print(f"Artifacts should be in: {expected_out_dir}")
            print("==== Map Generation Request Finished ====")

            if ret != 0:
                error_message = (
                    f"Map generation failed with exit code {ret}. See log: {log_path}"
                )
                print(f"Error: {error_message}")
                warning_messages = get_flashed_messages(with_categories=True)
                return render_template(
                    "index.html",
                    error_message=error_message,
                    warning_messages=warning_messages,
                    **_index_context(style),
                )

            # Redirect to GET request to display the new map
            return redirect(url_for("index"))
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            print(f"Error: {error_message}")
            warning_messages = get_flashed_messages(with_categories=True)
            return render_template(
                "index.html",
                error_message=error_message,
                warning_messages=warning_messages,
                **_index_context(style),
            )

    else:
        # GET (and non-generate POST): render the studio with the current style
        # and the latest run. discover_runs() is the single source of truth.
        style = load_style()
        warning_messages = get_flashed_messages(with_categories=True)
        return render_template(
            "index.html",
            warning_messages=warning_messages,
            **_index_context(style),
        )


@app.route("/output/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "5000")))
