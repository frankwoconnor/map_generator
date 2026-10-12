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


# --- Helper Functions ---


def load_style() -> Dict[str, Any]:
    """
    Load `style.json`, ensuring default values are present for missing keys.

    Returns:
        dict: The normalized style configuration dictionary suitable for the UI/backend.
    """
    raw = cfg.load_style(STYLE_FILE) or {}
    style_data = cfg.normalize_style(raw)

    layers_settings = style_data.setdefault("layers", {})
    # Ensure generic layers exist with sensible defaults
    layers_settings.setdefault(
        "streets",
        {
            "enabled": True,
            "edgecolor": "#000000",
            "linewidth": 0.5,
            "alpha": 1.0,
            "hatch": None,
            "zorder": 3,
            "simplify_tolerance": 0.0,
            "min_size_threshold": 0.0,
        },
    )
    layers_settings.setdefault(
        "water",
        {
            "enabled": True,
            "facecolor": "#a6cee3",
            "edgecolor": "#a6cee3",
            "linewidth": 0.0,
            "alpha": 1.0,
            "hatch": None,
            "zorder": 1,
            "simplify_tolerance": 0.0,
            "min_size_threshold": 0.0,
        },
    )
    layers_settings.setdefault(
        "green",
        {
            "enabled": False,
            "facecolor": "#b2df8a",
            "edgecolor": "#33a02c",
            "linewidth": 0.0,
            "alpha": 1.0,
            "hatch": None,
            "zorder": 1,
            "simplify_tolerance": 0.0,
            "min_size_threshold": 0.0,
        },
    )
    buildings_settings = layers_settings.setdefault("buildings", {})

    buildings_settings.setdefault("size_categories", [])
    buildings_settings.setdefault("auto_style_mode", "manual")
    buildings_settings.setdefault("auto_size_palette", "")
    buildings_settings.setdefault("auto_distance_palette", "")
    # Manual color only stores facecolor
    buildings_settings.setdefault("manual_color_settings", {"facecolor": "#000000"})
    # Common building params (apply to all modes)
    buildings_settings.setdefault("edgecolor", "#000000")
    buildings_settings.setdefault("linewidth", 0.5)
    buildings_settings.setdefault("alpha", 1.0)
    buildings_settings.setdefault("hatch", None)
    buildings_settings.setdefault("zorder", 2)

    # Distance is stored and displayed in meters consistently
    location_settings = style_data.setdefault("location", {})
    # Ensure persistent UI-related defaults
    location_settings.setdefault("data_source", "remote")  # 'remote' or 'local'

    # Provide sensible offline defaults if missing: Cork city center and ~1km bbox (1km across)
    # Cork center approx
    default_lat = 51.8986
    default_lon = -8.4756
    # Half-size of 1 km box: 0.5 km
    # Degrees per km approx: lat ~ 1/111, lon ~ 1/(111*cos(lat))
    lat_deg_per_km = 1.0 / 111.0
    from math import cos, radians

    lon_deg_per_km = 1.0 / (111.32 * max(0.0001, cos(radians(default_lat))))
    dlat = 0.5 * lat_deg_per_km
    dlon = 0.5 * lon_deg_per_km
    default_bbox = [
        default_lon - dlon,
        default_lat - dlat,
        default_lon + dlon,
        default_lat + dlat,
    ]

    # If no query present, set coords for Cork center
    if not location_settings.get("query"):
        location_settings["query"] = f"{default_lat} {default_lon}"
    # Do not persist a manual bbox in the new UX; it will be computed from distance
    location_settings["bbox"] = None

    return style_data


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


def _save_style_json(style_data: Dict[str, Any]) -> None:
    """Save the updated style data to `style.json`."""
    with open(STYLE_FILE, "w") as f:
        json.dump(style_data, f, indent=2)


def _update_style_from_form(
    style: Dict[str, Any], form: Mapping[str, str]
) -> Dict[str, Any]:
    """Orchestrate updating the style dictionary from form data.

    Args:
        style: Current style config (will be mutated in-place).
        form: Request form mapping with string keys/values.

    Returns:
        dict: The updated style config (same object as input).
    """
    _update_location_settings(style, form)
    _update_output_settings(style, form)
    _update_generic_layer_settings(style, form, "streets")
    _update_generic_layer_settings(style, form, "water")
    _update_generic_layer_settings(style, form, "green")
    _update_buildings_settings(style, form)
    _update_feature_visibility_from_form(form)  # New call
    # The new generic layer settings function handles filters, so the specific processing one is removed.
    # _update_processing_settings(style, form)
    return style


def _update_location_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update location settings per new Local/Remote model (coords + distance required)."""
    loc = style.setdefault("location", {})

    # Data source selection
    data_source = form.get("location_data_source", loc.get("data_source", "remote"))
    loc["data_source"] = "local" if data_source == "local" else "remote"

    # Combine separate latitude and longitude fields into query string
    lat = (form.get("location_latitude") or "").strip()
    lon = (form.get("location_longitude") or "").strip()
    if lat and lon:
        raw_query = f"{lat} {lon}"
    else:
        # Fallback to old location_query field or existing value
        raw_query = (form.get("location_query") or loc.get("query") or "").strip()

    # Distance in meters (UI and engine use meters consistently)
    distance_m_raw = (form.get("location_distance") or "").strip()
    if distance_m_raw:
        try:
            m_int = max(0, int(float(distance_m_raw)))
            loc["distance"] = float(m_int)
        except ValueError:
            loc["distance"] = None
            flash(
                "Invalid distance. Leave empty or provide an integer number of meters.",
                category="warning",
            )
    else:
        loc["distance"] = None

    # PBF options: only applicable in local mode
    if loc["data_source"] == "local":
        pbf_folder = (form.get("location_pbf_folder") or "").strip()
        if pbf_folder:
            loc["pbf_folder"] = pbf_folder
        elif "pbf_folder" not in loc:
            loc["pbf_folder"] = DEFAULT_PBF_FOLDER  # Default PBF folder

        pbf_file_selection = (form.get("location_pbf_file_selection") or "").strip()
        manual_pbf_path = (form.get("location_pbf_path") or "").strip()
        if pbf_file_selection and pbf_file_selection != "manual":
            loc["pbf_path"] = pbf_file_selection
        elif manual_pbf_path:
            loc["pbf_path"] = manual_pbf_path
        else:
            loc["pbf_path"] = None
            flash(
                "Local mode selected: please choose a local .osm.pbf file.",
                category="warning",
            )
    else:
        # Remote mode should not carry a PBF path
        loc["pbf_path"] = None

    # Manual bbox is no longer supported in the UI
    loc["bbox"] = None

    # Parse coordinates always (lat lon)
    q = raw_query.replace(",", " ").split()
    if len(q) >= 2:
        try:
            lat = float(q[0])
            lon = float(q[1])
            loc["query"] = f"{lat} {lon}"
        except Exception:
            loc["query"] = raw_query
            flash(
                "Coordinates must be numeric: expected 'lat lon'.", category="warning"
            )
    else:
        loc["query"] = raw_query
        flash(
            "Provide coordinates as 'lat lon' (e.g., '51.8944 -8.4827').",
            category="warning",
        )

    # If we have numeric coordinates but no distance, set a sensible default
    try:
        have_dist = loc.get("distance") is not None
        # Parse numeric coords from loc['query']
        q = (loc.get("query") or "").replace(",", " ").split()
        numeric_coords = False
        if len(q) >= 2:
            float(q[0])
            float(q[1])
            numeric_coords = True
        # Only set default if distance empty
        if numeric_coords and not have_dist:
            loc["distance"] = 1000.0
            flash(
                "Distance not provided; defaulting to 1000 meters around the coordinate center.",
                category="info",
            )
    except Exception:
        pass


def _update_output_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update output settings in the style dictionary."""
    output = style.setdefault("output", {})
    output["separate_layers"] = "separate_layers" in form
    filename_prefix = form.get(
        "filename_prefix", output.get("filename_prefix")
    )
    # Validate filename_prefix: alphanumeric and underscore only
    if filename_prefix and not re.match(r'^[a-zA-Z0-9_]+$', filename_prefix):
        flash(
            "Filename prefix must contain only letters, numbers, and underscores.",
            category="error",
        )
        # Keep existing valid prefix or default
        filename_prefix = output.get("filename_prefix", "map")
    output["filename_prefix"] = filename_prefix
    output["output_directory"] = form.get(
        "output_directory", output.get("output_directory", "../output")
    )
    width_str = form.get("figure_size_width")
    height_str = form.get("figure_size_height")
    output.setdefault("figure_size", [10.0, 10.0])
    output["figure_size"][0] = float(width_str) if width_str else 10.0
    output["figure_size"][1] = float(height_str) if height_str else 10.0
    output["background_color"] = form.get(
        "background_color", output.get("background_color")
    )
    # Transparent background flag overrides background color at save time
    output["transparent_background"] = "transparent_background" in form
    output["enable_debug_legends"] = "enable_debug_legends" in form
    dpi_str = form.get("figure_dpi")
    output["figure_dpi"] = int(dpi_str) if dpi_str else 300
    margin_str = form.get("margin")
    output["margin"] = float(margin_str) if margin_str else 0.05
    output["preview_type"] = form.get(
        "preview_type", output.get("preview_type", "embedded")
    )


def _parse_form_field(
    form: Mapping[str, str],
    field_name: str,
    default: Any = None,
    field_type: type = str,
) -> Any:
    """Parse a single form field with type casting and validation.

    Args:
        form: Request form mapping
        field_name: Name of the field to parse
        default: Default value if field is missing or invalid
        field_type: Type to cast the value to (str, float, int, bool)

    Returns:
        Parsed and type-cast value, or default if parsing fails
    """
    if field_type == bool:
        return field_name in form

    value = form.get(field_name)
    if value is None or value == "":
        return default

    try:
        if field_type == str:
            return value
        elif field_type == float:
            return float(value)
        elif field_type == int:
            return int(value)
        else:
            return field_type(value)
    except (ValueError, TypeError):
        return default


def _update_feature_visibility_from_form(form: Mapping[str, str]):
    """Update feature visibility states in the manager based on form data."""
    for category_id, category in feature_visibility_manager.get_all_categories():
        form_field_name = f"feature_{category_id.replace('.', '__')}"  # e.g., feature_streets__Motorways
        is_enabled = form_field_name in form
        feature_visibility_manager.set_visibility(category_id, is_enabled)


def _update_generic_layer_settings(
    style: Dict[str, Any], form: Mapping[str, str], layer_name: str
) -> None:
    """Update settings for generic layers like streets, water, and green."""
    layers = style.setdefault("layers", {})
    if layer_name not in layers:
        return

    layer = layers[layer_name]

    # Standard enabled checkbox for all layers
    layer["enabled"] = _parse_form_field(form, f"{layer_name}_enabled", False, bool)

    # Common layer settings
    layer["legend_enabled"] = _parse_form_field(
        form, f"{layer_name}_legend_enabled", False, bool
    )

    # Streets don't have facecolor (they're lines)
    if layer_name != "streets":
        layer["facecolor"] = _parse_form_field(
            form, f"{layer_name}_facecolor", layer.get("facecolor", "#000000"), str
        )

    layer["edgecolor"] = _parse_form_field(
        form, f"{layer_name}_edgecolor", layer.get("edgecolor", "#000000"), str
    )
    layer["linewidth"] = _parse_form_field(form, f"{layer_name}_linewidth", 0.5, float)
    layer["alpha"] = _parse_form_field(form, f"{layer_name}_alpha", 1.0, float)
    layer["simplify_tolerance"] = _parse_form_field(
        form, f"{layer_name}_simplify_tolerance", 0.0, float
    )
    layer["min_size_threshold"] = _parse_form_field(
        form, f"{layer_name}_min_size_threshold", 0.0, float
    )

    hatch_value = form.get(f"{layer_name}_hatch")
    layer["hatch"] = None if hatch_value == "null" else hatch_value

    layer["zorder"] = _parse_form_field(form, f"{layer_name}_zorder", 1, int)

    # Tag-based filters are now handled by the feature_visibility_manager
    # _update_layer_tag_filters(style, form, layer_name)


def _update_buildings_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update the complex, multi-mode settings for the buildings layer."""
    buildings = style.setdefault("layers", {}).setdefault("buildings", {})

    # Common settings using helper
    buildings["enabled"] = _parse_form_field(form, "buildings_enabled", False, bool)
    buildings["legend_enabled"] = _parse_form_field(
        form, "buildings_legend_enabled", False, bool
    )
    buildings["simplify_tolerance"] = _parse_form_field(
        form, "buildings_simplify_tolerance", 0.0, float
    )
    buildings["min_size_threshold"] = _parse_form_field(
        form, "buildings_min_size_threshold", 0.0, float
    )
    buildings["edgecolor"] = _parse_form_field(
        form, "buildings_edgecolor", buildings.get("edgecolor", "#000000"), str
    )
    buildings["linewidth"] = _parse_form_field(form, "buildings_linewidth", 0.5, float)
    buildings["alpha"] = _parse_form_field(form, "buildings_alpha", 1.0, float)
    buildings["zorder"] = _parse_form_field(form, "buildings_zorder", 2, int)

    hatch_value = form.get("buildings_hatch")
    buildings["hatch"] = None if hatch_value == "null" else hatch_value

    # Styling mode
    buildings_style_mode = _parse_form_field(
        form, "building_styling_mode", "manual", str
    )
    buildings["auto_style_mode"] = buildings_style_mode

    # Reset styling options
    buildings["size_categories"] = []
    buildings["size_categories_enabled"] = False
    buildings["auto_size_palette"] = ""
    buildings["auto_distance_palette"] = ""
    buildings.setdefault("manual_color_settings", {"facecolor": "#000000"})

    if buildings_style_mode == "manual":
        manual_settings = buildings.setdefault("manual_color_settings", {})
        manual_settings["facecolor"] = _parse_form_field(
            form,
            "buildings_manual_color_facecolor",
            manual_settings.get("facecolor", "#000000"),
            str,
        )

    elif buildings_style_mode == "auto_size":
        palette = _parse_form_field(form, "auto_size_palette", "YlGnBu_5", str)
        buildings["auto_size_palette"] = palette if palette else "YlGnBu_5"

    elif buildings_style_mode == "auto_distance":
        palette = _parse_form_field(form, "auto_distance_palette", "YlOrRd_5", str)
        buildings["auto_distance_palette"] = palette if palette else "YlOrRd_5"

    elif buildings_style_mode == "manual_floorsize":
        # Parse dynamic size categories
        categories = []
        indices = set()

        for key in form.keys():
            if key.startswith("buildings_size_category_"):
                try:
                    parts = key.split("_")
                    if len(parts) >= 5:
                        idx = int(parts[3])
                        indices.add(idx)
                except (ValueError, IndexError):
                    continue

        for idx in sorted(indices):
            prefix = f"buildings_size_category_{idx}"
            name = _parse_form_field(form, f"{prefix}_name", f"Category {idx+1}", str)
            min_area = _parse_form_field(form, f"{prefix}_min_area", 0.0, float)
            max_area = _parse_form_field(form, f"{prefix}_max_area", 0.0, float)
            facecolor = _parse_form_field(form, f"{prefix}_facecolor", "#000000", str)

            if name:
                categories.append(
                    {
                        "name": name,
                        "min_area": min_area,
                        "max_area": max_area,
                        "facecolor": facecolor,
                    }
                )

        if categories:
            buildings["size_categories"] = categories
            buildings["size_categories_enabled"] = True


def _validate_location_inputs(style: Dict[str, Any]) -> List[str]:
    """Validate location inputs after _update_style_from_form and normalization.

    - Coordinates must be 'lat lon' with lat in [-90, 90], lon in [-180, 180]
    - Distance must be between 10 and 20000 meters
    - If data_source == 'local', pbf_path must be present
    """
    errors: List[str] = []
    loc = style.get("location", {})
    query = (loc.get("query") or "").replace(",", " ").split()
    if len(query) < 2:
        errors.append("Coordinates required as 'lat lon'.")
    else:
        try:
            lat = float(query[0])
            lon = float(query[1])
            if not (-90.0 <= lat <= 90.0):
                errors.append("Latitude must be between -90 and 90.")
            if not (-180.0 <= lon <= 180.0):
                errors.append("Longitude must be between -180 and 180.")
        except Exception:
            errors.append("Coordinates must be numeric: expected 'lat lon'.")

    # Distance
    distance = loc.get("distance")
    try:
        if distance is None:
            errors.append("Distance is required (meters).")
        else:
            dval = float(distance)
            if not (10.0 <= dval <= 20000.0):
                errors.append("Distance must be between 10 and 20000 meters.")
    except Exception:
        errors.append("Distance must be a number (meters).")

    # Local PBF requirement
    data_source = loc.get("data_source")
    if data_source == "local":
        pbf_path = loc.get("pbf_path")
        if not pbf_path:
            errors.append("Local mode selected: please choose a local .osm.pbf file.")

    return errors


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
    """Get a list of all saved preset names."""
    presets = _load_presets()
    return jsonify({"success": True, "presets": list(presets.keys())})


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
    """Load a preset and save it as the current style."""
    presets = _load_presets()
    if preset_name not in presets:
        return jsonify({"success": False, "error": "Preset not found."}), 404

    _save_style_json(presets[preset_name])
    return jsonify(
        {"success": True, "message": f'Preset "{preset_name}" loaded successfully.'}
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
    style = _update_style_from_form(load_style(), request.form)
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

        style = _update_style_from_form(style, request.form)

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
