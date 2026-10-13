"""Style form service: apply HTTP form data onto a style dictionary.

Extracted from ``app.py`` so the transformation logic is testable without
Flask. Warnings that the web layer used to ``flash()`` are emitted through an
injected ``warn(message, category)`` callback (default: silently ignored), so
this module has no Flask dependency.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional

from config.manager import feature_visibility_manager
from map_core.core.street_styling import (STREET_CATEGORIES,
                                          street_category_colors)


def _no_warn(message: str, category: str = "warning") -> None:
    """Default warning sink when no callback is supplied (Flask-free use)."""


def _update_style_from_form(
    style: Dict[str, Any], form: Mapping[str, str], warn: Optional[Callable[[str, str], None]] = None
) -> Dict[str, Any]:
    """Orchestrate updating the style dictionary from form data.

    Args:
        style: Current style config (will be mutated in-place).
        form: Request form mapping with string keys/values.

    Returns:
        dict: The updated style config (same object as input).
    """
    _update_location_settings(style, form, warn)
    _update_output_settings(style, form, warn)
    _update_generic_layer_settings(style, form, "streets")
    _update_generic_layer_settings(style, form, "water")
    _update_generic_layer_settings(style, form, "green")
    _update_streets_settings(style, form)
    _update_buildings_settings(style, form)
    _update_feature_visibility_from_form(form)  # New call
    # The new generic layer settings function handles filters, so the specific processing one is removed.
    # _update_processing_settings(style, form)
    return style


def _update_location_settings(style: Dict[str, Any], form: Mapping[str, str], warn: Optional[Callable[[str, str], None]] = None) -> None:
    """Update location settings per new Local/Remote model (coords + distance required)."""
    if warn is None:
        warn = _no_warn
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
            warn(
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
            warn(
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
            warn(
                "Coordinates must be numeric: expected 'lat lon'.", category="warning"
            )
    else:
        loc["query"] = raw_query
        warn(
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
            warn(
                "Distance not provided; defaulting to 1000 meters around the coordinate center.",
                category="info",
            )
    except Exception:
        pass


def _update_output_settings(style: Dict[str, Any], form: Mapping[str, str], warn: Optional[Callable[[str, str], None]] = None) -> None:
    """Update output settings in the style dictionary."""
    if warn is None:
        warn = _no_warn
    output = style.setdefault("output", {})
    output["separate_layers"] = "separate_layers" in form
    filename_prefix = form.get(
        "filename_prefix", output.get("filename_prefix")
    )
    # Validate filename_prefix: alphanumeric and underscore only
    if filename_prefix and not re.match(r'^[a-zA-Z0-9_]+$', filename_prefix):
        warn(
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

    # Creative output controls (D4): mask / frame / title.
    output["mask"] = form.get("output_mask", output.get("mask", "none")) or "none"
    mask_options = output.setdefault("mask_options", {})
    mask_options["inset"] = _parse_form_field(
        form,
        "output_mask_inset",
        mask_options.get("inset", 0.0),
        float,
    )
    mask_options["rounded_radius"] = _parse_form_field(
        form,
        "output_mask_radius",
        mask_options.get("rounded_radius", 0.15),
        float,
    )
    frame = output.setdefault("frame", {})
    frame["width"] = _parse_form_field(
        form,
        "output_frame_width",
        frame.get("width", 0.0),
        float,
    )
    frame["color"] = form.get("output_frame_color", frame.get("color", "#000000"))
    output["title"] = form.get("output_title", output.get("title", ""))
    output["subtitle"] = form.get("output_subtitle", output.get("subtitle", ""))
    output["title_font_size"] = _parse_form_field(
        form, "output_title_font_size", output.get("title_font_size", 16), float
    )
    output["subtitle_font_size"] = _parse_form_field(
        form, "output_subtitle_font_size", output.get("subtitle_font_size", 11), float
    )
    output["title_color"] = form.get("output_title_color", output.get("title_color", "#000000"))
    output["subtitle_color"] = form.get("output_subtitle_color", output.get("subtitle_color", "#444444"))

    # Output format controls
    output["generate_png"] = "generate_png" in form
    output["png_dpi"] = _parse_form_field(form, "png_dpi", output.get("png_dpi", 300), int)
    output["png_scale"] = _parse_form_field(form, "png_scale", output.get("png_scale", 2.0), float)
    output["png_suffix"] = form.get("png_suffix", output.get("png_suffix", "_hq"))
    output["png_background"] = form.get("png_background", output.get("png_background", "#FFFFFF"))
    output["png_source"] = form.get("png_source", output.get("png_source", "original"))

    # SVG optimization controls
    output["svg_optimize_enabled"] = "svg_optimize_enabled" in form
    output["svg_optimizer"] = form.get("svg_optimizer", output.get("svg_optimizer", "scour"))
    output["svg_css_extract"] = "svg_css_extract" in form
    output["svg_css_merge_rules"] = "svg_css_merge_rules" in form
    output["svg_write_separate"] = "svg_write_separate" in form
    output["svg_optimized_suffix"] = form.get("svg_optimized_suffix", output.get("svg_optimized_suffix", "_opt"))
    output["svg_optimized_dir"] = form.get("svg_optimized_dir", output.get("svg_optimized_dir", "optimized"))
    # svg_optimize_config and svg_css_attributes are complex; keep existing or load from config file

    # SVG shape simplification controls
    output["svg_simplify_enabled"] = "svg_simplify_enabled" in form
    output["svg_simplify_tolerance"] = _parse_form_field(form, "svg_simplify_tolerance", output.get("svg_simplify_tolerance", 0.1), float)
    output["svg_simplify_preserve_topology"] = "svg_simplify_preserve_topology" in form
    output["svg_simplify_min_segment_length"] = _parse_form_field(form, "svg_simplify_min_segment_length", output.get("svg_simplify_min_segment_length", 0.5), float)
    output["svg_low_poly_mode"] = "svg_low_poly_mode" in form
    output["svg_low_poly_grid"] = _parse_form_field(form, "svg_low_poly_grid", output.get("svg_low_poly_grid", 1.0), float)


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


def _update_streets_settings(style: Dict[str, Any], form: Mapping[str, str]) -> None:
    """Update the per-class street styling settings (mode, palette, overrides)."""
    streets = style.setdefault("layers", {}).setdefault("streets", {})

    # Color mode: "uniform" (single edgecolor) or "by_class" (color per class).
    streets["style_mode"] = _parse_form_field(
        form, "streets_style_mode", "uniform", str
    )
    if streets["style_mode"] not in ("uniform", "by_class"):
        streets["style_mode"] = "uniform"

    # Variable per-class line widths.
    streets["variable_widths"] = "streets_variable_widths" in form

    # Theme palette ("" = default grey importance ramp).
    streets["street_palette"] = _parse_form_field(
        form, "streets_street_palette", "", str
    )

    # Per-category color overrides from the class table. The form always
    # submits every row, so an empty value clears a previous override.
    class_colors = {}
    for cat_id, _label, _highways in STREET_CATEGORIES:
        color = (form.get(f"streets_class_color_{cat_id}") or "").strip()
        if color:
            class_colors[cat_id] = color
    streets["class_colors"] = class_colors

    # Per-category absolute width overrides (blank = automatic hierarchy).
    category_widths = {}
    for cat_id, _label, _highways in STREET_CATEGORIES:
        raw = (form.get(f"streets_category_width_{cat_id}") or "").strip()
        if not raw:
            continue
        try:
            w = float(raw)
        except ValueError:
            continue
        if w > 0:
            category_widths[cat_id] = w
    streets["category_widths"] = category_widths


def _street_class_table(style: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the per-class color/width rows the studio renders for streets."""
    streets = style.get("layers", {}).get("streets", {}) or {}
    edgecolor = streets.get("edgecolor", "#000000")
    cat_colors = street_category_colors(edgecolor, streets)
    cat_widths = streets.get("category_widths", {}) or {}

    table = []
    for cat_id, label, _highways in STREET_CATEGORIES:
        table.append(
            {
                "id": cat_id,
                "label": label,
                "color": cat_colors.get(cat_id, edgecolor),
                "width": cat_widths.get(cat_id, ""),
            }
        )
    return table


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
