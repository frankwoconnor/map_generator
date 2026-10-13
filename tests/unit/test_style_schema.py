"""Tests for config/schemas/style_schema.py."""

import json

import pytest

from config.schemas.style_schema import StyleConfig


def test_style_config_round_trip_preserves_all_fields(tmp_path):
    """Load style.json, convert to StyleConfig, back to dict - verify no data loss."""
    from map_core.core.config import load_style

    style_path = tmp_path / "style.json"
    raw = {
        "location": {
            "query": "51.8986 -8.4756",
            "distance": 1000.0,
            "pbf_path": "../data/ireland.osm.pbf",
            "data_source": "local",
            "bbox": None,
            "pbf_folder": "../../data/",
        },
        "layers": {
            "streets": {
                "enabled": True,
                "facecolor": "#000000",
                "edgecolor": "#000000",
                "linewidth": 0.5,
                "alpha": 1.0,
                "simplify_tolerance": 0.0,
                "min_size_threshold": 0.0,
                "hatch": None,
                "zorder": 3,
                "filter": ["motorway", "primary"],
                "filters": {"highway": ["motorway", "primary"]},
                "legend_enabled": False,
                "style_mode": "uniform",
                "variable_widths": True,
                "street_palette": "",
                "class_colors": {},
                "category_widths": {},
                "linewidths": {},
            },
            "water": {
                "enabled": True,
                "facecolor": "#1f24bd",
                "edgecolor": "#0702f7",
                "linewidth": 0.5,
                "alpha": 1.0,
                "simplify_tolerance": 0.0,
                "min_size_threshold": 0.0,
                "hatch": None,
                "zorder": 4,
                "filter": ["water"],
                "filters": {"natural": ["water", "lake"]},
                "legend_enabled": False,
            },
            "green": {
                "enabled": True,
                "facecolor": "#2f7c0e",
                "edgecolor": "#116927",
                "linewidth": 0.5,
                "alpha": 1.0,
                "simplify_tolerance": 0.0,
                "min_size_threshold": 0.0,
                "hatch": None,
                "zorder": 1,
                "filters": {"leisure": ["park"]},
                "legend_enabled": False,
            },
            "buildings": {
                "enabled": True,
                "facecolor": "#000000",
                "edgecolor": "#000000",
                "linewidth": 0.2,
                "alpha": 1.0,
                "simplify_tolerance": 0.0,
                "min_size_threshold": 0.0,
                "hatch": None,
                "zorder": 4,
                "filters": {},
                "manual_color_settings": {"facecolor": "#feffff"},
                "size_categories": [],
                "size_categories_enabled": False,
                "auto_style_mode": "manual",
                "auto_size_palette": "",
                "auto_distance_palette": "",
                "legend_enabled": False,
            },
        },
        "output": {
            "separate_layers": True,
            "filename_prefix": "map",
            "figure_size": [10.0, 10.0],
            "background_color": "#ffffff",
            "transparent_background": True,
            "figure_dpi": 100,
            "margin": 0.05,
            "preview_type": "embedded",
            "output_directory": "../output",
            "enable_debug_legends": True,
            "generate_png": False,
            "png_dpi": 300,
            "png_scale": 2.0,
            "png_suffix": "_hq",
            "png_background": None,
            "png_source": "original",
            "svg_optimize_enabled": True,
            "svg_optimizer": "scour",
            "svg_optimize_config": {
                "remove_metadata": True,
                "strip_comments": True,
                "enable_viewboxing": True,
                "keep_editor_data": False,
                "renderer_workaround": True,
                "remove_descriptive_elements": False,
                "shorten_ids": True,
                "simple_colors": False,
                "digits": 3,
                "indent_type": "none",
                "nindent": 2,
            },
            "svg_css_extract": True,
            "svg_css_merge_rules": True,
            "svg_css_attributes": [
                "fill", "stroke", "stroke-width", "opacity", "fill-opacity",
                "stroke-opacity", "vector-effect", "stroke-linejoin",
                "stroke-linecap", "stroke-dasharray"
            ],
            "svg_write_separate": True,
            "svg_optimized_suffix": "_opt",
            "svg_optimized_dir": "optimized",
            "svg_simplify_enabled": False,
            "svg_simplify_tolerance": 0.1,
            "svg_simplify_preserve_topology": True,
            "svg_simplify_min_segment_length": 0.5,
            "svg_low_poly_mode": False,
            "mask": "none",
            "mask_options": {"inset": 0.0, "rounded_radius": 0.15},
            "frame": {"width": 0.0, "color": "#000000"},
            "title": "",
            "subtitle": "",
            "title_font_size": 16,
            "subtitle_font_size": 11,
            "title_color": "#000000",
            "subtitle_color": "#444444",
        },
        "processing": {
            "street_filter": [],
            "auto_clip": True,
            "auto_clip_threshold_mb": 20.0,
            "auto_clip_margin": 0.2,
        },
    }
    style_path.write_text(json.dumps(raw))

    loaded = load_style(str(style_path))
    config = StyleConfig.from_dict(loaded)
    result = config.to_dict()

    assert result["location"] == raw["location"]
    assert result["output"] == raw["output"]
    assert result["processing"] == raw["processing"]

    for layer_name in ["streets", "water", "green", "buildings"]:
        assert result["layers"][layer_name] == raw["layers"][layer_name]


def test_style_config_fills_defaults_for_missing_sections():
    """Missing sections should be filled with defaults."""
    config = StyleConfig.from_dict({})
    result = config.to_dict()

    assert "location" in result
    assert "layers" in result
    assert "output" in result
    assert "processing" in result

    assert result["location"]["query"] == ""
    assert result["location"]["distance"] is None
    assert result["output"]["filename_prefix"] == "map"
    assert result["output"]["figure_size"] == [10.0, 10.0]

    for layer_name in ["streets", "water", "green", "buildings"]:
        assert layer_name in result["layers"]
        assert result["layers"][layer_name]["enabled"] is True
        assert result["layers"][layer_name]["linewidth"] == 0.5


def test_street_styling_keys_round_trip_through_schema():
    """New streets styling keys survive a StyleConfig normalization round-trip."""
    raw = {
        "layers": {
            "streets": {
                "style_mode": "by_class",
                "variable_widths": True,
                "street_palette": "Blues_5",
                "class_colors": {"motorways": "#ff0000"},
                "category_widths": {"service": 0.2},
                "linewidths": {"motorway": 3.0},
            }
        }
    }
    result = StyleConfig.from_dict(raw).to_dict()
    streets = result["layers"]["streets"]
    assert streets["style_mode"] == "by_class"
    assert streets["variable_widths"] is True
    assert streets["street_palette"] == "Blues_5"
    assert streets["class_colors"] == {"motorways": "#ff0000"}
    assert streets["category_widths"] == {"service": 0.2}
    assert streets["linewidths"] == {"motorway": 3.0}


def test_buildings_size_categories_preserved():
    """Buildings size_categories list should round-trip correctly."""
    raw = {
        "layers": {
            "buildings": {
                "size_categories": [
                    {
                        "name": "Small",
                        "min_area": 0.0,
                        "max_area": 100.0,
                        "facecolor": "#000000",
                    }
                ],
                "size_categories_enabled": True,
            }
        }
    }
    config = StyleConfig.from_dict(raw)
    result = config.to_dict()

    assert result["layers"]["buildings"]["size_categories"] == raw["layers"]["buildings"]["size_categories"]
    assert result["layers"]["buildings"]["size_categories_enabled"] is True
