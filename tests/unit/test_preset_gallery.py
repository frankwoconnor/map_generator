"""Tests for the preset-gallery metadata helper (D2)."""

import importlib.util
import os
import sys
from pathlib import Path


def _load_app_module():
    """Import app.py by path (the `app/` package shadows the name)."""
    root = Path(__file__).parent.parent.parent
    spec = importlib.util.spec_from_file_location("studio_app", root / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["studio_app"] = module
    spec.loader.exec_module(module)
    return module


def test_preset_card_meta_collects_swatches_and_layers():
    app = _load_app_module()
    preset = {
        "layers": {
            "streets": {
                "enabled": True,
                "style_mode": "by_class",
                "class_colors": {"motorways": "#ff0000", "primary": "#00ff00"},
            },
            "buildings": {"enabled": True, "auto_style_mode": "manual"},
            "water": {"enabled": True, "facecolor": "#a6cee3"},
            "green": {"enabled": False, "facecolor": "#2f7c0e"},
        },
        "location": {"query": "51.8986 -8.4756", "distance": 1200},
    }
    meta = app._preset_card_meta("demo", preset)
    assert meta["name"] == "demo"
    assert meta["enabled"] == ["streets", "buildings", "water"]
    assert "#ff0000" in meta["colors"] and "#a6cee3" in meta["colors"]
    assert "#2f7c0e" not in meta["colors"]  # disabled layer excluded
    assert meta["region"] == "51.8986 -8.4756"
    assert meta["distance"] == 1200


def test_preset_card_meta_tolerates_missing_sections():
    app = _load_app_module()
    meta = app._preset_card_meta("bare", {"layers": {}})
    assert meta["enabled"] == []
    assert meta["colors"] == []
    assert meta["region"] == ""
    assert meta["street_mode"] == "uniform"


def test_form_output_settings_include_creative_keys():
    """_update_output_settings maps the new output_* form fields (style_form_service)."""
    from map_core.services.style_form_service import _update_output_settings

    style = {"output": {}}
    _update_output_settings(
        style,
        {
            "output_mask": "circle",
            "output_mask_inset": "0.12",
            "output_frame_width": "4",
            "output_frame_color": "#123456",
            "output_title": "Cork {coords}",
        },
    )
    out = style["output"]
    assert out["mask"] == "circle"
    assert out["mask_options"]["inset"] == 0.12
    assert out["frame"] == {"width": 4.0, "color": "#123456"}
    assert out["title"] == "Cork {coords}"