"""Integration tests for template rendering."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config.manager import feature_visibility_manager

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))


# Mock Flask's url_for function
def mock_url_for(endpoint, **values):
    """Mock implementation of Flask's url_for."""
    if endpoint == "static":
        return f"/static/{values.get('filename', '')}"
    return f"/{endpoint}"


class TestTemplateRendering:
    """Test suite for template rendering functionality."""

    @pytest.fixture
    def jinja_env(self):
        """Create a Jinja2 environment for testing."""
        return Environment(
            loader=FileSystemLoader(
                os.path.join(os.path.dirname(__file__), "../../templates")
            ),
            autoescape=select_autoescape(["html", "xml"]),
        )

    @pytest.fixture
    def dummy_style(self):
        """Provide a dummy style configuration for testing."""
        return {
            "location": {"query": "Test", "distance": None},
            "layers": {
                "streets": {
                    "enabled": True,
                    "facecolor": "#000000",
                    "edgecolor": "#000000",
                    "linewidth": 0.1,
                    "alpha": 1.0,
                    "simplify_tolerance": 0.00005,
                    "min_size_threshold": 0,
                },
                "buildings": {
                    "enabled": True,
                    "facecolor": "#000000",
                    "edgecolor": "#000000",
                    "linewidth": 0.2,
                    "alpha": 1.0,
                    "simplify_tolerance": 0.000001,
                    "hatch": "|",
                    "zorder": 2,
                    "size_categories": [],
                    "min_size_threshold": 10,
                    "manual_color_settings": {
                        "facecolor": "#000000",
                        "edgecolor": "#000000",
                    },
                },
                "water": {
                    "enabled": True,
                    "facecolor": "#000000",
                    "edgecolor": "#000000",
                    "linewidth": 0.3,
                    "alpha": 1.0,
                    "simplify_tolerance": 0.0001,
                    "hatch": "\\",
                    "zorder": 1,
                    "min_size_threshold": 0.000001,
                },
            },
            "output": {
                "separate_layers": False,
                "filename_prefix": "test_map",
                "output_directory": "../output",
                "figure_size": [10, 10],
                "background_color": "white",
                "figure_dpi": 300,
                "margin": 0.05,
            },
            "processing": {"street_filter": []},
        }

    def test_index_template_rendering(self, jinja_env, dummy_style):
        """Test that the index template renders without errors."""
        # Add url_for to the template globals
        jinja_env.globals["url_for"] = mock_url_for

        template = jinja_env.get_template("index.html")

        # Test rendering with dummy data
        rendered_html = template.render(
            style=dummy_style,
            generated_image_path=None,
            url_for=mock_url_for,  # Pass as a parameter as well for good measure
            feature_manager=feature_visibility_manager,
        )

        # Basic assertions
        assert rendered_html is not None
        assert len(rendered_html) > 0

        # Check for important elements in the rendered template
        assert "<!DOCTYPE html>" in rendered_html
        assert "<html" in rendered_html.lower()
        assert "</html>" in rendered_html.lower()

        # Check for a form element that should be present in the template
        assert "<form" in rendered_html.lower()
        assert 'method="post"' in rendered_html.lower()

    def test_wall_art_template_rendering(self, jinja_env):
        """Test that the wall art template renders without errors."""
        # Skip if the template doesn't exist yet
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "../../templates/wall_art.html")
        ):
            pytest.skip("Wall art template not found")

        template = jinja_env.get_template("wall_art.html")
        rendered_html = template.render()

        assert rendered_html is not None
        assert len(rendered_html) > 0
        assert "<!DOCTYPE html>" in rendered_html
