"""Tests for per-class street styling (colors/themes) in map_core.core.street_styling."""

from matplotlib.colors import to_hex as mpl_to_hex

from map_core.core.plot import (
    MIN_HIGHWAY_LINEWIDTH,
    _highway_widths_for_style,
    _plot_streets_with_class_widths,
    setup_figure_and_axes_isolated,
)
from map_core.core.street_styling import (
    DEFAULT_STREET_CATEGORY_COLORS,
    STREET_CATEGORIES,
    street_category_colors,
    street_class_colors,
)


# --- Category/color resolution -------------------------------------------------


def test_category_order_starts_motorways_ends_paths():
    assert STREET_CATEGORIES[0][0] == "motorways"
    assert STREET_CATEGORIES[-1][0] == "paths_trails"
    for _cat_id, _label, highways in STREET_CATEGORIES:
        assert highways


def test_uniform_mode_returns_no_class_colors():
    assert street_class_colors("#000000", {"style_mode": "uniform"}) == {}
    assert street_category_colors("#000000", {}) == {}
    assert street_category_colors("#000000") == {}


def test_by_class_defaults_to_grey_importance_ramp():
    colors = street_category_colors("#000000", {"style_mode": "by_class"})
    assert colors["motorways"] == DEFAULT_STREET_CATEGORY_COLORS["motorways"]
    assert colors["paths_trails"] == DEFAULT_STREET_CATEGORY_COLORS["paths_trails"]
    # Ramp is monotonic lightening: motorways darkest, paths lightest.
    assert (
        DEFAULT_STREET_CATEGORY_COLORS["motorways"]
        < DEFAULT_STREET_CATEGORY_COLORS["primary"]
        < DEFAULT_STREET_CATEGORY_COLORS["secondary"]
        < DEFAULT_STREET_CATEGORY_COLORS["paths_trails"]
    )


def test_palette_colors_darkest_first_and_clamped():
    style = {"style_mode": "by_class", "street_palette": "Greys_5"}
    colors = street_category_colors("#000000", style)
    # Greys_5 reversed puts the darkest grey on Motorways.
    assert colors["motorways"] == "#252525"
    assert colors["secondary"] == "#969696"
    # Categories past the palette's length keep the lightest color.
    assert colors["service"] == "#f7f7f7"
    assert colors["paths_trails"] == "#f7f7f7"


def test_category_override_wins_over_palette():
    style = {
        "style_mode": "by_class",
        "street_palette": "Greys_5",
        "class_colors": {"motorways": "#ff0000"},
    }
    colors = street_class_colors("#000000", style)
    assert colors["motorway"] == "#ff0000"
    assert colors["motorway_link"] == "#ff0000"
    assert colors["trunk"] == "#ff0000"
    # Non-overridden categories still come from the palette.
    assert colors["residential"] == "#f7f7f7"


def test_per_highway_override_wins_within_category():
    style = {
        "style_mode": "by_class",
        "street_palette": "Greys_5",
        "class_colors": {"paths_trails": "#0f0f0f", "footway": "#00ff00"},
    }
    colors = street_class_colors("#000000", style)
    assert colors["footway"] == "#00ff00"
    # The rest of paths & trails keeps the category color.
    assert colors["path"] == "#0f0f0f"
    assert colors["cycleway"] == "#0f0f0f"


# --- Width overrides (category_widths) ------------------------------------------


def test_category_widths_override_highway_defaults():
    style = {"category_widths": {"motorways": 2.0, "paths_trails": 0.1}}
    widths = _highway_widths_for_style(0.1, layer_style=style)
    assert widths["motorway"] == 2.0
    assert widths["trunk_link"] == 2.0
    assert widths["footway"] == 0.1
    assert widths["path"] == 0.1
    # Categories without a per-category width keep the default hierarchy.
    assert widths["residential"] == max(1.1 * 0.1, MIN_HIGHWAY_LINEWIDTH["residential"])


def test_explicit_linewidths_still_win_over_category_widths():
    style = {
        "linewidths": {"motorway": 5.0},
        "category_widths": {"motorways": 2.0},
    }
    widths = _highway_widths_for_style(0.5, layer_style=style)
    # The explicit per-highway override beats the coarser category width,
    assert widths["motorway"] == 5.0
    # ...but the category width fills the rest of that category.
    assert widths["trunk"] == 2.0
    assert widths["motorway_link"] == 2.0
    # An explicit linewidths dict replaces the defaults entirely.
    assert "residential" not in widths


# --- Plot-level behavior (variable widths flag + by_class colors) ---------------


def _sample_edges(highways):
    import geopandas as gpd
    from shapely.geometry import LineString

    return gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)]) for _ in highways],
            "highway": list(highways),
        },
        crs="EPSG:4326",
    )


def _line_collection_colors(ax):
    from matplotlib.collections import LineCollection

    return [
        mpl_to_hex(c.get_edgecolor()[0])
        for c in ax.collections
        if isinstance(c, LineCollection)
    ]


def test_variable_widths_false_draws_single_uniform_stroke():
    fig, ax = setup_figure_and_axes_isolated([2, 2], 100, "#ffffff", 0.0)
    edges = _sample_edges(["motorway", "footway", "motorway"])
    _plot_streets_with_class_widths(
        ax,
        edges,
        "#111111",
        0.5,
        1.0,
        1,
        layer_style={"variable_widths": False},
    )
    colors = _line_collection_colors(ax)
    assert len(colors) == 1
    assert colors[0] == "#111111"
    fig.clear()


def test_by_class_mode_draws_one_color_per_class():
    fig, ax = setup_figure_and_axes_isolated([2, 2], 100, "#ffffff", 0.0)
    edges = _sample_edges(["motorway", "footway", "trunk"])
    style = {"style_mode": "by_class", "street_palette": "Greys_5"}
    _plot_streets_with_class_widths(
        ax, edges, "#000000", 0.5, 1.0, 1, layer_style=style
    )
    colors = set(_line_collection_colors(ax))
    # motorway + trunk (darkest), footway (lightest grey).
    assert "#252525" in colors
    assert "#f7f7f7" in colors
    assert len(colors) == 2  # same palette color for both motorway and trunk
    fig.clear()


def test_by_class_mode_keeps_colors_when_widths_are_flat():
    """By Road Class without variable widths still colors each class (uniform width)."""
    fig, ax = setup_figure_and_axes_isolated([2, 2], 100, "#ffffff", 0.0)
    edges = _sample_edges(["motorway", "footway"])
    style = {
        "style_mode": "by_class",
        "street_palette": "Greys_5",
        "variable_widths": False,
    }
    _plot_streets_with_class_widths(
        ax, edges, "#000000", 0.5, 1.0, 1, layer_style=style
    )
    colors = set(_line_collection_colors(ax))
    assert "#252525" in colors
    assert "#f7f7f7" in colors
    fig.clear()