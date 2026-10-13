"""Tests for the street-class line width hierarchy in map_core.core.plot."""

from map_core.core.plot import (
    DEFAULT_HIGHWAY_WIDTH_MULTIPLIERS,
    MIN_HIGHWAY_LINEWIDTH,
    _highway_widths_for_style,
    setup_figure_and_axes_isolated,
)


def test_widths_scale_from_base_linewidth():
    widths = _highway_widths_for_style(0.5)
    assert widths["motorway"] == 3.2 * 0.5
    assert widths["service"] == 0.85 * 0.5
    # Hierarchy order is preserved: bigger roads are thicker than minor roads.
    assert widths["motorway"] > widths["trunk"] > widths["primary"] > widths["service"]


def test_widths_floor_keeps_major_roads_legible():
    widths = _highway_widths_for_style(0.1)
    # At a tiny base width the multiplier alone would shrink everything.
    assert widths["motorway"] == max(3.2 * 0.1, MIN_HIGHWAY_LINEWIDTH["motorway"])
    assert widths["service"] == max(0.85 * 0.1, MIN_HIGHWAY_LINEWIDTH["service"])
    # Major roads stay visually distinct even at base 0.1.
    assert widths["motorway"] >= 0.5
    assert widths["motorway"] / widths["service"] > 2


def test_widths_explicit_override_wins():
    overrides = {"motorway": 1.0, "residential": 0.2}
    widths = _highway_widths_for_style(0.5, layer_style={"linewidths": overrides})
    assert widths == overrides


def test_isolated_figure_can_save_svg(tmp_path):
    """The thread-safe figure factory must produce a savable SVG with an axes."""
    fig, ax = setup_figure_and_axes_isolated([2, 2], 100, "#ffffff", 0.0)
    assert ax is not None
    out = tmp_path / "t.svg"
    fig.savefig(str(out), format="svg")
    assert out.exists() and out.stat().st_size > 0
    fig.clear()