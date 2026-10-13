"""Tests for the creative output renderer (D4): crop mask / frame / title."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from map_core.core.plot import (
    _format_coords_text,
    _substitute_text,
    apply_creative_output,
)


def _render(output, query="51.8986 -8.4756"):
    """Plot a trivial map with the creative output applied and return the SVG."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([1, 9], [1, 9])
    apply_creative_output(fig, ax, output, query)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def _render_baseline():
    """SVG of the same plot with no creative output call."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([1, 9], [1, 9])
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def test_format_coords_text():
    assert _format_coords_text("51.8986 -8.4756") == "51.90° N / 8.48° W"
    assert _format_coords_text("51.5, -0.12") == "51.50° N / 0.12° W"
    assert _format_coords_text("Cork") == "Cork"  # not a coords string


def test_substitute_text_placeholders():
    out = _substitute_text("Map of {region} at {coords}", "51.8986 -8.4756")
    assert "{region}" not in out and "{coords}" not in out
    assert "51.90° N / 8.48° W" in out


def test_circle_mask_clips_and_frames():
    svg = _render(
        {
            "mask": "circle",
            "mask_options": {"inset": 0.1},
            "frame": {"width": 3, "color": "#ff0000"},
        }
    )
    assert "clip-path" in svg
    assert "#ff0000" in svg.lower() or "#ff0000;" in svg


def test_rounded_mask_and_title():
    svg = _render(
        {
            "mask": "rounded",
            "mask_options": {"inset": 0.0, "rounded_radius": 0.2},
            "title": "Cork {coords}",
            "subtitle": "{region} · generator",
            "title_color": "#0000ff",
        }
    )
    # Title text is drawn (matplotlib emits it as a commented text group).
    assert "Cork 51.90" in svg.replace("°", "")
    assert "generator" in svg


def test_no_mask_when_none_or_unknown():
    baseline_refs = _render_baseline().count("clip-path")
    for bad in (None, "none", "star", ""):
        svg = _render({"mask": bad} if bad else {})
        # Unknown/empty masks must leave the plot untouched.
        assert svg.count("clip-path") == baseline_refs
    # A real mask clips extra artists.
    assert _render({"mask": "circle"}) .count("clip-path") > baseline_refs