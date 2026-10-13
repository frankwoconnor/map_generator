"""Creative output: crop masks, frames, and titles baked into a figure.

Self-contained post-processing applied to a matplotlib figure just before
``savefig`` (see :func:`apply_creative_output`). Extracted from ``plot.py``
to keep that module focused on data plotting.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np


def _format_coords_text(query: Optional[str]) -> str:
    """Format a 'lat lon' query as e.g. '51.90° N / 8.47° W'."""
    parts = (str(query or "").replace(",", " ").split())
    if len(parts) < 2:
        return str(query or "")
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return str(query or "")
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}° {ns} / {abs(lon):.2f}° {ew}"


def _substitute_text(text: str, query: Optional[str]) -> str:
    """Substitute {coords}/{region} placeholders in title/subtitle text."""
    coords = _format_coords_text(query)
    return (
        str(text or "")
        .replace("{coords}", coords)
        .replace("{region}", str(query or ""))
        .replace("{name}", str(query or ""))
    )


def _rounded_rect_verts(
    cx: float, cy: float, hx: float, hy: float, r: float, n_arc: int = 12
) -> np.ndarray:
    """Vertices of a rounded rectangle centred at (cx, cy) with half extents.

    ``r`` is the corner radius; arcs are approximated with ``n_arc`` segments.
    """
    r = max(0.0, min(r, hx, hy))
    verts: List[Tuple[float, float]] = []

    def _arc(scx: float, scy: float, start: float, end: float) -> None:
        t = np.linspace(start, end, n_arc, endpoint=False)
        for a in t:
            verts.append((scx + r * math.cos(a), scy + r * math.sin(a)))

    # Top-left arc, then top-right, bottom-right, bottom-left.
    _arc(cx - hx + r, cy + hy - r, math.pi, 1.5 * math.pi)
    _arc(cx + hx - r, cy + hy - r, 1.5 * math.pi, 2 * math.pi)
    _arc(cx + hx - r, cy - hy + r, 0, 0.5 * math.pi)
    _arc(cx - hx + r, cy - hy + r, 0.5 * math.pi, math.pi)
    return np.asarray(verts, dtype=float)


def apply_creative_output(
    fig: Any,
    ax: Any,
    output: Optional[Dict[str, Any]] = None,
    location_query: Optional[str] = None,
) -> None:
    """Apply creative output controls to an axes before saving.

    - ``output["mask"]``: ``circle`` | ``hexagon`` | ``rounded`` (or ``none``)
      clips the map to a shape centred on the axes; ``mask_options["inset"]``
      shrinks it by a fraction of the map's half-extent.
    - ``output["frame"]``: a stroked outline around the mask edge
      (``width`` in points, ``color``).
    - ``output["title"]`` / ``output["subtitle"]``: text drawn above the map;
      ``{coords}`` and ``{region}`` placeholders are substituted from the
      location query.

    Call immediately before ``fig.savefig`` so the effects are baked in.
    """
    output = output or {}
    mask = (output.get("mask") or "none").strip().lower()
    if mask not in ("circle", "hexagon", "rounded"):
        return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half_w = max(1e-9, (xmax - xmin) / 2.0)
    half_h = max(1e-9, (ymax - ymin) / 2.0)

    opts = output.get("mask_options") or {}
    inset = max(0.0, min(0.9, float(opts.get("inset", 0.0) or 0.0)))

    if mask in ("circle", "hexagon"):
        radius = min(half_w, half_h) * (1.0 - inset)
        if radius <= 0:
            return
        if mask == "circle":
            t = np.linspace(0.0, 2 * math.pi, 129)[:-1]
            verts = np.column_stack(
                (cx + radius * np.cos(t), cy + radius * np.sin(t))
            )
        else:
            angles = np.deg2rad(np.arange(6) * 60.0 + 30.0)
            verts = np.column_stack(
                (cx + radius * np.cos(angles), cy + radius * np.sin(angles))
            )
    else:  # rounded rectangle
        hx = half_w * (1.0 - inset)
        hy = half_h * (1.0 - inset)
        corner = float(opts.get("rounded_radius", 0.15) or 0.15)
        verts = _rounded_rect_verts(cx, cy, hx, hy, corner * min(hx, hy))

    from matplotlib.path import Path as MplPath

    mask_path = MplPath(verts, closed=True)
    clip_patch = mpatches.PathPatch(mask_path, transform=ax.transData)
    for artist in ax.get_children():
        try:
            artist.set_clip_path(clip_patch)
        except (AttributeError, TypeError):
            pass

    # Outside the mask show the figure background (transparent or a colour).
    try:
        ax.patch.set_visible(False)
    except Exception:  # noqa: BLE001 - cosmetic
        pass

    # Frame: stroked outline around the mask edge.
    frame = output.get("frame") or {}
    try:
        fw = float(frame.get("width") or 0.0)
    except (TypeError, ValueError):
        fw = 0.0
    if fw > 0:
        ring = mpatches.PathPatch(
            mask_path,
            transform=ax.transData,
            fill=False,
            edgecolor=frame.get("color") or "#000000",
            linewidth=fw,
            zorder=1000,
        )
        ax.add_patch(ring)

    # Title / subtitle text above the map.
    raw_title = output.get("title")
    raw_subtitle = output.get("subtitle")
    if not raw_title and not raw_subtitle:
        return
    title = _substitute_text(raw_title, location_query)
    subtitle = _substitute_text(raw_subtitle, location_query)

    fig_h_pt = max(1.0, fig.get_size_inches()[1] * 72.0)
    tfs = float(output.get("title_font_size") or 16)
    sfs = float(output.get("subtitle_font_size") or 11)
    title_color = output.get("title_color") or "#000000"
    subtitle_color = output.get("subtitle_color") or "#444444"

    if title:
        ax.text(
            0.5,
            1.03,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=tfs,
            color=title_color,
            zorder=2000,
        )
    if subtitle:
        y = 1.03 if not title else 1.03 - (tfs * 1.45) / fig_h_pt
        ax.text(
            0.5,
            y,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=sfs,
            color=subtitle_color,
            zorder=2000,
        )
