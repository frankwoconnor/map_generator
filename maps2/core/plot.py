"""Plotting utilities for Map Art Generator.

Centralizes figure/axes setup and layer plotting to improve reuse and testability.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import osmnx as ox


def setup_figure_and_axes(
    figure_size: List[float],
    figure_dpi: int,
    background_color: str,
    margin: float,
    transparent: bool = False,
) -> Tuple[Any, Any]:
    """Create a matplotlib figure/axes with common settings.

    Args:
        figure_size: [width, height] in inches.
        figure_dpi: Dots per inch.
        background_color: Figure and axes background color.
        margin: Axes margin fraction.

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    # Set background handling: when transparent, use 'none' facecolors and allow savefig(transparent=True)
    if transparent:
        # Ensure both figure and axes patches are transparent
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
    else:
        ax.set_facecolor(background_color)
    ax.set_axis_off()
    ax.margins(margin)
    fig.tight_layout(pad=0)
    return fig, ax


def plot_map_layer(
    ax: Any,
    layer_name: str,
    data: Any,
    facecolor: Any,
    edgecolor: Any,
    linewidth: float,
    alpha: float,
    hatch: Optional[str] = None,
    linestyle: str = '-',
    zorder: int = 1,
) -> None:
    """Plot a street network or generic GeoDataFrame on the given axis.

    Streets are drawn using the edges GeoDataFrame to avoid background overrides.
    Other layers are plotted as polygons/lines via GeoPandas.
    """
    from maps2.core.util import has_data, log_progress  # local import to avoid cycles

    if not has_data(data):
        return

    if layer_name == 'streets':
        try:
            _, gdf_edges = ox.graph_to_gdfs(data)
            if has_data(gdf_edges):
                gdf_edges.plot(ax=ax, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
        except Exception as e:
            log_progress(f"Warning: failed to plot streets via edges GDF ({e}); falling back to ox.plot_graph.")
            ox.plot_graph(data, ax=ax, show=False, close=False,
                          edge_color=edgecolor, edge_linewidth=linewidth, edge_alpha=alpha, node_size=0)
    else:
        data.plot(ax=ax, fc=facecolor, ec=edgecolor, lw=linewidth, alpha=alpha, hatch=hatch, zorder=zorder)
