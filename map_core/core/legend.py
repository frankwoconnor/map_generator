"""Legend generation: standalone legend figures and in-axes legends.

Two entry points share the ``_collect_*_legend`` tag-inspection helpers and
:func:`_add_legend_item`:

* :func:`generate_layer_legend` / :func:`create_map_legend` render a dedicated
  legend figure/PNG for one layer (studio "Generate Debug Legends").
* :func:`add_map_legend` draws a legend onto an existing map axes.

Extracted from ``plot.py``; ``setup_figure_and_axes`` is imported lazily to
avoid a module import cycle.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from map_core.core.util import has_data, log_progress


def create_map_legend(
    style: Dict[str, Any],
    layer_data: Dict[str, Any],
    parent: bool = False,
    figure_size: List[float] = None,
    figure_dpi: int = 300,
    background_color: str = "white",
    margin: float = 0.05,
    transparent: bool = False,
) -> Tuple[Any, Any]:
    """Create a legend for the map based on the style configuration and layer data.

    Args:
        style: The style configuration dictionary
        layer_data: Dictionary containing the GeoDataFrames for each layer
        parent: If True, the legend will be created as a parent legend
        figure_size: List of [width, height] in inches
        figure_dpi: Dots per inch
        background_color: Figure and axes background color
        margin: Axes margin fraction
        transparent: If True, use transparent background

    Returns:
        (fig, ax)
    """
    if figure_size is None:
        figure_size = [10, 10]
    from map_core.core.plot import setup_figure_and_axes

    return setup_figure_and_axes(
        figure_size, figure_dpi, background_color, margin, transparent
    )


def _add_legend_item(
    handles: List,
    labels: List,
    color: str,
    edgecolor: str,
    linewidth: float,
    label: str,
    is_line: bool = False,
) -> None:
    """Helper to add a legend item to the handles and labels lists.

    Args:
        handles: List to append the patch/line to
        labels: List to append the label to
        color: Fill color for the patch
        edgecolor: Edge color for the patch
        linewidth: Line width
        label: Text label
        is_line: If True, create a line representation instead of patch
    """
    if is_line:
        item = mpatches.Rectangle(
            (0, 0), 1, 0.1, facecolor=color, edgecolor="none", label=label
        )
    else:
        item = mpatches.Patch(
            facecolor=color, edgecolor=edgecolor, linewidth=linewidth, label=label
        )
    handles.append(item)
    labels.append(label)


def generate_layer_legend(
    layer_name: str,
    data: Any,
    layer_style: Dict[str, Any],
    figure_size: List[float] = None,
    figure_dpi: int = 150,
    background_color: str = "white",
) -> Tuple[Any, Any]:
    """Generate a standalone legend showing unique features for a layer.

    Args:
        layer_name: Name of the layer (e.g., 'water', 'streets', 'buildings')
        data: The layer data (GeoDataFrame or NetworkX graph)
        layer_style: Style configuration for the layer
        figure_size: Figure size for the legend
        figure_dpi: DPI for the legend figure
        background_color: Background color for the legend

    Returns:
        (fig, ax) tuple with the legend figure
    """
    if figure_size is None:
        figure_size = [8, 6]

    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    ax.set_facecolor(background_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    legend_handles = []
    legend_labels = []

    if not has_data(data):
        ax.text(
            0.5,
            0.5,
            f"No data available for {layer_name}",
            ha="center",
            va="center",
            fontsize=12,
        )
        return fig, ax

    try:
        if layer_name == "water":
            # Extract unique water feature types
            if hasattr(data, "geometry"):
                # Group by natural/waterway tags if available
                unique_features = {}
                if "natural" in data.columns:
                    for val in data["natural"].dropna().unique():
                        unique_features[f"Natural: {val}"] = layer_style.get(
                            "facecolor", "#a6cee3"
                        )
                if "waterway" in data.columns:
                    for val in data["waterway"].dropna().unique():
                        unique_features[f"Waterway: {val}"] = layer_style.get(
                            "edgecolor", "#a6cee3"
                        )

                # If no tags, just show generic water
                if not unique_features:
                    unique_features = {
                        "Water features": layer_style.get("facecolor", "#a6cee3")
                    }

                for label, color in unique_features.items():
                    _add_legend_item(
                        legend_handles,
                        legend_labels,
                        color,
                        layer_style.get("edgecolor", color),
                        layer_style.get("linewidth", 0.5),
                        label,
                    )

        elif layer_name == "streets":
            # Extract unique highway types from street network
            try:
                # Convert graph to edges GeoDataFrame
                if hasattr(data, "edges"):
                    nodes, edges = ox.graph_to_gdfs(data)
                    if "highway" in edges.columns:
                        unique_highways = edges["highway"].dropna().unique()
                        # Handle list values
                        highway_types = set()
                        for hw in unique_highways:
                            if isinstance(hw, list):
                                highway_types.update(hw)
                            else:
                                highway_types.add(hw)

                        for hw_type in sorted(highway_types):
                            _add_legend_item(
                                legend_handles,
                                legend_labels,
                                layer_style.get("edgecolor", "#000000"),
                                "none",
                                0.5,
                                hw_type.capitalize(),
                                is_line=True,
                            )
            except Exception as e:
                log_progress(f"Warning: Could not extract street types: {e}")
                _add_legend_item(
                    legend_handles,
                    legend_labels,
                    layer_style.get("edgecolor", "#000000"),
                    "none",
                    0.5,
                    "Streets",
                    is_line=True,
                )

        elif layer_name == "buildings":
            # Show building color scheme
            auto_mode = layer_style.get("auto_style_mode", "manual")
            if auto_mode == "manual":
                color = layer_style.get("manual_color_settings", {}).get(
                    "facecolor", "#000000"
                )
                _add_legend_item(
                    legend_handles,
                    legend_labels,
                    color,
                    layer_style.get("edgecolor", "#000000"),
                    layer_style.get("linewidth", 0.5),
                    "Buildings",
                )
            elif auto_mode in ["auto_size", "auto_distance"]:
                # Show the palette colors
                from map_core.core.palettes import load_palettes

                palette_key = (
                    "auto_size_palette"
                    if auto_mode == "auto_size"
                    else "auto_distance_palette"
                )
                palette_name = layer_style.get(palette_key, "")
                palettes = load_palettes()
                colors = (
                    palettes.get(palette_name, []) if isinstance(palettes, dict) else []
                )

                if colors:
                    colors_reversed = list(colors)[::-1]
                    labels = ["Smallest" if auto_mode == "auto_size" else "Farthest"]
                    labels.extend([""] * (len(colors_reversed) - 2))
                    labels.append("Largest" if auto_mode == "auto_size" else "Nearest")

                    for color, label in zip(colors_reversed, labels):
                        _add_legend_item(
                            legend_handles,
                            legend_labels,
                            color,
                            layer_style.get("edgecolor", "#000000"),
                            layer_style.get("linewidth", 0.5),
                            label if label else "",
                        )
            elif auto_mode == "manual_floorsize":
                categories = layer_style.get("size_categories", [])
                for cat in categories:
                    min_area = cat.get("min_area", 0)
                    max_area = cat.get("max_area", "∞")
                    label = f"{min_area}-{max_area} m²"
                    _add_legend_item(
                        legend_handles,
                        legend_labels,
                        cat.get("facecolor", "#000000"),
                        layer_style.get("edgecolor", "#000000"),
                        layer_style.get("linewidth", 0.5),
                        label,
                    )

        elif layer_name == "green":
            # Check for leisure/landuse/natural tags
            unique_features = {}
            if hasattr(data, "geometry"):
                for tag in ["leisure", "landuse", "natural"]:
                    if tag in data.columns:
                        for val in data[tag].dropna().unique():
                            unique_features[f"{tag.capitalize()}: {val}"] = (
                                layer_style.get("facecolor", "#b2df8a")
                            )

                if not unique_features:
                    unique_features = {
                        "Green spaces": layer_style.get("facecolor", "#b2df8a")
                    }

                for label, color in unique_features.items():
                    _add_legend_item(
                        legend_handles,
                        legend_labels,
                        color,
                        layer_style.get("edgecolor", color),
                        layer_style.get("linewidth", 0.5),
                        label,
                    )

        # Create the legend
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title=f"{layer_name.capitalize()} Legend",
                loc="center",
                fontsize=10,
                title_fontsize=14,
                frameon=True,
                fancybox=True,
                shadow=True,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"No legend items for {layer_name}",
                ha="center",
                va="center",
                fontsize=12,
            )

    except Exception as e:
        log_progress(f"Error generating legend for {layer_name}: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error generating legend:\n{str(e)}",
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
        )

    return fig, ax


def add_map_legend(ax: Any, layer_data: Dict[str, Any], style: Dict[str, Any]) -> None:
    """Add a legend to the map based on the style configuration and layer data.

    Args:
        ax: The matplotlib axes to add the legend to
        layer_data: Dictionary containing the GeoDataFrames for each layer
        style: The style configuration dictionary
    """
    if not style.get("legend", {}).get("enabled", True):
        return

    legend_config = style.get("legend", {})
    legend_handles = []

    # Add water features to legend
    if "water" in layer_data and has_data(layer_data["water"]):
        water_style = style.get("layers", {}).get("water", {})

        # Add water bodies (polygons)
        water_patches = []
        if "filters" in water_style and "natural" in water_style["filters"]:
            for water_type in water_style["filters"]["natural"]:
                if water_type in ["lake", "pond", "water", "reservoir"]:
                    patch = mpatches.Patch(
                        facecolor=water_style.get("facecolor", "#479711"),
                        edgecolor=water_style.get("edgecolor", "#f70202"),
                        linewidth=water_style.get("linewidth", 1.0),
                        label=f"{water_type.capitalize()}",
                    )
                    water_patches.append(patch)

        # Add waterways (lines)
        if "filters" in water_style and "waterway" in water_style["filters"]:
            for waterway_type in water_style["filters"]["waterway"]:
                if waterway_type in ["river", "stream", "canal"]:
                    line = mpatches.Rectangle(
                        (0, 0),
                        1,
                        0.2,  # width, height
                        facecolor=water_style.get("edgecolor", "#479711"),
                        edgecolor="none",
                        label=f"{waterway_type.capitalize()}",
                    )
                    water_patches.append(line)

        if water_patches:
            legend_handles.extend(water_patches)

    # Add green spaces to legend
    if "green" in layer_data and has_data(layer_data["green"]):
        green_style = style.get("layers", {}).get("green", {})
        if green_style.get("enabled", False):
            patch = mpatches.Patch(
                facecolor=green_style.get("facecolor", "#4CAF50"),
                edgecolor=green_style.get("edgecolor", "#000000"),
                linewidth=green_style.get("linewidth", 0.5),
                label="Green Space",
            )
            legend_handles.append(patch)

    # Add buildings to legend
    if "buildings" in layer_data and has_data(layer_data["buildings"]):
        building_style = style.get("layers", {}).get("buildings", {})
        if building_style.get("enabled", False):
            patch = mpatches.Patch(
                facecolor=building_style.get("facecolor", "#808080"),
                edgecolor=building_style.get("edgecolor", "#000000"),
                linewidth=building_style.get("linewidth", 0.5),
                label="Buildings",
            )
            legend_handles.append(patch)

    # Add streets to legend
    if "streets" in layer_data and has_data(layer_data["streets"]):
        street_style = style.get("layers", {}).get("streets", {})
        if street_style.get("enabled", False):
            line = mpatches.Rectangle(
                (0, 0),
                1,
                0.1,  # width, height
                facecolor=street_style.get("edgecolor", "#000000"),
                edgecolor="none",
                label="Streets",
            )
            legend_handles.append(line)

    # Only add legend if we have items to show
    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            title=legend_config.get("title", "Map Legend"),
            loc=legend_config.get("position", "lower right"),
            fontsize=legend_config.get("font_size", 10),
            title_fontsize=legend_config.get("title_size", 12),
            frameon=legend_config.get("fancybox", True),
            fancybox=legend_config.get("fancybox", True),
            shadow=legend_config.get("shadow", True),
            borderpad=legend_config.get("borderpad", 0.8),
            labelspacing=legend_config.get("labelspacing", 0.8),
            handlelength=legend_config.get("handlelength", 1.5),
            handletextpad=legend_config.get("handletextpad", 0.8),
            borderaxespad=legend_config.get("borderaxespad", 0.8),
        )

        # Set legend background color and transparency
        frame = legend.get_frame()
        frame.set_facecolor(legend_config.get("background_color", "#ffffff"))
        frame.set_edgecolor(legend_config.get("edge_color", "#000000"))
        frame.set_alpha(legend_config.get("alpha", 0.9))



# Legend-only collection functions (no plotting)
def _collect_water_legend(data, legend_handles, legend_labels):
    """Collect water feature types for legend without plotting."""
    if not hasattr(data, "geometry"):
        return

    cmap = cm.get_cmap("tab20")
    feature_types = {}
    color_idx = 0

    for tag in ["natural", "waterway"]:
        if tag in data.columns:
            for val in data[tag].dropna().unique():
                feature_key = f"{tag}={val}"
                if feature_key not in feature_types:
                    feature_types[feature_key] = cmap(color_idx % 20)
                    color_idx += 1

    for feature_key, color in feature_types.items():
        legend_handles.append(mpatches.Patch(color=color))
        legend_labels.append(feature_key)


def _collect_streets_legend(data, legend_handles, legend_labels):
    """Collect street highway types for legend without plotting."""
    try:
        edges = None
        if isinstance(data, gpd.GeoDataFrame):
            edges = data
        elif isinstance(data, (nx.MultiDiGraph, nx.MultiGraph, nx.DiGraph, nx.Graph)):
            _, edges = ox.graph_to_gdfs(data)
        else:
            log_progress(
                f"Warning: Expected NetworkX graph or GeoDataFrame for streets legend, got {type(data)}"
            )
            return

        if not has_data(edges):
            log_progress("[debug] No street edges for legend.")
            return

        if "highway" not in edges.columns or edges["highway"].dropna().empty:
            log_progress(
                "[debug] 'highway' column not found or empty in street edges. Adding generic legend entry."
            )
            legend_handles.append(mpatches.Patch(color="gray"))
            legend_labels.append("Streets (no type info)")
            return

        cmap = cm.get_cmap("Set3")
        highway_types = {}
        color_idx = 0

        for hw in edges["highway"].dropna().unique():
            if isinstance(hw, list):
                for h in hw:
                    if h not in highway_types:
                        highway_types[h] = cmap(color_idx % 12)
                        color_idx += 1
            else:
                if hw not in highway_types:
                    highway_types[hw] = cmap(color_idx % 12)
                    color_idx += 1

        for hw_type, color in sorted(highway_types.items()):
            legend_handles.append(mpatches.Patch(color=color))
            legend_labels.append(hw_type)

    except Exception as e:
        log_progress(f"Warning: Could not collect streets legend: {e}")


def _collect_buildings_legend(data, legend_handles, legend_labels):
    """Collect building types for legend without plotting."""
    if not hasattr(data, "geometry"):
        return

    cmap = cm.get_cmap("Paired")
    building_types = {}
    color_idx = 0

    if "building" in data.columns:
        for val in data["building"].dropna().unique():
            if val not in building_types:
                building_types[val] = cmap(color_idx % 12)
                color_idx += 1

    if building_types:
        for btype, color in sorted(building_types.items()):
            legend_handles.append(mpatches.Patch(color=color))
            legend_labels.append(f"building={btype}")
    else:
        legend_handles.append(mpatches.Patch(color="#ff7f0e"))
        legend_labels.append("buildings")


def _collect_green_legend(data, legend_handles, legend_labels):
    """Collect green space types for legend without plotting."""
    if not hasattr(data, "geometry"):
        return

    cmap = cm.get_cmap("Set2")
    feature_types = {}
    color_idx = 0

    for tag in ["leisure", "landuse", "natural"]:
        if tag in data.columns:
            for val in data[tag].dropna().unique():
                feature_key = f"{tag}={val}"
                if feature_key not in feature_types:
                    feature_types[feature_key] = cmap(color_idx % 8)
                    color_idx += 1

    if feature_types:
        for feature_key, color in feature_types.items():
            legend_handles.append(mpatches.Patch(color=color))
            legend_labels.append(feature_key)
    else:
        legend_handles.append(mpatches.Patch(color="#90ee90"))
