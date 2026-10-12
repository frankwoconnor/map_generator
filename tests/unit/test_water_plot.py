"""Tests for waterway line plotting vectorization."""

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, MultiLineString

matplotlib.use("Agg")

from map_core.core.plot import _line_collection_for_waterways


def _make_gdf(geoms, widths=None):
    data = {"geometry": geoms}
    if widths is not None:
        data["width"] = widths
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def test_line_collection_for_linestrings():
    geoms = [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
    gdf = _make_gdf(geoms, widths=["10", "3m"])

    lc = _line_collection_for_waterways(
        gdf,
        default_width_m=5.0,
        min_linewidth=0.5,
        meters_per_point=5.0,
        color="#5599cc",
        alpha=1.0,
        zorder=1,
    )

    assert isinstance(lc, LineCollection)
    assert len(lc.get_segments()) == 2
    # widths: 10/5=2.0, 3/5=0.6 -> max with min 0.5 -> [2.0, 0.6]
    assert list(lc.get_linewidths()) == pytest.approx([2.0, 0.6])


def test_line_collection_for_multilinestring():
    geom = MultiLineString(
        [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
    )
    gdf = _make_gdf([geom])

    lc = _line_collection_for_waterways(
        gdf,
        default_width_m=5.0,
        min_linewidth=0.5,
        meters_per_point=5.0,
        color="#5599cc",
        alpha=1.0,
        zorder=1,
    )

    assert isinstance(lc, LineCollection)
    # MultiLineString produces one segment per part
    assert len(lc.get_segments()) == 2


def test_line_collection_empty_gdf():
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    lc = _line_collection_for_waterways(
        gdf,
        default_width_m=5.0,
        min_linewidth=0.5,
        meters_per_point=5.0,
        color="#5599cc",
        alpha=1.0,
        zorder=1,
    )
    assert lc is None


def test_line_collection_adds_to_axes():
    geoms = [LineString([(0, 0), (1, 1)])]
    gdf = _make_gdf(geoms)
    lc = _line_collection_for_waterways(
        gdf,
        default_width_m=5.0,
        min_linewidth=0.5,
        meters_per_point=5.0,
        color="#5599cc",
        alpha=1.0,
        zorder=1,
    )

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale_view()
    collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(collections) == 1
    plt.close(fig)
