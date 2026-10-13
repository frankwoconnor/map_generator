"""Tests for the min-width (sliver) filter in fetch.post_process_features."""

import geopandas as gpd
from shapely.geometry import Polygon

from map_core.core.fetch import post_process_features


def _gdf():
    # A chunky 40x40 square (avg width ~10 m) and a long, thin 100x1 sliver
    # (avg width ~0.5 m). Same projected CRS (EPSG:32630, metres) so no
    # reprojection is applied.
    big = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    sliver = Polygon([(0, 0), (100, 0), (100, 1), (0, 1)])
    return gpd.GeoDataFrame(
        {"name": ["big", "sliver"], "geometry": [big, sliver]},
        crs="EPSG:32630",
    )


def test_min_width_drops_thin_sliver_only():
    gdf = _gdf()
    out = post_process_features(gdf, min_width_m=5.0)
    names = set(out["name"])
    assert "big" in names
    assert "sliver" not in names  # ~2 m wide, below 5 m threshold


def test_min_width_zero_keeps_all():
    out = post_process_features(_gdf(), min_width_m=0.0)
    assert set(out["name"]) == {"big", "sliver"}


def test_lines_are_not_width_filtered():
    from shapely.geometry import LineString

    line = LineString([(0, 0), (100, 0)])
    gdf = gpd.GeoDataFrame({"name": ["road"], "geometry": [line]}, crs="EPSG:32630")
    out = post_process_features(gdf, min_width_m=50.0)
    assert set(out["name"]) == {"road"}  # lines have no width; must survive
