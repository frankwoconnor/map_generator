"""Regression tests for the point-feature filter.

OSM feature queries can return node-tagged points inside polygon layers.
When they reach the plotter they render as stray circle markers in the SVG
(matplotlib scatter glyphs). ``drop_point_features`` must strip them from
every GeoDataFrame before plotting while leaving lines/polygons and graph
data untouched.
"""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from map_core.core.util import drop_point_features, has_data


def _gdf(*geoms):
    return gpd.GeoDataFrame(geometry=list(geoms), crs="EPSG:4326")


def test_drop_point_features_removes_points_only():
    gdf = _gdf(
        Point(0, 0),
        Polygon([(0, 0), (1, 0), (1, 1)]),
        LineString([(0, 0), (1, 1)]),
    )
    out = drop_point_features(gdf, "green")
    assert list(out.geom_type) == ["Polygon", "LineString"]
    assert len(out) == 2


def test_drop_point_features_handles_multipoint():
    from shapely.geometry import MultiPoint

    gdf = _gdf(
        MultiPoint([(0, 0), (1, 1)]),
        Polygon([(0, 0), (1, 0), (1, 1)]),
    )
    out = drop_point_features(gdf, "buildings")
    assert list(out.geom_type) == ["Polygon"]


def test_drop_point_features_leaves_pure_polygon_layer_unchanged():
    gdf = _gdf(Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 2), (3, 2), (3, 3)]))
    out = drop_point_features(gdf, "water")
    assert len(out) == 2


def test_drop_point_features_passes_non_gdf_through():
    graph_like = object()
    assert drop_point_features(graph_like, "streets") is graph_like
    assert drop_point_features(None, "streets") is None
    assert not has_data(drop_point_features(_gdf(Point(0, 0)), "green"))


def test_drop_point_features_normalises_empty_and_null():
    gdf = _gdf(Point(0, 0), Polygon([(0, 0), (1, 0), (1, 1)]))
    out = drop_point_features(gdf, "green")
    assert list(out.geom_type) == ["Polygon"]