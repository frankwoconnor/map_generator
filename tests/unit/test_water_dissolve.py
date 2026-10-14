"""Tests for the topology-safe water dissolve/heal pipeline."""

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Polygon

from map_core.core.water import clean_water_layer

CRS = "EPSG:32630"  # UTM metres


def _areas(*polys):
    return gpd.GeoDataFrame({"geometry": list(polys)}, crs=CRS)


def test_adjacent_polygons_dissolve_into_one():
    a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    b = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])  # shares edge with a
    areas, _ = clean_water_layer(_areas(a, b), seam_heal_m=0.0)
    geom = areas.geometry.iloc[0]
    assert geom.geom_type in ("Polygon", "MultiPolygon")
    # unioned area ~200 (set_precision snaps to a 0.05 m grid -> sub-metre drift)
    assert abs(geom.area - 200.0) < 2.0
    # single topology -> 1 polygon part
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    assert len(parts) == 1


def test_sub_heal_gap_closes_big_gap_stays():
    near = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    close = Polygon([(10.1, 0), (20, 0), (20, 10), (10.1, 10)])  # 0.1 m gap
    far = Polygon([(0, 20), (10, 20), (10, 30), (0, 30)])       # 10 m gap
    areas, _ = clean_water_layer(_areas(near, close, far), seam_heal_m=0.15)
    geom = areas.geometry.iloc[0]
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    # near(+close) healed together; far stays separate -> 2 parts
    assert len(parts) == 2


def test_min_area_drops_tiny():
    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])       # 100 m2
    tiny = Polygon([(50, 50), (50.3, 50), (50.3, 50.3), (50, 50.3)])  # 0.09 m2
    areas, _ = clean_water_layer(_areas(big, tiny), seam_heal_m=0.0, min_area_m2=1.0)
    geom = areas.geometry.iloc[0]
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    assert len(parts) == 1
    assert areas.geometry.iloc[0].area >= 99.0


def test_min_width_drops_thin_sliver():
    chunky = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])     # avg width ~10 m
    sliver = Polygon([(0, 100), (100, 100), (100, 101), (0, 101)])  # ~1 m wide
    areas, _ = clean_water_layer(_areas(chunky, sliver), seam_heal_m=0.0, min_width_m=5.0)
    geom = areas.geometry.iloc[0]
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    assert len(parts) == 1  # sliver dropped, chunky kept


def test_lines_separated_with_kind():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    river = LineString([(-5, -5), (30, 30)])
    gdf = gpd.GeoDataFrame(
        {"waterway": [None, "river"], "geometry": [poly, river]}, crs=CRS
    )
    areas, lines = clean_water_layer(gdf, seam_heal_m=0.0)
    assert len(areas) and not areas.geometry.iloc[0].is_empty
    assert len(lines) == 1
    assert set(lines["waterway"]) == {"river"}
    # both outputs back in the source CRS
    assert areas.crs.to_epsg() == int(CRS.split(":")[1])
    assert lines.crs.to_epsg() == int(CRS.split(":")[1])


def test_empty_input_ok():
    areas, lines = clean_water_layer(gpd.GeoDataFrame(geometry=[], crs=CRS))
    assert len(areas) == 0
    assert len(lines) == 0
