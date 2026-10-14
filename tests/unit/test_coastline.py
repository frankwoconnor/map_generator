"""Tests for the optional coastline overlay loader."""

import os

import geopandas as gpd
import pytest

from map_core.core.coastline import (
    COASTLINE_DEFAULT_PATH,
    has_coastlines,
    load_coastlines,
    resolve_coastline_path,
)

# A coastal bbox around Cork harbour (sea to the south/west).
COASTAL_BBOX = (-8.6, 51.85, -8.35, 51.95)
# A clearly inland bbox (central England).
INLAND_BBOX = (-2.0, 52.0, -1.0, 53.0)


def _shapefile_present() -> bool:
    from map_core import paths

    try:
        return paths.resolve_data_path(COASTLINE_DEFAULT_PATH).exists()
    except Exception:
        return False


requires_shapefile = pytest.mark.skipif(
    not _shapefile_present(),
    reason="coastline shapefile not present",
)


def test_resolve_path_returns_none_or_str():
    path = resolve_coastline_path()
    assert path is None or isinstance(path, str)


@requires_shapefile
def test_coastal_bbox_returns_lines():
    gdf = load_coastlines(COASTAL_BBOX)
    assert gdf is not None
    assert len(gdf) > 0
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326
    # Only line geometries survive the loader.
    assert set(gdf.geometry.type).issubset({"LineString", "MultiLineString"})


@requires_shapefile
def test_inland_bbox_returns_none():
    assert load_coastlines(INLAND_BBOX) is None
    assert has_coastlines(INLAND_BBOX) is False


@requires_shapefile
def test_invalid_bbox_returns_none():
    assert load_coastlines(None) is None
    assert load_coastlines((0, 0, 0, 0)) is None  # zero width
    assert load_coastlines(("a", "b", "c", "d")) is None
