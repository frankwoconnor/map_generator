import main
import pytest
gpd = pytest.importorskip("geopandas")


def test_has_data_none():
    assert main.has_data(None) is False


def test_has_data_empty_gdf():
    gdf = gpd.GeoDataFrame(geometry=[])
    assert main.has_data(gdf) is False


def test_has_data_nonempty_gdf():
    gdf = gpd.GeoDataFrame(geometry=[None])
    # Non-empty length, even if geometry is None; function only checks emptiness
    assert main.has_data(gdf) is True
