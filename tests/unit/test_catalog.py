"""Unit tests for the region catalog (map_core.core.regions.RegionCatalog)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from map_core.core import regions as r


def _make_catalog(tmp_path):
    cat = r.RegionCatalog(tmp_path / "regions.json")
    pbf = tmp_path / "cork.osm.pbf"
    pbf.write_bytes(b"x" * 100)
    cat.add_from_file("Cork", pbf, source="big.pbf", bbox=(-8.6, 51.8, -8.3, 51.9), inspect=False)
    return cat, pbf


def test_region_from_dict_ignores_unknown_keys():
    region = r.Region.from_dict(
        {"slug": "cork", "name": "Cork", "path": "cork.osm.pbf", "future_field": 1}
    )
    assert region.slug == "cork" and not hasattr(region, "future_field")


def test_add_get_list_remove_round_trip(tmp_path):
    cat, _ = _make_catalog(tmp_path)
    assert cat.get("CORK").name == "Cork"  # slugified lookup
    assert [x.slug for x in cat.list()] == ["cork"]

    reloaded = r.RegionCatalog(tmp_path / "regions.json")
    assert reloaded.get("cork").bbox == (-8.6, 51.8, -8.3, 51.9)

    assert cat.remove("cork") is True
    assert cat.remove("cork") is False
    assert r.RegionCatalog(tmp_path / "regions.json").get("cork") is None


def test_duplicate_add_requires_overwrite(tmp_path):
    cat, pbf = _make_catalog(tmp_path)
    with pytest.raises(FileExistsError):
        cat.add_from_file("Cork", pbf, inspect=False)
    cat.add_from_file("Cork", pbf, inspect=False, overwrite=True)
    assert len(cat.list()) == 1


def test_status_missing_file(tmp_path):
    cat, pbf = _make_catalog(tmp_path)
    assert cat.get("cork").status(cat.dir) == "ok"
    pbf.unlink()
    assert cat.get("cork").status(cat.dir) == "missing"


def test_add_from_file_inspects(tmp_path):
    info = r.PbfInfo(
        path="x", size_bytes=42, bbox=(-1, -2, 3, 4),
        nodes=5, ways=6, relations=7, timestamp=None, generator="osmium",
    )
    cat = r.RegionCatalog(tmp_path / "regions.json")
    pbf = tmp_path / "galway.osm.pbf"
    pbf.write_bytes(b"x")
    with patch.object(r, "pbf_info", return_value=info):
        region = cat.add_from_file("Galway", pbf)
    assert region.size_bytes == 42
    assert region.bbox == (-1, -2, 3, 4)
    assert region.counts == {"nodes": 5, "ways": 6, "relations": 7}


def test_refresh_updates_from_disk(tmp_path):
    cat, pbf = _make_catalog(tmp_path)
    pbf.write_bytes(b"y" * 500)
    info = r.PbfInfo(
        path=str(pbf), size_bytes=500, bbox=(-8.5, 51.7, -8.2, 52.0),
        nodes=1, ways=2, relations=3, timestamp=None, generator="osmium",
    )
    with patch.object(r, "pbf_info", return_value=info):
        region = cat.refresh("cork")
    assert region.size_bytes == 500
    assert region.bbox == (-8.5, 51.7, -8.2, 52.0)
    assert region.counts["ways"] == 2


def test_refresh_unknown_region(tmp_path):
    cat = r.RegionCatalog(tmp_path / "regions.json")
    with pytest.raises(KeyError):
        cat.refresh("nope")


def test_load_tolerates_corrupt_catalog(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text("{not json")
    cat = r.RegionCatalog(path)
    assert cat.list() == []


def test_default_catalog_path():
    from map_core import paths

    assert Path(r.DEFAULT_CATALOG) == paths.region_catalog_path()
    assert Path(r.DEFAULT_CITY_TABLE) == paths.REPO_ROOT / "config" / "regions" / "ireland_cities.json"
