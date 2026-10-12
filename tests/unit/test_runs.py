"""Unit tests for map_core.runs (output-directory discovery)."""

import json
from pathlib import Path

import pytest

from map_core import runs


@pytest.fixture(autouse=True)
def _data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPGEN_DATA_DIR", str(tmp_path))
    return tmp_path


def _make_run(base: Path, name: str, *, combined=True, layers=("streets", "water")):
    run = base / name
    run.mkdir(parents=True)
    for layer in layers:
        (run / f"{name}_{layer}.svg").write_text("<svg/>")
    if combined:
        (run / f"{name}_combined.svg").write_text("<svg/>")
        (run / f"{name}_combined.png").write_bytes(b"\x89PNG")
    (run / "config.json").write_text(
        json.dumps({"location": {"query": "51.9 -8.47", "pbf_path": "/data/regions/cork.osm.pbf"}})
    )
    return run


def test_parse_run_timestamp():
    assert runs.parse_run_timestamp("map_20260101_120000") is not None
    assert runs.parse_run_timestamp("map_20260101_120000").year == 2026
    assert runs.parse_run_timestamp("cli_cork") is None
    assert runs.parse_run_timestamp("no_stamp_here_2026") is None


def test_region_name_strips_map_suffixes():
    assert runs.region_name("/data/regions/cork.osm.pbf") == "cork"
    assert runs.region_name("ireland-and-northern-ireland-latest.osm.pbf") == (
        "ireland-and-northern-ireland-latest"
    )
    assert runs.region_name("cork.pbf") == "cork"
    assert runs.region_name(None) is None
    assert runs.region_name("") is None


def test_discover_runs_newest_first_with_layers(_data_root):
    base = _data_root / "output"
    _make_run(base, "map_20260101_120000")
    _make_run(base, "map_20260102_090000", layers=("streets",))

    records = runs.discover_runs()
    assert [r["id"] for r in records] == [
        "map_20260102_090000",
        "map_20260101_120000",
    ]
    top = records[0]
    assert top["combined_svg"].endswith("_combined.svg")
    assert top["combined_png"].endswith("_combined.png")
    assert top["thumbnail"].endswith(".png")
    assert [l["name"] for l in top["layers"]] == ["streets"]
    assert top["region"] == "cork"
    assert top["data_source"] is None


def test_discover_runs_ignores_empty_dirs(_data_root):
    base = _data_root / "output"
    (base / "not_a_run").mkdir(parents=True)  # no timestamp, no artifacts
    _make_run(base, "map_20260101_120000")

    records = runs.discover_runs()
    assert [r["id"] for r in records] == ["map_20260101_120000"]


def test_discover_runs_limit(_data_root):
    base = _data_root / "output"
    for day in (1, 2, 3):
        _make_run(base, f"map_2026010{day}_120000")
    assert len(runs.discover_runs(limit=2)) == 2


def test_discover_runs_missing_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPGEN_DATA_DIR", str(tmp_path / "empty"))
    assert runs.discover_runs() == []
