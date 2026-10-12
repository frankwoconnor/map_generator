"""Unit tests for map_core.paths - the single filesystem resolver."""

from pathlib import Path

import pytest

from map_core import paths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MAPGEN_DATA_DIR", raising=False)


def test_default_data_root_is_repo_parent():
    assert paths.REPO_ROOT == Path(__file__).resolve().parents[2]
    assert paths.data_root() == paths.REPO_ROOT.parent


def test_env_override_relocates_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPGEN_DATA_DIR", str(tmp_path))
    root = tmp_path.resolve()
    assert paths.data_root() == root
    assert paths.sources_dir() == root / "data" / "sources"
    assert paths.regions_dir() == root / "data" / "regions"
    assert paths.region_catalog_path() == root / "data" / "regions" / "regions.json"
    assert paths.clips_dir() == root / "cache" / "clips"
    assert paths.features_dir() == root / "cache" / "features"
    assert paths.osmnx_cache_dir() == root / "cache" / "osmnx"
    assert paths.outputs_dir() == root / "output"


def test_env_override_expands_user(monkeypatch):
    monkeypatch.setenv("MAPGEN_DATA_DIR", "~/mapgen-test")
    assert "~" not in str(paths.data_root())


def test_ensure_dirs_creates_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPGEN_DATA_DIR", str(tmp_path / "root"))
    paths.ensure_dirs()
    for getter in paths._DIR_GETTERS.values():
        assert getter().is_dir()
    # Idempotent
    paths.ensure_dirs()


def test_resolve_output_dir_anchors_to_repo_root(tmp_path):
    # A sibling path like "../output" must resolve against the repo root, not CWD.
    assert paths.resolve_output_dir("../output") == (paths.REPO_ROOT / "../output").resolve()
    assert paths.resolve_output_dir("output") == (paths.REPO_ROOT / "output").resolve()
    absolute = tmp_path / "abs"
    assert paths.resolve_output_dir(absolute) == absolute


def test_resolve_data_path_finds_file_in_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPGEN_DATA_DIR", str(tmp_path))
    src = paths.sources_dir()
    src.mkdir(parents=True)
    pbf = src / "somewhere.osm.pbf"
    pbf.write_bytes(b"x")
    # A bare filename should be found under sources/, regardless of CWD.
    assert paths.resolve_data_path("somewhere.osm.pbf") == pbf


def test_resolve_data_path_returns_cwd_candidate_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPGEN_DATA_DIR", str(tmp_path / "root"))
    monkeypatch.chdir(tmp_path)
    result = paths.resolve_data_path("does-not-exist.pbf")
    assert result == (tmp_path / "does-not-exist.pbf")


def test_describe_reports_layout_and_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPGEN_DATA_DIR", str(tmp_path))
    paths.ensure_dirs()
    info = paths.describe()
    assert info["data_root"] == str(tmp_path.resolve())
    assert info["env_override"] == str(tmp_path)
    assert set(info["dirs"]) == {"sources", "regions", "clips", "features", "osmnx", "output"}
    assert isinstance(info["free_space_bytes"], int)
    assert isinstance(info["osmium_available"], bool)
