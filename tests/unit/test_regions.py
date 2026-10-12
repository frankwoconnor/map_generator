"""Unit tests for map_core.core.regions (osmium is mocked)."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from map_core.core import regions as r


def test_bbox_from_center_is_square_and_contains_center():
    w, s, e, n = r.bbox_from_center(51.9, -8.47, 1000)
    assert w < -8.47 < e
    assert s < 51.9 < n
    # ~1 km half-width in latitude degrees
    assert abs((n - s) / 2 - 1000 / 111_320) < 1e-6
    # longitude span wider than latitude span at 52N
    assert (e - w) > (n - s)


def test_bbox_from_center_margin_grows_box():
    base = r.bbox_from_center(51.9, -8.47, 1000)
    grown = r.bbox_from_center(51.9, -8.47, 1000, margin=0.5)
    assert grown[0] < base[0] and grown[2] > base[2]
    assert grown[1] < base[1] and grown[3] > base[3]


def test_normalize_bbox_fixes_order():
    assert r.normalize_bbox((2, 4, 1, 3)) == (1.0, 3.0, 2.0, 4.0)


def test_bbox_to_osmium_format():
    assert r.bbox_to_osmium((-8.5, 51.8, -8.4, 51.9)) == "-8.5000000,51.8000000,-8.4000000,51.9000000"


@pytest.mark.parametrize(
    "name,slug",
    [("Cork", "cork"), ("Dún Laoghaire", "d-n-laoghaire"), ("  Derry / Londonderry ", "derry-londonderry"), ("", "region")],
)
def test_slugify(name, slug):
    assert r.slugify(name) == slug


def _fileinfo_doc(header_boxes=None, data_bbox=None):
    return {
        "file": {"name": "x.pbf", "format": "PBF", "size": 1234},
        "header": {
            "boxes": header_boxes or [],
            "option": {"generator": "osmium/1.19.1", "osmosis_replication_timestamp": "2025-08-08T20:20:30Z"},
        },
        "data": {
            "bbox": data_bbox,
            "timestamp": {"first": "", "last": "2025-08-08T19:54:57Z"},
            "count": {"changesets": 0, "nodes": 10, "ways": 5, "relations": 1},
        },
    }


def test_pbf_info_prefers_header_bbox(tmp_path):
    pbf = tmp_path / "a.pbf"
    pbf.write_bytes(b"x")
    doc = _fileinfo_doc(header_boxes=[[-8.6, 51.8, -8.3, 51.9]], data_bbox=[-10, 51, -5, 55])
    with patch.object(r, "_require_osmium", return_value="osmium"), patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(doc), stderr=""),
    ):
        info = r.pbf_info(pbf)
    assert info.bbox == (-8.6, 51.8, -8.3, 51.9)
    assert (info.nodes, info.ways, info.relations) == (10, 5, 1)
    assert info.timestamp == "2025-08-08T20:20:30Z"
    assert info.size_bytes == 1234
    assert info.to_dict()["size_mb"] == 0.0


def test_pbf_info_falls_back_to_data_bbox(tmp_path):
    pbf = tmp_path / "a.pbf"
    pbf.write_bytes(b"x")
    doc = _fileinfo_doc(data_bbox=[-8.57, 51.89, -8.44, 51.90])
    with patch.object(r, "_require_osmium", return_value="osmium"), patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(doc), stderr=""),
    ):
        info = r.pbf_info(pbf)
    assert info.bbox == (-8.57, 51.89, -8.44, 51.90)


def test_pbf_info_missing_file(tmp_path):
    with patch.object(r, "_require_osmium", return_value="osmium"):
        with pytest.raises(FileNotFoundError):
            r.pbf_info(tmp_path / "missing.pbf")


def test_extract_bbox_builds_expected_command(tmp_path):
    src = tmp_path / "src.pbf"
    src.write_bytes(b"x")
    dst = tmp_path / "out" / "cork.osm.pbf"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"y")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch.object(r, "_require_osmium", return_value="osmium"), patch(
        "subprocess.run", side_effect=fake_run
    ):
        out = r.extract_bbox(src, (-8.6, 51.8, -8.3, 51.9), dst, progress_cb=lambda m: None)

    assert out == dst and dst.exists()
    cmd = calls[0]
    assert cmd[:2] == ["osmium", "extract"]
    assert "--bbox" in cmd and cmd[cmd.index("--bbox") + 1] == "-8.6000000,51.8000000,-8.3000000,51.9000000"
    assert "--strategy=smart" in cmd and "--set-bounds" in cmd
    assert cmd[-1] == str(src)


def test_extract_bbox_refuses_overwrite_by_default(tmp_path):
    src = tmp_path / "src.pbf"
    src.write_bytes(b"x")
    dst = tmp_path / "exists.pbf"
    dst.write_bytes(b"y")
    with patch.object(r, "_require_osmium", return_value="osmium"):
        with pytest.raises(FileExistsError):
            r.extract_bbox(src, (-8.6, 51.8, -8.3, 51.9), dst)


def test_extract_bbox_reports_osmium_failure(tmp_path):
    src = tmp_path / "src.pbf"
    src.write_bytes(b"x")
    with patch.object(r, "_require_osmium", return_value="osmium"), patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            r.extract_bbox(src, (-8.6, 51.8, -8.3, 51.9), tmp_path / "o.pbf", progress_cb=lambda m: None)


def test_require_osmium_missing():
    with patch.object(r, "osmium_path", return_value=None):
        with pytest.raises(r.OsmiumNotAvailable, match="brew install osmium-tool"):
            r._require_osmium()
