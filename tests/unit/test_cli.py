"""Unit tests for the mapgen CLI (argument parsing and command construction)."""

import json
from unittest.mock import patch

import pytest

from map_core.core import regions as r
from mapgen import __main__ as m


def test_parser_info():
    args = m.build_parser().parse_args(["info", "x.pbf", "--json"])
    assert args.func is m.cmd_info
    assert args.pbf == "x.pbf" and args.json is True


def test_parser_generate_repeatable_set():
    args = m.build_parser().parse_args(
        ["generate", "--region", "cork", "--set", "a.b=1", "--set", "c=2"]
    )
    assert args.set == ["a.b=1", "c=2"]
    assert args.func is m.cmd_generate


def test_parser_regions_list_uses_default_catalog():
    args = m.build_parser().parse_args(["regions", "list"])
    assert args.catalog == str(r.DEFAULT_CATALOG)
    assert args.func is m.cmd_regions_list


def test_cmd_info_text_and_json(capsys):
    info = r.PbfInfo(
        path="x.pbf", size_bytes=1024 * 1024, bbox=(-1, -2, 3, 4),
        nodes=10, ways=5, relations=1, timestamp="2025-01-01T00:00:00Z", generator="osmium",
    )
    with patch.object(m.R, "pbf_info", return_value=info):
        assert m.cmd_info(m.build_parser().parse_args(["info", "x.pbf"])) == 0
        out = capsys.readouterr().out
        assert "1.0 MB" in out and "nodes:" in out

        assert m.cmd_info(m.build_parser().parse_args(["info", "x.pbf", "--json"])) == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["nodes"] == 10 and doc["size_mb"] == 1.0


def _catalog_with_cork(tmp_path):
    cat = r.RegionCatalog(tmp_path / "regions.json")
    pbf = tmp_path / "cork.osm.pbf"
    pbf.write_bytes(b"x" * 10)
    cat.add_from_file("Cork", pbf, bbox=(-8.6, 51.8, -8.3, 51.9), inspect=False)
    return cat


def test_cmd_regions_list(tmp_path, capsys):
    _catalog_with_cork(tmp_path)
    args = m.build_parser().parse_args(
        ["regions", "list", "--catalog", str(tmp_path / "regions.json")]
    )
    assert m.cmd_regions_list(args) == 0
    out = capsys.readouterr().out
    assert "cork" in out and "Cork" in out and "ok" in out


def test_cmd_regions_list_empty(tmp_path, capsys):
    args = m.build_parser().parse_args(
        ["regions", "list", "--catalog", str(tmp_path / "regions.json")]
    )
    assert m.cmd_regions_list(args) == 0
    assert "No regions" in capsys.readouterr().out


def test_cmd_regions_info_unknown(tmp_path, capsys):
    args = m.build_parser().parse_args(
        ["regions", "info", "nope", "--catalog", str(tmp_path / "regions.json")]
    )
    assert m.cmd_regions_info(args) == 1
    assert "No such region" in capsys.readouterr().err


def test_cmd_generate_builds_command(tmp_path):
    _catalog_with_cork(tmp_path)
    style = tmp_path / "style.json"
    style.write_text(json.dumps({"layers": {"streets": {"linewidth": 0.5}}}))

    captured = {}

    def fake_call(cmd, cwd=None):
        captured["cmd"] = cmd
        if "--config" in cmd:
            captured["config"] = json.loads(open(cmd[cmd.index("--config") + 1]).read())
        return 0

    args = m.build_parser().parse_args(
        [
            "generate",
            "--catalog", str(tmp_path / "regions.json"),
            "--region", "cork",
            "--out", "OUT", "--prefix", "p",
            "--layers", "streets,water",
            "--set", "layers.streets.linewidth=2",
        ]
    )
    with patch.object(m, "STYLE_PATH", style), patch.object(
        m.subprocess, "call", side_effect=fake_call
    ):
        assert m.cmd_generate(args) == 0

    cmd = captured["cmd"]
    assert "--pbf" in cmd
    pbf_path = cmd[cmd.index("--pbf") + 1]
    assert pbf_path.endswith("cork.osm.pbf")
    # dist derived from the region bbox height (0.1 deg * 111320 / 2)
    assert "--dist" in cmd
    assert float(cmd[cmd.index("--dist") + 1]) == pytest.approx(5566.0)
    assert cmd[cmd.index("--out") + 1] == "OUT"
    assert cmd[cmd.index("--layers") + 1] == "streets,water"
    assert "--config" in cmd
    assert captured["config"]["layers"]["streets"]["linewidth"] == 2
    # temp config cleaned up
    import os

    config_path = cmd[cmd.index("--config") + 1]
    assert not os.path.exists(config_path)


def test_cmd_generate_requires_source(capsys):
    args = m.build_parser().parse_args(["generate", "--out", "OUT"])
    assert m.cmd_generate(args) == 1
    assert "Provide --region" in capsys.readouterr().err
