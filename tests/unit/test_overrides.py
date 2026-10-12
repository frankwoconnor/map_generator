"""Unit tests for mapgen.overrides."""

import pytest

from mapgen.overrides import apply_overrides, coerce, set_path


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("False", False),
        ("on", True),
        ("off", False),
        ("null", None),
        ("none", None),
        ("5000", 5000),
        ("-3", -3),
        ("0.5", 0.5),
        ("1e3", 1000.0),
        ("#ff0066", "#ff0066"),
        ("primary", "primary"),
        ("[1, 2, 3]", [1, 2, 3]),
        ('{"a": 1}', {"a": 1}),
    ],
)
def test_coerce(raw, expected):
    result = coerce(raw)
    assert result == expected
    assert type(result) is type(expected)


def test_set_path_nested():
    style = {"layers": {"buildings": {"facecolor": "#000"}}}
    set_path(style, "layers.buildings.facecolor", "#fff")
    assert style["layers"]["buildings"]["facecolor"] == "#fff"


def test_set_path_unknown_key_lists_valid():
    style = {"layers": {"buildings": {"facecolor": "#000"}}}
    with pytest.raises(KeyError, match="facecolor"):
        set_path(style, "layers.buildings.nope", 1)
    with pytest.raises(KeyError, match="buildings"):
        set_path(style, "layers.nope.x", 1)


def test_set_path_empty():
    with pytest.raises(KeyError):
        set_path({}, "", 1)


def test_apply_overrides_multiple_and_coerces():
    style = {"layers": {"streets": {"linewidth": 0.5, "enabled": True}}}
    apply_overrides(style, ["layers.streets.linewidth=2", "layers.streets.enabled=false"])
    assert style["layers"]["streets"]["linewidth"] == 2
    assert style["layers"]["streets"]["enabled"] is False


def test_apply_overrides_requires_equals():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["layers.streets.linewidth"])


def test_apply_overrides_no_coerce():
    style = {"a": {"b": 1}}
    apply_overrides(style, ["a.b=2"], coerce_values=False)
    assert style["a"]["b"] == "2"
