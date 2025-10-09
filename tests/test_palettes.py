import importlib

import main


def test_load_palettes_returns_dict_and_caches(monkeypatch):
    # First call returns a dict
    palettes1 = main.load_palettes()
    assert isinstance(palettes1, dict)

    # Capture current cache object id
    palettes2 = main.load_palettes()
    assert palettes1 is palettes2, "Palettes should be cached and identical object"
