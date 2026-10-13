"""Unit tests for palette persistence (metadata preservation)."""

import json

from map_core.core import palettes


def test_save_palettes_preserves_metadata_keys(tmp_path):
    """Create/rename/delete writes must keep curated _comment metadata."""
    original_path = palettes.get_palettes_file_path()
    palettes.set_palettes_file_path(str(tmp_path / "palettes.json"))
    try:
        original = {
            "_comment": "Curated schemes",
            "_sequential": "Sequential palettes",
            "Alpha_2": ["#101010", "#202020"],
            "Beta_2": ["#303030", "#404040"],
        }
        (tmp_path / "palettes.json").write_text(json.dumps(original), encoding="utf-8")

        loaded = palettes.load_palettes()
        assert set(loaded) == {"Alpha_2", "Beta_2"}, "metadata keys are not palettes"

        # Create a new palette (mirrors POST /api/palettes)
        assert palettes.add_palette("Gamma_2", ["#505050", "#606060"])

        saved = json.loads((tmp_path / "palettes.json").read_text(encoding="utf-8"))
        assert saved["_comment"] == "Curated schemes", "comment key preserved"
        assert saved["_sequential"] == "Sequential palettes"
        assert set(saved) == {"_comment", "_sequential", "_custom", "Alpha_2", "Beta_2", "Gamma_2"}

        # Rename via remove + add keeps metadata too
        assert palettes.remove_palette("Alpha_2")
        assert palettes.add_palette("Delta_2", ["#707070", "#808080"])
        saved = json.loads((tmp_path / "palettes.json").read_text(encoding="utf-8"))
        assert saved["_comment"] == "Curated schemes"
        assert "Alpha_2" not in saved and "Delta_2" in saved

        # New palettes land under the _custom section, existing palettes keep
        # their compact per-line layout, and repeated saves are byte-stable.
        text = (tmp_path / "palettes.json").read_text(encoding="utf-8")
        assert '"_custom": "User-created palettes",' in text
        assert '  "Beta_2": ["#303030", "#404040"],' in text, "compact per-line palette layout"
        first = (tmp_path / "palettes.json").read_bytes()
        assert palettes.save_palettes(palettes.load_palettes())
        second = (tmp_path / "palettes.json").read_bytes()
        assert first == second, "repeated saves are byte-identical"
    finally:
        palettes.set_palettes_file_path(original_path)
        palettes.reset_cache()