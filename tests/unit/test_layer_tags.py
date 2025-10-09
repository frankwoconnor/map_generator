"""Unit tests for the layer tags configuration."""

import os
import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from config.manager import feature_visibility_manager
from config.schemas.feature_config import FeatureLayerConfig


class TestLayerTags:
    """Test suite for layer tags configuration."""

    def test_get_all_layer_tags(self):
        """Test retrieving all layer tags."""
        all_configs = feature_visibility_manager.layer_configs
        assert isinstance(all_configs, dict)
        assert len(all_configs) > 0

        # Verify expected layers exist
        expected_layers = ["buildings", "water", "green", "streets"]
        for layer in expected_layers:
            assert layer in all_configs

    def test_get_specific_layer(self):
        """Test retrieving a specific layer's tags."""
        water_config = feature_visibility_manager.get_feature_config_for_layer("water")
        assert water_config is not None
        assert isinstance(water_config, FeatureLayerConfig)
        assert len(water_config.categories) > 0  # Check if it has categories

    # def test_update_layer_tags(self):
    #     """Test updating layer tags."""
    #     # Store original state
    #     original_config = feature_visibility_manager.get_feature_config_for_layer('water')

    #     try:
    #         # Test update
    #         new_tags = {
    #             'natural': ['water', 'bay', 'wetland', 'lake', 'pond'],
    #             'waterway': ['river', 'stream', 'canal']
    #         }
    #         exclude_tags = ['waterway=ditch', 'waterway=drain']

    #         feature_visibility_manager.update_feature_config(
    #             'water',
    #             tags=new_tags,
    #             exclude_tags=exclude_tags
    #         )

    #         # Verify update
    #         updated = feature_visibility_manager.get_feature_config_for_layer('water')
    #         assert updated.tags == new_tags
    #         assert updated.exclude_tags == exclude_tags

    #     finally:
    #         # Restore original state
    #         if original_config:
    #             feature_visibility_manager.update_feature_config(
    #                 'water',
    #                 tags=original_config.tags,
    #                 exclude_tags=original_config.exclude_tags,
    #                 custom_filter=original_config.custom_filter
    #             )
    #             # No explicit save needed, manager handles persistence

    def test_nonexistent_layer(self):
        """Test retrieving a non-existent layer returns None."""
        assert (
            feature_visibility_manager.get_feature_config_for_layer("nonexistent_layer")
            is None
        )
