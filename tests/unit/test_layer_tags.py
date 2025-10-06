"""Unit tests for the layer tags configuration."""
import sys
import os
from pathlib import Path
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from config import get_layer_tags, get_layer_tag_config, config_manager, LayerTagConfig

class TestLayerTags:
    """Test suite for layer tags configuration."""
    
    def test_get_all_layer_tags(self):
        """Test retrieving all layer tags."""
        layer_tags = get_layer_tags()
        assert isinstance(layer_tags.layers, dict)
        assert len(layer_tags.layers) > 0
        
        # Verify expected layers exist
        expected_layers = ['buildings', 'water', 'green', 'streets']
        for layer in expected_layers:
            assert layer in layer_tags.layers
    
    def test_get_specific_layer(self):
        """Test retrieving a specific layer's tags."""
        water_config = get_layer_tag_config('water')
        assert water_config is not None
        assert isinstance(water_config, LayerTagConfig)
        assert 'natural' in water_config.tags
        
    def test_update_layer_tags(self):
        """Test updating layer tags."""
        # Store original state
        original_config = get_layer_tag_config('water')
        
        try:
            # Test update
            new_tags = {
                'natural': ['water', 'bay', 'wetland', 'lake', 'pond'],
                'waterway': ['river', 'stream', 'canal']
            }
            exclude_tags = ['waterway=ditch', 'waterway=drain']
            
            config_manager.update_layer_tag_config(
                'water',
                tags=new_tags,
                exclude_tags=exclude_tags
            )
            
            # Verify update
            updated = get_layer_tag_config('water')
            assert updated.tags == new_tags
            assert updated.exclude_tags == exclude_tags
            
        finally:
            # Restore original state
            if original_config:
                config_manager.update_layer_tag_config(
                    'water',
                    tags=original_config.tags,
                    exclude_tags=original_config.exclude_tags,
                    custom_filter=original_config.custom_filter
                )
                config_manager.save_layer_tags()
    
    def test_nonexistent_layer(self):
        """Test retrieving a non-existent layer returns None."""
        assert get_layer_tag_config('nonexistent_layer') is None
