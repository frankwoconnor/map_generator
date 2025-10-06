"""Schema definitions for layer tag configurations."""
from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Optional

@dataclass
class LayerTagConfig:
    """Configuration for a single layer's OSM tags."""
    tags: Dict[str, Union[bool, str, List[str]]] = field(default_factory=dict)
    exclude_tags: List[str] = field(default_factory=list)
    custom_filter: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerTagConfig':
        """Create a LayerTagConfig from a dictionary."""
        return cls(
            tags=data.get('tags', {}),
            exclude_tags=data.get('exclude_tags', []),
            custom_filter=data.get('custom_filter')
        )

@dataclass
class LayerTags:
    """Collection of layer tag configurations."""
    layers: Dict[str, LayerTagConfig]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerTags':
        """Create a LayerTags from a dictionary."""
        layers = {}
        for layer_name, layer_data in data.items():
            if isinstance(layer_data, dict):
                if 'tags' in layer_data:
                    layers[layer_name] = LayerTagConfig.from_dict(layer_data)
                else:
                    # Handle the case where the value is directly the tags dict
                    layers[layer_name] = LayerTagConfig(tags=layer_data)
        return cls(layers=layers)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        result = {}
        for name, config in self.layers.items():
            result[name] = {
                'tags': config.tags,
                'exclude_tags': config.exclude_tags
            }
            if config.custom_filter:
                result[name]['custom_filter'] = config.custom_filter
        return result

# Default layer tags that match the current implementation
DEFAULT_LAYER_TAGS = {
    'buildings': {'building': True},
    'water': {
        'natural': ['water'],
        'landuse': ['reservoir', 'basin']
    },
    'waterways': {
        'waterway': ['river', 'stream', 'canal', 'drain', 'ditch']
    },
    'green': {
        'leisure': ['park', 'garden', 'pitch', 'recreation_ground'],
        'landuse': ['grass', 'meadow', 'recreation_ground'],
        'natural': ['grassland', 'heath']
    },
    'aeroway': {'aeroway': ['runway', 'taxiway', 'apron', 'terminal']},
    'rail': {'railway': True},
    'amenities': {'amenity': True},
    'shops': {'shop': True},
}
