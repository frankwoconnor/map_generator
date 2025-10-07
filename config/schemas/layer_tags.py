"""Schema definitions for layer tag configurations."""
from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Optional

@dataclass
class TagValueConfig:
    """Configuration for a single tag value."""
    label: str
    default: bool = True

@dataclass
class TagConfig:
    """Configuration for a single OSM tag."""
    key: str
    description: str
    values: Dict[str, Union[bool, str, Dict[str, Any]]]
    default: Union[bool, str, List[str]] = True

@dataclass
class LayerTagConfig:
    """Configuration for a single layer's OSM tags."""
    description: str = ""
    tags: Dict[str, Union[bool, str, Dict[str, Any]]] = field(default_factory=dict)
    exclude_tags: List[str] = field(default_factory=list)
    custom_filter: Optional[str] = None
    tag_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerTagConfig':
        """Create a LayerTagConfig from a dictionary."""
        return cls(
            description=data.get('description', ''),
            tags=data.get('tags', {}),
            exclude_tags=data.get('exclude_tags', []),
            custom_filter=data.get('custom_filter'),
            tag_configs=data.get('tag_configs', {})
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
            if not isinstance(layer_data, dict):
                continue
                
            # Handle both old and new format
            if 'tags' in layer_data or 'tag_configs' in layer_data:
                layers[layer_name] = LayerTagConfig.from_dict(layer_data)
            else:
                # Convert old format to new format
                layers[layer_name] = LayerTagConfig(
                    tags=layer_data,
                    tag_configs={
                        key: {
                            'key': key,
                            'description': key.replace('_', ' ').title(),
                            'values': {v: str(v).title() for v in values} if isinstance(values, list) else {},
                            'default': values if isinstance(values, (bool, str)) else True
                        }
                        for key, values in layer_data.items()
                        if not isinstance(values, bool)  # Skip boolean values
                    }
                )
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
