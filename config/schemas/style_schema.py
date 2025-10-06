from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class SizeCategory(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"

@dataclass
class LayerConfig:
    enabled: bool = True
    facecolor: str = "#ffffff"
    edgecolor: str = "#000000"
    linewidth: float = 0.1
    alpha: float = 1.0
    simplify_tolerance: Optional[float] = None
    hatch: Optional[str] = None
    zorder: int = 1
    min_size_threshold: Optional[float] = None

@dataclass
class BuildingsConfig(LayerConfig):
    size_categories: Dict[SizeCategory, Dict[str, Any]] = field(default_factory=dict)
    max_area: Optional[float] = None

@dataclass
class WaterConfig(LayerConfig):
    pass

@dataclass
class OutputConfig:
    separate_layers: bool = False
    filename_prefix: str = "map_output"
    output_directory: str = "output"

@dataclass
class StyleConfig:
    buildings: BuildingsConfig = field(default_factory=BuildingsConfig)
    water: WaterConfig = field(default_factory=WaterConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StyleConfig':
        """Create a StyleConfig from a dictionary."""
        return cls(
            buildings=BuildingsConfig(**data.get('buildings', {})),
            water=WaterConfig(**data.get('water', {})),
            output=OutputConfig(**data.get('output', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'buildings': {
                'enabled': self.buildings.enabled,
                'facecolor': self.buildings.facecolor,
                'edgecolor': self.buildings.edgecolor,
                'linewidth': self.buildings.linewidth,
                'alpha': self.buildings.alpha,
                'simplify_tolerance': self.buildings.simplify_tolerance,
                'hatch': self.buildings.hatch,
                'zorder': self.buildings.zorder,
                'min_size_threshold': self.buildings.min_size_threshold,
                'size_categories': self.buildings.size_categories,
                'max_area': self.buildings.max_area
            },
            'water': {
                'enabled': self.water.enabled,
                'facecolor': self.water.facecolor,
                'edgecolor': self.water.edgecolor,
                'linewidth': self.water.linewidth,
                'alpha': self.water.alpha,
                'simplify_tolerance': self.water.simplify_tolerance,
                'hatch': self.water.hatch,
                'zorder': self.water.zorder,
                'min_size_threshold': self.water.min_size_threshold
            },
            'output': {
                'separate_layers': self.output.separate_layers,
                'filename_prefix': self.output.filename_prefix,
                'output_directory': self.output.output_directory
            }
        }
