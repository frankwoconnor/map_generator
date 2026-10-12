from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SizeCategory(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"


@dataclass
class LocationConfig:
    query: str = ""
    distance: Optional[float] = None
    pbf_path: Optional[str] = None
    data_source: str = "remote"
    bbox: Optional[List[float]] = None
    pbf_folder: Optional[str] = None


@dataclass
class LayerConfig:
    enabled: bool = True
    facecolor: str = "#000000"
    edgecolor: str = "#000000"
    linewidth: float = 0.5
    alpha: float = 1.0
    simplify_tolerance: Optional[float] = 0.0
    min_size_threshold: Optional[float] = 0.0
    hatch: Optional[str] = None
    zorder: int = 1
    legend_enabled: bool = False


@dataclass
class StreetsConfig(LayerConfig):
    filter: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaterConfig(LayerConfig):
    filter: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GreenConfig(LayerConfig):
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManualColorSettings:
    facecolor: str = "#000000"


@dataclass
class BuildingsConfig(LayerConfig):
    manual_color_settings: ManualColorSettings = field(
        default_factory=ManualColorSettings
    )
    size_categories: List[Dict[str, Any]] = field(default_factory=list)
    size_categories_enabled: bool = False
    auto_style_mode: str = "manual"
    auto_size_palette: str = ""
    auto_distance_palette: str = ""


@dataclass
class OutputConfig:
    separate_layers: bool = True
    filename_prefix: str = "map"
    figure_size: List[float] = field(default_factory=lambda: [10.0, 10.0])
    background_color: str = "#FFFFFF"
    transparent_background: bool = False
    figure_dpi: int = 100
    margin: float = 0.05
    preview_type: str = "embedded"
    output_directory: str = "../output"
    enable_debug_legends: bool = False


@dataclass
class ProcessingConfig:
    street_filter: List[str] = field(default_factory=list)


@dataclass
class StyleConfig:
    location: LocationConfig = field(default_factory=LocationConfig)
    layers: Dict[str, Any] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleConfig":
        """Create a StyleConfig from a dictionary.

        Preserves all data from style.json; layer dictionaries are normalized
        with their corresponding typed configs where a schema exists. Missing
        sections are filled with defaults.
        """
        layers_raw = data.get("layers", {}) or {}

        normalized_layers: Dict[str, Any] = {}
        known_layers = {
            "streets": StreetsConfig,
            "water": WaterConfig,
            "green": GreenConfig,
            "buildings": BuildingsConfig,
        }
        for key, cls_ in known_layers.items():
            cfg = layers_raw.get(key, {}) or {}
            if cls_ is BuildingsConfig:
                buildings_data = dict(cfg)
                if "manual_color_settings" in buildings_data:
                    buildings_data["manual_color_settings"] = ManualColorSettings(
                        **buildings_data["manual_color_settings"]
                    )
                normalized_layers[key] = BuildingsConfig(**buildings_data)
            else:
                normalized_layers[key] = cls_(**cfg)

        # Preserve any user-defined layers beyond the known set
        for key, cfg in layers_raw.items():
            if key not in known_layers:
                normalized_layers[key] = LayerConfig(**(cfg or {}))

        return cls(
            location=LocationConfig(**data.get("location", {})),
            layers=normalized_layers,
            output=OutputConfig(**data.get("output", {})),
            processing=ProcessingConfig(**data.get("processing", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Round-trips all fields back to the style.json shape expected by the
        rest of the application.
        """
        layers_out: Dict[str, Any] = {}
        for key, cfg in self.layers.items():
            if hasattr(cfg, "__dataclass_fields__"):
                layers_out[key] = {
                    f: self._field_to_plain(getattr(cfg, f))
                    for f in cfg.__dataclass_fields__
                }
            elif isinstance(cfg, dict):
                layers_out[key] = cfg
            else:
                layers_out[key] = cfg

        return {
            "location": {
                "query": self.location.query,
                "distance": self.location.distance,
                "pbf_path": self.location.pbf_path,
                "data_source": self.location.data_source,
                "bbox": self.location.bbox,
                "pbf_folder": self.location.pbf_folder,
            },
            "layers": layers_out,
            "output": {
                "separate_layers": self.output.separate_layers,
                "filename_prefix": self.output.filename_prefix,
                "figure_size": self.output.figure_size,
                "background_color": self.output.background_color,
                "transparent_background": self.output.transparent_background,
                "figure_dpi": self.output.figure_dpi,
                "margin": self.output.margin,
                "preview_type": self.output.preview_type,
                "output_directory": self.output.output_directory,
                "enable_debug_legends": self.output.enable_debug_legends,
            },
            "processing": {
                "street_filter": self.processing.street_filter,
            },
        }

    @staticmethod
    def _field_to_plain(value: Any) -> Any:
        """Convert dataclass instances in nested fields to plain dicts."""
        if hasattr(value, "__dataclass_fields__"):
            return {
                f: StyleConfig._field_to_plain(getattr(value, f))
                for f in value.__dataclass_fields__
            }
        if isinstance(value, list):
            return [StyleConfig._field_to_plain(v) for v in value]
        if isinstance(value, dict):
            return {k: StyleConfig._field_to_plain(v) for k, v in value.items()}
        return value
