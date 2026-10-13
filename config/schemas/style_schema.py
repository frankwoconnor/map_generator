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
    # Per-class street styling (see map_core.core.street_styling). style_mode is
    # "uniform" (single edgecolor) or "by_class" (color per road category).
    style_mode: str = "uniform"
    variable_widths: bool = True
    street_palette: str = ""
    class_colors: Dict[str, str] = field(default_factory=dict)
    category_widths: Dict[str, float] = field(default_factory=dict)
    linewidths: Dict[str, float] = field(default_factory=dict)


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
    # Creative output controls (D4): crop mask / frame / title.
    mask: str = "none"  # none | circle | hexagon | rounded
    mask_options: Dict[str, Any] = field(
        default_factory=lambda: {"inset": 0.0, "rounded_radius": 0.15}
    )
    frame: Dict[str, Any] = field(
        default_factory=lambda: {"width": 0.0, "color": "#000000"}
    )
    title: str = ""
    subtitle: str = ""
    title_font_size: float = 16
    subtitle_font_size: float = 11
    title_color: str = "#000000"
    subtitle_color: str = "#444444"


@dataclass
class ProcessingConfig:
    street_filter: List[str] = field(default_factory=list)
    # Clip large local PBFs to the map bbox (plus margin) with osmium before
    # reading. The clip is cached under ../cache/clips and reused across runs.
    auto_clip: bool = True
    auto_clip_threshold_mb: float = 20.0
    auto_clip_margin: float = 0.2


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
                "mask": self.output.mask,
                "mask_options": self.output.mask_options,
                "frame": self.output.frame,
                "title": self.output.title,
                "subtitle": self.output.subtitle,
                "title_font_size": self.output.title_font_size,
                "subtitle_font_size": self.output.subtitle_font_size,
                "title_color": self.output.title_color,
                "subtitle_color": self.output.subtitle_color,
            },
            "processing": {
                "street_filter": self.processing.street_filter,
                "auto_clip": self.processing.auto_clip,
                "auto_clip_threshold_mb": self.processing.auto_clip_threshold_mb,
                "auto_clip_margin": self.processing.auto_clip_margin,
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
