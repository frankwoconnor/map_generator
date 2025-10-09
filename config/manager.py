import json
import logging
from pathlib import Path
from typing import (Any, Dict, List, Optional, Type, TypedDict, TypeVar, Union,
                    cast)

from .loader import ConfigLoader
from .schemas.feature_config import (DEFAULT_FEATURE_CONFIGS,
                                     FeatureLayerConfig,
                                     FeatureVisibilityManager,
                                     load_feature_configs,
                                     save_feature_configs)
from .schemas.layer_tags import LayerTagConfig, LayerTags
from .schemas.style_schema import SizeCategory, StyleConfig

# Type definitions for palettes
ColorPalette = List[str]
PaletteCollection = Dict[str, Union[ColorPalette, Dict[str, ColorPalette]]]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files. If None, uses default.
        """
        self.loader = ConfigLoader(config_dir)
        self._style_config: Optional[StyleConfig] = None
        self._palettes: Optional[PaletteCollection] = None
        self._feature_configs: Optional[List[FeatureLayerConfig]] = (
            None  # New attribute
        )
        self._layer_tags: Optional[LayerTags] = None  # Keep for now, will be removed

    def get_style_config(self, validate: bool = True) -> StyleConfig:
        """Get the style configuration.

        Args:
            validate: If True, validates the configuration against the schema.

        Returns:
            StyleConfig: The loaded and validated style configuration.
        """
        if self._style_config is None:
            config_data = self.loader.load_config("style.json")
            self._style_config = self._validate_style_config(config_data)
        return self._style_config

    def _validate_style_config(self, config_data: Dict[str, Any]) -> StyleConfig:
        """Validate the style configuration against the schema.

        Args:
            config_data: The raw configuration data.

        Returns:
            StyleConfig: The validated configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        try:
            return StyleConfig.from_dict(config_data)
        except Exception as e:
            logger.error(f"Invalid style configuration: {e}")
            raise ValueError(f"Invalid style configuration: {e}") from e

    def update_style_config(self, config_data: Dict[str, Any]) -> None:
        """Update the style configuration.

        Args:
            config_data: The new configuration data.
        """
        self._style_config = self._validate_style_config(config_data)

    def save_style_config(self, filename: Optional[str] = None) -> None:
        """Save the current style configuration to a file.

        Args:
            filename: The output filename. If None, uses 'style.json'.
        """
        if self._style_config is None:
            raise ValueError("No style configuration loaded")

        if filename is None:
            filename = "style.json"

        self.loader.save_config(self._style_config.to_dict(), filename)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot notation.

        Example:
            get_config_value('buildings.facecolor')

        Args:
            key: The configuration key in dot notation.
            default: Default value if key is not found.

        Returns:
            The configuration value or default if not found.
        """
        config = self.get_style_config().to_dict()
        keys = key.split(".")
        value = config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_palettes(self) -> PaletteCollection:
        """Get all color palettes.

        Returns:
            A dictionary of palette names to color arrays or nested palette collections.

        Raises:
            FileNotFoundError: If the palettes file doesn't exist
            json.JSONDecodeError: If the palettes file contains invalid JSON
        """
        if self._palettes is None:
            try:
                self._palettes = self.loader.load_config("palettes/palettes.json")
            except FileNotFoundError as e:
                logger.error(f"Palettes file not found: {e}")
                # Return a default set of palettes if the file is missing
                self._palettes = {
                    "default": {
                        "OrRd_3": ["#fee8c8", "#fdbb84", "#e34a33"],
                        "Blues_5": [
                            "#eff3ff",
                            "#bdd7e7",
                            "#6baed6",
                            "#3182bd",
                            "#08519c",
                        ],
                    }
                }
                logger.warning("Using default palettes")

        return self._palettes

    def get_palette(self, name: str) -> Optional[ColorPalette]:
        """Get a specific color palette by name.

        Args:
            name: Name of the palette (e.g., 'OrRd_3' or 'default.OrRd_3')

        Returns:
            The color palette if found, None otherwise.
        """
        palettes = self.get_palettes()

        # Handle nested palette names (e.g., 'default.OrRd_3')
        if "." in name:
            parts = name.split(".")
            current = palettes
            try:
                for part in parts[:-1]:
                    current = current[part]
                return current[parts[-1]]
            except (KeyError, TypeError):
                return None

        # Try to find the palette in the root
        if name in palettes and isinstance(palettes[name], list):
            return palettes[name]

        # If not found, try to find in any category
        for category in palettes.values():
            if isinstance(category, dict) and name in category:
                return category[name]

        return None

    def add_palette(self, name: str, colors: List[str], category: str = None) -> None:
        """Add or update a color palette.

        Args:
            name: Name of the palette
            colors: List of color codes
            category: Optional category name (creates nested structure if provided)
        """
        if self._palettes is None:
            self.get_palettes()  # Initialize palettes if not loaded

        if category:
            if category not in self._palettes or not isinstance(
                self._palettes[category], dict
            ):
                self._palettes[category] = {}
            self._palettes[category][name] = colors
        else:
            self._palettes[name] = colors

    def save_palettes(self) -> None:
        """Save the current palettes to the configuration file."""
        if self._palettes is not None:
            self.loader.save_config(self._palettes, "palettes/palettes.json")

    def get_feature_configs(self) -> List[FeatureLayerConfig]:
        """Get the feature layer configurations.

        Returns:
            List[FeatureLayerConfig]: The loaded feature configurations.
        """
        if self._feature_configs is None:
            self._feature_configs = load_feature_configs(self.loader, "features.json")
        return self._feature_configs

    def save_feature_configs(self, configs: List[FeatureLayerConfig]) -> None:
        """Save the feature layer configurations back to disk."""
        self._feature_configs = configs
        save_feature_configs(self.loader, "features.json", configs)

    def get_layer_tags(self) -> LayerTags:
        """Get the layer tags configuration (deprecated, will be removed)."""
        # This function is being replaced by the new feature_config system.
        # For now, return an empty LayerTags object to avoid breaking existing code.
        return LayerTags(layers={})

    def save_layer_tags(self) -> None:
        """Save the layer tags configuration back to disk (deprecated, will be removed)."""
        pass  # No-op

    def get_layer_tag_config(self, layer_name: str) -> Optional[LayerTagConfig]:
        """Get the tag configuration for a specific layer (deprecated, will be removed)."""
        return None

    def update_layer_tag_config(
        self,
        layer_name: str,
        tags: Optional[Dict[str, Any]] = None,
        exclude_tags: Optional[List[str]] = None,
        custom_filter: Optional[str] = None,
    ) -> None:
        """Update the tag configuration for a layer (deprecated, will be removed)."""
        pass  # No-op


# Create a default instance
config_manager = ConfigManager()

# Global instance of the feature visibility manager
feature_visibility_manager = FeatureVisibilityManager()

# Initialize the feature visibility manager with default or loaded configurations
_loaded_feature_configs = config_manager.get_feature_configs()
for config in _loaded_feature_configs:
    feature_visibility_manager.register_feature_config(config)
