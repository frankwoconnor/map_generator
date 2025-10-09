from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class FeatureCategory:
    """Represents a hierarchical category of features.

    Attributes:
        name: A unique name for the category (e.g., "Motorways").
        description: A human-readable description.
        osm_tags: A list of dictionaries, where each dictionary is a set of OSM tag key-value pairs
                  that define this category. E.g., `[{"highway": "motorway"}, {"highway": "motorway_link"}]`.
                  A value of `True` for an OSM key means any value for that key is included.
        sub_categories: Child categories for hierarchical organization.
        default_enabled: Default visibility state for this category.
        geom_type_preference: Preferred geometric type for this category (e.g., "line", "polygon", "point", "any").
    """

    name: str
    description: Optional[str] = None
    osm_tags: List[Dict[str, Union[str, bool, List[str]]]] = field(default_factory=list)
    sub_categories: List["FeatureCategory"] = field(default_factory=list)
    default_enabled: bool = True
    geom_type_preference: str = "any"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureCategory":
        sub_categories = [cls.from_dict(sc) for sc in data.get("sub_categories", [])]
        return cls(
            name=data["name"],
            description=data.get("description"),
            osm_tags=data.get("osm_tags", []),
            sub_categories=sub_categories,
            default_enabled=data.get("default_enabled", True),
            geom_type_preference=data.get("geom_type_preference", "any"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "osm_tags": self.osm_tags,
            "sub_categories": [sc.to_dict() for sc in self.sub_categories],
            "default_enabled": self.default_enabled,
            "geom_type_preference": self.geom_type_preference,
        }


@dataclass
class FeatureLayerConfig:
    """Configuration for features within a specific map layer.

    Attributes:
        layer_name: The name of the map layer (e.g., "streets", "buildings").
        categories: Top-level feature categories for this layer.
    """

    layer_name: str
    categories: List[FeatureCategory] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureLayerConfig":
        categories = [FeatureCategory.from_dict(c) for c in data.get("categories", [])]
        return cls(layer_name=data["layer_name"], categories=categories)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "categories": [c.to_dict() for c in self.categories],
        }


@dataclass
class FeatureVisibilityManager:
    """Manages the visibility state of all features across layers.

    Attributes:
        layer_configs: Maps layer_name to FeatureLayerConfig.
        _visibility_states: Internal dictionary mapping category_id (e.g., "streets.roads.residential") to boolean.
    """

    layer_configs: Dict[str, FeatureLayerConfig] = field(default_factory=dict)
    _visibility_states: Dict[str, bool] = field(default_factory=dict)

    def register_feature_config(self, config: FeatureLayerConfig):
        """Registers feature categories for a given layer and initializes their visibility states."""
        self.layer_configs[config.layer_name] = config
        self._initialize_visibility_states(config.layer_name, config.categories, [])

    def _initialize_visibility_states(
        self, layer_name: str, categories: List[FeatureCategory], parent_path: List[str]
    ):
        """Recursively initializes visibility states based on default_enabled."""
        for category in categories:
            current_path = parent_path + [category.name]
            category_id = f"{layer_name}.{'.'.join(current_path)}"
            self._visibility_states[category_id] = category.default_enabled
            if category.sub_categories:
                self._initialize_visibility_states(
                    layer_name, category.sub_categories, current_path
                )

    def get_visibility(self, category_id: str) -> bool:
        """Returns the current visibility state of a feature category."""
        return self._visibility_states.get(category_id, False)

    def set_visibility(self, category_id: str, enabled: bool):
        """Sets the visibility state of a feature category."""
        if category_id in self._visibility_states:
            self._visibility_states[category_id] = enabled
        else:
            # log_progress(f"Warning: Attempted to set visibility for unregistered category: {category_id}")
            pass  # Allow setting for potentially new categories, they will be registered on next load

    def get_all_categories(
        self, layer_name: Optional[str] = None
    ) -> List[Tuple[str, FeatureCategory]]:
        """Returns a flat list of all (category_id, FeatureCategory) pairs, optionally filtered by layer_name."""
        all_cats = []
        for ln, l_config in self.layer_configs.items():
            if layer_name is None or ln == layer_name:
                self._flatten_categories(ln, l_config.categories, [], all_cats)
        return all_cats

    def _flatten_categories(
        self,
        layer_name: str,
        categories: List[FeatureCategory],
        parent_path: List[str],
        result: List[Tuple[str, FeatureCategory]],
    ):
        """Helper to flatten hierarchical categories for iteration."""
        for category in categories:
            current_path = parent_path + [category.name]
            category_id = f"{layer_name}.{'.'.join(current_path)}"
            result.append((category_id, category))
            if category.sub_categories:
                self._flatten_categories(
                    layer_name, category.sub_categories, current_path, result
                )

    def get_active_osm_filters(self, layer_name: str) -> Dict[str, Any]:
        """
        Generates a combined OSM tag filter dictionary for a given layer
        based on currently enabled feature categories.

        Returns a dictionary suitable for osmnx/pyrosm tags parameter.
        """
        combined_tags: Dict[str, Any] = {}
        for category_id, category in self.get_all_categories(layer_name):
            if self.get_visibility(category_id):
                for osm_tag_pair in category.osm_tags:
                    for key, value in osm_tag_pair.items():
                        # Handle boolean True for a tag key (e.g., 'building': True)
                        if value is True:
                            combined_tags[key] = True
                        # Handle list of values or single string value
                        elif isinstance(value, (str, list)):
                            if key not in combined_tags or combined_tags[key] is True:
                                combined_tags[key] = []
                            if isinstance(value, str):
                                if value not in combined_tags[key]:
                                    combined_tags[key].append(value)
                            elif isinstance(value, list):
                                for v in value:
                                    if v not in combined_tags[key]:
                                        combined_tags[key].append(v)
                        # If a tag is explicitly set to False, it should override any True/list settings
                        elif value is False:
                            combined_tags[key] = False

        # Clean up: remove tags explicitly set to False, or empty lists if no specific values were added
        tags_to_remove = []
        for key, value in combined_tags.items():
            if value is False or (isinstance(value, list) and not value):
                tags_to_remove.append(key)
        for key in tags_to_remove:
            del combined_tags[key]

        return combined_tags

    def get_feature_config_for_layer(
        self, layer_name: str
    ) -> Optional[FeatureLayerConfig]:
        """Returns the FeatureLayerConfig for a given layer name."""
        return self.layer_configs.get(layer_name)


# Global instance of the manager
feature_visibility_manager = FeatureVisibilityManager()

# Example default feature configurations (can be loaded from JSON later)
# This will replace the logic in load_layer_tags() in main.py
DEFAULT_FEATURE_CONFIGS = [
    FeatureLayerConfig(
        layer_name="streets",
        categories=[
            FeatureCategory(
                name="Motorways",
                description="Major highways",
                osm_tags=[
                    {"highway": "motorway"},
                    {"highway": "trunk"},
                    {"highway": "motorway_link"},
                    {"highway": "trunk_link"},
                ],
            ),
            FeatureCategory(
                name="Primary Roads",
                description="Main roads",
                osm_tags=[{"highway": "primary"}, {"highway": "primary_link"}],
            ),
            FeatureCategory(
                name="Secondary Roads",
                description="Secondary roads",
                osm_tags=[{"highway": "secondary"}, {"highway": "secondary_link"}],
            ),
            FeatureCategory(
                name="Residential Roads",
                description="Local streets",
                osm_tags=[
                    {"highway": "residential"},
                    {"highway": "living_street"},
                    {"highway": "unclassified"},
                    {"highway": "road"},
                ],
            ),
            FeatureCategory(
                name="Service Roads",
                description="Service access roads",
                osm_tags=[{"highway": "service"}],
            ),
            FeatureCategory(
                name="Paths & Trails",
                description="Footpaths, cycleways, tracks",
                osm_tags=[
                    {"highway": "path"},
                    {"highway": "footway"},
                    {"highway": "cycleway"},
                    {"highway": "track"},
                ],
                default_enabled=False,
            ),
        ],
    ),
    FeatureLayerConfig(
        layer_name="buildings",
        categories=[
            FeatureCategory(
                name="All Buildings",
                description="All building types",
                osm_tags=[{"building": True}],
            )
        ],
    ),
    FeatureLayerConfig(
        layer_name="water",
        categories=[
            FeatureCategory(
                name="Lakes & Ponds",
                description="Natural water bodies",
                osm_tags=[
                    {"natural": "water"},
                    {"natural": "lake"},
                    {"natural": "pond"},
                ],
                geom_type_preference="polygon",
            ),
            FeatureCategory(
                name="Rivers & Canals",
                description="Major waterways",
                osm_tags=[{"waterway": "river"}, {"waterway": "canal"}],
                geom_type_preference="line",
            ),
            FeatureCategory(
                name="Streams & Ditches",
                description="Minor waterways",
                osm_tags=[
                    {"waterway": "stream"},
                    {"waterway": "drain"},
                    {"waterway": "ditch"},
                ],
                default_enabled=False,
                geom_type_preference="line",
            ),
        ],
    ),
    FeatureLayerConfig(
        layer_name="green",
        categories=[
            FeatureCategory(
                name="Parks & Gardens",
                description="Recreational green spaces",
                osm_tags=[
                    {"leisure": "park"},
                    {"leisure": "garden"},
                    {"leisure": "pitch"},
                    {"leisure": "recreation_ground"},
                ],
            ),
            FeatureCategory(
                name="Forests & Woods",
                description="Natural wooded areas",
                osm_tags=[{"natural": "wood"}, {"natural": "forest"}],
            ),
            FeatureCategory(
                name="Grasslands",
                description="Open grassy areas",
                osm_tags=[
                    {"landuse": "grass"},
                    {"natural": "grassland"},
                    {"landuse": "meadow"},
                ],
            ),
        ],
    ),
]

from config.loader import ConfigLoader  # Import ConfigLoader


def load_feature_configs(
    loader: ConfigLoader, filename: str
) -> List[FeatureLayerConfig]:
    try:
        data = loader.load_config(filename)
        return [FeatureLayerConfig.from_dict(item) for item in data]
    except FileNotFoundError:
        # log_progress(f"Feature config file not found: {filename}. Using default configurations.")
        return DEFAULT_FEATURE_CONFIGS
    except Exception as e:
        # log_progress(f"Error loading feature config from {filename}: {e}. Using default configurations.")
        return DEFAULT_FEATURE_CONFIGS


# Function to save feature configurations to a JSON file
def save_feature_configs(
    loader: ConfigLoader, filename: str, configs: List[FeatureLayerConfig]
):
    try:
        loader.save_config([c.to_dict() for c in configs], filename)
    except Exception as e:
        # log_progress(f"Error saving feature config to {filename}: {e}")
        pass
