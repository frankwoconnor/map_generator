# Try to load environment variables (optional dependency)
try:
    from pathlib import Path

    from dotenv import load_dotenv

    # Load environment variables
    basedir = Path(__file__).parent
    load_dotenv(basedir.parent / ".env")
except ImportError:
    # dotenv not available, continue without it
    from pathlib import Path

    basedir = Path(__file__).parent

import os


class Config:
    """Base configuration class."""

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-key-please-change"
    TEMPLATES_AUTO_RELOAD = True
    UPLOAD_FOLDER = basedir.parent / "uploads"
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "svg"}
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB max file size

    # Ensure upload directory exists
    UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False


# Configuration presets
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}

# Style configuration
from .manager import (ColorPalette, PaletteCollection, StyleConfig,
                      config_manager, feature_visibility_manager)
from .schemas import SizeCategory
from .schemas.layer_tags import LayerTagConfig, LayerTags

# Public API
__all__ = [
    "config",
    "Config",
    "DevelopmentConfig",
    "ProductionConfig",
    "config_manager",
    "StyleConfig",
    "SizeCategory",
    "ColorPalette",
    "PaletteCollection",
    "LayerTags",
    "LayerTagConfig",
    "feature_visibility_manager",
]
