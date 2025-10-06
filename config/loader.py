import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    def __init__(self, config_dir: str = None):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files. Defaults to the 'config' directory
                      in the same directory as this module.
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent
        else:
            self.config_dir = Path(config_dir)
        
        self._config_cache: Dict[str, Any] = {}
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load a JSON configuration file.
        
        Args:
            filename: Name of the configuration file (with or without .json extension).
                   Can include subdirectories (e.g., 'palettes/my_palettes.json')
            
        Returns:
            Dict containing the configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
            
        # Check cache first
        if filename in self._config_cache:
            return self._config_cache[filename].copy()  # Return a copy to prevent modification of cached data
            
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Cache the loaded config
        self._config_cache[filename] = config
        return config.copy()  # Return a copy to prevent modification of cached data
    
    def get_style_config(self) -> Dict[str, Any]:
        """Get the style configuration."""
        return self.load_config('style.json')
    
    def save_config(self, config: Dict[str, Any], filename: str) -> None:
        """Save a configuration to a JSON file.
        
        Args:
            config: Configuration dictionary to save
            filename: Name of the output file (with or without .json extension).
                   Can include subdirectories (e.g., 'palettes/my_palettes.json')
        """
        if not filename.endswith('.json'):
            filename += '.json'
            
        # Handle nested paths
        config_path = self.config_dir / filename
        
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update cache with the full path as the key
        self._config_cache[filename] = config

# Create a default instance
config_loader = ConfigLoader()

def get_style_config() -> Dict[str, Any]:
    """Get the style configuration."""
    return config_loader.get_style_config()
