"""
config_management_module.py - Configuration management for the Lego sorting application

This module provides a centralized way to load, access, and modify configuration settings
used throughout the Lego sorting application. It handles loading from JSON files,
providing default values when settings are missing, and saving changes.
"""

import json
import os
from typing import Any, Dict


class ConfigManager:
    """Manages configuration settings for the Lego sorting application.

    The ConfigManager provides a unified interface for all modules to access
    their configuration settings, with reasonable defaults when settings aren't
    specified. Configuration is stored in a JSON file with sections for
    different components (camera, detector, sorting, etc.).
    """

    def __init__(self, config_path: str = "config.json"):
        """Initialize configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file or create default if not exists.

        This internal method attempts to read the config file from disk.
        If the file doesn't exist or has errors, it falls back to creating
        a default configuration.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
                self._create_default_config()
        else:
            print(f"Configuration file not found at {self.config_path}")
            self._create_default_config()

    def _create_default_config(self) -> None:
        """Create default configuration.

        This internal method builds a default configuration with reasonable
        values for all components. It's called when no valid configuration
        file exists. After creating the default config, it saves it to disk.
        """
        self.config = {
            "camera": {
                "device_id": 0,
                "directory": "LegoPictures",
                "filename_prefix": "Lego"
            },
            "detector": {
                "min_piece_area": 300,
                "max_piece_area": 100000,
                "buffer_percent": 0.1,
                "crop_padding": 50,
                "history_length": 2,
                "grid_size": [10, 10],
                "color_margin": 40,
                "show_grid": True,
                "track_timeout": 1.0,
                "capture_min_interval": 1.0,
                "spatial_cooldown_radius": 80,
                "spatial_cooldown_time": 2.0
            },
            "piece_identifier": {
                "csv_path": "Lego_Categories.csv",
                "confidence_threshold": 0.7
            },
            "sorting": {
                "strategy": "primary",
                "target_primary_category": "",
                "target_secondary_category": "",
                "max_bins": 9,
                "overflow_bin": 9
            }
        }
        self.save_config()

    def save_config(self) -> None:
        """Save current configuration to file.

        Writes the current configuration dictionary to the JSON file
        specified in self.config_path. This preserves any changes made
        to the configuration during runtime.
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        This is the primary method for modules to access their settings.
        It safely retrieves a value from the config, or returns a default
        if the requested setting doesn't exist.

        Args:
            section: Configuration section (e.g., "camera", "detector")
            key: Configuration key within the section
            default: Default value to return if the key is not found

        Returns:
            The configuration value or the provided default
        """
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section.

        This method retrieves all settings within a specific section.
        Useful when a module needs to access multiple related settings
        at once rather than retrieving them individually.

        Args:
            section: Configuration section (e.g., "camera", "detector")

        Returns:
            Dictionary containing the section or empty dict if not found
        """
        return self.config.get(section, {})

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value.

        Updates a single configuration setting. This method ensures the
        section exists before trying to update a value within it. Changes
        are stored in memory and must be saved with save_config() to persist.

        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update an entire section with new values.

        Efficiently updates multiple settings within a section at once.
        This is useful for updating related settings together, like when
        reconfiguring a component based on user input or calibration.

        Args:
            section: Configuration section
            values: Dictionary of values to update
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)


# Factory function
def create_config_manager(config_path: str = "config.json") -> ConfigManager:
    """Create a configuration manager instance.

    This factory function provides a consistent way to create config managers
    across the application, following the same pattern used in other modules.

    Args:
        config_path: Path to the configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


# Example usage
if __name__ == "__main__":
    # Create config manager
    config = create_config_manager()

    # Read some values
    camera_dir = config.get("camera", "directory", "default_dir")
    print(f"Camera directory: {camera_dir}")

    # Update a value
    config.set("camera", "device_id", 1)

    # Save changes
    config.save_config()