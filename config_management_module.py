"""
config_management_module.py - Configuration management for the Lego sorting application

This module provides a centralized way to load, access, and modify configuration settings
used throughout the Lego sorting application. It handles loading from JSON files,
providing default values when settings are missing, and saving changes. Now includes
support for multithreaded operations.
"""

import json
import os
import threading
from typing import Any, Dict, Optional


class ConfigManager:
    """Manages configuration settings for the Lego sorting application.

    The ConfigManager provides a unified interface for all modules to access
    their configuration settings, with reasonable defaults when settings aren't
    specified. Configuration is stored in a JSON file with sections for
    different components (camera, detector, sorting, etc.).

    This implementation is thread-safe, allowing concurrent access from
    multiple threads.
    """

    def __init__(self, config_path: str = "config.json"):
        """Initialize configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._config_lock = threading.RLock()  # Reentrant lock for thread safety
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file or create default if not exists."""
        with self._config_lock:
            print(f"Attempting to load configuration from: {os.path.abspath(self.config_path)}")
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r') as f:
                        self.config = json.load(f)
                    print(f"Configuration loaded from {self.config_path}")
                    print(f"Loaded config contents: {self.config}")
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
        with self._config_lock:
            self.config = {
                "camera": {
                    "device_id": 0,
                    "directory": "LegoPictures",
                    "filename_prefix": "Lego"
                },
                "detector": {
                    "min_piece_area": 300,
                    "max_piece_area": 100000,
                    # Separate entry and exit zone percentages
                    "entry_zone_percent": 0.1,  # Entry zone width as percentage of ROI width
                    "exit_zone_percent": 0.1,  # Exit zone width as percentage of ROI width
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
                    "overflow_bin": 9,
                    "calibrate_sorting_strategy": False
                },
                "threading": {
                    "max_queue_size": 100,
                    "worker_count": 1,
                    "api_timeout": 30.0,
                    "processing_timeout": 60.0,
                    "shutdown_timeout": 5.0,
                    "polling_interval": 0.01
                },
                "servo": {
                    "channel": 0,
                    "min_pulse": 150,
                    "max_pulse": 600,
                    "frequency": 50,
                    "default_position": 90,
                    "calibration_mode": False,
                    "bin_positions": {
                        "0": 20,
                        "1": 40,
                        "2": 60,
                        "3": 80,
                        "4": 100,
                        "5": 120,
                        "6": 140,
                        "7": 160,
                        "8": 180,
                        "9": 90
                    }
                },
                # New section for exit zone trigger configuration
                "exit_zone_trigger": {
                    "enabled": True,  # Enable exit zone trigger mechanism
                    "fall_time": 1.0,  # Estimated time (seconds) for piece to fall through chute
                    "priority_method": "rightmost",
                    # Method for prioritizing pieces in exit zone: rightmost, first_in, etc.
                    "min_piece_spacing": 50,  # Minimum pixel spacing between pieces
                    "cooldown_time": 0.5  # Time to wait after piece falls before next servo movement
                }
            }
            self.save_config()

    def save_config(self) -> None:
        """Save current configuration to file.

        Writes the current configuration dictionary to the JSON file
        specified in self.config_path. This preserves any changes made
        to the configuration during runtime.
        """
        with self._config_lock:
            try:
                # Create directory if it doesn't exist
                config_dir = os.path.dirname(self.config_path)
                if config_dir and not os.path.exists(config_dir):
                    os.makedirs(config_dir)

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
        with self._config_lock:
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
        with self._config_lock:
            return self.config.get(section, {}).copy()  # Return a copy to prevent modification

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
        with self._config_lock:
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
        with self._config_lock:
            if section not in self.config:
                self.config[section] = {}
            self.config[section].update(values)

    def set_default_threading_config(self) -> None:
        """Set default threading configuration if it doesn't exist.

        This method ensures that all necessary threading configuration
        options are present with reasonable default values.
        """
        with self._config_lock:
            if "threading" not in self.config:
                self.config["threading"] = {}

            threading_defaults = {
                "max_queue_size": 100,
                "worker_count": 1,
                "api_timeout": 30.0,
                "processing_timeout": 60.0,
                "shutdown_timeout": 5.0,
                "polling_interval": 0.01
            }

            # Only add missing keys, don't overwrite existing settings
            for key, value in threading_defaults.items():
                if key not in self.config["threading"]:
                    self.config["threading"][key] = value

    def set_default_exit_zone_trigger_config(self) -> None:
        """Set default exit zone trigger configuration if it doesn't exist.

        This method ensures that all necessary exit zone trigger configuration
        options are present with reasonable default values.
        """
        with self._config_lock:
            if "exit_zone_trigger" not in self.config:
                self.config["exit_zone_trigger"] = {}

            trigger_defaults = {
                "enabled": True,
                "fall_time": 1.0,
                "priority_method": "rightmost",
                "min_piece_spacing": 50,
                "cooldown_time": 0.5
            }

            # Only add missing keys, don't overwrite existing settings
            for key, value in trigger_defaults.items():
                if key not in self.config["exit_zone_trigger"]:
                    self.config["exit_zone_trigger"][key] = value

    def create_backup(self, suffix: str = "backup") -> Optional[str]:
        """Create a backup of the current configuration file.

        Args:
            suffix: Suffix to add to the backup filename

        Returns:
            str: Path to the backup file, or None if backup failed
        """
        with self._config_lock:
            if not os.path.exists(self.config_path):
                return None

            backup_path = f"{self.config_path}.{suffix}"
            try:
                with open(self.config_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                return backup_path
            except Exception as e:
                print(f"Error creating backup: {e}")
                return None


def calibrate_sorting_strategy(self):
    """
    Guide the user through selecting a sorting strategy and update configuration.
    This function runs directly in the terminal and updates the config file.
    """
    print("\n=== SORTING STRATEGY CALIBRATION ===")
    print("1. Primary (sort by main categories)")
    print("2. Secondary (sort by subcategories within one main category)")
    print("3. Tertiary (sort by sub-subcategories)")

    # Get strategy level selection
    strategy_choice = 0
    while strategy_choice not in [1, 2, 3]:
        try:
            strategy_choice = int(input("Select strategy (1-3): "))
        except ValueError:
            print("Please enter a number between 1 and 3")

    # Convert to strategy name
    strategy_names = {1: "primary", 2: "secondary", 3: "tertiary"}
    strategy = strategy_names[strategy_choice]

    # Set initial values
    primary_category = ""
    secondary_category = ""

    # If secondary or tertiary, need primary category
    if strategy_choice in [2, 3]:
        # Parse CSV once to get categories
        csv_path = self.get("piece_identifier", "csv_path", "Lego_Categories.csv")
        categories = self._parse_categories_from_csv(csv_path)

        # Show primary options
        print("\nSelect primary category:")
        primary_options = sorted(list(categories["primary"]))
        for idx, category in enumerate(primary_options, 1):
            print(f"{idx}. {category}")

        # Get primary selection
        primary_idx = 0
        while primary_idx < 1 or primary_idx > len(primary_options):
            try:
                primary_idx = int(input(f"Enter selection (1-{len(primary_options)}): "))
            except ValueError:
                print("Please enter a valid number")

        primary_category = primary_options[primary_idx - 1]

        # If tertiary, also need secondary category
        if strategy_choice == 3:
            # Get secondary options for this primary
            secondary_options = sorted(list(categories["primary_to_secondary"].get(primary_category, [])))

            if not secondary_options:
                print(f"No secondary categories exist for {primary_category}")
                print("Falling back to primary sorting")
                strategy = "primary"
                primary_category = ""
            else:
                # Show secondary options
                print(f"\nSelect secondary category within {primary_category}:")
                for idx, category in enumerate(secondary_options, 1):
                    print(f"{idx}. {category}")

                # Get secondary selection
                while True:
                    try:
                        secondary_idx = int(input(f"Enter selection (1-{len(secondary_options)}): "))
                        if 1 <= secondary_idx <= len(secondary_options):
                            secondary_category = secondary_options[secondary_idx - 1]

                            # Check if tertiary categories exist
                            key = (primary_category, secondary_category)
                            if key not in categories["secondary_to_tertiary"] or not \
                                    categories["secondary_to_tertiary"][key]:
                                print(f"No tertiary categories exist for {secondary_category} category.")
                                retry = input("Try another secondary category? (y/n): ")
                                if retry.lower() != 'y':
                                    # Fall back to secondary
                                    strategy = "secondary"
                                    secondary_category = ""
                                    break
                            else:
                                break  # Valid selection
                        else:
                            print(f"Please enter a number between 1 and {len(secondary_options)}")
                    except ValueError:
                        print("Please enter a valid number")

    # Update configuration
    self.set("sorting", "strategy", strategy)
    self.set("sorting", "target_primary_category", primary_category)
    self.set("sorting", "target_secondary_category", secondary_category)
    self.save_config()

    print(f"\nSorting configuration updated:")
    print(f"Strategy: {strategy}")
    if primary_category:
        print(f"Target primary category: {primary_category}")
    if secondary_category:
        print(f"Target secondary category: {secondary_category}")


def _parse_categories_from_csv(self, csv_path):
    """
    Simple helper to parse the categories CSV.
    Only used by the calibration function.
    """
    import csv
    categories = {
        "primary": set(),
        "primary_to_secondary": {},
        "secondary_to_tertiary": {}
    }

    try:
        with open(csv_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                primary = row['primary_category']
                secondary = row['secondary_category']
                tertiary = row['tertiary_category']

                categories["primary"].add(primary)

                if primary not in categories["primary_to_secondary"]:
                    categories["primary_to_secondary"][primary] = set()
                if secondary:
                    categories["primary_to_secondary"][primary].add(secondary)

                key = (primary, secondary)
                if key not in categories["secondary_to_tertiary"]:
                    categories["secondary_to_tertiary"][key] = set()
                if tertiary:
                    categories["secondary_to_tertiary"][key].add(tertiary)

        return categories
    except Exception as e:
        print(f"Error parsing categories: {e}")
        return {"primary": set(), "primary_to_secondary": {}, "secondary_to_tertiary": {}}


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
    config_manager = ConfigManager(config_path)

    # Ensure threading configuration exists
    config_manager.set_default_threading_config()

    # Ensure exit zone trigger configuration exists
    config_manager.set_default_exit_zone_trigger_config()

    return config_manager


# Example usage
if __name__ == "__main__":
    # Create config manager
    config = create_config_manager()

    # Read some values
    camera_dir = config.get("camera", "directory", "default_dir")
    print(f"Camera directory: {camera_dir}")

    # Check threading configuration
    worker_count = config.get("threading", "worker_count", 1)
    print(f"Worker count: {worker_count}")

    # Update a value
    config.set("camera", "device_id", 1)

    # Save changes
    config.save_config()
