"""
enhanced_config_manager.py - Unified configuration management with validation

This module provides centralized configuration management with schema validation,
defaults handling, migration support, and standardized module interfaces.

UPDATED: Added parse_categories_from_csv function for category hierarchy
"""

import json
import os
import csv
import shutil
import threading
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigVersion:
    """Configuration schema versions"""
    V1_0 = "1.0"  # Original schema
    V2_0 = "2.0"  # GUI integration schema
    V2_1 = "2.1"  # Enhanced with validation
    CURRENT = V2_1


class ModuleConfig(Enum):
    """Standard module names for configuration"""
    CAMERA = "camera"
    DETECTOR = "detector"
    DETECTOR_ROI = "detector_roi"
    SORTING = "sorting"
    SERVO = "servo"
    ARDUINO_SERVO = "arduino_servo"
    API = "api"
    THREADING = "threading"
    UI = "ui"
    GUI = "gui_settings"
    EXIT_ZONE = "exit_zone_trigger"
    PIECE_IDENTIFIER = "piece_identifier"


class ConfigSchema:
    """Configuration schema definitions and defaults"""

    @staticmethod
    def get_module_schema(module_name: str) -> Dict[str, Any]:
        """Get schema with defaults for a specific module"""

        schemas = {
            ModuleConfig.CAMERA.value: {
                "device_id": 0,
                "buffer_size": 1,
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "auto_exposure": True
            },

            ModuleConfig.DETECTOR.value: {
                "detector_type": "conveyor",
                "min_area": 1000,
                "max_area": 50000,
                "min_aspect_ratio": 0.3,
                "max_aspect_ratio": 3.0,
                "edge_margin": 10,
                "debug_mode": False
            },

            ModuleConfig.DETECTOR_ROI.value: {
                "x": 125,
                "y": 200,
                "w": 1550,
                "h": 500,
                "entry_zone_width": 150,
                "exit_zone_width": 150
            },

            ModuleConfig.SORTING.value: {
                "strategy": "primary",
                "target_primary_category": "",
                "target_secondary_category": "",
                "max_bins": 9,
                "overflow_bin": 0,
                "confidence_threshold": 0.7,
                "pre_assignments": {}  # For GUI pre-assignment feature
            },

            ModuleConfig.SERVO.value: {
                "min_pulse": 500,
                "max_pulse": 2500,
                "default_position": 90,
                "speed": 100,
                "calibration_positions": {}
            },

            ModuleConfig.ARDUINO_SERVO.value: {
                "port": "",
                "baud_rate": 57600,
                "timeout": 1.0,
                "servo_count": 9,
                "servo_pins": [2, 3, 4, 5, 6, 7, 8, 9, 10]
            },

            ModuleConfig.API.value: {
                "api_type": "rebrickable",
                "api_key": "",
                "cache_enabled": True,
                "cache_dir": "api_cache",
                "timeout": 10.0,
                "retry_count": 3,
                "rate_limit": 5.0
            },

            ModuleConfig.THREADING.value: {
                "max_workers": 4,
                "queue_size": 100,
                "timeout": 30.0,
                "priority_levels": 3
            },

            ModuleConfig.UI.value: {
                "display_enabled": True,
                "window_width": 1280,
                "window_height": 720,
                "show_debug_info": False,
                "show_fps": True,
                "overlay_alpha": 0.7
            },

            ModuleConfig.GUI.value: {
                "theme": "dark",
                "auto_save": True,
                "save_interval": 60,
                "show_tooltips": True,
                "confirm_exit": True
            },

            ModuleConfig.EXIT_ZONE.value: {
                "enabled": True,
                "fall_time": 1.0,
                "priority_method": "rightmost",
                "min_piece_spacing": 50,
                "cooldown_time": 0.5
            },

            ModuleConfig.PIECE_IDENTIFIER.value: {
                "csv_path": "Lego_Categories.csv",
                "confidence_threshold": 0.7,
                "save_unknown": True,
                "unknown_dir": "unknown_pieces"
            }
        }

        return schemas.get(module_name, {})

    @staticmethod
    def get_required_fields(module_name: str) -> List[str]:
        """Get required fields for a module"""

        required = {
            ModuleConfig.CAMERA.value: ["device_id"],
            ModuleConfig.DETECTOR.value: ["detector_type"],
            ModuleConfig.SORTING.value: ["strategy", "max_bins"],
            ModuleConfig.API.value: ["api_type"],
            ModuleConfig.ARDUINO_SERVO.value: ["port", "baud_rate"],
            ModuleConfig.PIECE_IDENTIFIER.value: ["csv_path"]
        }

        return required.get(module_name, [])


class ConfigValidator:
    """Validates configuration values"""

    @staticmethod
    def validate_module(module_name: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a module's configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        required = ConfigSchema.get_required_fields(module_name)
        for field in required:
            if field not in config or config[field] is None:
                errors.append(f"Missing required field '{field}' in {module_name}")

        # Module-specific validation
        if module_name == ModuleConfig.SORTING.value:
            errors.extend(ConfigValidator._validate_sorting(config))
        elif module_name == ModuleConfig.CAMERA.value:
            errors.extend(ConfigValidator._validate_camera(config))
        elif module_name == ModuleConfig.DETECTOR_ROI.value:
            errors.extend(ConfigValidator._validate_roi(config))

        return (len(errors) == 0, errors)

    @staticmethod
    def _validate_sorting(config: Dict[str, Any]) -> List[str]:
        """Validate sorting configuration"""
        errors = []

        strategy = config.get("strategy")
        if strategy not in ["primary", "secondary", "tertiary"]:
            errors.append(f"Invalid sorting strategy: {strategy}")

        if strategy == "secondary" and not config.get("target_primary_category"):
            errors.append("Secondary sorting requires target_primary_category")

        if strategy == "tertiary":
            if not config.get("target_primary_category"):
                errors.append("Tertiary sorting requires target_primary_category")
            if not config.get("target_secondary_category"):
                errors.append("Tertiary sorting requires target_secondary_category")

        max_bins = config.get("max_bins", 0)
        if not (1 <= max_bins <= 20):
            errors.append(f"Invalid max_bins: {max_bins} (must be 1-20)")

        # Validate pre-assignments if present
        pre_assignments = config.get("pre_assignments", {})
        for category, bin_num in pre_assignments.items():
            if not isinstance(bin_num, int):
                errors.append(f"Invalid bin number for {category}: {bin_num} (must be integer)")
            elif not (1 <= bin_num <= max_bins):
                errors.append(f"Bin {bin_num} for {category} out of range (1-{max_bins})")

        return errors

    @staticmethod
    def _validate_camera(config: Dict[str, Any]) -> List[str]:
        """Validate camera configuration"""
        errors = []

        device_id = config.get("device_id")
        if not isinstance(device_id, int) or device_id < 0:
            errors.append(f"Invalid device_id: {device_id} (must be non-negative integer)")

        fps = config.get("fps", 30)
        if not (1 <= fps <= 120):
            errors.append(f"Invalid fps: {fps} (must be 1-120)")

        return errors

    @staticmethod
    def _validate_roi(config: Dict[str, Any]) -> List[str]:
        """Validate ROI configuration"""
        errors = []

        # Check dimensions are positive
        for key in ["w", "h"]:
            value = config.get(key, 0)
            if value <= 0:
                errors.append(f"ROI {key} must be positive: {value}")

        # Check positions are non-negative
        for key in ["x", "y"]:
            value = config.get(key, 0)
            if value < 0:
                errors.append(f"ROI {key} must be non-negative: {value}")

        return errors


class ConfigMigration:
    """Handles configuration migration between versions"""

    @staticmethod
    def migrate(config: Dict[str, Any], from_version: str) -> Dict[str, Any]:
        """
        Migrate configuration from old version to current

        Args:
            config: Configuration dictionary
            from_version: Version to migrate from

        Returns:
            Migrated configuration
        """
        migrations = {
            ConfigVersion.V1_0: ConfigMigration._migrate_v1_to_v2,
            ConfigVersion.V2_0: ConfigMigration._migrate_v2_to_v2_1
        }

        current_version = from_version

        # Apply migrations sequentially
        while current_version != ConfigVersion.CURRENT:
            if current_version in migrations:
                config = migrations[current_version](config)
                # Update version after each migration
                if current_version == ConfigVersion.V1_0:
                    current_version = ConfigVersion.V2_0
                elif current_version == ConfigVersion.V2_0:
                    current_version = ConfigVersion.V2_1
            else:
                break

        config["version"] = ConfigVersion.CURRENT
        return config

    @staticmethod
    def _migrate_v1_to_v2(config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from v1.0 to v2.0"""
        # Add GUI settings if not present
        if ModuleConfig.GUI.value not in config:
            config[ModuleConfig.GUI.value] = ConfigSchema.get_module_schema(ModuleConfig.GUI.value)

        # Add pre_assignments to sorting if not present
        if ModuleConfig.SORTING.value in config:
            if "pre_assignments" not in config[ModuleConfig.SORTING.value]:
                config[ModuleConfig.SORTING.value]["pre_assignments"] = {}

        return config

    @staticmethod
    def _migrate_v2_to_v2_1(config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from v2.0 to v2.1"""
        # Add validation fields
        if "validation" not in config:
            config["validation"] = {
                "strict_mode": False,
                "auto_fix": True
            }

        return config


class EnhancedConfigManager:
    """
    Enhanced configuration manager with validation and standardized interfaces

    This manager provides:
    - Thread-safe operations
    - Schema validation
    - Automatic defaults
    - Version migration
    - Module-specific helpers
    - Change notifications
    - Category hierarchy parsing
    """

    def __init__(self, config_path: str = "config.json", auto_migrate: bool = True):
        """
        Initialize enhanced configuration manager

        Args:
            config_path: Path to configuration file
            auto_migrate: Automatically migrate old configs
        """
        self.config_path = config_path
        self.auto_migrate = auto_migrate
        self.config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._observers: List[Callable] = []
        self._backup_dir = "config_backups"

        # Cache for category hierarchy
        self._category_hierarchy: Optional[Dict[str, Any]] = None
        self._category_hierarchy_lock = threading.RLock()

        # Create backup directory
        os.makedirs(self._backup_dir, exist_ok=True)

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load and validate configuration from file"""
        with self._lock:
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r') as f:
                        self.config = json.load(f)

                    # Check version and migrate if needed
                    config_version = self.config.get("version", ConfigVersion.V1_0)

                    if config_version != ConfigVersion.CURRENT and self.auto_migrate:
                        logger.info(f"Migrating config from {config_version} to {ConfigVersion.CURRENT}")
                        self._create_backup(suffix=f"_pre_migration_{config_version}")
                        self.config = ConfigMigration.migrate(self.config, config_version)
                        self.save_config()

                    logger.info(f"Configuration loaded from {self.config_path}")

                except Exception as e:
                    logger.error(f"Error loading configuration: {e}")
                    self._create_default_config()
            else:
                logger.info(f"No configuration file found at {self.config_path}")
                self._create_default_config()

    def _create_default_config(self) -> None:
        """Create default configuration with all modules"""
        with self._lock:
            self.config = {
                "version": ConfigVersion.CURRENT,
                "validation": {
                    "strict_mode": False,
                    "auto_fix": True
                }
            }

            # Add all module defaults
            for module in ModuleConfig:
                module_name = module.value
                self.config[module_name] = ConfigSchema.get_module_schema(module_name)

            self.save_config()
            logger.info("Created default configuration")

    def parse_categories_from_csv(self, csv_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
        """
        Parse the Lego categories CSV file to build category hierarchy.

        This function reads the CSV and builds a hierarchical structure of categories
        that can be used by the GUI and sorting modules to understand relationships
        between primary, secondary, and tertiary categories.

        Args:
            csv_path: Path to CSV file (uses default from config if not provided)
            force_reload: Force reload even if cached

        Returns:
            dict: Category hierarchy with the following structure:
                {
                    "primary": set of primary category names,
                    "primary_to_secondary": dict mapping primary -> set of secondary,
                    "secondary_to_tertiary": dict mapping (primary, secondary) -> set of tertiary,
                    "all_categories": dict mapping element_id -> full category info
                }

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            Exception: If error parsing CSV
        """
        with self._category_hierarchy_lock:
            # Return cached hierarchy if available and not forcing reload
            if self._category_hierarchy is not None and not force_reload:
                return self._category_hierarchy

            # Get CSV path from config if not provided
            if csv_path is None:
                piece_config = self.get_module_config(ModuleConfig.PIECE_IDENTIFIER.value)
                csv_path = piece_config.get("csv_path", "Lego_Categories.csv")

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Categories CSV not found at: {csv_path}")

            # Initialize the hierarchy structure
            categories = {
                "primary": set(),
                "primary_to_secondary": {},
                "secondary_to_tertiary": {},
                "all_categories": {}  # Maps element_id to full category info
            }

            try:
                with open(csv_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.DictReader(file)

                    for row in csv_reader:
                        # Get element ID and category levels
                        element_id = row.get('element_id', '')
                        primary = row.get('primary_category', '')
                        secondary = row.get('secondary_category', '')
                        tertiary = row.get('tertiary_category', '')

                        # Skip rows without element_id or primary category
                        if not element_id or not primary:
                            continue

                        # Store full category info
                        categories["all_categories"][element_id] = {
                            'name': row.get('name', ''),
                            'primary_category': primary,
                            'secondary_category': secondary,
                            'tertiary_category': tertiary
                        }

                        # Add to primary categories set
                        categories["primary"].add(primary)

                        # Build primary to secondary mapping
                        if primary not in categories["primary_to_secondary"]:
                            categories["primary_to_secondary"][primary] = set()

                        if secondary:  # Only add if secondary exists
                            categories["primary_to_secondary"][primary].add(secondary)

                        # Build secondary to tertiary mapping
                        if secondary:  # Only process if secondary exists
                            key = (primary, secondary)
                            if key not in categories["secondary_to_tertiary"]:
                                categories["secondary_to_tertiary"][key] = set()

                            if tertiary:  # Only add if tertiary exists
                                categories["secondary_to_tertiary"][key].add(tertiary)

                # Convert sets to sorted lists for easier use in GUI
                categories["primary"] = sorted(list(categories["primary"]))

                for primary in categories["primary_to_secondary"]:
                    categories["primary_to_secondary"][primary] = sorted(
                        list(categories["primary_to_secondary"][primary])
                    )

                for key in categories["secondary_to_tertiary"]:
                    categories["secondary_to_tertiary"][key] = sorted(
                        list(categories["secondary_to_tertiary"][key])
                    )

                # Cache the parsed hierarchy
                self._category_hierarchy = categories

                # Log summary statistics
                logger.info(f"Parsed category hierarchy from {csv_path}:")
                logger.info(f"  - {len(categories['primary'])} primary categories")
                logger.info(f"  - {len(categories['all_categories'])} total elements")

                return categories

            except Exception as e:
                error_msg = f"Error parsing categories CSV: {e}"
                logger.error(error_msg)
                raise Exception(error_msg)

    def get_category_hierarchy(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Get the cached category hierarchy, parsing if necessary.

        This is a convenience method that wraps parse_categories_from_csv
        for modules that need quick access to the category hierarchy.

        Args:
            force_reload: Force reload from CSV even if cached

        Returns:
            dict: Category hierarchy structure
        """
        return self.parse_categories_from_csv(force_reload=force_reload)

    def get_categories_for_strategy(self, strategy: str,
                                    primary_category: Optional[str] = None,
                                    secondary_category: Optional[str] = None) -> List[str]:
        """
        Get the list of categories available for a given sorting strategy.

        This helper method returns the appropriate categories based on the
        sorting strategy and selected parent categories.

        Args:
            strategy: "primary", "secondary", or "tertiary"
            primary_category: Required for secondary/tertiary strategies
            secondary_category: Required for tertiary strategy

        Returns:
            List of category names available for the given strategy
        """
        hierarchy = self.get_category_hierarchy()

        if strategy == "primary":
            return hierarchy.get("primary", [])

        elif strategy == "secondary":
            if not primary_category:
                return []
            return hierarchy.get("primary_to_secondary", {}).get(primary_category, [])

        elif strategy == "tertiary":
            if not primary_category or not secondary_category:
                return []
            key = (primary_category, secondary_category)
            return hierarchy.get("secondary_to_tertiary", {}).get(key, [])

        return []

    def get_module_config(self, module_name: str, validate: bool = True) -> Dict[str, Any]:
        """
        Get complete configuration for a module with defaults

        This is the PRIMARY method modules should use to get their config.
        It ensures all expected fields are present with appropriate defaults.

        Args:
            module_name: Name of the module (use ModuleConfig enum values)
            validate: Whether to validate the configuration

        Returns:
            Complete module configuration with defaults applied

        Raises:
            ValueError: If validation fails and strict_mode is enabled
        """
        with self._lock:
            # Get stored config for module
            stored_config = self.config.get(module_name, {})

            # Get schema defaults
            defaults = ConfigSchema.get_module_schema(module_name)

            # Merge stored config over defaults
            module_config = {**defaults, **stored_config}

            # Validate if requested
            if validate:
                is_valid, errors = ConfigValidator.validate_module(module_name, module_config)

                if not is_valid:
                    error_msg = f"Invalid config for {module_name}: " + "; ".join(errors)

                    if self.config.get("validation", {}).get("strict_mode", False):
                        raise ValueError(error_msg)
                    else:
                        logger.warning(error_msg)

                        # Auto-fix if enabled
                        if self.config.get("validation", {}).get("auto_fix", True):
                            # Use defaults for invalid fields
                            module_config = defaults
                            logger.info(f"Using defaults for {module_name} due to validation errors")

            return module_config

    def update_module_config(self, module_name: str, updates: Dict[str, Any],
                             validate: bool = True) -> bool:
        """
        Update a module's configuration

        Args:
            module_name: Name of the module
            updates: Dictionary of updates to apply
            validate: Whether to validate before applying

        Returns:
            True if update successful, False otherwise
        """
        with self._lock:
            # Get current module config
            current = self.get_module_config(module_name, validate=False)

            # Apply updates
            updated = {**current, **updates}

            # Validate if requested
            if validate:
                is_valid, errors = ConfigValidator.validate_module(module_name, updated)

                if not is_valid:
                    logger.error(f"Validation failed for {module_name}: {errors}")
                    return False

            # Store updated config
            self.config[module_name] = updated

            # Notify observers
            self._notify_observers(module_name, updates)

            # Save to file
            self.save_config()

            return True

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get complete validation report for all modules

        Returns:
            Dictionary with validation results for each module
        """
        report = {
            "valid": True,
            "modules": {},
            "timestamp": datetime.now().isoformat()
        }

        with self._lock:
            for module in ModuleConfig:
                module_name = module.value
                module_config = self.get_module_config(module_name, validate=False)

                is_valid, errors = ConfigValidator.validate_module(module_name, module_config)

                report["modules"][module_name] = {
                    "valid": is_valid,
                    "errors": errors
                }

                if not is_valid:
                    report["valid"] = False

        return report

    def save_config(self) -> bool:
        """
        Save configuration to file

        Returns:
            True if save successful, False otherwise
        """
        with self._lock:
            try:
                # Create backup before saving
                self._create_backup()

                # Save configuration
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)

                logger.info(f"Configuration saved to {self.config_path}")
                return True

            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                return False

    def _create_backup(self, suffix: str = None) -> Optional[str]:
        """Create timestamped backup of configuration"""
        with self._lock:
            if not os.path.exists(self.config_path):
                return None

            if suffix is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix = timestamp

            backup_path = os.path.join(self._backup_dir, f"config_{suffix}.json")

            try:
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Backup created at {backup_path}")
                return backup_path
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
                return None

    def register_observer(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register callback for configuration changes

        Args:
            callback: Function to call with (module_name, changes) on config change
        """
        with self._lock:
            self._observers.append(callback)

    def _notify_observers(self, module_name: str, changes: Dict[str, Any]) -> None:
        """Notify all observers of configuration changes"""
        for observer in self._observers:
            try:
                observer(module_name, changes)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")

    # Legacy compatibility methods
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Legacy compatibility: Get single value"""
        module_config = self.get_module_config(section, validate=False)
        return module_config.get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Legacy compatibility: Set single value"""
        with self._lock:
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Legacy compatibility: Get entire section"""
        return self.get_module_config(section, validate=False)

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Legacy compatibility: Update entire section"""
        self.update_module_config(section, values, validate=False)


# Factory function for compatibility
def create_config_manager(config_path: str = "config.json") -> EnhancedConfigManager:
    """
    Create an enhanced configuration manager instance

    Args:
        config_path: Path to configuration file

    Returns:
        EnhancedConfigManager instance
    """
    return EnhancedConfigManager(config_path)


# Example usage and migration guide
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create config manager
    config = create_config_manager()

    # Parse and display category hierarchy
    print("\n=== Category Hierarchy ===")
    try:
        hierarchy = config.parse_categories_from_csv()
        print(f"Primary categories: {hierarchy['primary'][:5]}...")  # Show first 5

        # Show example of hierarchy
        if hierarchy['primary']:
            first_primary = hierarchy['primary'][0]
            print(f"\nSecondary categories in '{first_primary}':")
            secondaries = hierarchy['primary_to_secondary'].get(first_primary, [])
            print(f"  {secondaries[:3]}...")  # Show first 3

            if secondaries:
                first_secondary = secondaries[0]
                key = (first_primary, first_secondary)
                tertiaries = hierarchy['secondary_to_tertiary'].get(key, [])
                print(f"\nTertiary categories in '{first_primary}' -> '{first_secondary}':")
                print(f"  {tertiaries[:3]}...")  # Show first 3
    except Exception as e:
        print(f"Error parsing categories: {e}")

    # Modern usage - recommended approach
    print("\n=== Modern Usage ===")

    # Get complete module config with validation
    camera_config = config.get_module_config(ModuleConfig.CAMERA.value)
    print(f"Camera config: {camera_config}")

    # Get categories for different strategies
    print("\n=== Categories for Strategies ===")
    primary_cats = config.get_categories_for_strategy("primary")
    print(f"Primary strategy categories: {len(primary_cats)} total")

    if primary_cats:
        secondary_cats = config.get_categories_for_strategy("secondary", primary_cats[0])
        print(f"Secondary categories within '{primary_cats[0]}': {len(secondary_cats)} total")

    # Save configuration
    config.save_config()