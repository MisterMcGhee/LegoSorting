"""
enhanced_config_manager.py - Unified configuration management with validation

This module provides centralized configuration management with schema validation,
defaults handling, migration support, and standardized module interfaces.
"""

import json
import os
import shutil
import threading
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
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
                "calibration_mode": False,
                "min_bin_separation": 20,
                "movement_speed": 0.5,
                "settling_time": 0.2
            },

            ModuleConfig.ARDUINO_SERVO.value: {
                "port": "/dev/ttyACM0",
                "baud_rate": 9600,
                "timeout": 2.0,
                "connection_retries": 3,
                "retry_delay": 1.0,
                "simulation_mode": False
            },

            ModuleConfig.API.value: {
                "api_type": "brickognize",
                "api_key": "",
                "base_url": "https://api.brickognize.info/predict/",
                "timeout": 30,
                "retry_count": 3,
                "cache_enabled": True,
                "cache_ttl": 3600
            },

            ModuleConfig.THREADING.value: {
                "max_queue_size": 100,
                "worker_count": 1,
                "api_timeout": 30.0,
                "processing_timeout": 60.0,
                "shutdown_timeout": 5.0,
                "polling_interval": 0.01
            },

            ModuleConfig.UI.value: {
                "panels": {
                    "metrics_dashboard": True,
                    "processed_piece": True,
                    "sorting_information": True,
                    "help_overlay": True
                },
                "opacity": 0.7,
                "colors": {
                    "panel_bg": [30, 30, 30],
                    "text": [255, 255, 255],
                    "roi": [0, 255, 0],
                    "entry_zone": [255, 0, 0],
                    "exit_zone": [0, 0, 255]
                }
            },

            ModuleConfig.GUI.value: {
                "window_width": 1200,
                "window_height": 800,
                "theme": "dark",
                "auto_save": True,
                "save_interval": 300,
                "show_tooltips": True,
                "confirm_exit": True
            },

            ModuleConfig.EXIT_ZONE.value: {
                "enabled": True,
                "fall_time": 1.0,
                "cooldown_time": 0.5,
                "min_piece_spacing": 50,
                "trigger_threshold": 0.8
            },

            ModuleConfig.PIECE_IDENTIFIER.value: {
                "csv_path": "Lego_Categories.csv",
                "cache_predictions": True,
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

            # Apply updates
            if module_name not in self.config:
                self.config[module_name] = {}

            self.config[module_name] = updated

            # Notify observers
            self._notify_observers(module_name, updates)

            return True

    def save_config(self, create_backup: bool = True) -> bool:
        """
        Save configuration to file

        Args:
            create_backup: Whether to create backup before saving

        Returns:
            True if save successful
        """
        with self._lock:
            try:
                if create_backup and os.path.exists(self.config_path):
                    self._create_backup()

                # Write to temporary file first (atomic write)
                temp_path = f"{self.config_path}.tmp"
                with open(temp_path, 'w') as f:
                    json.dump(self.config, f, indent=4)

                # Move temp file to actual path
                shutil.move(temp_path, self.config_path)

                logger.info(f"Configuration saved to {self.config_path}")
                return True

            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                return False

    def _create_backup(self, suffix: str = "") -> Optional[str]:
        """Create backup of current configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_{timestamp}{suffix}.json"
        backup_path = os.path.join(self._backup_dir, backup_name)

        try:
            shutil.copy2(self.config_path, backup_path)
            logger.info(f"Created backup: {backup_path}")

            # Keep only last 10 backups
            self._cleanup_old_backups()

            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def _cleanup_old_backups(self, keep_count: int = 10) -> None:
        """Remove old backup files"""
        try:
            backups = sorted([
                f for f in os.listdir(self._backup_dir)
                if f.startswith("config_") and f.endswith(".json")
            ])

            if len(backups) > keep_count:
                for old_backup in backups[:-keep_count]:
                    os.remove(os.path.join(self._backup_dir, old_backup))
                    logger.debug(f"Removed old backup: {old_backup}")

        except Exception as e:
            logger.warning(f"Error cleaning up backups: {e}")

    def register_observer(self, callback: Callable) -> None:
        """
        Register a callback for configuration changes

        Args:
            callback: Function to call on config changes
                     Signature: callback(module_name: str, changes: dict)
        """
        with self._lock:
            if callback not in self._observers:
                self._observers.append(callback)

    def unregister_observer(self, callback: Callable) -> None:
        """Unregister a configuration change callback"""
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)

    def _notify_observers(self, module_name: str, changes: Dict[str, Any]) -> None:
        """Notify all observers of configuration changes"""
        for observer in self._observers:
            try:
                observer(module_name, changes)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")

    def export_config(self, path: str, modules: Optional[List[str]] = None) -> bool:
        """
        Export configuration to file

        Args:
            path: Path to export to
            modules: List of modules to export (None for all)

        Returns:
            True if export successful
        """
        with self._lock:
            try:
                export_data = {"version": self.config.get("version", ConfigVersion.CURRENT)}

                if modules:
                    for module in modules:
                        if module in self.config:
                            export_data[module] = self.config[module]
                else:
                    export_data = self.config.copy()

                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=4)

                logger.info(f"Configuration exported to {path}")
                return True

            except Exception as e:
                logger.error(f"Error exporting configuration: {e}")
                return False

    def import_config(self, path: str, modules: Optional[List[str]] = None,
                      merge: bool = True) -> bool:
        """
        Import configuration from file

        Args:
            path: Path to import from
            modules: List of modules to import (None for all)
            merge: Whether to merge or replace

        Returns:
            True if import successful
        """
        with self._lock:
            try:
                with open(path, 'r') as f:
                    import_data = json.load(f)

                # Create backup before import
                self._create_backup(suffix="_pre_import")

                if merge:
                    # Merge imported data
                    if modules:
                        for module in modules:
                            if module in import_data:
                                if module not in self.config:
                                    self.config[module] = {}
                                self.config[module].update(import_data[module])
                    else:
                        for key, value in import_data.items():
                            if key != "version":
                                if key not in self.config:
                                    self.config[key] = {}
                                if isinstance(value, dict):
                                    self.config[key].update(value)
                                else:
                                    self.config[key] = value
                else:
                    # Replace configuration
                    if modules:
                        for module in modules:
                            if module in import_data:
                                self.config[module] = import_data[module]
                    else:
                        self.config = import_data

                self.save_config(create_backup=False)
                logger.info(f"Configuration imported from {path}")
                return True

            except Exception as e:
                logger.error(f"Error importing configuration: {e}")
                return False

    def reset_module(self, module_name: str) -> bool:
        """
        Reset a module to default configuration

        Args:
            module_name: Name of module to reset

        Returns:
            True if reset successful
        """
        with self._lock:
            try:
                defaults = ConfigSchema.get_module_schema(module_name)
                self.config[module_name] = defaults

                self._notify_observers(module_name, defaults)

                logger.info(f"Reset {module_name} to defaults")
                return True

            except Exception as e:
                logger.error(f"Error resetting module {module_name}: {e}")
                return False

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get validation report for entire configuration

        Returns:
            Dictionary with validation results for each module
        """
        report = {
            "valid": True,
            "modules": {}
        }

        with self._lock:
            for module in ModuleConfig:
                module_name = module.value
                if module_name in self.config:
                    is_valid, errors = ConfigValidator.validate_module(
                        module_name,
                        self.config[module_name]
                    )

                    report["modules"][module_name] = {
                        "valid": is_valid,
                        "errors": errors
                    }

                    if not is_valid:
                        report["valid"] = False

        return report

    # Compatibility methods for existing code
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Legacy compatibility: Get a single configuration value"""
        with self._lock:
            if section in self.config and key in self.config[section]:
                return self.config[section][key]
            return default

    def set(self, section: str, key: str, value: Any) -> None:
        """Legacy compatibility: Set a single configuration value"""
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

    # Modern usage - recommended approach
    print("\n=== Modern Usage ===")

    # Get complete module config with validation
    camera_config = config.get_module_config(ModuleConfig.CAMERA.value)
    print(f"Camera config: {camera_config}")

    # Update module config with validation
    success = config.update_module_config(
        ModuleConfig.SORTING.value,
        {
            "strategy": "secondary",
            "target_primary_category": "Technic",
            "pre_assignments": {"Technic": 1, "Basic": 2}
        }
    )
    print(f"Update success: {success}")

    # Get validation report
    report = config.get_validation_report()
    print(f"Config valid: {report['valid']}")


    # Register for changes
    def on_config_change(module_name: str, changes: dict):
        print(f"Config changed - Module: {module_name}, Changes: {changes}")


    config.register_observer(on_config_change)

    print("\n=== Legacy Compatibility ===")

    # Legacy usage - still works for backward compatibility
    device_id = config.get("camera", "device_id", 0)
    print(f"Device ID: {device_id}")

    sorting_config = config.get_section("sorting")
    print(f"Sorting config: {sorting_config}")

    # Save configuration
    config.save_config()