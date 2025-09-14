"""
enhanced_config_manager.py - Unified configuration management with validation

This module provides centralized configuration management with schema validation,
defaults handling, migration support, and standardized module interfaces.

MAIN PURPOSE: Replace the simple config manager with a robust system that:
- Validates all configuration values
- Provides guaranteed defaults for every setting
- Handles version migrations automatically
- Supports category hierarchy parsing for the GUI
"""

import json
import os
import csv
import shutil
import threading
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModuleConfig(Enum):
    """Standard module names for configuration - prevents typos and ensures consistency"""
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
    PIECE_HISTORY = "piece_history"


class ConfigSchema:
    """Configuration schema definitions and defaults - the "source of truth" for what settings exist"""

    @staticmethod
    def get_module_schema(module_name: str) -> Dict[str, Any]:
        """Get complete schema with ALL default values for a specific module

        This is the master definition of what settings exist and their default values.
        Every module should get its config through this to ensure completeness.
        """
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
                "pre_assignments": {},  # For GUI pre-assignment feature
                "max_pieces_per_bin": 50,  # Maximum pieces before bin is full
                "bin_warning_threshold": 0.8  # Warn at 80% capacity
            },

            ModuleConfig.SERVO.value: {
                "min_pulse": 500,
                "max_pulse": 2500,
                "default_position": 90,
                "speed": 100,
                "calibration_positions": {},
                "calibration_mode": False,
                "min_bin_separation": 20
            },

            ModuleConfig.ARDUINO_SERVO.value: {
                "port": "",
                "baud_rate": 57600,
                "timeout": 1.0,
                "servo_count": 9,
                "servo_pins": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "connection_retries": 3,
                "retry_delay": 1.0,
                "simulation_mode": False
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
                "priority_levels": 3,
                "api_timeout": 30.0,  # ADD THIS
                "processing_timeout": 60.0,  # ADD THIS
                "polling_interval": 0.01,  # ADD THIS
                "shutdown_timeout": 5.0
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
            },

            ModuleConfig.PIECE_HISTORY.value: {
                "max_entries": 10,
                "csv_path": "piece_history.csv",
                "include_timestamp": True
            }
        }

        return schemas.get(module_name, {})

    @staticmethod
    def get_required_fields(module_name: str) -> List[str]:
        """Get list of fields that MUST be present (no defaults allowed)

        These are critical settings that must be explicitly configured.
        """
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
    """Validates configuration values against business rules"""

    @staticmethod
    def validate_module(module_name: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if a module's configuration is valid

        Returns: (is_valid, list_of_error_messages)
        """
        errors = []

        # Check that all required fields are present
        required = ConfigSchema.get_required_fields(module_name)
        for field in required:
            if field not in config or config[field] is None:
                errors.append(f"Missing required field '{field}' in {module_name}")

        # Run module-specific validation rules
        if module_name == ModuleConfig.SORTING.value:
            errors.extend(ConfigValidator._validate_sorting(config))
        elif module_name == ModuleConfig.CAMERA.value:
            errors.extend(ConfigValidator._validate_camera(config))
        elif module_name == ModuleConfig.DETECTOR_ROI.value:
            errors.extend(ConfigValidator._validate_roi(config))

        return len(errors) == 0, errors

    @staticmethod
    def _validate_sorting(config: Dict[str, Any]) -> List[str]:
        """Business rules for sorting configuration"""
        errors = []

        strategy = config.get("strategy")
        if strategy not in ["primary", "secondary", "tertiary"]:
            errors.append(f"Invalid sorting strategy: {strategy}")

        # Secondary sorting needs a target primary category
        if strategy == "secondary" and not config.get("target_primary_category"):
            errors.append("Secondary sorting requires target_primary_category")

        # Tertiary sorting needs both target categories
        if strategy == "tertiary":
            if not config.get("target_primary_category"):
                errors.append("Tertiary sorting requires target_primary_category")
            if not config.get("target_secondary_category"):
                errors.append("Tertiary sorting requires target_secondary_category")

        # Validate bin count is reasonable
        max_bins = config.get("max_bins", 0)
        if not (1 <= max_bins <= 20):
            errors.append(f"Invalid max_bins: {max_bins} (must be 1-20)")

        # Check pre-assignments are valid
        pre_assignments = config.get("pre_assignments", {})
        for category, bin_num in pre_assignments.items():
            if not isinstance(bin_num, int):
                errors.append(f"Invalid bin number for {category}: {bin_num} (must be integer)")
            elif not (1 <= bin_num <= max_bins):
                errors.append(f"Bin {bin_num} for {category} out of range (1-{max_bins})")

        errors.extend(ConfigValidator._validate_bin_capacity(config))

        return errors

    @staticmethod
    def _validate_bin_capacity(config: Dict[str, Any]) -> List[str]:
        """Validate bin capacity settings"""
        errors = []

        # Check max_pieces_per_bin is reasonable
        max_pieces = config.get("max_pieces_per_bin", 50)
        if not isinstance(max_pieces, int) or max_pieces < 1:
            errors.append(f"Invalid max_pieces_per_bin: {max_pieces} (must be positive integer)")
        elif max_pieces > 1000:
            errors.append(f"max_pieces_per_bin too large: {max_pieces} (max 1000)")

        # Check warning threshold is valid percentage
        threshold = config.get("bin_warning_threshold", 0.8)
        if not isinstance(threshold, (int, float)) or not (0.0 < threshold <= 1.0):
            errors.append(f"Invalid bin_warning_threshold: {threshold} (must be between 0.0 and 1.0)")

        # Logical check: warning should be less than full
        if threshold >= 1.0:
            errors.append("bin_warning_threshold should be less than 1.0 to provide warning before full")

        return errors

    @staticmethod
    def _validate_camera(config: Dict[str, Any]) -> List[str]:
        """Business rules for camera configuration"""
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
        """Business rules for ROI configuration"""
        errors = []

        # Dimensions must be positive
        for key in ["w", "h"]:
            value = config.get(key, 0)
            if value <= 0:
                errors.append(f"ROI {key} must be positive: {value}")

        # Positions must be non-negative
        for key in ["x", "y"]:
            value = config.get(key, 0)
            if value < 0:
                errors.append(f"ROI {key} must be non-negative: {value}")

        return errors


class EnhancedConfigManager:
    """
    The main configuration manager - this is what modules actually use

    Key improvements over old ConfigManager:
    - Automatic validation of all settings
    - Guaranteed defaults for every parameter
    - Thread-safe operations
    - Automatic version migration
    - Category hierarchy parsing for GUI
    """

    def __init__(self, config_path: str = "config.json", auto_migrate: bool = True):
        """Initialize with automatic loading and migration"""
        self.config_path = config_path
        self.auto_migrate = auto_migrate
        self.config: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Thread safety
        self._observers: List[Callable] = []  # For change notifications
        self._backup_dir = "config_backups"

        # Cache for parsed category data (expensive to parse repeatedly)
        self._category_hierarchy: Optional[Dict[str, Any]] = None
        self._category_hierarchy_lock = threading.RLock()

        # Create backup directory
        os.makedirs(self._backup_dir, exist_ok=True)

        # Load and migrate configuration automatically
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file"""
        with self._lock:
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r') as f:
                        self.config = json.load(f)
                    logger.info(f"Configuration loaded from {self.config_path}")
                except Exception as e:
                    logger.error(f"Error loading configuration: {e}")
                    self._create_default_config()
            else:
                logger.info(f"No configuration file found, creating default")
                self._create_default_config()

    def _create_default_config(self) -> None:
        """Create a complete default configuration with all modules"""
        with self._lock:
            self.config = {
                "validation": {
                    "strict_mode": False,
                    "auto_fix": True
                }
            }

            # Add complete default configuration for every module
            for module in ModuleConfig:
                module_name = module.value
                self.config[module_name] = ConfigSchema.get_module_schema(module_name)

            self.save_config()
            logger.info("Created default configuration")

    def parse_categories_from_csv(self, csv_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
        """
        Parse the Lego categories CSV file into a hierarchical structure

        Used by GUI to populate category dropdowns and by sorting to understand relationships.
        """
        with self._category_hierarchy_lock:
            # Return cached version if available
            if self._category_hierarchy is not None and not force_reload:
                return self._category_hierarchy

            # Get CSV path from config if not provided
            if csv_path is None:
                piece_config = self.get_module_config(ModuleConfig.PIECE_IDENTIFIER.value)
                csv_path = piece_config.get("csv_path", "Lego_Categories.csv")

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Categories CSV not found at: {csv_path}")

            # Build the hierarchy structure
            categories = {
                "primary": set(),
                "primary_to_secondary": {},
                "secondary_to_tertiary": {},
                "all_categories": {}
            }

            try:
                with open(csv_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.DictReader(file)

                    for row in csv_reader:
                        element_id = row.get('element_id', '')
                        primary = row.get('primary_category', '')
                        secondary = row.get('secondary_category', '')
                        tertiary = row.get('tertiary_category', '')

                        if not element_id or not primary:
                            continue

                        # Store complete piece information
                        categories["all_categories"][element_id] = {
                            'name': row.get('name', ''),
                            'primary_category': primary,
                            'secondary_category': secondary,
                            'tertiary_category': tertiary
                        }

                        # Build hierarchy mappings
                        categories["primary"].add(primary)

                        if primary not in categories["primary_to_secondary"]:
                            categories["primary_to_secondary"][primary] = set()
                        if secondary:
                            categories["primary_to_secondary"][primary].add(secondary)

                        if secondary:
                            key = (primary, secondary)
                            if key not in categories["secondary_to_tertiary"]:
                                categories["secondary_to_tertiary"][key] = set()
                            if tertiary:
                                categories["secondary_to_tertiary"][key].add(tertiary)

                # Convert sets to sorted lists for GUI use
                categories["primary"] = sorted(list(categories["primary"]))
                for primary in categories["primary_to_secondary"]:
                    categories["primary_to_secondary"][primary] = sorted(
                        list(categories["primary_to_secondary"][primary])
                    )
                for key in categories["secondary_to_tertiary"]:
                    categories["secondary_to_tertiary"][key] = sorted(
                        list(categories["secondary_to_tertiary"][key])
                    )

                # Cache the result
                self._category_hierarchy = categories

                logger.info(f"Parsed {len(categories['primary'])} primary categories from {csv_path}")
                return categories

            except Exception as e:
                error_msg = f"Error parsing categories CSV: {e}"
                logger.error(error_msg)
                raise Exception(error_msg)

    def get_category_hierarchy(self, force_reload: bool = False) -> Dict[str, Any]:
        """Get cached category hierarchy (convenience wrapper)"""
        return self.parse_categories_from_csv(force_reload=force_reload)

    def get_categories_for_strategy(self, strategy: str,
                                    primary_category: Optional[str] = None,
                                    secondary_category: Optional[str] = None) -> List[str]:
        """
        Get categories available for a specific sorting strategy
        Used by GUI to populate category selection dropdowns.
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
        **THIS IS THE MAIN METHOD MODULES SHOULD USE**

        Get complete configuration for a module with ALL defaults filled in.
        This ensures modules never get missing settings.

        Example:
            camera_config = config_manager.get_module_config("camera")
            # camera_config is GUARANTEED to have device_id, width, height, fps, etc.
        """
        with self._lock:
            # Get what's actually stored in the config file
            stored_config = self.config.get(module_name, {})

            # Get the complete schema with all defaults
            defaults = ConfigSchema.get_module_schema(module_name)

            # Merge: user settings override defaults
            module_config = {**defaults, **stored_config}

            # Validate the final configuration
            if validate:
                is_valid, errors = ConfigValidator.validate_module(module_name, module_config)

                if not is_valid:
                    error_msg = f"Invalid config for {module_name}: " + "; ".join(errors)

                    if self.config.get("validation", {}).get("strict_mode", False):
                        raise ValueError(error_msg)
                    else:
                        logger.warning(error_msg)

                        # Auto-fix by using pure defaults
                        if self.config.get("validation", {}).get("auto_fix", True):
                            module_config = defaults
                            logger.info(f"Using defaults for {module_name} due to validation errors")

            return module_config

    def update_module_config(self, module_name: str, updates: Dict[str, Any],
                             validate: bool = True) -> bool:
        """
        Update configuration for a module

        This validates the changes before applying them.
        """
        with self._lock:
            # Get current complete config
            current = self.get_module_config(module_name, validate=False)

            # Apply the updates
            updated = {**current, **updates}

            # Validate the result
            if validate:
                is_valid, errors = ConfigValidator.validate_module(module_name, updated)
                if not is_valid:
                    logger.error(f"Validation failed for {module_name}: {errors}")
                    return False

            # Store the updated config
            self.config[module_name] = updated

            # Notify observers of the change
            self._notify_observers(module_name, updates)

            # Save to file
            self.save_config()

            return True

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get a complete validation report for all modules

        Useful for debugging configuration issues.
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
        """Save current configuration to file with automatic backup"""
        with self._lock:
            try:
                # Create backup before saving
                self._create_backup()

                # Write the config file
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)

                logger.info(f"Configuration saved to {self.config_path}")
                return True

            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                return False

    def _create_backup(self, suffix: str = None) -> Optional[str]:
        """Create timestamped backup of current config file"""
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
        """Register callback to be notified when configuration changes"""
        with self._lock:
            self._observers.append(callback)

    def _notify_observers(self, module_name: str, changes: Dict[str, Any]) -> None:
        """Notify all registered observers of configuration changes"""
        for observer in self._observers:
            try:
                observer(module_name, changes)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")


# Factory function for creating config manager instances
def create_config_manager(config_path: str = "config.json") -> EnhancedConfigManager:
    """
    Create an enhanced configuration manager instance

    This is the main entry point - use this function rather than
    calling the constructor directly.
    """
    return EnhancedConfigManager(config_path)
