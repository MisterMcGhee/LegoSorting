"""
enhanced_config_manager.py - Unified configuration management with validation

This module provides centralized configuration management with schema validation,
defaults handling, and standardized module interfaces.

MAIN PURPOSE:
- Validates all configuration values against business rules
- Provides guaranteed defaults for every setting
- Supports configuration changes with automatic backup
- Thread-safe access for multi-threaded applications

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


class ModuleConfig(Enum):
    """
    Standard module names for configuration.

    Using an enum prevents typos and provides autocomplete in IDEs.
    Every module should reference its config using these constants.

    Example:
        config = config_manager.get_module_config(ModuleConfig.CAMERA.value)
    """
    CAMERA = "camera"
    DETECTOR = "detector"
    DETECTOR_ROI = "detector_roi"
    SORTING = "sorting"
    SERVO = "servo"
    ARDUINO_SERVO = "arduino_servo"
    API = "api"
    SYSTEM = "system"
    UI = "ui"
    GUI = "gui_settings"
    PIECE_IDENTIFIER = "piece_identifier"
    PROCESSING_QUEUE = "processing_queue"
    PROCESSING_WORKERS = "processing_workers"


class ConfigSchema:
    """
    Configuration schema definitions and defaults.

    This is the "source of truth" for what settings exist and their default values.
    Every module should get its config through this to ensure completeness.
    """

    @staticmethod
    def get_module_schema(module_name: str) -> Dict[str, Any]:
        """
        Get complete schema with ALL default values for a specific module.

        This ensures modules never encounter missing configuration values.
        User settings will override these defaults when config.json is loaded.

        Args:
            module_name: Name of the module (use ModuleConfig enum values)

        Returns:
            Dictionary with all configuration keys and default values

        Example:
            camera_defaults = ConfigSchema.get_module_schema("camera")
            # Returns: {"device_id": 0, "width": 1920, "height": 1080, ...}
        """
        schemas = {
            # =================================================================
            # CAMERA MODULE
            # Used by: camera_handler.py
            # Controls: OpenCV camera interface settings
            # =================================================================
            ModuleConfig.CAMERA.value: {
                "device_id": 0,  # Camera index (0 = first camera)
                "buffer_size": 1,  # Frame buffer size (1 = no buffering for real-time)
                "width": 1920,  # Frame width in pixels
                "height": 1080,  # Frame height in pixels
                "fps": 30,  # Target frames per second
                "auto_exposure": True  # Enable camera auto-exposure
            },

            # =================================================================
            # DETECTOR MODULE
            # Used by: All detector pipeline modules
            # Controls: Computer vision detection parameters
            # =================================================================
            ModuleConfig.DETECTOR.value: {
                "detector_type": "conveyor",  # Detection mode (conveyor/static)

                # Blob detection - filters what counts as a piece
                "min_area": 1000,  # Minimum contour area in pixels
                "max_area": 50000,  # Maximum contour area in pixels
                "min_aspect_ratio": 0.3,  # Minimum width/height ratio
                "max_aspect_ratio": 3.0,  # Maximum width/height ratio
                "edge_margin": 10,  # Pixels from frame edge to ignore
                "debug_mode": False,  # Show debug visualizations

                # Background subtraction - removes static background
                "bg_history": 500,  # Frames to average for background model
                "bg_threshold": 800.0,  # Threshold for foreground detection
                "learn_rate": 0.005,  # How fast background adapts

                # Noise reduction - cleans up detection mask
                "gaussian_blur_size": 7,  # Blur kernel size (must be odd)
                "morph_kernel_size": 5,  # Morphological operation kernel size

                # Quality filtering - rejects bad detections
                "min_contour_points": 5,  # Minimum points in contour

                # Piece tracker - tracks pieces across frames
                "match_distance_threshold": 100.0,  # Max pixels to match same piece
                "match_x_weight": 2.0,  # Horizontal position weight
                "match_y_weight": 1.0,  # Vertical position weight
                "min_updates_for_stability": 3,  # Updates before piece stable
                "piece_timeout_seconds": 2.0,  # Seconds before piece expires
                "fully_in_frame_margin": 20,  # Pixels from edge = "in frame"
                "min_velocity_samples": 2,  # Samples needed to calc velocity
                "max_velocity_change": 50.0,  # Max velocity change per frame

                # ROI zones - divides frame into entry/middle/exit zones
                "entry_zone_percent": 0.15,  # Entry zone as % of ROI width
                "exit_zone_percent": 0.15,  # Exit zone as % of ROI width
                "zone_transition_debounce": 0.1,  # Seconds to debounce zone changes

                # Capture coordinator - when to capture piece images
                "min_stability_updates": 5,  # Updates before capture allowed
                "capture_cooldown_seconds": 0.5,  # Seconds between captures
                "max_concurrent_processing": 3,  # Max pieces processing at once
                "crop_padding": 20,  # Padding around piece crop (pixels)
                "id_label_height": 40,  # Height of ID label on saved images
                "id_font_scale": 1.5,  # Font size for ID labels
                "id_font_thickness": 2,  # Font thickness for ID labels
                "save_captured_images": True,  # Save images to disk
                "capture_directory": "LegoPictures"  # Directory for saved images
            },

            # =================================================================
            # DETECTOR ROI (Region of Interest)
            # Used by: detector modules
            # Controls: Where in the frame to look for pieces
            # =================================================================
            ModuleConfig.DETECTOR_ROI.value: {
                "x": 125,  # ROI left edge (pixels from left)
                "y": 200,  # ROI top edge (pixels from top)
                "w": 1550,  # ROI width in pixels
                "h": 500,  # ROI height in pixels
                "entry_zone_width": 150,  # Width of entry zone (pixels)
                "exit_zone_width": 150  # Width of exit zone (pixels)
            },

            # =================================================================
            # SORTING MODULE
            # Used by: bin_assignment_module.py
            # Controls: How pieces are sorted into bins
            # =================================================================
            ModuleConfig.SORTING.value: {
                "strategy": "primary",  # Sorting level: "primary", "secondary", or "tertiary"
                "target_primary_category": "",  # For secondary/tertiary: which primary to sort
                "target_secondary_category": "",  # For tertiary: which secondary to sort
                "max_bins": 7,  # Total bins available (bin 0 is overflow)
                "overflow_bin": 0,  # Bin number for unknown/overflow pieces
                "confidence_threshold": 0.7,  # Minimum API confidence to accept (0.0-1.0)
                "pre_assignments": {},  # Pre-assign categories to bins: {"Basic": 1, "Technic": 4}
                "max_pieces_per_bin": 50,  # ADD THIS
                "bin_warning_threshold": 0.8
            },

            # =================================================================
            # ARDUINO SERVO MODULE
            # Used by: arduino_servo_controller.py (future hardware package)
            # Controls: Arduino communication for servo control
            # =================================================================
            # In enhanced_config_manager.py - Replace BOTH servo sections with this ONE:

            ModuleConfig.ARDUINO_SERVO.value: {
                # Connection settings
                "port": "",  # Serial port (e.g., "COM3" or "/dev/cu.usbmodem...")
                "baud_rate": 57600,  # Serial communication speed
                "timeout": 1.0,  # Serial timeout in seconds
                "connection_retries": 3,  # Connection retry attempts
                "retry_delay": 1.0,  # Seconds between retries
                "simulation_mode": False,  # Run without real Arduino (testing)

                # Servo hardware parameters
                "min_pulse": 500,  # Minimum servo pulse width (microseconds)
                "max_pulse": 2500,  # Maximum servo pulse width (microseconds)
                "default_position": 90,  # Default/home servo angle (degrees)
                "min_bin_separation": 20,  # Minimum degrees between bins

                # Bin position mapping
                "bin_positions": {}  # Calibrated positions: {bin_number: angle}
            },

            # =================================================================
            # API MODULE
            # Used by: identification_api_handler.py
            # Controls: Brickognize API communication
            # =================================================================
            ModuleConfig.API.value: {
                "api_type": "rebrickable",  # API service type (for future expansion)
                "api_key": "",  # API key (if required)
                "cache_enabled": True,  # Cache API responses
                "cache_dir": "api_cache",  # Directory for cached responses
                "timeout": 10.0,  # API request timeout (seconds)
                "retry_count": 3,  # Number of retry attempts
                "rate_limit": 5.0  # Requests per second limit
            },

            # =================================================================
            # System MODULE
            # Used by: system_tab.py in the configuration_gui
            # Controls:
            # =================================================================
            ModuleConfig.SYSTEM.value: {
                "threading_enabled": True,
                "log_level": "INFO"
            },
            # =================================================================
            # UI MODULE
            # Used by: display modules (visualization)
            # Controls: On-screen display settings
            # =================================================================
            ModuleConfig.UI.value: {
                "display_enabled": True,  # Show live video display
                "window_width": 1280,  # Display window width
                "window_height": 720,  # Display window height
                "show_debug_info": False,  # Show debug overlays
                "show_fps": True,  # Show FPS counter
                "overlay_alpha": 0.7  # Transparency of overlays (0.0-1.0)
            },

            # =================================================================
            # GUI MODULE
            # Used by: config_gui_module.py
            # Controls: Configuration GUI settings
            # =================================================================
            ModuleConfig.GUI.value: {
                "theme": "dark",  # GUI theme ("dark" or "light")
                "auto_save": True,  # Auto-save config changes
                "save_interval": 60,  # Auto-save interval (seconds)
                "show_tooltips": True,  # Show helpful tooltips
                "confirm_exit": True  # Confirm before closing GUI
            },

            # =================================================================
            # PIECE IDENTIFIER MODULE
            # Used by: category_lookup_module.py and category_hierarchy_service.py
            # Controls: Category database settings
            # =================================================================
            ModuleConfig.PIECE_IDENTIFIER.value: {
                "csv_path": "./Lego_Categories.csv",
                "confidence_threshold": 0.7,  # Minimum confidence for identification
                "save_unknown": True,  # Log unknown pieces to CSV
                "unknown_dir": "unknown_pieces",  # Directory for unknown piece logs
                "log_sorted_pieces": True,  # Enable/disable sorted piece logging
                "sorted_pieces_path": "sorted_pieces.csv"  # Path to sorted pieces log file
            },

            # =================================================================
            # PROCESSING QUEUE MODULE
            # Used by: processing_queue_manager.py
            # Controls: Queue between detection and processing pipelines
            # =================================================================
            ModuleConfig.PROCESSING_QUEUE.value: {
                "queue_size": 100,  # <-- VERIFY THIS EXISTS
                "priority_strategy": "exit_zone_first",
                "enable_batching": False,
                "batch_size": 5,
                "batch_timeout": 2.0,
            },

            # =================================================================
            # PROCESSING WORKERS MODULE
            # Used by: processing_worker.py
            # Controls: Worker threads for processing pipeline
            # =================================================================
            ModuleConfig.PROCESSING_WORKERS.value: {
                "num_workers": 3,  # Number of worker threads
                "worker_timeout": 1.0  # Timeout for queue.get() calls (seconds)
            }
        }

        return schemas.get(module_name, {})

    @staticmethod
    def get_required_fields(module_name: str) -> List[str]:
        """
        Get list of fields that MUST be present in configuration.

        These are critical settings that must be explicitly configured.
        The validator will fail if these are missing.

        Args:
            module_name: Name of the module to check

        Returns:
            List of required field names

        Example:
            required = ConfigSchema.get_required_fields("camera")
            # Returns: ["device_id"]
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
    """
    Validates configuration values against business rules.

    This ensures configurations are logically valid before they're used.
    For example, it checks that bin numbers are in valid ranges, required
    fields are present, and strategy settings are consistent.
    """

    @staticmethod
    def validate_module(module_name: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if a module's configuration is valid.

        Runs both generic validation (required fields present) and
        module-specific business logic validation.

        Args:
            module_name: Name of module to validate
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)

        Example:
            is_valid, errors = ConfigValidator.validate_module("sorting", config)
            if not is_valid:
                print(f"Errors: {errors}")
        """
        errors = []

        # Check required fields are present
        required = ConfigSchema.get_required_fields(module_name)
        for field in required:
            if field not in config or config[field] is None:
                errors.append(f"Missing required field '{field}' in {module_name}")

        # Run module-specific validation
        if module_name == ModuleConfig.SORTING.value:
            errors.extend(ConfigValidator._validate_sorting(config))
        elif module_name == ModuleConfig.CAMERA.value:
            errors.extend(ConfigValidator._validate_camera(config))
        elif module_name == ModuleConfig.DETECTOR_ROI.value:
            errors.extend(ConfigValidator._validate_roi(config))

        return len(errors) == 0, errors

    @staticmethod
    def _validate_sorting(config: Dict[str, Any]) -> List[str]:
        """
        Validate sorting configuration business rules.

        Checks:
        - Strategy is valid ("primary", "secondary", or "tertiary")
        - Secondary strategy has target_primary_category set
        - Tertiary strategy has both target categories set
        - Bin count is reasonable (1-20)
        - Pre-assignments reference valid bin numbers

        Args:
            config: Sorting module configuration

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        strategy = config.get("strategy")
        if strategy not in ["primary", "secondary", "tertiary"]:
            errors.append(f"Invalid sorting strategy: {strategy}")

        # Secondary sorting requires specifying which primary to sort
        if strategy == "secondary" and not config.get("target_primary_category"):
            errors.append("Secondary sorting requires target_primary_category")

        # Tertiary sorting requires both primary and secondary targets
        if strategy == "tertiary":
            if not config.get("target_primary_category"):
                errors.append("Tertiary sorting requires target_primary_category")
            if not config.get("target_secondary_category"):
                errors.append("Tertiary sorting requires target_secondary_category")

        # Validate bin count is reasonable
        max_bins = config.get("max_bins", 0)
        if not (1 <= max_bins <= 20):
            errors.append(f"Invalid max_bins: {max_bins} (must be 1-20)")

        # Check pre-assignments reference valid bins
        pre_assignments = config.get("pre_assignments", {})
        for category, bin_num in pre_assignments.items():
            if not isinstance(bin_num, int):
                errors.append(f"Invalid bin number for {category}: {bin_num}")
            elif not (1 <= bin_num <= max_bins):
                errors.append(f"Bin {bin_num} for {category} out of range (1-{max_bins})")

        return errors

    @staticmethod
    def _validate_camera(config: Dict[str, Any]) -> List[str]:
        """
        Validate camera configuration business rules.

        Checks:
        - Device ID is a non-negative integer
        - FPS is reasonable (1-120)

        Args:
            config: Camera module configuration

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        device_id = config.get("device_id")
        if not isinstance(device_id, int) or device_id < 0:
            errors.append(f"Invalid device_id: {device_id}")

        fps = config.get("fps", 30)
        if not (1 <= fps <= 120):
            errors.append(f"Invalid fps: {fps}")

        return errors

    @staticmethod
    def _validate_roi(config: Dict[str, Any]) -> List[str]:
        """
        Validate ROI (Region of Interest) configuration business rules.

        Checks:
        - Width and height are positive
        - X and Y positions are non-negative

        Args:
            config: ROI module configuration

        Returns:
            List of error messages (empty if valid)
        """
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
    Main configuration manager - this is what modules use to access configuration.

    Provides:
    - Thread-safe configuration access
    - Automatic validation
    - Defaults for all settings
    - Configuration change notifications
    - Automatic backup before saves

    Usage:
        config_manager = create_config_manager()
        camera_config = config_manager.get_module_config("camera")
    """

    def __init__(self, config_path: str = "config.json", auto_migrate: bool = True):
        """
        Initialize configuration manager with automatic loading.

        Called once during application startup. Loads existing config.json
        or creates a default one if it doesn't exist.

        Args:
            config_path: Path to configuration file
            auto_migrate: Automatically handle config version migrations (future use)
        """
        self.config_path = config_path
        self.auto_migrate = auto_migrate
        self.config: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Thread-safe access
        self._observers: List[Callable] = []  # Change notification callbacks
        self._backup_dir = "config_backups"

        # Create backup directory if it doesn't exist
        os.makedirs(self._backup_dir, exist_ok=True)

        # Load configuration from file
        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from file.

        Called during initialization. If config.json doesn't exist or is
        invalid, creates a new default configuration.
        """
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
        """
        Create a complete default configuration with all modules.

        Called when config.json is missing or corrupted. Creates a new
        configuration with all default values from schemas.
        """
        with self._lock:
            self.config = {
                "validation": {
                    "strict_mode": False,  # Fail on validation errors vs. use defaults
                    "auto_fix": True  # Automatically fix invalid configs
                }
            }

            # Add defaults for every module
            for module in ModuleConfig:
                module_name = module.value
                self.config[module_name] = ConfigSchema.get_module_schema(module_name)

            self.save_config()
            logger.info("Created default configuration")

    def get_module_config(self, module_name: str, validate: bool = True) -> Dict[str, Any]:
        """
        Get complete configuration for a module with ALL defaults filled in.

        THIS IS THE PRIMARY METHOD MODULES SHOULD USE.

        Returns a complete configuration by merging:
        1. Schema defaults (guaranteed to have all keys)
        2. User settings from config.json (overrides defaults)

        The result is validated against business rules unless validate=False.

        Args:
            module_name: Name of module (use ModuleConfig enum values)
            validate: Whether to validate the configuration

        Returns:
            Complete configuration dictionary with all keys present

        Example:
            config_manager = create_config_manager()
            camera_config = config_manager.get_module_config("camera")
            device_id = camera_config["device_id"]  # Guaranteed to exist
        """
        with self._lock:
            # Get what user has configured
            stored_config = self.config.get(module_name, {})

            # Get complete schema with all defaults
            defaults = ConfigSchema.get_module_schema(module_name)

            # Merge: user settings override defaults
            module_config = {**defaults, **stored_config}

            # Validate the final configuration
            if validate:
                is_valid, errors = ConfigValidator.validate_module(module_name, module_config)

                if not is_valid:
                    error_msg = f"Invalid config for {module_name}: " + "; ".join(errors)

                    # In strict mode, fail hard
                    if self.config.get("validation", {}).get("strict_mode", False):
                        raise ValueError(error_msg)
                    else:
                        logger.warning(error_msg)

                        # Auto-fix by falling back to pure defaults
                        if self.config.get("validation", {}).get("auto_fix", True):
                            module_config = defaults
                            logger.info(f"Using defaults for {module_name}")

            return module_config

    def update_module_config(self, module_name: str, updates: Dict[str, Any],
                             validate: bool = True) -> bool:
        """
        Update configuration for a module.

        Validates the changes before applying them. If validation fails,
        the configuration is not updated.

        Args:
            module_name: Name of module to update
            updates: Dictionary of settings to update (partial updates allowed)
            validate: Whether to validate before saving

        Returns:
            True if update successful, False if validation failed

        Example:
            success = config_manager.update_module_config(
                "camera",
                {"fps": 60, "width": 1280}
            )
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

            # Notify any registered observers
            self._notify_observers(module_name, updates)

            # Save to disk
            self.save_config()

            return True

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get a complete validation report for all modules.

        Useful for debugging configuration issues or for displaying
        in a configuration GUI.

        Returns:
            Dictionary with validation status for each module

        Example:
            report = config_manager.get_validation_report()
            if not report["valid"]:
                for module, info in report["modules"].items():
                    if not info["valid"]:
                        print(f"{module} errors: {info['errors']}")
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
        Save current configuration to file with automatic backup.

        Creates a timestamped backup before saving. This prevents
        data loss if something goes wrong during save.

        Returns:
            True if save successful, False if error occurred
        """
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
        """
        Create timestamped backup of current config file.

        Called automatically before saving. Backups allow recovery
        if a save corrupts the configuration.

        Args:
            suffix: Optional suffix for backup filename (uses timestamp if None)

        Returns:
            Path to backup file, or None if backup failed
        """
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
        Register callback to be notified when configuration changes.

        Useful for modules that need to react to configuration changes
        without polling.

        Args:
            callback: Function to call when config changes
                      Signature: callback(module_name: str, changes: Dict[str, Any])

        Example:
            def on_config_change(module_name, changes):
                print(f"{module_name} updated: {changes}")

            config_manager.register_observer(on_config_change)
        """
        with self._lock:
            self._observers.append(callback)

    def _notify_observers(self, module_name: str, changes: Dict[str, Any]) -> None:
        """
        Notify all registered observers of configuration changes.

        Called automatically after successful config updates.
        Catches exceptions in callbacks to prevent one bad callback
        from breaking the notification system.

        Args:
            module_name: Name of module that changed
            changes: Dictionary of changed values
        """
        for observer in self._observers:
            try:
                observer(module_name, changes)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_config_manager(config_path: str = "config.json") -> EnhancedConfigManager:
    """
    Create an enhanced configuration manager instance.

    This is the main entry point - use this function rather than
    calling the constructor directly. It provides a consistent
    interface across the application.

    Args:
        config_path: Path to configuration file (default: "config.json")

    Returns:
        Initialized EnhancedConfigManager instance

    Example:
        # During application startup
        config_manager = create_config_manager()

        # Pass to modules that need configuration
        camera = CameraHandler(config_manager)
        detector = ConveyorDetector(config_manager)
    """
    return EnhancedConfigManager(config_path)