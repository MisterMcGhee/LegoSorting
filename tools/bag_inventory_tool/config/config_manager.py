"""
Configuration manager for bag inventory tool

Follows the same pattern as enhanced_config_manager.py in the parent project
This is a standalone version - does NOT integrate with the main config yet
"""

import json
import os
import logging
from typing import Any, Dict, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BagInventoryModule(Enum):
    """
    Module names for configuration.

    Using enum prevents typos and provides IDE autocomplete.
    """
    PDF = "pdf"
    REFERENCE = "reference"
    OCR = "ocr"
    VALIDATION = "validation"
    OUTPUT = "output"


class ConfigSchema:
    """
    Configuration schema definitions and defaults.

    This is the source of truth for settings and their default values.
    """

    @staticmethod
    def get_module_schema(module_name: str) -> Dict[str, Any]:
        """
        Get complete schema with ALL default values for a module.

        Args:
            module_name: Name of the module

        Returns:
            Dictionary with all configuration keys and default values
        """
        schemas = {
            # =================================================================
            # PDF MODULE
            # Controls: PDF loading and page extraction
            # =================================================================
            BagInventoryModule.PDF.value: {
                "dpi": 300,  # Resolution for PDF rendering
                "color_mode": "RGB",  # Color mode for extracted pages
                "cache_enabled": True  # Cache rendered pages in memory
            },

            # =================================================================
            # REFERENCE MODULE
            # Controls: Reference page extraction settings
            # =================================================================
            BagInventoryModule.REFERENCE.value: {
                # Piece detection
                "min_piece_area": 500,  # Minimum piece contour area (pixels)
                "max_piece_area": 50000,  # Maximum piece contour area (pixels)
                "min_aspect_ratio": 0.2,  # Minimum width/height ratio
                "max_aspect_ratio": 5.0,  # Maximum width/height ratio

                # Grid detection
                "expected_columns": 3,  # Typical number of columns on reference page
                "column_tolerance": 0.15,  # Tolerance for column alignment (15%)

                # Image preprocessing
                "gaussian_blur_size": 5,  # Blur kernel for noise reduction
                "binary_threshold": 200,  # Threshold for binary conversion

                # Template creation
                "template_padding": 10,  # Padding around piece template (pixels)
                "normalize_size": True,  # Normalize template sizes
                "target_template_height": 200,  # Target height for normalized templates

                # CSV-first approach settings
                "piece_search_radius": 150  # How far to search for piece image from element ID text (pixels)
            },

            # =================================================================
            # OCR MODULE
            # Controls: OCR engine settings
            # =================================================================
            BagInventoryModule.OCR.value: {
                "engine": "easyocr",  # OCR engine to use
                "languages": ["en"],  # Languages for OCR
                "gpu": False,  # Use GPU acceleration (if available)
                "confidence_threshold": 0.3,  # Minimum confidence to accept OCR result

                # Text preprocessing
                "contrast_enhancement": True,  # Enhance contrast before OCR
                "denoise": True,  # Denoise image before OCR

                # Element ID validation
                "allowed_characters": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "min_element_id_length": 4,  # Minimum valid element ID length
                "max_element_id_length": 10,  # Maximum valid element ID length

                # CSV-first approach settings
                "fuzzy_match_threshold": 0.8  # Minimum similarity for fuzzy matching (0.0-1.0)
            },

            # =================================================================
            # VALIDATION MODULE
            # Controls: CSV validation and checking
            # =================================================================
            BagInventoryModule.VALIDATION.value: {
                "strict_mode": False,  # Fail if any piece doesn't validate
                "require_all_csv_pieces": False,  # Require all CSV pieces found in reference
                "allow_extra_pieces": True,  # Allow pieces in reference not in CSV
                "log_validation_failures": True  # Log pieces that fail validation
            },

            # =================================================================
            # OUTPUT MODULE
            # Controls: Output file generation
            # =================================================================
            BagInventoryModule.OUTPUT.value: {
                "base_directory": "output",  # Base output directory
                "save_templates": True,  # Save individual piece templates
                "save_originals": True,  # Save original piece images
                "save_debug_images": True,  # Save debug visualizations
                "generate_report": True,  # Generate extraction report
                "image_format": "png"  # Format for saved images
            }
        }

        return schemas.get(module_name, {})


class BagInventoryConfigManager:
    """
    Configuration manager for bag inventory tool.

    Provides thread-safe access to configuration with validation and defaults.
    Follows the same pattern as EnhancedConfigManager.
    """

    def __init__(self, config_path: str = "bag_inventory_config.json"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        # Load or create configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self._create_default_config()
        else:
            logger.info("No configuration file found, creating default")
            self._create_default_config()

    def _create_default_config(self) -> None:
        """Create a complete default configuration."""
        self.config = {
            "version": "0.1.0",
            "created": datetime.now().isoformat()
        }

        # Add defaults for every module
        for module in BagInventoryModule:
            module_name = module.value
            self.config[module_name] = ConfigSchema.get_module_schema(module_name)

        self.save_config()
        logger.info("Created default configuration")

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a module.

        Args:
            module_name: Name of module (use BagInventoryModule enum)

        Returns:
            Complete configuration dictionary with all keys present
        """
        # Get user configuration
        stored_config = self.config.get(module_name, {})

        # Get defaults
        defaults = ConfigSchema.get_module_schema(module_name)

        # Merge: user settings override defaults
        module_config = {**defaults, **stored_config}

        return module_config

    def update_module_config(self, module_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration for a module.

        Args:
            module_name: Name of module to update
            updates: Dictionary of settings to update

        Returns:
            True if update successful
        """
        # Get current config
        current = self.get_module_config(module_name)

        # Apply updates
        updated = {**current, **updates}

        # Store updated config
        self.config[module_name] = updated

        # Save to disk
        return self.save_config()

    def save_config(self) -> bool:
        """
        Save current configuration to file.

        Returns:
            True if save successful
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def get_all_config(self) -> Dict[str, Any]:
        """
        Get complete configuration for all modules.

        Returns:
            Complete configuration dictionary
        """
        complete_config = {}
        for module in BagInventoryModule:
            module_name = module.value
            complete_config[module_name] = self.get_module_config(module_name)
        return complete_config


def create_config_manager(config_path: str = "bag_inventory_config.json") -> BagInventoryConfigManager:
    """
    Create a configuration manager instance.

    This is the main entry point for getting configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized BagInventoryConfigManager instance
    """
    return BagInventoryConfigManager(config_path)
