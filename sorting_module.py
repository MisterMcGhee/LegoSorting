"""
sorting_module.py - Module for sorting Lego pieces into bins by category

This module handles the sorting logic for identified Lego pieces,
determining which bin each piece should go into based on various
sorting strategies and configuration.
"""

import csv
import logging
from typing import Dict, Any, Optional, List
import os
import time

from error_module import SortingError

# Set up module logger
logger = logging.getLogger(__name__)


class SortingStrategy:
    """Base class for Lego piece sorting with dynamic bin assignment."""

    def __init__(self, config: Dict[str, Any], categories_data: Dict[str, Dict[str, str]]):
        """Initialize sorting strategy.

        Args:
            config: Configuration dictionary
            categories_data: Dictionary mapping element_ids to category information
        """
        self.config = config
        self.categories_data = categories_data
        self.overflow_bin = config.get("overflow_bin", 9)

        # For dynamic bin assignment
        self.category_to_bin = {}  # Maps category keys to bin numbers
        self.next_available_bin = 0
        self.max_bins = config.get("max_bins", 9)  # Maximum number of bins (excluding overflow)

        logger.debug(f"Sorting strategy initialized: max_bins={self.max_bins}, overflow_bin={self.overflow_bin}")

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Determine bin for a piece (to be implemented by specific strategies).

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number
        """
        raise NotImplementedError("Subclasses must implement get_bin method")


class PrimaryCategorySorter(SortingStrategy):
    """Sort Lego pieces by their primary category."""

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Assign bin based on primary category.

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number
        """
        if element_id not in self.categories_data:
            logger.warning(f"Element ID {element_id} not found in categories data, using overflow bin")
            return self.overflow_bin

        piece_info = self.categories_data[element_id]
        category_key = piece_info.get("primary_category")

        # If we've already assigned a bin to this category, use it
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using existing bin {bin_number} for primary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin < self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.next_available_bin += 1
            logger.info(f"Assigned new bin {bin_number} to primary category '{category_key}'")
            return bin_number

        # Otherwise, use the overflow bin
        logger.info(
            f"Using overflow bin {self.overflow_bin} for primary category '{category_key}' (no more bins available)")
        return self.overflow_bin

    def get_description(self) -> str:
        """Get description of this sorting strategy.

        Returns:
            Human-readable description
        """
        return "Sorting by primary category"


class SecondaryCategorySorter(SortingStrategy):
    """Sort Lego pieces by their secondary category within a specific primary category."""

    def __init__(self, config: Dict[str, Any], categories_data: Dict[str, Dict[str, str]]):
        super().__init__(config, categories_data)
        self.target_primary = config.get("target_primary_category")

        if not self.target_primary:
            error_msg = "Secondary category sorting requires a target primary category"
            logger.error(error_msg)
            raise SortingError(error_msg)

        logger.info(f"Secondary category sorter targeting primary category: '{self.target_primary}'")

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Assign bin based on secondary category within target primary category.

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number
        """
        if element_id not in self.categories_data:
            logger.warning(f"Element ID {element_id} not found in categories data, using overflow bin")
            return self.overflow_bin

        piece_info = self.categories_data[element_id]

        # Check if this piece belongs to the target primary category
        if piece_info.get("primary_category") != self.target_primary:
            logger.debug(
                f"Element ID {element_id} not in target primary category '{self.target_primary}', using overflow bin")
            return self.overflow_bin

        # Use secondary category as the key
        category_key = piece_info.get("secondary_category")

        # If we've already assigned a bin to this category, use it
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using existing bin {bin_number} for secondary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin < self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.next_available_bin += 1
            logger.info(f"Assigned new bin {bin_number} to secondary category '{category_key}'")
            return bin_number

        # Otherwise, use the overflow bin
        logger.info(
            f"Using overflow bin {self.overflow_bin} for secondary category '{category_key}' (no more bins available)")
        return self.overflow_bin

    def get_description(self) -> str:
        """Get description of this sorting strategy.

        Returns:
            Human-readable description
        """
        return f"Sorting by secondary category within primary category '{self.target_primary}'"


class TertiaryCategorySorter(SortingStrategy):
    """Sort Lego pieces by their tertiary category within specific primary and secondary categories."""

    def __init__(self, config: Dict[str, Any], categories_data: Dict[str, Dict[str, str]]):
        super().__init__(config, categories_data)
        self.target_primary = config.get("target_primary_category")
        self.target_secondary = config.get("target_secondary_category")

        if not self.target_primary or not self.target_secondary:
            error_msg = "Tertiary category sorting requires target primary and secondary categories"
            logger.error(error_msg)
            raise SortingError(error_msg)

        logger.info(f"Tertiary category sorter targeting: '{self.target_primary}/{self.target_secondary}'")

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Assign bin based on tertiary category within target primary and secondary categories.

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number
        """
        if element_id not in self.categories_data:
            logger.warning(f"Element ID {element_id} not found in categories data, using overflow bin")
            return self.overflow_bin

        piece_info = self.categories_data[element_id]

        # Check if this piece belongs to the target primary and secondary categories
        if (piece_info.get("primary_category") != self.target_primary or
                piece_info.get("secondary_category") != self.target_secondary):
            logger.debug(f"Element ID {element_id} not in target primary/secondary categories, using overflow bin")
            return self.overflow_bin

        # Use tertiary category as the key
        category_key = piece_info.get("tertiary_category")

        # If we've already assigned a bin to this category, use it
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using existing bin {bin_number} for tertiary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin < self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.next_available_bin += 1
            logger.info(f"Assigned new bin {bin_number} to tertiary category '{category_key}'")
            return bin_number

        # Otherwise, use the overflow bin
        logger.info(
            f"Using overflow bin {self.overflow_bin} for tertiary category '{category_key}' (no more bins available)")
        return self.overflow_bin

    def get_description(self) -> str:
        """Get description of this sorting strategy.

        Returns:
            Human-readable description
        """
        return f"Sorting by tertiary category within '{self.target_primary}/{self.target_secondary}'"


class SortingManager:
    """Manages piece sorting based on configured strategies."""

    def __init__(self, config_manager):
        """Initialize sorting manager.

        Args:
            config_manager: Configuration manager object
        """
        self.config_manager = config_manager
        self.sorting_config = config_manager.get_section("sorting")
        self.categories_data = {}

        logger.info("Initializing sorting manager")

        # Load categories data
        csv_path = config_manager.get("piece_identifier", "csv_path", "Lego_Categories.csv")

        try:
            self._load_categories(csv_path)
        except Exception as e:
            error_msg = f"Failed to load categories from CSV: {str(e)}"
            logger.error(error_msg)
            raise SortingError(error_msg)

        # Set up sorting strategy based on config
        try:
            self.strategy = self._configure_strategy()
            logger.info(f"Sorting strategy configured: {self.get_strategy_description()}")
        except Exception as e:
            error_msg = f"Failed to configure sorting strategy: {str(e)}"
            logger.error(error_msg)
            raise SortingError(error_msg)

    def _load_categories(self, filepath: str) -> None:
        """Load piece categories from CSV file.

        Args:
            filepath: Path to CSV file

        Raises:
            SortingError: If categories cannot be loaded
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    self.categories_data[row['element_id']] = {
                        'name': row['name'],
                        'primary_category': row['primary_category'],
                        'secondary_category': row['secondary_category'],
                        'tertiary_category': row.get('tertiary_category', '')
                    }
            logger.info(f"Loaded {len(self.categories_data)} pieces from {filepath}")

            # Log some sample categories for debugging
            sample_keys = list(self.categories_data.keys())[:3]
            if sample_keys:
                for key in sample_keys:
                    logger.debug(f"Sample piece: {key} - {self.categories_data[key]['name']}")

        except FileNotFoundError:
            error_msg = f"Could not find categories file at {filepath}"
            logger.error(error_msg)
            raise SortingError(error_msg)
        except Exception as e:
            error_msg = f"Error reading CSV file: {str(e)}"
            logger.error(error_msg)
            raise SortingError(error_msg)

    def _configure_strategy(self) -> SortingStrategy:
        """Configure sorting strategy based on config settings.

        Returns:
            Configured SortingStrategy

        Raises:
            SortingError: If strategy cannot be configured
        """
        # Get the strategy type or default to primary
        strategy_type = self.sorting_config.get("strategy", "primary")
        logger.info(f"Configuring sorting strategy of type: {strategy_type}")

        try:
            if strategy_type == "primary":
                return PrimaryCategorySorter(self.sorting_config, self.categories_data)

            elif strategy_type == "secondary":
                return SecondaryCategorySorter(self.sorting_config, self.categories_data)

            elif strategy_type == "tertiary":
                return TertiaryCategorySorter(self.sorting_config, self.categories_data)

            else:
                logger.warning(f"Unknown sorting strategy '{strategy_type}'. Defaulting to primary category sorting.")
                return PrimaryCategorySorter(self.sorting_config, self.categories_data)

        except ValueError as e:
            error_msg = f"Error configuring sorting strategy: {str(e)}"
            logger.error(error_msg)
            logger.info("Defaulting to primary category sorting")
            return PrimaryCategorySorter(self.sorting_config, self.categories_data)

    def _record_missing_piece(self, element_id: str, api_result: Dict[str, Any]) -> None:
        """Record missing piece data to a CSV file for later addition to the categories CSV.

        Args:
            element_id: The element ID of the missing piece
            api_result: The full API result containing piece information
        """
        missing_file_path = "missing_piece_data.csv"
        try:
            # Get piece name from API result if available
            piece_name = api_result.get("name", "Unknown")

            # Get current image number if available in the API result
            image_number = api_result.get("image_number", 0)

            # Check if the file exists, if not create it with headers
            file_exists = os.path.isfile(missing_file_path)

            with open(missing_file_path, 'a', encoding='utf-8', newline='') as file:
                fieldnames = ['element_id', 'name', 'primary_category', 'secondary_category',
                              'tertiary_category', 'image_number', 'timestamp']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()

                # Write the piece data
                writer.writerow({
                    'element_id': element_id,
                    'name': piece_name,
                    'primary_category': '',  # Empty fields for user to fill in
                    'secondary_category': '',
                    'tertiary_category': '',
                    'image_number': image_number,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })

            logger.info(f"Recorded missing piece data for element ID: {element_id} (image #{image_number})")
        except Exception as e:
            logger.error(f"Failed to record missing piece data: {str(e)}")

    def identify_piece(self, api_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process API result and determine sorting bin.

        Args:
            api_result: Result from Brickognize API

        Returns:
            Dictionary with identification results and bin number
        """
        # Extract information from API result
        element_id = api_result.get("id")
        confidence = api_result.get("score", 0)

        logger.debug(f"Identifying piece: element_id={element_id}, confidence={confidence}")

        # Create result dictionary
        result = {
            "id": element_id,
            "confidence": confidence,
            "bin_number": self.sorting_config.get("overflow_bin", 9)  # Default to overflow bin
        }

        # Check confidence threshold
        confidence_threshold = self.config_manager.get(
            "piece_identifier", "confidence_threshold", 0.7
        )
        if confidence < confidence_threshold:
            error_msg = f"Low confidence score: {confidence} < {confidence_threshold}"
            logger.warning(error_msg)
            result["error"] = error_msg
            return result

        # Check if piece exists in our database
        if not element_id or element_id not in self.categories_data:
            error_msg = f"Piece not found in dictionary: {element_id}"
            logger.warning(error_msg)
            result["error"] = error_msg

            # Add new code to record missing piece data
            self._record_missing_piece(element_id, api_result)

            return result

    def get_strategy_description(self) -> str:
        """Get description of the current sorting strategy.

        Returns:
            Human-readable description of the sorting strategy
        """
        return self.strategy.get_description()

    def get_bin_mapping(self) -> Dict[str, int]:
        """Get current mapping of categories to bins.

        Returns:
            Dictionary mapping category names to bin numbers
        """
        return self.strategy.category_to_bin.copy()

    def release(self) -> None:
        """Release any resources used by the sorting manager"""
        logger.info("Releasing sorting manager resources")
        # Currently no resources to release, but adding for API consistency


def create_sorting_manager(config_manager):
    """Factory function to create a sorting manager.

    Args:
        config_manager: Configuration manager object

    Returns:
        SortingManager instance

    Raises:
        SortingError: If manager creation fails
    """
    logger.info("Creating sorting manager")

    try:
        return SortingManager(config_manager)
    except Exception as e:
        if isinstance(e, SortingError):
            raise
        error_msg = f"Failed to create sorting manager: {str(e)}"
        logger.error(error_msg)
        raise SortingError(error_msg)
