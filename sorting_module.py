"""
Enhanced sorting_module.py - Added support for pre-assigned bin allocations

Key Changes:
1. Added pre_assignments parameter to UnifiedSortingStrategy
2. Added validation for pre-assigned categories
3. Updated bin allocation logic to skip pre-assigned bins
4. Added helper methods for bin management
"""

import csv
import logging
import os
import time
from typing import Dict, Any, Optional, List

from error_module import SortingError, log_and_return

# Set up module logger
logger = logging.getLogger(__name__)


class UnifiedSortingStrategy:
    """Unified strategy for Lego piece sorting with dynamic bin assignment and pre-assignments."""

    def __init__(self, config: Dict[str, Any], categories_data: Dict[str, Dict[str, str]],
                 pre_assignments: Optional[Dict[str, int]] = None):
        """Initialize sorting strategy.

        Args:
            config: Configuration dictionary
            categories_data: Dictionary mapping element_ids to category information
            pre_assignments: Optional dictionary mapping category names to bin numbers
                           Format: {"Basic": 2, "Technic": 3, ...}
        """
        self.config = config
        self.categories_data = categories_data
        self.overflow_bin = 0  # Always use bin 0 as overflow bin

        # Strategy type can be 'primary', 'secondary', or 'tertiary'
        self.strategy_type = config.get("strategy", "primary")
        logger.info(f"Initialized sorting with strategy: {self.strategy_type}")

        # Target categories for secondary/tertiary sorts
        self.target_primary = config.get("target_primary_category", "")
        self.target_secondary = config.get("target_secondary_category", "")

        # Validate strategy configuration
        if self.strategy_type == "secondary" and not self.target_primary:
            error_msg = "Secondary sorting requires a target primary category"
            logger.error(error_msg)
            raise SortingError(error_msg)

        if self.strategy_type == "tertiary" and (not self.target_primary or not self.target_secondary):
            error_msg = "Tertiary sorting requires target primary and secondary categories"
            logger.error(error_msg)
            raise SortingError(error_msg)

        # Initialize bin assignment tracking
        self.category_to_bin = {}  # Maps category keys to bin numbers
        self.max_bins = config.get("max_bins", 9)  # Maximum number of bins (excluding overflow)
        self.used_bins = {self.overflow_bin}  # Track which bins are unavailable for dynamic assignment

        # Process pre-assignments first
        if pre_assignments:
            self._apply_pre_assignments(pre_assignments)

        # Find next available bin for dynamic assignment
        self.next_available_bin = self._find_next_available_bin()

        logger.debug(f"Sorting strategy initialized: max_bins={self.max_bins}, overflow_bin={self.overflow_bin}")
        logger.debug(f"Pre-assigned bins: {dict(self.category_to_bin)}")
        logger.debug(f"Next available bin for dynamic assignment: {self.next_available_bin}")

        if self.strategy_type != "primary":
            logger.debug(f"Target categories: primary='{self.target_primary}', secondary='{self.target_secondary}'")

    def _apply_pre_assignments(self, pre_assignments: Dict[str, int]) -> None:
        """Apply pre-assigned category-to-bin mappings.

        Args:
            pre_assignments: Dictionary mapping category names to bin numbers
        """
        logger.info(f"Applying pre-assignments: {pre_assignments}")

        for category, bin_number in pre_assignments.items():
            # Validate bin number
            if not (1 <= bin_number <= self.max_bins):
                logger.warning(f"Invalid bin number {bin_number} for category '{category}', skipping")
                continue

            # Validate category for current strategy
            if not self._is_valid_category_for_strategy(category):
                logger.warning(f"Category '{category}' is not valid for {self.strategy_type} strategy, skipping")
                continue

            # Check if bin is already assigned
            if bin_number in self.used_bins:
                logger.warning(f"Bin {bin_number} already assigned, skipping category '{category}'")
                continue

            # Apply the assignment
            self.category_to_bin[category] = bin_number
            self.used_bins.add(bin_number)
            logger.info(f"Pre-assigned category '{category}' to bin {bin_number}")

    def _is_valid_category_for_strategy(self, category: str) -> bool:
        """Check if a category is valid for the current sorting strategy.

        Args:
            category: Category name to validate

        Returns:
            bool: True if category is valid for current strategy
        """
        if self.strategy_type == "primary":
            # For primary sorting, check if category exists in any piece's primary_category
            return any(piece_data.get("primary_category") == category
                       for piece_data in self.categories_data.values())

        elif self.strategy_type == "secondary":
            # For secondary sorting, check if category exists as secondary within target primary
            return any(piece_data.get("primary_category") == self.target_primary and
                       piece_data.get("secondary_category") == category
                       for piece_data in self.categories_data.values())

        elif self.strategy_type == "tertiary":
            # For tertiary sorting, check if category exists as tertiary within target primary/secondary
            return any(piece_data.get("primary_category") == self.target_primary and
                       piece_data.get("secondary_category") == self.target_secondary and
                       piece_data.get("tertiary_category") == category
                       for piece_data in self.categories_data.values())

        return False

    def _find_next_available_bin(self) -> Optional[int]:
        """Find the next available bin number for dynamic assignment.

        Returns:
            int: Next available bin number, or None if all bins are used
        """
        for bin_num in range(1, self.max_bins + 1):
            if bin_num not in self.used_bins:
                return bin_num
        return None

    def get_bin(self, element_id: str, confidence: float) -> int:
        """Determine bin for a piece using the configured sorting strategy.

        Args:
            element_id: Element ID of the Lego piece
            confidence: Confidence score of the identification

        Returns:
            Bin number (0 = overflow bin)
        """
        if element_id not in self.categories_data:
            logger.warning(f"Element ID {element_id} not found in categories data, using overflow bin")
            return self.overflow_bin

        piece_info = self.categories_data[element_id]

        # Handle strategy-specific logic
        if self.strategy_type == "primary":
            return self._assign_bin_for_primary(piece_info)
        elif self.strategy_type == "secondary":
            return self._assign_bin_for_secondary(piece_info)
        elif self.strategy_type == "tertiary":
            return self._assign_bin_for_tertiary(piece_info)
        else:
            logger.warning(f"Unknown strategy type: {self.strategy_type}, falling back to primary sorting")
            return self._assign_bin_for_primary(piece_info)

    def _assign_bin_for_primary(self, piece_info: Dict[str, str]) -> int:
        """Assign bin based on primary category.

        Args:
            piece_info: Dictionary with piece category information

        Returns:
            Bin number
        """
        category_key = piece_info.get("primary_category", "")
        if not category_key:
            logger.warning("Piece has no primary category, using overflow bin")
            return self.overflow_bin

        # Check if this category has a pre-assigned bin
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using pre-assigned bin {bin_number} for primary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin is not None and self.next_available_bin <= self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.used_bins.add(bin_number)

            # Find next available bin
            self.next_available_bin = self._find_next_available_bin()

            logger.info(f"Assigned new bin {bin_number} to primary category '{category_key}'")
            return bin_number

        # Otherwise, use the overflow bin
        logger.info(
            f"Using overflow bin {self.overflow_bin} for primary category '{category_key}' (no more bins available)")
        return self.overflow_bin

    def _assign_bin_for_secondary(self, piece_info: Dict[str, str]) -> int:
        """Assign bin based on secondary category within target primary category.

        Args:
            piece_info: Dictionary with piece category information

        Returns:
            Bin number
        """
        # Check if this piece belongs to the target primary category
        if piece_info.get("primary_category", "") != self.target_primary:
            logger.debug(
                f"Piece primary category '{piece_info.get('primary_category', '')}' doesn't match target '{self.target_primary}', using overflow bin")
            return self.overflow_bin

        # Use secondary category as the key
        category_key = piece_info.get("secondary_category", "")
        if not category_key:
            logger.warning("Piece has no secondary category, using overflow bin")
            return self.overflow_bin

        # Check if this category has a pre-assigned bin
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using pre-assigned bin {bin_number} for secondary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin is not None and self.next_available_bin <= self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.used_bins.add(bin_number)

            # Find next available bin
            self.next_available_bin = self._find_next_available_bin()

            logger.info(f"Assigned new bin {bin_number} to secondary category '{category_key}'")
            return bin_number

        # Otherwise, use the overflow bin
        logger.info(
            f"Using overflow bin {self.overflow_bin} for secondary category '{category_key}' (no more bins available)")
        return self.overflow_bin

    def _assign_bin_for_tertiary(self, piece_info: Dict[str, str]) -> int:
        """Assign bin based on tertiary category within target primary and secondary categories.

        Args:
            piece_info: Dictionary with piece category information

        Returns:
            Bin number
        """
        # Check if this piece belongs to the target primary and secondary categories
        if (piece_info.get("primary_category", "") != self.target_primary or
                piece_info.get("secondary_category", "") != self.target_secondary):
            logger.debug(f"Piece categories don't match targets, using overflow bin")
            return self.overflow_bin

        # Use tertiary category as the key
        category_key = piece_info.get("tertiary_category", "")
        if not category_key:
            logger.debug("Piece has no tertiary category, using overflow bin")
            return self.overflow_bin

        # Check if this category has a pre-assigned bin
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using pre-assigned bin {bin_number} for tertiary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin is not None and self.next_available_bin <= self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.used_bins.add(bin_number)

            # Find next available bin
            self.next_available_bin = self._find_next_available_bin()

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
        desc = ""
        if self.strategy_type == "primary":
            desc = "Sorting by primary category"
        elif self.strategy_type == "secondary":
            desc = f"Sorting by secondary category within primary category '{self.target_primary}'"
        elif self.strategy_type == "tertiary":
            desc = f"Sorting by tertiary category within '{self.target_primary}/{self.target_secondary}'"
        else:
            desc = "Unknown sorting strategy"

        # Add pre-assignment info
        if self.category_to_bin:
            pre_assigned_count = len([bin_num for bin_num in self.category_to_bin.values()
                                      if bin_num != self.overflow_bin])
            if pre_assigned_count > 0:
                desc += f" (with {pre_assigned_count} pre-assigned bins)"

        return desc

    def get_available_categories(self) -> List[str]:
        """Get list of categories available for the current sorting strategy.

        Returns:
            List of category names that can be used with current strategy
        """
        categories = set()

        if self.strategy_type == "primary":
            # Get all primary categories
            for piece_data in self.categories_data.values():
                primary = piece_data.get("primary_category")
                if primary:
                    categories.add(primary)

        elif self.strategy_type == "secondary":
            # Get secondary categories within target primary
            for piece_data in self.categories_data.values():
                if piece_data.get("primary_category") == self.target_primary:
                    secondary = piece_data.get("secondary_category")
                    if secondary:
                        categories.add(secondary)

        elif self.strategy_type == "tertiary":
            # Get tertiary categories within target primary/secondary
            for piece_data in self.categories_data.values():
                if (piece_data.get("primary_category") == self.target_primary and
                        piece_data.get("secondary_category") == self.target_secondary):
                    tertiary = piece_data.get("tertiary_category")
                    if tertiary:
                        categories.add(tertiary)

        return sorted(list(categories))

    def get_pre_assigned_bins(self) -> Dict[str, int]:
        """Get dictionary of pre-assigned category-to-bin mappings.

        Returns:
            Dictionary mapping category names to bin numbers (excluding overflow assignments)
        """
        return {category: bin_num for category, bin_num in self.category_to_bin.items()
                if bin_num != self.overflow_bin}

class SortingManager:
    """Manages piece sorting based on configured strategies."""

    def __init__(self, config_manager, pre_assignments: Optional[Dict[str, int]] = None):
        """Initialize sorting manager.

        Args:
            config_manager: Configuration manager object
            pre_assignments: Optional dictionary of category-to-bin pre-assignments
        """
        self.config_manager = config_manager
        self.sorting_config = config_manager.get_section("sorting")
        self.categories_data = {}
        self.pre_assignments = pre_assignments or {}

        logger.info("Initializing sorting manager")
        if self.pre_assignments:
            logger.info(f"Pre-assignments provided: {self.pre_assignments}")

        # Load categories data
        csv_path = config_manager.get("piece_identifier", "csv_path", "Lego_Categories.csv")

        try:
            self._load_categories(csv_path)
        except Exception as e:
            error_msg = f"Failed to load categories from CSV: {str(e)}"
            logger.error(error_msg)
            raise SortingError(error_msg)

        # Set up sorting strategy with pre-assignments
        try:
            self.strategy = UnifiedSortingStrategy(self.sorting_config, self.categories_data, self.pre_assignments)
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

        # Create base result dictionary with API data
        result = {
            "element_id": element_id,
            "confidence": confidence,
            "bin_number": self.sorting_config.get("overflow_bin", 0)  # Default to overflow bin
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

            # Record missing piece data
            self._record_missing_piece(element_id, api_result)

            return result

        # Piece exists in database, get its categories
        piece_data = self.categories_data[element_id]

        # Add category information to result
        result["name"] = piece_data.get("name", api_result.get("name", "Unknown"))
        result["primary_category"] = piece_data.get("primary_category", "")
        result["secondary_category"] = piece_data.get("secondary_category", "")
        result["tertiary_category"] = piece_data.get("tertiary_category", "")

        # Determine bin using sorting strategy
        bin_number = self.strategy.get_bin(element_id, confidence)
        result["bin_number"] = bin_number

        logger.info(f"Piece {element_id} ({result['name']}) assigned to bin {bin_number}")

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

    def get_available_categories(self) -> List[str]:
        """Get list of categories available for pre-assignment with current strategy.

        Returns:
            List of category names that can be pre-assigned
        """
        return self.strategy.get_available_categories()

    def get_pre_assigned_bins(self) -> Dict[str, int]:
        """Get current pre-assigned bin mappings.

        Returns:
            Dictionary mapping category names to pre-assigned bin numbers
        """
        return self.strategy.get_pre_assigned_bins()

    def release(self) -> None:
        """Release any resources used by the sorting manager"""
        logger.info("Releasing sorting manager resources")
        # Currently no resources to release, but adding for API consistency


def create_sorting_manager(config_manager, pre_assignments: Optional[Dict[str, int]] = None):
    """Factory function to create a sorting manager.

    Args:
        config_manager: Configuration manager object
        pre_assignments: Optional dictionary of category-to-bin pre-assignments

    Returns:
        SortingManager instance

    Raises:
        SortingError: If manager creation fails
    """
    logger.info("Creating sorting manager")
    if pre_assignments:
        logger.info(f"With pre-assignments: {pre_assignments}")

    try:
        return SortingManager(config_manager, pre_assignments)
    except Exception as e:
        if isinstance(e, SortingError):
            raise
        error_msg = f"Failed to create sorting manager: {str(e)}"
        logger.error(error_msg)
        raise SortingError(error_msg)