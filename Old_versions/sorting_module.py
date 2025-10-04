"""
Enhanced sorting_module.py - Added support for bin capacity management

This module handles the sorting logic for Lego pieces with:
1. Hierarchical sorting strategies (primary, secondary, tertiary)
2. Pre-assigned bin allocations
3. Dynamic bin assignment
4. Bin capacity tracking and management (NEW)
5. Missing piece recording

Key Changes:
- Added BinCapacityManager class for tracking pieces per bin
- Integrated capacity tracking into sorting decisions
- Added callbacks for bin full/warning conditions
- Maintains all original functionality
"""

import csv
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from error_module import SortingError, log_and_return
from enhanced_config_manager import ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# BIN CAPACITY MANAGEMENT SECTION
# ============================================================================
# This section handles tracking how many pieces are assigned to each bin
# and triggers warnings/pauses when bins reach capacity

class BinCapacityManager:
    """
    Tracks the number of pieces assigned to each bin and triggers pause when full.

    This class monitors bin usage to prevent overflow and alerts the system
    when human intervention is needed to empty bins.
    """

    def __init__(self, max_pieces_per_bin: int = 50, warning_threshold: float = 0.8):
        """
        Initialize bin capacity tracking.

        Args:
            max_pieces_per_bin: Maximum pieces before bin is considered full
            warning_threshold: Percentage (0.0-1.0) at which to issue warning
        """
        # Configuration
        self.max_pieces = max_pieces_per_bin
        self.warning_threshold = warning_threshold

        # Tracking dictionaries
        self.bin_counts = {}  # Track count for each bin {bin_number: count}

        # Callback functions for external notifications
        self.pause_callback = None  # Function to call when bin is full
        self.warning_callback = None  # Function to call for warnings

        # Statistics
        self.total_pieces_sorted = 0
        self.bins_filled_count = 0  # How many times bins have been filled

        logger.info(f"BinCapacityManager initialized: max={max_pieces_per_bin} pieces/bin, "
                    f"warning at {warning_threshold * 100}%")

    def record_piece(self, bin_number: int) -> bool:
        """
        Record a piece being assigned to a bin and check capacity.

        This method increments the counter for the specified bin and checks
        if the bin has reached warning or full capacity.

        Args:
            bin_number: The bin number the piece was assigned to

        Returns:
            True if bin is now full, False otherwise
        """
        # Increment counter for this bin
        self.bin_counts[bin_number] = self.bin_counts.get(bin_number, 0) + 1
        current_count = self.bin_counts[bin_number]
        self.total_pieces_sorted += 1

        logger.debug(f"Bin {bin_number} now has {current_count} pieces "
                     f"(total sorted: {self.total_pieces_sorted})")

        # Check if bin is full
        if current_count >= self.max_pieces:
            logger.warning(f"BIN {bin_number} IS FULL! ({current_count}/{self.max_pieces})")
            self.bins_filled_count += 1

            # Trigger pause callback if registered
            if self.pause_callback:
                self.pause_callback(bin_number, current_count, self.max_pieces)

            return True  # Bin is full

        # Check if approaching capacity (warning threshold)
        elif current_count >= (self.max_pieces * self.warning_threshold):
            percentage = (current_count / self.max_pieces) * 100
            warning_msg = (f"Bin {bin_number} approaching capacity: "
                           f"{current_count}/{self.max_pieces} ({percentage:.0f}%)")
            logger.warning(warning_msg)

            # Trigger warning callback if registered
            if self.warning_callback:
                self.warning_callback(bin_number, current_count, self.max_pieces)

        return False  # Bin is not full

    def reset_bin(self, bin_number: int):
        """
        Reset the count for a specific bin (when emptied by user).

        This should be called when the user confirms they have emptied a bin.

        Args:
            bin_number: The bin number to reset
        """
        if bin_number in self.bin_counts:
            previous_count = self.bin_counts[bin_number]
            logger.info(f"Resetting bin {bin_number} count (was {previous_count} pieces)")
            self.bin_counts[bin_number] = 0
        else:
            logger.debug(f"Bin {bin_number} was already at 0 or never used")

    def reset_all_bins(self):
        """
        Reset all bin counts (for new session).

        This should be called at the start of a new sorting session.
        """
        logger.info(f"Resetting all bin counts (total pieces sorted: {self.total_pieces_sorted})")
        self.bin_counts = {}
        self.total_pieces_sorted = 0
        self.bins_filled_count = 0

    def get_bin_status(self, bin_number: int) -> Dict[str, Any]:
        """
        Get current status of a specific bin.

        Args:
            bin_number: The bin to check

        Returns:
            Dictionary with bin status information
        """
        count = self.bin_counts.get(bin_number, 0)
        percentage = (count / self.max_pieces * 100) if self.max_pieces > 0 else 0

        return {
            "current": count,
            "max": self.max_pieces,
            "percentage": percentage,
            "is_full": count >= self.max_pieces,
            "is_warning": count >= (self.max_pieces * self.warning_threshold),
            "remaining": max(0, self.max_pieces - count)
        }

    def get_all_bin_status(self) -> Dict[int, Dict[str, Any]]:
        """
        Get status of all bins that have received pieces.

        Returns:
            Dictionary mapping bin numbers to their status
        """
        return {bin_num: self.get_bin_status(bin_num)
                for bin_num in self.bin_counts.keys()}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics for the capacity manager.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_pieces_sorted": self.total_pieces_sorted,
            "bins_filled_count": self.bins_filled_count,
            "active_bins": len(self.bin_counts),
            "fullest_bin": self._get_fullest_bin(),
            "average_fill": self._calculate_average_fill()
        }

    def _get_fullest_bin(self) -> Tuple[Optional[int], int]:
        """Find the bin with the most pieces."""
        if not self.bin_counts:
            return None, 0

        fullest_bin = max(self.bin_counts.items(), key=lambda x: x[1])
        return fullest_bin

    def _calculate_average_fill(self) -> float:
        """Calculate average fill percentage across all used bins."""
        if not self.bin_counts:
            return 0.0

        total_percentage = sum(
            (count / self.max_pieces * 100)
            for count in self.bin_counts.values()
        )
        return total_percentage / len(self.bin_counts)


# ============================================================================
# SORTING STRATEGY SECTION
# ============================================================================
# This section contains the main sorting logic and bin assignment algorithms

class UnifiedSortingStrategy:
    """
    Unified strategy for Lego piece sorting with dynamic bin assignment and pre-assignments.

    This class implements the core sorting logic that determines which bin each
    piece should go to based on the configured strategy (primary, secondary, or tertiary).
    """

    def __init__(self, config: Dict[str, Any], categories_data: Dict[str, Dict[str, str]],
                 pre_assignments: Optional[Dict[str, int]] = None):
        """
        Initialize sorting strategy with configuration and category data.

        Args:
            config: Configuration dictionary containing sorting parameters
            categories_data: Dictionary mapping element_ids to category information
            pre_assignments: Optional dictionary mapping category names to bin numbers
                           Format: {"Basic": 2, "Technic": 3, ...}
        """
        # ====== CORE CONFIGURATION ======
        self.config = config
        self.categories_data = categories_data
        self.overflow_bin = 0  # Always use bin 0 as overflow bin

        # ====== STRATEGY CONFIGURATION ======
        # Strategy type can be 'primary', 'secondary', or 'tertiary'
        self.strategy_type = config.get("strategy", "primary")
        logger.info(f"Initialized sorting with strategy: {self.strategy_type}")

        # Target categories for filtered sorting (secondary/tertiary modes)
        self.target_primary = config.get("target_primary_category", "")
        self.target_secondary = config.get("target_secondary_category", "")

        # ====== VALIDATION OF STRATEGY ======
        # Ensure required target categories are specified for secondary/tertiary sorting
        if self.strategy_type == "secondary" and not self.target_primary:
            error_msg = "Secondary sorting requires a target primary category"
            logger.error(error_msg)
            raise SortingError(error_msg)

        if self.strategy_type == "tertiary" and (not self.target_primary or not self.target_secondary):
            error_msg = "Tertiary sorting requires target primary and secondary categories"
            logger.error(error_msg)
            raise SortingError(error_msg)

        # ====== BIN ASSIGNMENT TRACKING ======
        self.category_to_bin = {}  # Maps category keys to bin numbers
        self.max_bins = config.get("max_bins", 9)  # Maximum number of bins (excluding overflow)
        self.used_bins = {self.overflow_bin}  # Track which bins are unavailable for dynamic assignment

        # ====== PRE-ASSIGNMENTS PROCESSING ======
        # Process pre-assignments first (user-defined category-to-bin mappings)
        if pre_assignments:
            self._apply_pre_assignments(pre_assignments)

        # Find next available bin for dynamic assignment
        self.next_available_bin = self._find_next_available_bin()

        # ====== BIN CAPACITY MANAGEMENT (NEW) ======
        # Initialize capacity tracking for bins
        max_pieces_per_bin = config.get("max_pieces_per_bin", 50)
        warning_threshold = config.get("bin_warning_threshold", 0.8)
        self.capacity_manager = BinCapacityManager(max_pieces_per_bin, warning_threshold)

        logger.debug(f"Bin capacity tracking enabled: max {max_pieces_per_bin} pieces/bin, "
                     f"warning at {warning_threshold * 100}%")

        # ====== LOGGING CONFIGURATION STATE ======
        logger.debug(f"Sorting strategy initialized: max_bins={self.max_bins}, overflow_bin={self.overflow_bin}")
        logger.debug(f"Pre-assigned bins: {dict(self.category_to_bin)}")
        logger.debug(f"Next available bin for dynamic assignment: {self.next_available_bin}")

        if self.strategy_type != "primary":
            logger.debug(f"Target categories: primary='{self.target_primary}', secondary='{self.target_secondary}'")

    # ========================================================================
    # PRE-ASSIGNMENT METHODS
    # ========================================================================
    # These methods handle user-defined bin assignments

    def _apply_pre_assignments(self, pre_assignments: Dict[str, int]) -> None:
        """
        Apply pre-assigned category-to-bin mappings.

        This allows users to specify that certain categories should always
        go to specific bins, overriding dynamic assignment.

        Args:
            pre_assignments: Dictionary mapping category names to bin numbers
        """
        logger.info(f"Applying pre-assignments: {pre_assignments}")

        for category, bin_number in pre_assignments.items():
            # Validate bin number is within valid range
            if not (1 <= bin_number <= self.max_bins):
                logger.warning(f"Invalid bin number {bin_number} for category '{category}', skipping")
                continue

            # Validate category exists for current sorting strategy
            if not self._is_valid_category_for_strategy(category):
                logger.warning(f"Category '{category}' is not valid for {self.strategy_type} strategy, skipping")
                continue

            # Check if bin is already assigned to another category
            if bin_number in self.used_bins:
                logger.warning(f"Bin {bin_number} already assigned, skipping category '{category}'")
                continue

            # Apply the assignment
            self.category_to_bin[category] = bin_number
            self.used_bins.add(bin_number)
            logger.info(f"Pre-assigned category '{category}' to bin {bin_number}")

    def _is_valid_category_for_strategy(self, category: str) -> bool:
        """
        Check if a category is valid for the current sorting strategy.

        This validates that the category exists in the database and matches
        the current sorting level (primary, secondary, or tertiary).

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

    # ========================================================================
    # BIN MANAGEMENT METHODS
    # ========================================================================
    # These methods handle finding and assigning bins

    def _find_next_available_bin(self) -> Optional[int]:
        """
        Find the next available bin number for dynamic assignment.

        This scans through bin numbers to find the first one that hasn't
        been assigned yet (either through pre-assignment or dynamic assignment).

        Returns:
            int: Next available bin number, or None if all bins are used
        """
        for bin_num in range(1, self.max_bins + 1):
            if bin_num not in self.used_bins:
                return bin_num
        return None

    # ========================================================================
    # MAIN SORTING METHOD
    # ========================================================================
    # This is the primary public interface for getting bin assignments

    def get_bin(self, element_id: str, confidence: float) -> int:
        """
        Determine bin for a piece using the configured sorting strategy.

        This is the main method called for each piece to determine its bin.
        It also tracks capacity for each bin.

        Args:
            element_id: Element ID of the Lego piece (from API)
            confidence: Confidence score of the identification (0.0-1.0)

        Returns:
            Bin number (0 = overflow bin)
        """
        # Check if piece exists in our database
        if element_id not in self.categories_data:
            logger.warning(f"Element ID {element_id} not found in categories data, using overflow bin")
            bin_number = self.overflow_bin
        else:
            piece_info = self.categories_data[element_id]

            # Route to appropriate strategy-specific method
            if self.strategy_type == "primary":
                bin_number = self._assign_bin_for_primary(piece_info)
            elif self.strategy_type == "secondary":
                bin_number = self._assign_bin_for_secondary(piece_info)
            elif self.strategy_type == "tertiary":
                bin_number = self._assign_bin_for_tertiary(piece_info)
            else:
                logger.warning(f"Unknown strategy type: {self.strategy_type}, falling back to primary sorting")
                bin_number = self._assign_bin_for_primary(piece_info)

        # ====== CAPACITY TRACKING (NEW) ======
        # Record this piece assignment and check if bin is full
        is_full = self.capacity_manager.record_piece(bin_number)

        if is_full:
            logger.critical(f"BIN {bin_number} IS FULL - System should pause for bin emptying")
            # The pause will be triggered via callback registered by the application

        return bin_number

    # ========================================================================
    # STRATEGY-SPECIFIC SORTING METHODS
    # ========================================================================
    # These methods implement the actual sorting logic for each strategy level

    def _assign_bin_for_primary(self, piece_info: Dict[str, str]) -> int:
        """
        Assign bin based on primary category only.

        In primary sorting, all pieces are sorted by their top-level category
        (e.g., Basic, Technic, Duplo, etc.)

        Args:
            piece_info: Dictionary with piece category information

        Returns:
            Bin number
        """
        category_key = piece_info.get("primary_category", "")
        if not category_key:
            logger.warning("Piece has no primary category, using overflow bin")
            return self.overflow_bin

        # Check if this category already has an assigned bin
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using existing bin {bin_number} for primary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin is not None and self.next_available_bin <= self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.used_bins.add(bin_number)

            # Find next available bin for future assignments
            self.next_available_bin = self._find_next_available_bin()

            logger.info(f"Assigned new bin {bin_number} to primary category '{category_key}'")
            return bin_number

        # No more bins available, use overflow
        logger.info(f"Using overflow bin {self.overflow_bin} for primary category '{category_key}' "
                    f"(no more bins available)")
        return self.overflow_bin

    def _assign_bin_for_secondary(self, piece_info: Dict[str, str]) -> int:
        """
        Assign bin based on secondary category within target primary category.

        In secondary sorting, only pieces matching the target primary category
        are sorted, and they're sorted by their secondary category
        (e.g., within "Basic": Brick, Plate, Tile, etc.)

        Args:
            piece_info: Dictionary with piece category information

        Returns:
            Bin number
        """
        # First check if this piece belongs to the target primary category
        if piece_info.get("primary_category", "") != self.target_primary:
            logger.debug(f"Piece primary category '{piece_info.get('primary_category', '')}' "
                         f"doesn't match target '{self.target_primary}', using overflow bin")
            return self.overflow_bin

        # Piece matches target primary, now sort by secondary category
        category_key = piece_info.get("secondary_category", "")
        if not category_key:
            logger.warning("Piece has no secondary category, using overflow bin")
            return self.overflow_bin

        # Check if this category already has an assigned bin
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using existing bin {bin_number} for secondary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin is not None and self.next_available_bin <= self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.used_bins.add(bin_number)

            # Find next available bin for future assignments
            self.next_available_bin = self._find_next_available_bin()

            logger.info(f"Assigned new bin {bin_number} to secondary category '{category_key}'")
            return bin_number

        # No more bins available, use overflow
        logger.info(f"Using overflow bin {self.overflow_bin} for secondary category '{category_key}' "
                    f"(no more bins available)")
        return self.overflow_bin

    def _assign_bin_for_tertiary(self, piece_info: Dict[str, str]) -> int:
        """
        Assign bin based on tertiary category within target primary and secondary categories.

        In tertiary sorting, only pieces matching both target primary AND secondary
        categories are sorted, and they're sorted by their tertiary category
        (e.g., within "Basic/Brick": 1x1, 2x2, 2x4, etc.)

        Args:
            piece_info: Dictionary with piece category information

        Returns:
            Bin number
        """
        # Check if this piece belongs to both target primary and secondary categories
        if (piece_info.get("primary_category", "") != self.target_primary or
                piece_info.get("secondary_category", "") != self.target_secondary):
            logger.debug(f"Piece categories don't match targets "
                         f"({self.target_primary}/{self.target_secondary}), using overflow bin")
            return self.overflow_bin

        # Piece matches both targets, now sort by tertiary category
        category_key = piece_info.get("tertiary_category", "")
        if not category_key:
            logger.debug("Piece has no tertiary category, using overflow bin")
            return self.overflow_bin

        # Check if this category already has an assigned bin
        if category_key in self.category_to_bin:
            bin_number = self.category_to_bin[category_key]
            logger.debug(f"Using existing bin {bin_number} for tertiary category '{category_key}'")
            return bin_number

        # If we have room for a new bin, assign the next available one
        if self.next_available_bin is not None and self.next_available_bin <= self.max_bins:
            bin_number = self.next_available_bin
            self.category_to_bin[category_key] = bin_number
            self.used_bins.add(bin_number)

            # Find next available bin for future assignments
            self.next_available_bin = self._find_next_available_bin()

            logger.info(f"Assigned new bin {bin_number} to tertiary category '{category_key}'")
            return bin_number

        # No more bins available, use overflow
        logger.info(f"Using overflow bin {self.overflow_bin} for tertiary category '{category_key}' "
                    f"(no more bins available)")
        return self.overflow_bin

    # ========================================================================
    # INFORMATION AND STATUS METHODS
    # ========================================================================
    # These methods provide information about the sorting strategy and its state

    def get_description(self) -> str:
        """
        Get human-readable description of this sorting strategy.

        Returns:
            Human-readable description string
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

        # Add pre-assignment information
        if self.category_to_bin:
            pre_assigned_count = len([bin_num for bin_num in self.category_to_bin.values()
                                      if bin_num != self.overflow_bin])
            if pre_assigned_count > 0:
                desc += f" (with {pre_assigned_count} assigned bins)"

        return desc

    def get_available_categories(self) -> List[str]:
        """
        Get list of categories available for the current sorting strategy.

        This returns all categories that could potentially be assigned bins
        based on the current strategy and target categories.

        Returns:
            List of category names that can be used with current strategy
        """
        categories = set()

        if self.strategy_type == "primary":
            # Get all unique primary categories
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
        """
        Get dictionary of pre-assigned category-to-bin mappings.

        This excludes overflow bin assignments.

        Returns:
            Dictionary mapping category names to bin numbers
        """
        return {category: bin_num for category, bin_num in self.category_to_bin.items()
                if bin_num != self.overflow_bin}

    # ========================================================================
    # CAPACITY MANAGEMENT METHODS (NEW)
    # ========================================================================
    # These methods provide access to bin capacity information

    def get_bin_capacity_status(self, bin_number: int) -> Dict[str, Any]:
        """
        Get capacity status for a specific bin.

        Args:
            bin_number: The bin to check

        Returns:
            Dictionary with bin status information
        """
        return self.capacity_manager.get_bin_status(bin_number)

    def get_all_bins_capacity_status(self) -> Dict[int, Dict[str, Any]]:
        """
        Get capacity status for all bins.

        Returns:
            Dictionary mapping bin numbers to their status
        """
        return self.capacity_manager.get_all_bin_status()

    def reset_bin_count(self, bin_number: int):
        """
        Reset count for a specific bin (after user empties it).

        Args:
            bin_number: The bin number to reset
        """
        self.capacity_manager.reset_bin(bin_number)

    def set_pause_callback(self, callback: Callable):
        """
        Set callback function to trigger when a bin is full.

        The callback should accept: (bin_number, current_count, max_count)

        Args:
            callback: Function to call when bin reaches capacity
        """
        self.capacity_manager.pause_callback = callback

    def set_warning_callback(self, callback: Callable):
        """
        Set callback function for bin capacity warnings.

        The callback should accept: (bin_number, current_count, max_count)

        Args:
            callback: Function to call for warnings
        """
        self.capacity_manager.warning_callback = callback


# ============================================================================
# SORTING MANAGER SECTION
# ============================================================================
# This section contains the main manager class that orchestrates sorting

class SortingManager:
    """
    Manages piece sorting based on configured strategies.

    This is the main interface used by other modules to identify and sort pieces.
    It handles loading category data, configuring strategies, and processing pieces.
    """

    def __init__(self, config_manager):
        """
        Initialize sorting manager with configuration.

        Args:
            config_manager: Configuration manager object
        """
        self.config_manager = config_manager

        # Get validated sorting configuration
        self.sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)

        # Get piece identifier configuration for CSV path
        identifier_config = config_manager.get_module_config(ModuleConfig.PIECE_IDENTIFIER.value)
        csv_path = identifier_config["csv_path"]

        # Initialize categories database
        self.categories_data = {}

        logger.info("Initializing sorting manager")

        # Load categories data from CSV file
        try:
            self._load_categories(csv_path)
        except Exception as e:
            error_msg = f"Failed to load categories from CSV: {str(e)}"
            logger.error(error_msg)
            raise SortingError(error_msg)

        # Set up sorting strategy with loaded configuration
        try:
            # Get pre-assignments from config if they exist
            pre_assignments = self.sorting_config.get("pre_assignments", {})

            # Create sorting strategy
            self.strategy = UnifiedSortingStrategy(
                self.sorting_config,
                self.categories_data,
                pre_assignments
            )

            # Store reference to capacity manager for external access
            self.capacity_manager = self.strategy.capacity_manager

            logger.info(f"Sorting strategy configured: {self.get_strategy_description()}")
        except Exception as e:
            error_msg = f"Failed to configure sorting strategy: {str(e)}"
            logger.error(error_msg)
            raise SortingError(error_msg)

    # ========================================================================
    # DATA LOADING METHODS
    # ========================================================================

    def _load_categories(self, filepath: str) -> None:
        """
        Load piece categories from CSV file into memory.

        This creates the database that maps element IDs to their category hierarchy.

        Args:
            filepath: Path to CSV file containing category data

        Raises:
            SortingError: If categories cannot be loaded
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)

                # Load each row into our categories database
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

    # ========================================================================
    # PIECE IDENTIFICATION AND SORTING
    # ========================================================================

    def identify_piece(self, api_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process API result and determine sorting bin.

        This is the main method called by the processing module to sort a piece.

        Args:
            api_result: Result from Brickognize API containing element_id and confidence

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
        confidence_threshold = self.sorting_config.get("confidence_threshold", 0.7)
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

            # Record missing piece data for future database updates
            self._record_missing_piece(element_id, api_result)

            return result

        # Piece exists in database, get its category information
        piece_data = self.categories_data[element_id]

        # Add category information to result
        result["name"] = piece_data.get("name", api_result.get("name", "Unknown"))
        result["primary_category"] = piece_data.get("primary_category", "")
        result["secondary_category"] = piece_data.get("secondary_category", "")
        result["tertiary_category"] = piece_data.get("tertiary_category", "")

        # Determine bin using sorting strategy (includes capacity tracking)
        bin_number = self.strategy.get_bin(element_id, confidence)
        result["bin_number"] = bin_number

        logger.info(f"Piece {element_id} ({result['name']}) assigned to bin {bin_number}")

        return result

    def _record_missing_piece(self, element_id: str, api_result: Dict[str, Any]) -> None:
        """
        Record missing piece data to a CSV file for later addition to the categories CSV.

        This helps identify pieces that aren't in the database yet.

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

    # ========================================================================
    # INFORMATION AND CONTROL METHODS
    # ========================================================================

    def get_strategy_description(self) -> str:
        """
        Get description of the current sorting strategy.

        Returns:
            Human-readable description of the sorting strategy
        """
        return self.strategy.get_description()

    def get_bin_mapping(self) -> Dict[str, int]:
        """
        Get current mapping of categories to bins.

        Returns:
            Dictionary mapping category names to bin numbers
        """
        return self.strategy.category_to_bin.copy()

    def get_available_categories(self) -> List[str]:
        """
        Get list of categories available for pre-assignment with current strategy.

        Returns:
            List of category names that can be pre-assigned
        """
        return self.strategy.get_available_categories()

    def get_pre_assigned_bins(self) -> Dict[str, int]:
        """
        Get current pre-assigned bin mappings.

        Returns:
            Dictionary mapping category names to pre-assigned bin numbers
        """
        return self.strategy.get_pre_assigned_bins()

    # ========================================================================
    # CAPACITY MANAGEMENT METHODS (NEW)
    # ========================================================================

    def get_bin_capacity_status(self, bin_number: int) -> Dict[str, Any]:
        """
        Get current capacity status of a specific bin.

        Args:
            bin_number: The bin to check

        Returns:
            Dictionary with bin status information
        """
        return self.strategy.get_bin_capacity_status(bin_number)

    def get_all_bins_capacity_status(self) -> Dict[int, Dict[str, Any]]:
        """
        Get capacity status of all bins.

        Returns:
            Dictionary mapping bin numbers to their status
        """
        return self.strategy.get_all_bins_capacity_status()

    def reset_bin(self, bin_number: int):
        """
        Reset bin count after user empties it.

        This should be called when the user confirms they have emptied a bin.

        Args:
            bin_number: The bin number to reset
        """
        self.strategy.reset_bin_count(bin_number)
        logger.info(f"Bin {bin_number} count reset")

    def reset_all_bins(self):
        """
        Reset all bin counts for a new session.

        This should be called at the start of a new sorting session.
        """
        self.capacity_manager.reset_all_bins()
        logger.info("All bin counts reset for new session")

    def set_pause_callback(self, callback: Callable):
        """
        Register callback for when bins are full.

        The callback will be called with: (bin_number, current_count, max_count)

        Args:
            callback: Function to call when a bin reaches capacity
        """
        self.strategy.set_pause_callback(callback)

    def set_warning_callback(self, callback: Callable):
        """
        Register callback for bin capacity warnings.

        The callback will be called with: (bin_number, current_count, max_count)

        Args:
            callback: Function to call for warnings
        """
        self.strategy.set_warning_callback(callback)

    def get_capacity_statistics(self) -> Dict[str, Any]:
        """
        Get overall capacity management statistics.

        Returns:
            Dictionary with capacity statistics
        """
        return self.capacity_manager.get_statistics()

    def release(self) -> None:
        """
        Release any resources used by the sorting manager.

        Currently no resources to release, but included for API consistency.
        """
        logger.info("Releasing sorting manager resources")
        # Future: Could save statistics or state here if needed


# ============================================================================
# FACTORY FUNCTION SECTION
# ============================================================================

def create_sorting_manager(config_manager, pre_assignments: Optional[Dict[str, int]] = None):
    """
    Factory function to create a sorting manager.

    This is the recommended way to create a SortingManager instance.

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

        # Add pre-assignments to config if provided
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)
        sorting_config["pre_assignments"] = pre_assignments
        config_manager.update_module_config(ModuleConfig.SORTING.value, sorting_config)

    try:
        return SortingManager(config_manager)
    except Exception as e:
        if isinstance(e, SortingError):
            raise
        error_msg = f"Failed to create sorting manager: {str(e)}"
        logger.error(error_msg)
        raise SortingError(error_msg)


# ============================================================================
# MODULE TESTING SECTION
# ============================================================================

if __name__ == "__main__":
    """
    Test the sorting module with capacity management.

    This demonstrates the bin capacity tracking and warning system.
    """
    import logging
    from enhanced_config_manager import create_config_manager

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config manager with test settings
    config_manager = create_config_manager()
    config_manager.update_module_config("sorting", {
        "strategy": "primary",
        "max_bins": 5,
        "overflow_bin": 0,
        "confidence_threshold": 0.7,
        "max_pieces_per_bin": 5,  # Small for testing
        "bin_warning_threshold": 0.6,  # Warning at 60%
        "pre_assignments": {
            "Basic": 1,
            "Technic": 2
        }
    })

    # Create sorting manager
    sorting_mgr = create_sorting_manager(config_manager)


    # Set up callbacks for capacity events
    def on_bin_full(bin_num, current, maximum):
        print(f"\n🛑 BIN {bin_num} IS FULL! ({current}/{maximum} pieces)")
        print("   System should pause for bin emptying!")


    def on_bin_warning(bin_num, current, maximum):
        percentage = (current / maximum) * 100
        print(f"\n⚠️  BIN {bin_num} WARNING: {current}/{maximum} pieces ({percentage:.0f}%)")


    sorting_mgr.set_pause_callback(on_bin_full)
    sorting_mgr.set_warning_callback(on_bin_warning)

    # Simulate processing pieces
    print("\n" + "=" * 60)
    print("SORTING MODULE TEST - Capacity Management")
    print("=" * 60)

    test_pieces = [
        # Basic pieces (pre-assigned to bin 1)
        {"id": "3001", "score": 0.95},  # Basic Brick
        {"id": "3001", "score": 0.95},  # Basic Brick
        {"id": "3001", "score": 0.95},  # Basic Brick (warning at 60%)
        {"id": "3001", "score": 0.95},  # Basic Brick
        {"id": "3001", "score": 0.95},  # Basic Brick (FULL at 5)
        {"id": "3001", "score": 0.95},  # Basic Brick (still full)

        # Technic pieces (pre-assigned to bin 2)
        {"id": "32524", "score": 0.90},  # Technic Beam
        {"id": "32524", "score": 0.90},  # Technic Beam

        # Unknown piece (goes to overflow)
        {"id": "99999", "score": 0.85},
    ]

    print("\nProcessing test pieces:")
    print("-" * 40)

    for i, api_result in enumerate(test_pieces, 1):
        # Process piece
        result = sorting_mgr.identify_piece(api_result)

        # Display result
        print(f"\nPiece {i}: ID={api_result['id']}")
        print(f"  → Bin {result['bin_number']} ({result.get('primary_category', 'Unknown')})")

        # Show bin status
        if result['bin_number'] != 0:  # Not overflow
            status = sorting_mgr.get_bin_capacity_status(result['bin_number'])
            print(f"  → Bin status: {status['current']}/{status['max']} "
                  f"({status['percentage']:.0f}%)")

    # Show final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    stats = sorting_mgr.get_capacity_statistics()
    print(f"Total pieces sorted: {stats['total_pieces_sorted']}")
    print(f"Bins filled: {stats['bins_filled_count']}")
    print(f"Active bins: {stats['active_bins']}")
    print(f"Average fill: {stats['average_fill']:.1f}%")

    # Show all bin statuses
    print("\nBin Status Summary:")
    print("-" * 40)
    all_status = sorting_mgr.get_all_bins_capacity_status()
    for bin_num in sorted(all_status.keys()):
        status = all_status[bin_num]
        bar_length = 20
        filled = int((status['percentage'] / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"Bin {bin_num}: [{bar}] {status['current']}/{status['max']} "
              f"({status['percentage']:.0f}%)")

    print("\nTest complete!")