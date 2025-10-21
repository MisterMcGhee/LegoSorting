# processing/bin_assignment_module.py
"""
bin_assignment_module.py - Bin assignment logic for Lego pieces

This module handles assigning pieces to bins based on their categories.
It implements three sorting strategies (primary, secondary, tertiary) with
pre-assigned bins and dynamic bin allocation.

RESPONSIBILITIES:
- Assign bins based on category hierarchy and strategy
- Handle pre-assigned bins from configuration
- Dynamically assign new categories to available bins
- Route unknown/overflow pieces to bin 0
- Notify registered callbacks when bin assignments occur (for GUI updates)

DOES NOT:
- Track bin capacity (stateless assignment)
- Look up categories (that's category_lookup_module)
- Identify pieces (that's identification_api_handler)
- Control servos or hardware

CALLBACK SYSTEM:
This module implements the Observer pattern to notify external systems
(primarily the GUI) when bins are assigned to categories. External systems
register callback functions during initialization, and these callbacks are
automatically invoked when assignments occur.
"""

import logging
from typing import Dict, List, Optional, Callable
from processing.processing_data_models import CategoryInfo, BinAssignment
from processing.category_hierarchy_service import CategoryHierarchyService
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class BinAssignmentModule:
    """
    Assigns pieces to bins based on configured sorting strategy.

    This module implements stateless bin assignment - it doesn't track
    how many pieces are in each bin, only which categories map to which bins.

    Three sorting strategies:
    - Primary: Sort all pieces by primary category (Basic, Technic, etc.)
    - Secondary: Sort pieces matching target primary by secondary category
    - Tertiary: Sort pieces matching target primary+secondary by tertiary category

    Callback Support:
    External systems can register callbacks to be notified when bin assignments
    occur. This is primarily used by the GUI to update bin status displays.
    """

    def __init__(self, config_manager: EnhancedConfigManager,
                 category_service: CategoryHierarchyService):
        """
        Initialize bin assignment module.

        Called during application startup before any sorting begins.

        Args:
            config_manager: Configuration manager instance
            category_service: Category hierarchy service for validation
        """
        self.config_manager = config_manager
        self.category_service = category_service

        # Load sorting configuration
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)

        # Basic configuration
        self.strategy = sorting_config["strategy"]  # "primary", "secondary", or "tertiary"
        self.max_bins = sorting_config["max_bins"]  # Total bins (excluding overflow)
        self.overflow_bin = sorting_config["overflow_bin"]  # Bin 0 for unknowns

        # Target categories for secondary/tertiary strategies
        self.target_primary = sorting_config.get("target_primary_category", "")
        self.target_secondary = sorting_config.get("target_secondary_category", "")

        # Validate strategy configuration
        self._validate_strategy_config()

        # Bin assignment tracking
        self.category_to_bin: Dict[str, int] = {}  # Maps category names to bin numbers
        self.used_bins = {self.overflow_bin}  # Track which bins are unavailable
        self.next_available_bin = 1  # Start looking from bin 1

        # ============================================================
        # CALLBACK SYSTEM FOR GUI UPDATES
        # ============================================================
        # Callback tracking for notifying external systems (like the GUI)
        # when bins are assigned to categories. Each callback should be
        # a function that accepts (bin_number: int, category: str)
        self.assignment_callbacks: List[Callable[[int, str], None]] = []

        # Apply pre-assignments from configuration
        pre_assignments = sorting_config.get("pre_assignments", {})
        if pre_assignments:
            self._apply_pre_assignments(pre_assignments)

        # Find first available bin for dynamic assignment
        self.next_available_bin = self._find_next_available_bin()

        logger.info(
            f"Bin assignment module initialized: strategy={self.strategy}, "
            f"max_bins={self.max_bins}, overflow_bin={self.overflow_bin}"
        )
        if pre_assignments:
            logger.info(f"Pre-assigned bins: {pre_assignments}")
        logger.debug(f"Next available bin for dynamic assignment: {self.next_available_bin}")

    # ========================================================================
    # CALLBACK REGISTRATION AND NOTIFICATION
    # ========================================================================
    # These methods allow external systems (GUI, statistics modules, etc.)
    # to be notified when bin assignments occur. This implements the Observer
    # pattern where the GUI "subscribes" to bin assignment events.

    def register_assignment_callback(self, callback_function: Callable[[int, str], None]):
        """
        Register a callback function to be notified of bin assignments.

        PURPOSE:
        This allows external systems (especially the GUI) to be notified
        immediately when a bin is assigned to a category. The callback
        is called synchronously, so it happens as part of the assignment
        process.

        WHEN TO USE:
        Call this during initialization to register any listeners that
        need to know about bin assignments. The sorting_gui calls this
        during its startup to register its update method.

        CALLBACK SIGNATURE:
        The registered function must accept two arguments:
            callback(bin_number: int, category: str)

        WHERE CALLBACKS ARE TRIGGERED:
        Callbacks are triggered in two scenarios:
        1. During initialization when pre-assignments are loaded
        2. During runtime when new categories are dynamically assigned

        THREAD SAFETY:
        Callbacks are called synchronously on the same thread that
        performs the assignment. If your callback does heavy work,
        consider using a queue or thread pool.

        Args:
            callback_function: Function to call when bin is assigned.
                              Must accept (bin_number: int, category: str)

        Example:
            # In sorting_gui.py __init__:
            bin_assignment.register_assignment_callback(self.on_bin_assigned)

            # Later, when assignment happens:
            # bin_assignment assigns Bin 3 to "Brick"
            # GUI's on_bin_assigned(3, "Brick") is called automatically
        """
        self.assignment_callbacks.append(callback_function)
        logger.debug(
            f"Registered bin assignment callback: {callback_function.__name__}"
        )

    def _notify_assignment_callbacks(self, bin_number: int, category: str):
        """
        Notify all registered callbacks that a bin has been assigned.

        PURPOSE:
        This is an internal method that calls all registered callbacks
        when a bin assignment occurs. It handles errors gracefully so
        one failing callback doesn't break the entire system.

        WHEN CALLED:
        This is called internally by:
        1. _apply_pre_assignments() - when loading pre-assignments
        2. _assign_or_create_bin() - when dynamically assigning new bins

        ERROR HANDLING:
        If a callback raises an exception, it is caught and logged,
        but other callbacks still execute. This prevents one bad
        callback from breaking the whole notification system.

        SYNCHRONOUS EXECUTION:
        All callbacks are executed synchronously in the order they
        were registered. The assignment operation waits for all
        callbacks to complete before continuing.

        Args:
            bin_number: The bin that was assigned (0-based index)
            category: The category name assigned to the bin

        Example Flow:
            # Somewhere in the code:
            self._notify_assignment_callbacks(3, "Brick")

            # This calls:
            # 1. sorting_gui.on_bin_assigned(3, "Brick")
            # 2. any_other_registered_callback(3, "Brick")
            # 3. etc.
        """
        for callback in self.assignment_callbacks:
            try:
                # Call the callback with bin number and category
                callback(bin_number, category)
            except Exception as e:
                # Log error but don't let it crash the system
                logger.error(
                    f"Error in bin assignment callback {callback.__name__}: {e}",
                    exc_info=True
                )

    # ========================================================================
    # CONFIGURATION AND VALIDATION
    # ========================================================================

    def _validate_strategy_config(self) -> None:
        """
        Validate that strategy configuration is complete and correct.

        Called during initialization. Raises error if configuration is invalid.

        Raises:
            ValueError: If strategy configuration is incomplete or invalid
        """
        # Secondary strategy requires target primary category
        if self.strategy == "secondary" and not self.target_primary:
            raise ValueError(
                "Secondary sorting strategy requires target_primary_category in config"
            )

        # Tertiary strategy requires both target categories
        if self.strategy == "tertiary":
            if not self.target_primary or not self.target_secondary:
                raise ValueError(
                    "Tertiary sorting strategy requires both target_primary_category "
                    "and target_secondary_category in config"
                )

        # Validate strategy value
        if self.strategy not in ["primary", "secondary", "tertiary"]:
            raise ValueError(
                f"Invalid sorting strategy: {self.strategy}. "
                f"Must be 'primary', 'secondary', or 'tertiary'"
            )

    def _apply_pre_assignments(self, pre_assignments: Dict[str, int]) -> None:
        """
        Apply pre-assigned category-to-bin mappings from configuration.

        Called during initialization. Pre-assignments let users specify that
        certain categories should always go to specific bins.

        CALLBACK NOTIFICATION:
        Each successful pre-assignment triggers registered callbacks, allowing
        the GUI to display pre-assigned bins at startup.

        Args:
            pre_assignments: Dictionary mapping category names to bin numbers
                           Example: {"Basic": 1, "Technic": 4}
        """
        logger.info(f"Applying {len(pre_assignments)} pre-assignments")

        for category, bin_number in pre_assignments.items():
            # Validate bin number is in valid range
            if not (1 <= bin_number <= self.max_bins):
                logger.warning(
                    f"Pre-assignment ignored: bin {bin_number} for '{category}' "
                    f"is out of range (1-{self.max_bins})"
                )
                continue

            # Validate category exists for current strategy
            if not self._is_valid_category(category):
                logger.warning(
                    f"Pre-assignment ignored: category '{category}' is not valid "
                    f"for {self.strategy} strategy"
                )
                continue

            # Check if bin already assigned to another category
            if bin_number in self.used_bins:
                existing = [cat for cat, bin_num in self.category_to_bin.items()
                            if bin_num == bin_number]
                logger.warning(
                    f"Pre-assignment ignored: bin {bin_number} already assigned "
                    f"to {existing[0] if existing else 'overflow'}"
                )
                continue

            # Apply the pre-assignment
            self.category_to_bin[category] = bin_number
            self.used_bins.add(bin_number)
            logger.info(f"Pre-assigned '{category}' → Bin {bin_number}")

            # NEW: Notify callbacks about this pre-assignment
            # This ensures the GUI shows pre-assigned bins at startup
            self._notify_assignment_callbacks(bin_number, category)

    def _is_valid_category(self, category: str) -> bool:
        """
        Check if a category name is valid for the current sorting strategy.

        Used during pre-assignment validation to ensure user hasn't specified
        categories that don't exist or don't match the current strategy level.

        Args:
            category: Category name to validate

        Returns:
            True if category exists at the current strategy level

        Example:
            # With primary strategy
            _is_valid_category("Basic") -> True
            _is_valid_category("Brick") -> False (Brick is secondary, not primary)
        """
        # Use category service to check if category exists at strategy level
        available_categories = self.category_service.get_categories_for_strategy(
            self.strategy,
            primary_category=self.target_primary if self.strategy != "primary" else None,
            secondary_category=self.target_secondary if self.strategy == "tertiary" else None
        )

        return category in available_categories

    def _find_next_available_bin(self) -> Optional[int]:
        """
        Find the next bin number available for dynamic assignment.

        Called during initialization and after each dynamic assignment to find
        the next free bin. Scans from 1 to max_bins looking for unused bins.

        Returns:
            Next available bin number (1-max_bins), or None if all bins used
        """
        for bin_num in range(1, self.max_bins + 1):
            if bin_num not in self.used_bins:
                return bin_num
        return None

    # ========================================================================
    # BIN ASSIGNMENT LOGIC
    # ========================================================================

    def assign_bin(self, categories: CategoryInfo) -> BinAssignment:
        """
        Assign a bin number for a piece based on its categories.

        This is the main public method called by processing_coordinator for each
        piece. It filters the piece based on target categories (for secondary/tertiary
        strategies), extracts the appropriate category level, and assigns a bin.

        Called by: processing_coordinator.py during piece processing

        Args:
            categories: CategoryInfo with all category levels populated

        Returns:
            BinAssignment with bin number (0 = overflow)

        Example:
            # Primary strategy
            categories = CategoryInfo(element_id="3001", primary_category="Basic", ...)
            result = assign_bin(categories)
            # result.bin_number = 1 (or whatever bin "Basic" is assigned to)

            # Secondary strategy (target_primary="Basic")
            categories = CategoryInfo(element_id="3001", primary_category="Basic",
                                     secondary_category="Brick", ...)
            result = assign_bin(categories)
            # result.bin_number = 2 (or whatever bin "Brick" is assigned to)
        """
        # Filter and extract the category to sort by
        category = self.filter_category_for_sorting(categories)

        # Assign bin for this category (handles pre-assigned, dynamic, and overflow)
        bin_number = self._assign_or_create_bin(category)

        logger.debug(
            f"Piece {categories.element_id}: category='{category}' → Bin {bin_number}"
        )

        return BinAssignment(bin_number=bin_number)

    def filter_category_for_sorting(self, categories: CategoryInfo) -> str:
        """
        Filter and return the category to sort by based on strategy.

        For primary strategy, returns the primary category directly.

        For secondary/tertiary strategies, this filters out pieces that don't
        match the target categories by returning "Unknown" (which routes them
        to overflow bin).

        Called by: assign_bin() for every piece

        Args:
            categories: CategoryInfo with all category levels

        Returns:
            Category string to use for bin assignment, or "Unknown" if filtered out

        Examples:
            # Primary strategy
            categories = CategoryInfo(primary_category="Basic", ...)
            filter_category_for_sorting(categories) -> "Basic"

            # Secondary strategy (target_primary="Basic")
            categories = CategoryInfo(primary_category="Basic", secondary_category="Brick", ...)
            filter_category_for_sorting(categories) -> "Brick"

            categories = CategoryInfo(primary_category="Technic", ...)
            filter_category_for_sorting(categories) -> "Unknown" (filtered out)

            # Tertiary strategy (target_primary="Basic", target_secondary="Brick")
            categories = CategoryInfo(primary="Basic", secondary="Brick", tertiary="2x4", ...)
            filter_category_for_sorting(categories) -> "2x4"

            categories = CategoryInfo(primary="Basic", secondary="Plate", ...)
            filter_category_for_sorting(categories) -> "Unknown" (filtered out)
        """
        if self.strategy == "primary":
            # Primary strategy: sort all pieces by primary category
            return categories.primary_category

        elif self.strategy == "secondary":
            # Secondary strategy: only sort pieces matching target primary
            if categories.primary_category != self.target_primary:
                logger.debug(
                    f"Piece filtered out: primary '{categories.primary_category}' "
                    f"!= target '{self.target_primary}'"
                )
                return "Unknown"

            # Piece matches target, use secondary category for sorting
            return categories.secondary_category or "Unknown"

        elif self.strategy == "tertiary":
            # Tertiary strategy: only sort pieces matching both target categories
            if (categories.primary_category != self.target_primary or
                    categories.secondary_category != self.target_secondary):
                logger.debug(
                    f"Piece filtered out: categories '{categories.primary_category}/"
                    f"{categories.secondary_category}' != targets "
                    f"'{self.target_primary}/{self.target_secondary}'"
                )
                return "Unknown"

            # Piece matches both targets, use tertiary category for sorting
            return categories.tertiary_category or "Unknown"

        # Should never reach here due to validation in __init__
        logger.error(f"Unknown strategy: {self.strategy}")
        return "Unknown"

    def _assign_or_create_bin(self, category: str) -> int:
        """
        Assign bin for a category, creating new assignment if needed.

        This is the unified bin assignment logic that works for any category level.
        It handles three cases:
        1. Unknown categories → overflow bin
        2. Already assigned categories → existing bin
        3. New categories → next available bin (or overflow if bins full)

        CALLBACK NOTIFICATION:
        When a NEW category is assigned to a bin (case 3), registered callbacks
        are notified. This allows the GUI to update in real-time as new categories
        are encountered during sorting.

        Called by: assign_bin() after filtering

        Args:
            category: Category name to assign bin for

        Returns:
            Bin number (0 = overflow)

        Example flow:
            # First "Basic" piece
            _assign_or_create_bin("Basic") -> 1 (assigns to first available bin)
            # Callbacks notified: on_bin_assigned(1, "Basic")

            # Second "Basic" piece
            _assign_or_create_bin("Basic") -> 1 (returns existing assignment)
            # No callbacks (already assigned)

            # Unknown piece
            _assign_or_create_bin("Unknown") -> 0 (overflow)
            # No callbacks (overflow is not a "real" assignment)

            # When all bins full
            _assign_or_create_bin("Plates") -> 0 (overflow)
            # No callbacks (overflow fallback)
        """
        # Case 1: Unknown or empty category goes to overflow
        if not category or category == "Unknown":
            logger.debug("Category is Unknown → overflow bin")
            return self.overflow_bin

        # Case 2: Category already has an assigned bin
        if category in self.category_to_bin:
            bin_number = self.category_to_bin[category]
            logger.debug(f"Category '{category}' already assigned → Bin {bin_number}")
            return bin_number

        # Case 3: New category needs a bin
        if self.next_available_bin is not None:
            # We have bins available - assign the next one
            bin_number = self.next_available_bin
            self.category_to_bin[category] = bin_number
            self.used_bins.add(bin_number)

            logger.info(f"New category '{category}' assigned → Bin {bin_number}")

            # NEW: Notify callbacks about this NEW dynamic assignment
            # This updates the GUI in real-time when a new category appears
            self._notify_assignment_callbacks(bin_number, category)

            # Find next available bin for future assignments
            self.next_available_bin = self._find_next_available_bin()

            return bin_number

        # No bins available - overflow
        logger.warning(
            f"No bins available for category '{category}' → overflow bin "
            f"({len(self.category_to_bin)} categories already assigned)"
        )
        return self.overflow_bin

    # ========================================================================
    # QUERY AND STATISTICS METHODS
    # ========================================================================

    def get_available_categories(self) -> List[str]:
        """
        Get list of categories that can be sorted with current strategy.

        Uses category_service to get categories at the appropriate level.
        Useful for GUI to show what categories can be pre-assigned.

        Returns:
            Sorted list of category names available for current strategy

        Example:
            # Primary strategy
            get_available_categories() -> ["Basic", "Duplo", "Plates", "Technic", ...]

            # Secondary strategy (target_primary="Basic")
            get_available_categories() -> ["Brick", "Plate", "Slope", "Tile", ...]
        """
        return self.category_service.get_categories_for_strategy(
            self.strategy,
            primary_category=self.target_primary if self.strategy != "primary" else None,
            secondary_category=self.target_secondary if self.strategy == "tertiary" else None
        )

    def get_bin_assignments(self) -> Dict[str, int]:
        """
        Get current mapping of categories to bins.

        Returns a copy of the internal mapping. Useful for displaying
        current assignments in GUI or logging.

        Returns:
            Dictionary mapping category names to bin numbers

        Example:
            get_bin_assignments() -> {"Basic": 1, "Technic": 4, "Plates": 2}
        """
        return self.category_to_bin.copy()

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about bin assignment state.

        Useful for monitoring and debugging.

        Returns:
            Dictionary with assignment statistics
        """
        return {
            "strategy": self.strategy,
            "max_bins": self.max_bins,
            "assigned_categories": len(self.category_to_bin),
            "used_bins": len(self.used_bins) - 1,  # Exclude overflow bin
            "available_bins": self.max_bins - (len(self.used_bins) - 1),
            "next_available": self.next_available_bin,
            "target_primary": self.target_primary,
            "target_secondary": self.target_secondary
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_bin_assignment_module(
        config_manager: EnhancedConfigManager,
        category_service: CategoryHierarchyService
) -> BinAssignmentModule:
    """
    Factory function to create a bin assignment module.

    Called during application startup to create the bin assignment module
    that will be used by the processing coordinator.

    Args:
        config_manager: Configuration manager instance
        category_service: Category hierarchy service for validation

    Returns:
        Initialized BinAssignmentModule

    Example:
        config_manager = create_config_manager()
        category_service = create_category_hierarchy_service(config_manager)
        bin_assignment = create_bin_assignment_module(config_manager, category_service)

        # Use in processing coordinator
        categories = category_lookup.get_categories(element_id)
        bin_result = bin_assignment.assign_bin(categories)
    """
    return BinAssignmentModule(config_manager, category_service)
