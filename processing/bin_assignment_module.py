# processing/bin_assignment_module.py
"""
bin_assignment_module.py - Bin assignment logic for Lego pieces

This module owns the bin bookkeeping for a sorting run: which sort keys map to
which bins, which bins are still free, pre-assignments from config, and the
observer callbacks the GUI uses to mirror assignments.

The decision of *what* sort key a given piece produces is delegated to a
SortingMode (see processing/sorting_modes.py). The mode handles mode-specific
filtering (e.g. design-ID tier targets, bag-inventory lookup); this module is
mode-agnostic bookkeeping on top of whatever key the mode returns.

RESPONSIBILITIES:
- Map sort keys (category name, "bag_N", etc.) to bin numbers
- Honor pre-assignments from configuration
- Dynamically assign new keys to the next available bin
- Route unknown/overflow pieces to the overflow bin
- Notify registered callbacks when bin assignments occur (for GUI updates)

DOES NOT:
- Decide how a piece becomes a sort key (that's SortingMode)
- Track bin capacity (that's bin_capacity_module)
- Identify pieces (that's the processing coordinator and its sub-modules)
- Control servos or hardware

CALLBACK SYSTEM:
This module implements the Observer pattern to notify external systems
(primarily the GUI) when bins are assigned to sort keys. External systems
register callback functions during initialization, and these callbacks are
automatically invoked when assignments occur.
"""

import logging
from typing import Dict, List, Optional, Callable

from processing.processing_data_models import IdentifiedPiece, BinAssignment
from processing.category_hierarchy_service import CategoryHierarchyService
from processing.sorting_modes import SortingMode, create_sorting_mode
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class BinAssignmentModule:
    """
    Maps identified pieces to bin numbers.

    The mode-specific routing rule lives on `self.mode` (a SortingMode). This
    class turns the mode's sort keys into bin numbers, manages the running
    key→bin table, and notifies observers on new assignments.
    """

    def __init__(self, config_manager: EnhancedConfigManager,
                 category_service: CategoryHierarchyService):
        """
        Initialize bin assignment module.

        Args:
            config_manager: Configuration manager instance
            category_service: Category hierarchy service — passed through to
                the sorting mode for modes that enumerate category keys
                (currently DesignIdMode).
        """
        if config_manager is None:
            raise ValueError("config_manager cannot be None")

        self.config_manager = config_manager
        self.category_service = category_service

        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)

        self.max_bins = sorting_config["max_bins"]
        self.overflow_bin = sorting_config["overflow_bin"]

        # The mode owns the policy: "given a piece, what is the sort key?"
        self.mode: SortingMode = create_sorting_mode(config_manager, category_service)

        # Bin assignment tracking (mode-agnostic)
        self.key_to_bin: Dict[str, int] = {}
        self.used_bins = {self.overflow_bin}
        self.next_available_bin: Optional[int] = 1

        self.assignment_callbacks: List[Callable[[int, str], None]] = []

        pre_assignments = sorting_config.get("pre_assignments", {})
        if pre_assignments:
            self._apply_pre_assignments(pre_assignments)

        self.next_available_bin = self._find_next_available_bin()

        logger.info(
            f"Bin assignment module initialized: mode={type(self.mode).__name__}, "
            f"max_bins={self.max_bins}, overflow_bin={self.overflow_bin}"
        )
        if pre_assignments:
            logger.info(f"Pre-assigned bins: {pre_assignments}")
        logger.debug(f"Next available bin for dynamic assignment: {self.next_available_bin}")

    # ========================================================================
    # CALLBACK REGISTRATION AND NOTIFICATION
    # ========================================================================

    def register_assignment_callback(self, callback_function: Callable[[int, str], None]):
        """
        Register a callback invoked whenever a sort key is bound to a bin.

        Signature: callback(bin_number: int, sort_key: str)

        Callbacks run synchronously on the assigning thread. Exceptions raised
        by a callback are logged but do not affect other callbacks.
        """
        self.assignment_callbacks.append(callback_function)
        logger.debug(
            f"Registered bin assignment callback: {callback_function.__name__}"
        )

    def _notify_assignment_callbacks(self, bin_number: int, sort_key: str):
        """Invoke every registered callback with the new assignment."""
        for callback in self.assignment_callbacks:
            try:
                callback(bin_number, sort_key)
            except Exception as e:
                logger.error(
                    f"Error in bin assignment callback {callback.__name__}: {e} "
                    f"(bin_number={bin_number}, sort_key={sort_key!r})",
                    exc_info=True
                )

    # ========================================================================
    # CONFIGURATION AND VALIDATION
    # ========================================================================

    def _apply_pre_assignments(self, pre_assignments: Dict[str, int]) -> None:
        """
        Bind sort keys to bins from the `pre_assignments` config section.

        Keys are whatever the active mode considers a sort key — primary
        category names for DesignIdMode.tier="primary", "bag_N" for BagMode, etc.
        """
        logger.info(f"Applying {len(pre_assignments)} pre-assignments")

        for sort_key, bin_number in pre_assignments.items():
            if not (1 <= bin_number <= self.max_bins):
                logger.warning(
                    f"Pre-assignment ignored: bin {bin_number} for {sort_key!r} "
                    f"is out of range (1-{self.max_bins})"
                )
                continue

            if not self.mode.is_valid_key(sort_key):
                logger.warning(
                    f"Pre-assignment ignored: key {sort_key!r} is not valid "
                    f"for {type(self.mode).__name__}"
                )
                continue

            if bin_number in self.used_bins:
                existing = [k for k, b in self.key_to_bin.items() if b == bin_number]
                logger.warning(
                    f"Pre-assignment ignored: bin {bin_number} already assigned "
                    f"to {existing[0] if existing else 'overflow'}"
                )
                continue

            self.key_to_bin[sort_key] = bin_number
            self.used_bins.add(bin_number)
            logger.info(f"Pre-assigned {sort_key!r} → Bin {bin_number}")

            self._notify_assignment_callbacks(bin_number, sort_key)

    def _find_next_available_bin(self) -> Optional[int]:
        """Return the lowest-numbered bin (1..max_bins) not yet claimed."""
        for bin_num in range(1, self.max_bins + 1):
            if bin_num not in self.used_bins:
                return bin_num
        return None

    # ========================================================================
    # BIN ASSIGNMENT LOGIC
    # ========================================================================

    def assign_bin(self, piece: IdentifiedPiece) -> BinAssignment:
        """
        Assign a bin number for a piece.

        Delegates to the active SortingMode to compute a sort key, then maps
        that key to a bin (existing assignment, new dynamic assignment, or
        overflow if bins are exhausted or the key is "Unknown").

        Args:
            piece: Fully-populated IdentifiedPiece from the processing pipeline.

        Returns:
            BinAssignment with bin number (0 = overflow)
        """
        sort_key = self.mode.sort_key_for(piece)
        bin_number = self._assign_or_create_bin(sort_key)

        logger.debug(
            f"Piece {piece.piece_id}: design_id={piece.design_id}, "
            f"element_id={piece.element_id}, key={sort_key!r} → Bin {bin_number}"
        )

        return BinAssignment(bin_number=bin_number)

    def _assign_or_create_bin(self, sort_key: str) -> int:
        """
        Return the bin for `sort_key`, creating a new assignment if needed.

        Three cases:
          1. sort_key is empty or "Unknown" → overflow bin.
          2. sort_key already has a bound bin → return it.
          3. New sort_key → claim the next available bin (or overflow if full).

        Callbacks fire only on case 3 when a new dynamic binding is created.
        """
        if not sort_key or sort_key == "Unknown":
            logger.debug(f"Sort key {sort_key!r} → overflow bin")
            return self.overflow_bin

        if sort_key in self.key_to_bin:
            bin_number = self.key_to_bin[sort_key]
            logger.debug(f"Key {sort_key!r} already assigned → Bin {bin_number}")
            return bin_number

        if self.next_available_bin is not None:
            bin_number = self.next_available_bin
            self.key_to_bin[sort_key] = bin_number
            self.used_bins.add(bin_number)

            logger.info(f"New key {sort_key!r} assigned → Bin {bin_number}")
            self._notify_assignment_callbacks(bin_number, sort_key)

            self.next_available_bin = self._find_next_available_bin()
            return bin_number

        logger.warning(
            f"No bins available for key {sort_key!r} → overflow "
            f"({len(self.key_to_bin)} keys already assigned)"
        )
        return self.overflow_bin

    # ========================================================================
    # QUERY AND STATISTICS METHODS
    # ========================================================================

    def get_available_categories(self) -> List[str]:
        """
        Return sort keys the active mode knows about.

        Kept under the old name for GUI compatibility. For BagMode this lists
        the bags found in the inventory; for DesignIdMode it lists categories
        at the configured tier.
        """
        return self.mode.available_keys()

    def get_bin_assignments(self) -> Dict[str, int]:
        """Return a copy of the current sort-key → bin mapping."""
        return self.key_to_bin.copy()

    def get_statistics(self) -> Dict[str, any]:
        """Return bookkeeping statistics for monitoring and debugging."""
        return {
            "mode": type(self.mode).__name__,
            "max_bins": self.max_bins,
            "assigned_keys": len(self.key_to_bin),
            "used_bins": len(self.used_bins) - 1,  # Exclude overflow bin
            "available_bins": self.max_bins - (len(self.used_bins) - 1),
            "next_available": self.next_available_bin,
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

    Args:
        config_manager: Configuration manager instance
        category_service: Category hierarchy service (used by DesignIdMode)

    Returns:
        Initialized BinAssignmentModule
    """
    return BinAssignmentModule(config_manager, category_service)
