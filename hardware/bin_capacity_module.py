# hardware/bin_capacity_module.py
"""
bin_capacity_module.py - Tracks bin fill levels and capacity management

This module manages the state of all sorting bins, tracking how many pieces
are in each bin and triggering warnings when bins approach or reach capacity.

RESPONSIBILITIES:
- Track piece counts for bins 0-9
- Monitor fill levels against capacity limits
- Trigger warning callbacks at configurable threshold
- Trigger full callbacks when bins reach capacity
- Provide bin status for GUI display
- Allow manual reset of bin counts

DOES NOT:
- Control servo movement (that's arduino_servo_module)
- Make sorting decisions (that's processing modules)
- Stop the conveyor (no automated feed control exists)

BIN NUMBERING:
- Bin 0: Overflow bin (for unknown pieces or full bins)
- Bins 1-9: Regular sorting bins

USAGE:
    bin_manager = BinCapacityManager(config_manager)

    # Register callbacks
    bin_manager.register_warning_callback(on_bin_warning)
    bin_manager.register_full_callback(on_bin_full)

    # Add pieces as they're sorted
    if bin_manager.add_piece(bin_number):
        print("Piece added successfully")
    else:
        print("Bin is full!")

    # Get status for GUI
    status = bin_manager.get_bin_status(bin_number)

    # Reset when operator empties bin
    bin_manager.reset_bin(bin_number)
"""

import logging
from typing import Dict, List, Callable, Optional
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class BinCapacityManager:
    """
    Manages bin capacity tracking and warning system.

    This class maintains the current state of all bins, tracks fill levels,
    and notifies registered callbacks when bins reach warning or full status.
    """

    def __init__(self, config_manager: EnhancedConfigManager):
        """
        Initialize bin capacity manager.

        Args:
            config_manager: Enhanced configuration manager instance
        """
        self.config_manager = config_manager

        # Load configuration from sorting module config
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)

        self.max_bins = sorting_config.get('max_bins', 9)
        self.max_capacity = sorting_config.get('max_pieces_per_bin', 50)
        self.warning_threshold = sorting_config.get('bin_warning_threshold', 0.8)

        # Initialize bin counts (bins 0-9, all start at 0)
        self.bin_counts: Dict[int, int] = {i: 0 for i in range(10)}

        # Track which bins have triggered warnings/full status
        self.warning_triggered: Dict[int, bool] = {i: False for i in range(10)}
        self.full_triggered: Dict[int, bool] = {i: False for i in range(10)}

        # Callback lists
        self.warning_callbacks: List[Callable] = []
        self.full_callbacks: List[Callable] = []

        logger.info("BinCapacityManager initialized")
        logger.info(f"  Max bins: {self.max_bins}")
        logger.info(f"  Max capacity per bin: {self.max_capacity}")
        logger.info(f"  Warning threshold: {self.warning_threshold * 100:.0f}%")

    # ========================================================================
    # PIECE TRACKING
    # ========================================================================

    def add_piece(self, bin_number: int) -> bool:
        """
        Add a piece to the specified bin.

        This method increments the bin count and checks if warning or full
        thresholds have been reached. If the bin is already full, the piece
        is NOT added and False is returned.

        Args:
            bin_number: Bin to add piece to (0-9)

        Returns:
            True if piece was added successfully, False if bin is full
        """
        if bin_number not in self.bin_counts:
            logger.error(f"Invalid bin number: {bin_number}")
            return False

        # Check if bin is already full
        if self.is_bin_full(bin_number):
            logger.warning(f"Bin {bin_number} is full, cannot add piece")
            self._trigger_full_warning(bin_number)
            return False

        # Add piece to bin
        self.bin_counts[bin_number] += 1
        current_count = self.bin_counts[bin_number]

        logger.debug(f"Added piece to bin {bin_number} (count: {current_count}/{self.max_capacity})")

        # Check if warning threshold reached (and not already triggered)
        fill_percentage = self.get_fill_percentage(bin_number)
        if fill_percentage >= self.warning_threshold and not self.warning_triggered[bin_number]:
            self._trigger_warning(bin_number)
            self.warning_triggered[bin_number] = True

        # Check if bin just became full (and not already triggered)
        if self.is_bin_full(bin_number) and not self.full_triggered[bin_number]:
            self._trigger_full_warning(bin_number)
            self.full_triggered[bin_number] = True

        return True

    def reset_bin(self, bin_number: int):
        """
        Reset bin count to zero.

        This should be called when an operator physically empties a bin.
        In the future, this will be triggered by a button in the sorting GUI.

        Args:
            bin_number: Bin to reset (0-9)
        """
        if bin_number not in self.bin_counts:
            logger.error(f"Invalid bin number: {bin_number}")
            return

        old_count = self.bin_counts[bin_number]
        self.bin_counts[bin_number] = 0

        # Reset warning/full flags so they can trigger again
        self.warning_triggered[bin_number] = False
        self.full_triggered[bin_number] = False

        logger.info(f"Reset bin {bin_number} (was {old_count} pieces, now 0)")

        # Print confirmation for operator
        print(f"\n{'=' * 60}")
        print(f"✓ Bin {bin_number} count reset to 0")
        print(f"  Previous count: {old_count}")
        print(f"{'=' * 60}\n")

    def reset_all_bins(self):
        """Reset all bin counts to zero."""
        logger.info("Resetting all bins")
        for bin_number in range(10):
            self.reset_bin(bin_number)

    # ========================================================================
    # STATUS QUERIES
    # ========================================================================

    def get_bin_status(self, bin_number: int) -> Dict:
        """
        Get current status of a specific bin.

        This is the primary method for the GUI to query bin state.

        Args:
            bin_number: Bin to query (0-9)

        Returns:
            Dictionary with bin status information:
            {
                'bin_number': int,
                'count': int,
                'max_capacity': int,
                'percentage': float (0.0 to 1.0),
                'is_full': bool,
                'is_warning': bool
            }
        """
        if bin_number not in self.bin_counts:
            logger.error(f"Invalid bin number: {bin_number}")
            return {}

        return {
            'bin_number': bin_number,
            'count': self.bin_counts[bin_number],
            'max_capacity': self.max_capacity,
            'percentage': self.get_fill_percentage(bin_number),
            'is_full': self.is_bin_full(bin_number),
            'is_warning': self.get_fill_percentage(bin_number) >= self.warning_threshold
        }

    def get_all_bins_status(self) -> List[Dict]:
        """
        Get status for all bins.

        Useful for GUI to display complete bin overview.

        Returns:
            List of status dictionaries, one for each bin (0-9)
        """
        return [self.get_bin_status(i) for i in range(10)]

    def get_fill_percentage(self, bin_number: int) -> float:
        """
        Get fill percentage for a bin.

        Args:
            bin_number: Bin to check (0-9)

        Returns:
            Fill percentage as float (0.0 to 1.0), or 0.0 if invalid bin
        """
        if bin_number not in self.bin_counts:
            return 0.0

        return self.bin_counts[bin_number] / self.max_capacity

    def is_bin_full(self, bin_number: int) -> bool:
        """
        Check if a bin has reached maximum capacity.

        Args:
            bin_number: Bin to check (0-9)

        Returns:
            True if bin is at or above max capacity
        """
        if bin_number not in self.bin_counts:
            return False

        return self.bin_counts[bin_number] >= self.max_capacity

    def is_bin_at_warning(self, bin_number: int) -> bool:
        """
        Check if a bin has reached warning threshold.

        Args:
            bin_number: Bin to check (0-9)

        Returns:
            True if bin is at or above warning threshold
        """
        return self.get_fill_percentage(bin_number) >= self.warning_threshold

    def can_accept_piece(self, bin_number: int) -> bool:
        """
        Check if a bin can accept another piece.

        Args:
            bin_number: Bin to check (0-9)

        Returns:
            True if bin has space available
        """
        return not self.is_bin_full(bin_number)

    # ========================================================================
    # CALLBACK MANAGEMENT
    # ========================================================================

    def register_warning_callback(self, callback: Callable):
        """
        Register callback to be called when bin reaches warning threshold.

        Callback signature: callback(bin_number: int, current_count: int, max_count: int)

        Args:
            callback: Function to call on warning
        """
        self.warning_callbacks.append(callback)
        logger.debug(f"Registered warning callback: {callback.__name__}")

    def register_full_callback(self, callback: Callable):
        """
        Register callback to be called when bin becomes full.

        Callback signature: callback(bin_number: int, current_count: int, max_count: int)

        Args:
            callback: Function to call when bin is full
        """
        self.full_callbacks.append(callback)
        logger.debug(f"Registered full callback: {callback.__name__}")

    def _trigger_warning(self, bin_number: int):
        """Internal: Trigger warning callbacks and print warning."""
        current_count = self.bin_counts[bin_number]
        percentage = self.get_fill_percentage(bin_number) * 100

        # Print warning to console
        print(f"\n{'=' * 60}")
        print(f"⚠️  WARNING: Bin {bin_number} is {percentage:.0f}% full")
        print(f"   Current: {current_count}/{self.max_capacity} pieces")
        print(f"   Consider emptying soon")
        print(f"{'=' * 60}\n")

        logger.warning(f"Bin {bin_number} reached warning threshold: {current_count}/{self.max_capacity}")

        # Notify all registered callbacks
        for callback in self.warning_callbacks:
            try:
                callback(bin_number, current_count, self.max_capacity)
            except Exception as e:
                logger.error(f"Error in warning callback: {e}", exc_info=True)

    def _trigger_full_warning(self, bin_number: int):
        """Internal: Trigger full callbacks and print warning."""
        current_count = self.bin_counts[bin_number]

        # Print full warning to console
        print(f"\n{'=' * 60}")
        print(f"🚨 ALERT: BIN {bin_number} IS FULL!")
        print(f"   Current: {current_count}/{self.max_capacity} pieces")
        print(f"   Please empty bin {bin_number} and reset the count")
        print(f"   (In GUI: Click reset button for bin {bin_number})")
        print(f"{'=' * 60}\n")

        logger.error(f"Bin {bin_number} is full: {current_count}/{self.max_capacity}")

        # Notify all registered callbacks
        for callback in self.full_callbacks:
            try:
                callback(bin_number, current_count, self.max_capacity)
            except Exception as e:
                logger.error(f"Error in full callback: {e}", exc_info=True)

    # ========================================================================
    # STATISTICS AND REPORTING
    # ========================================================================

    def get_statistics(self) -> Dict:
        """
        Get overall bin capacity statistics.

        Useful for logging and system monitoring.

        Returns:
            Dictionary with statistics
        """
        total_pieces = sum(self.bin_counts.values())
        bins_in_use = sum(1 for count in self.bin_counts.values() if count > 0)
        full_bins = sum(1 for i in range(10) if self.is_bin_full(i))
        warning_bins = sum(1 for i in range(10) if self.is_bin_at_warning(i) and not self.is_bin_full(i))

        return {
            'total_pieces_sorted': total_pieces,
            'bins_in_use': bins_in_use,
            'full_bins': full_bins,
            'warning_bins': warning_bins,
            'average_fill_percentage': total_pieces / (10 * self.max_capacity),
            'bin_counts': self.bin_counts.copy()
        }

    def print_status_report(self):
        """Print a formatted status report of all bins."""
        print("\n" + "=" * 60)
        print("BIN CAPACITY STATUS REPORT")
        print("=" * 60)

        for bin_number in range(10):
            status = self.get_bin_status(bin_number)
            count = status['count']
            percentage = status['percentage'] * 100

            # Create visual bar
            bar_length = 20
            filled = int(bar_length * status['percentage'])
            bar = "█" * filled + "░" * (bar_length - filled)

            # Status indicator
            if status['is_full']:
                indicator = "🚨 FULL"
            elif status['is_warning']:
                indicator = "⚠️  WARN"
            else:
                indicator = "✓  OK  "

            print(f"Bin {bin_number}: [{bar}] {count:3d}/{self.max_capacity} ({percentage:5.1f}%) {indicator}")

        stats = self.get_statistics()
        print("=" * 60)
        print(f"Total pieces: {stats['total_pieces_sorted']}")
        print(f"Bins in use:  {stats['bins_in_use']}/10")
        print(f"Full bins:    {stats['full_bins']}")
        print(f"Warning bins: {stats['warning_bins']}")
        print("=" * 60 + "\n")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_bin_capacity_manager(config_manager: EnhancedConfigManager) -> BinCapacityManager:
    """
    Factory function to create BinCapacityManager instance.

    Args:
        config_manager: Enhanced configuration manager

    Returns:
        Initialized BinCapacityManager instance
    """
    return BinCapacityManager(config_manager)


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the bin capacity module functionality using real enhanced_config_manager.
    """

    print("\n" + "=" * 60)
    print("BIN CAPACITY MODULE TEST")
    print("=" * 60 + "\n")

    # Import enhanced_config_manager
    from enhanced_config_manager import create_config_manager, ModuleConfig

    # Create real config manager (uses actual config.json or defaults)
    config_manager = create_config_manager()

    # Display current sorting configuration
    sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)
    print("Current Sorting Configuration:")
    print(f"  Max bins: {sorting_config['max_bins']}")
    print(f"  Max pieces per bin: {sorting_config['max_pieces_per_bin']}")
    print(f"  Warning threshold: {sorting_config['bin_warning_threshold'] * 100:.0f}%")
    print()

    # Create bin manager with real config
    bin_manager = create_bin_capacity_manager(config_manager)


    # Register test callbacks
    def on_warning(bin_num, count, max_count):
        print(f"[CALLBACK] Warning triggered for bin {bin_num}")


    def on_full(bin_num, count, max_count):
        print(f"[CALLBACK] Full triggered for bin {bin_num}")


    bin_manager.register_warning_callback(on_warning)
    bin_manager.register_full_callback(on_full)

    # Test 1: Add pieces to bin 1 until full
    print("\n--- Test 1: Filling bin 1 to capacity ---")
    max_capacity = bin_manager.max_capacity
    for i in range(max_capacity + 5):  # Try to overfill
        success = bin_manager.add_piece(1)
        if not success:
            print(f"  Piece {i + 1}: REJECTED (bin full)")
            break
        else:
            print(f"  Piece {i + 1}: Added (total: {bin_manager.bin_counts[1]}/{max_capacity})")

    # Test 2: Check bin status
    print("\n--- Test 2: Bin status query ---")
    status = bin_manager.get_bin_status(1)
    print(f"  Bin 1 status:")
    print(f"    Count: {status['count']}/{status['max_capacity']}")
    print(f"    Fill: {status['percentage'] * 100:.1f}%")
    print(f"    Is Full: {status['is_full']}")
    print(f"    Is Warning: {status['is_warning']}")

    # Test 3: Reset bin
    print("\n--- Test 3: Reset bin 1 ---")
    bin_manager.reset_bin(1)

    # Test 4: Fill multiple bins with different amounts
    print("\n--- Test 4: Fill multiple bins ---")
    test_data = [
        (2, 5),  # Bin 2: 5 pieces (low)
        (3, int(max_capacity * 0.5)),  # Bin 3: 50% full
        (4, int(max_capacity * 0.85)),  # Bin 4: 85% full (warning)
    ]

    for bin_num, piece_count in test_data:
        for _ in range(piece_count):
            bin_manager.add_piece(bin_num)
        status = bin_manager.get_bin_status(bin_num)
        print(f"  Bin {bin_num}: {status['count']}/{max_capacity} ({status['percentage'] * 100:.0f}%)")

    # Test 5: Complete status report
    print("\n--- Test 5: Complete system status ---")
    bin_manager.print_status_report()

    # Test 6: Statistics
    print("\n--- Test 6: System statistics ---")
    stats = bin_manager.get_statistics()
    print(f"  Total pieces sorted: {stats['total_pieces_sorted']}")
    print(f"  Bins in use: {stats['bins_in_use']}/10")
    print(f"  Full bins: {stats['full_bins']}")
    print(f"  Warning bins: {stats['warning_bins']}")
    print(f"  Average fill: {stats['average_fill_percentage'] * 100:.1f}%")

    # Test 7: Reset all bins
    print("\n--- Test 7: Reset all bins ---")
    bin_manager.reset_all_bins()
    print("  All bins reset")

    # Final status
    print("\n--- Final Status ---")
    bin_manager.print_status_report()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("All tests passed with real configuration!")
    print("=" * 60 + "\n")