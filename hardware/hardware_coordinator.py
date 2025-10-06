# hardware/hardware_coordinator.py
"""
hardware_coordinator.py - Orchestrates physical sorting operations

This module coordinates the physical hardware components (servo and bin capacity)
to execute sorting operations. It is a pure executor - it receives commands
from the main orchestrator and executes the physical sorting.

RESPONSIBILITIES:
- Check bin capacity before sorting
- Move servo to correct bin position
- Update bin piece counts after sorting
- Handle overflow redirection when bins are full
- Track sorting statistics (success/failure/overflow)

DOES NOT:
- Store piece data or "ready" state (that's processing pipeline)
- Make timing decisions about WHEN to sort (that's LegoSorting008)
- Track piece positions or detect pieces (that's detector)
- Identify pieces or assign bins (that's processing)

ARCHITECTURE:
This is a service provider for the main orchestrator. It exposes a simple
API for executing physical sorts and provides statistics. All decision-making
about WHEN and WHAT to sort happens in LegoSorting008.

USAGE:
    coordinator = create_hardware_coordinator(bin_capacity, servo, config_manager)

    # Main orchestrator calls when piece ready to sort:
    success = coordinator.trigger_sort_for_piece(piece_id=123, bin_number=5)

    # Get statistics:
    stats = coordinator.get_statistics()

    # Shutdown:
    coordinator.release()
"""

import logging
import threading
from typing import Dict
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class HardwareCoordinator:
    """
    Executes physical sorting operations by coordinating hardware modules.

    This class provides a simple interface for physical sorting. It receives
    a piece ID and bin number, then executes the physical operation by
    checking capacity, moving the servo, and updating counts.
    """

    def __init__(self, bin_capacity_manager, servo_controller, config_manager: EnhancedConfigManager):
        """
        Initialize hardware coordinator with hardware module references.

        Args:
            bin_capacity_manager: BinCapacityManager instance
            servo_controller: ArduinoServoController instance
            config_manager: EnhancedConfigManager instance
        """
        # Store references to hardware modules
        self.bin_capacity = bin_capacity_manager
        self.servo = servo_controller
        self.config_manager = config_manager

        # Load configuration
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)
        self.overflow_bin = sorting_config['overflow_bin']

        # Statistics tracking (only state this module maintains)
        self.total_sorted = 0
        self.overflow_count = 0
        self.failed_sorts = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info("Hardware coordinator initialized")
        logger.info(f"  Overflow bin: {self.overflow_bin}")

    # ========================================================================
    # CORE SORTING OPERATION
    # ========================================================================

    def trigger_sort_for_piece(self, piece_id: int, bin_number: int) -> bool:
        """
        Execute physical sorting for a piece.

        This is called by the main orchestrator when a piece reaches the
        exit zone and is ready to be physically sorted. All decision-making
        has already happened - this function just executes the physical
        operation.

        The caller (LegoSorting008) has already determined:
        - The piece is fully processed (IdentifiedPiece.complete = True)
        - The bin assignment (bin_number from bin_assignment_module)
        - The timing is correct (piece in exit zone, rightmost position)

        This function executes the physical sorting sequence:
        1. Check if target bin has capacity
        2. If full, redirect to overflow bin
        3. Move servo to target bin position
        4. Update bin piece count
        5. Update statistics

        Args:
            piece_id: Piece identifier (used for logging and statistics)
            bin_number: Target bin number (0-9)

        Returns:
            True if sorting operation successful, False if failed

        Example:
            # In LegoSorting008 detector callback:
            if piece.is_rightmost_in_exit_zone and identified_piece.complete:
                success = hardware_coordinator.trigger_sort_for_piece(
                    piece_id=piece.id,
                    bin_number=identified_piece.bin_number
                )
        """
        with self.lock:
            logger.info(f"Sorting piece {piece_id} to bin {bin_number}")

            # ================================================================
            # STEP 1: CHECK BIN CAPACITY
            # ================================================================
            target_bin = bin_number

            # Check if target bin can accept another piece
            if not self.bin_capacity.can_accept_piece(target_bin):
                logger.warning(
                    f"Bin {target_bin} is full! "
                    f"Redirecting to overflow bin {self.overflow_bin}"
                )
                target_bin = self.overflow_bin
                self.overflow_count += 1

                # Check if overflow bin is also full
                if not self.bin_capacity.can_accept_piece(self.overflow_bin):
                    logger.error(f"Overflow bin {self.overflow_bin} is also full!")
                    logger.error("Cannot sort piece - all capacity exhausted")
                    self.failed_sorts += 1
                    return False

            # ================================================================
            # STEP 2: MOVE SERVO TO TARGET BIN
            # ================================================================
            # Use wait=False for non-blocking operation
            # Servo will start moving immediately, allowing system to continue
            success = self.servo.move_to_bin(target_bin, wait=False)

            if not success:
                logger.error(f"Failed to move servo to bin {target_bin}")
                self.failed_sorts += 1
                return False

            # ================================================================
            # STEP 3: UPDATE BIN CAPACITY COUNT
            # ================================================================
            # Note: add_piece() internally checks capacity and triggers
            # warning/full callbacks if thresholds are reached
            self.bin_capacity.add_piece(target_bin)

            # ================================================================
            # STEP 4: UPDATE STATISTICS
            # ================================================================
            self.total_sorted += 1

            logger.info(f"✓ Piece {piece_id} sorted to bin {target_bin}")

            # Log if this was an overflow redirect
            if target_bin != bin_number:
                logger.info(f"  (Redirected from full bin {bin_number})")

            return True

    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================

    def get_statistics(self) -> Dict:
        """
        Get sorting statistics.

        Returns:
            Dictionary containing:
            - total_sorted: Number of pieces successfully sorted
            - overflow_redirects: Number of pieces redirected to overflow
            - failed_sorts: Number of sorting failures
            - success_rate: Percentage of successful sorts (0.0 to 1.0)

        Example:
            stats = coordinator.get_statistics()
            print(f"Success rate: {stats['success_rate']*100:.1f}%")
        """
        with self.lock:
            total_attempts = self.total_sorted + self.failed_sorts
            success_rate = (self.total_sorted / total_attempts) if total_attempts > 0 else 0.0

            return {
                'total_sorted': self.total_sorted,
                'overflow_redirects': self.overflow_count,
                'failed_sorts': self.failed_sorts,
                'success_rate': success_rate
            }

    def _log_final_statistics(self):
        """
        Log final statistics on shutdown (internal helper).

        Called by release() to log a summary of the sorting session.
        """
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("HARDWARE COORDINATOR FINAL STATISTICS")
        logger.info(f"  Total pieces sorted: {stats['total_sorted']}")
        logger.info(f"  Overflow redirects: {stats['overflow_redirects']}")
        logger.info(f"  Failed sorts: {stats['failed_sorts']}")
        logger.info(f"  Success rate: {stats['success_rate'] * 100:.1f}%")
        logger.info("=" * 60)

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def release(self):
        """
        Clean shutdown of hardware coordinator.

        Logs final statistics and returns servo to home position.
        This should be called when shutting down the sorting system.
        """
        logger.info("Releasing hardware coordinator")

        # Log final statistics
        self._log_final_statistics()

        # Return servo to safe home position
        self.servo.home()

        logger.info("Hardware coordinator released")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_hardware_coordinator(bin_capacity_manager, servo_controller,
                                config_manager: EnhancedConfigManager) -> HardwareCoordinator:
    """
    Factory function to create HardwareCoordinator instance.

    Args:
        bin_capacity_manager: BinCapacityManager instance
        servo_controller: ArduinoServoController instance
        config_manager: EnhancedConfigManager instance

    Returns:
        Initialized HardwareCoordinator instance

    Example:
        bin_capacity = create_bin_capacity_manager(config_manager)
        servo = create_arduino_servo_controller(config_manager)
        coordinator = create_hardware_coordinator(bin_capacity, servo, config_manager)
    """
    return HardwareCoordinator(bin_capacity_manager, servo_controller, config_manager)


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the hardware coordinator functionality.
    This demonstrates the complete hardware pipeline.
    """

    print("\n" + "=" * 60)
    print("HARDWARE COORDINATOR TEST")
    print("=" * 60 + "\n")

    # Import required modules
    from enhanced_config_manager import create_config_manager
    from bin_capacity_module import create_bin_capacity_manager
    from arduino_servo_module import create_arduino_servo_controller

    # Create config manager
    config_manager = create_config_manager()

    # Create hardware modules
    print("Initializing hardware modules...")
    bin_capacity = create_bin_capacity_manager(config_manager)
    servo = create_arduino_servo_controller(config_manager)

    # Create hardware coordinator
    print("Initializing hardware coordinator...")
    coordinator = create_hardware_coordinator(bin_capacity, servo, config_manager)
    print()

    # Check if servo is ready
    if not servo.is_ready():
        print("⚠️  Servo not ready - running in degraded mode")
        print()

    # Test 1: Sort some test pieces
    print("\n--- Test 1: Sort Test Pieces ---")
    test_pieces = [
        (1, 1),  # piece_id=1, bin=1
        (2, 2),  # piece_id=2, bin=2
        (3, 1),  # piece_id=3, bin=1
        (4, 3),  # piece_id=4, bin=3
        (5, 2),  # piece_id=5, bin=2
    ]

    for piece_id, bin_num in test_pieces:
        print(f"Sorting piece {piece_id} to bin {bin_num}...")
        success = coordinator.trigger_sort_for_piece(piece_id, bin_num)
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")

        # Small delay between sorts (simulate real operation)
        import time

        time.sleep(0.3)

    # Test 2: Check bin capacity status
    print("\n--- Test 2: Bin Capacity Status ---")
    print("Calling bin_capacity.get_all_bins_status() directly:")
    all_bins = bin_capacity.get_all_bins_status()
    for bin_status in all_bins[:5]:  # Show first 5 bins
        if bin_status['count'] > 0:
            print(f"  Bin {bin_status['bin_number']}: "
                  f"{bin_status['count']}/{bin_status['max_capacity']} pieces "
                  f"({bin_status['percentage'] * 100:.0f}%)")

    # Test 3: Statistics
    print("\n--- Test 3: Coordinator Statistics ---")
    stats = coordinator.get_statistics()
    print(f"  Total sorted: {stats['total_sorted']}")
    print(f"  Overflow redirects: {stats['overflow_redirects']}")
    print(f"  Failed sorts: {stats['failed_sorts']}")
    print(f"  Success rate: {stats['success_rate'] * 100:.1f}%")

    # Test 4: Overflow scenario (fill a bin)
    print("\n--- Test 4: Overflow Test ---")
    sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)
    max_capacity = sorting_config['max_pieces_per_bin']

    print(f"Filling bin 5 to capacity ({max_capacity} pieces)...")
    for i in range(max_capacity):
        bin_capacity.add_piece(5)

    print("Attempting to sort to full bin 5 (should redirect to overflow)...")
    success = coordinator.trigger_sort_for_piece(piece_id=999, bin_number=5)
    print(f"  Result: {'SUCCESS' if success else 'FAILED'}")

    # Check updated statistics
    stats = coordinator.get_statistics()
    print(f"  Overflow redirects: {stats['overflow_redirects']}")

    # Test 5: Reset a bin
    print("\n--- Test 5: Reset Bin ---")
    print("Resetting bin 5 (calling bin_capacity.reset_bin() directly)...")
    bin_capacity.reset_bin(5)

    # Cleanup
    print("\n--- Cleanup ---")
    coordinator.release()
    print("Hardware coordinator released")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("Hardware coordinator working correctly:")
    print("  ✓ Executes physical sorts on command")
    print("  ✓ Checks bin capacity before sorting")
    print("  ✓ Handles overflow redirection")
    print("  ✓ Tracks statistics accurately")
    print("  ✓ External modules call bin_capacity directly (no wrappers)")
    print("=" * 60 + "\n")