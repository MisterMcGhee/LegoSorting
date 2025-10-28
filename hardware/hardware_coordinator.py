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
- Notify registered callbacks when sorting completes (for GUI updates)

DOES NOT:
- Store piece data or "ready" state (that's processing pipeline)
- Make timing decisions about WHEN to sort (that's LegoSorting008)
- Track piece positions or detect pieces (that's detector)
- Identify pieces or assign bins (that's processing)

ARCHITECTURE:
This is a service provider for the main orchestrator. It exposes a simple
API for executing physical sorts and provides statistics. All decision-making
about WHEN and WHAT to sort happens in LegoSorting008.

CALLBACK SYSTEM:
This module implements the Observer pattern to notify external systems
(primarily the GUI) when physical sorting operations complete. External systems
register callback functions during initialization, and these callbacks are
automatically invoked when pieces are sorted into bins.

USAGE:
    coordinator = create_hardware_coordinator(bin_capacity, servo, config_manager)

    # Register GUI callback
    coordinator.register_sort_callback(gui.on_piece_sorted)

    # Main orchestrator positions chute when piece identified:
    success = coordinator.position_chute_for_piece(piece_id=123, bin_number=5)

    # Main orchestrator notifies when piece exits:
    coordinator.notify_piece_exited(piece_id=123)

    # Get statistics:
    stats = coordinator.get_statistics()

    # Shutdown:
    coordinator.release()
"""

import logging
import threading
from typing import Dict, List, Callable
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig
from hardware.chute_state_manager import create_chute_state_manager, ChuteStateManager

logger = logging.getLogger(__name__)


class HardwareCoordinator:
    """
    Executes physical sorting operations by coordinating hardware modules.

    This class provides a simple interface for physical sorting. It receives
    a piece ID and bin number, then executes the physical operation by
    checking capacity, moving the servo, and updating counts.

    Callback Support:
    External systems can register callbacks to be notified when physical
    sorting completes. This is primarily used by the GUI to update bin
    capacity displays in real-time.
    """

    def __init__(self, bin_capacity_manager, servo_controller, config_manager: EnhancedConfigManager):
        """
        Initialize hardware coordinator with hardware module references.

        Args:
            bin_capacity_manager: BinCapacityManager instance for tracking bin fill levels
            servo_controller: ArduinoServoController instance for moving servo
            config_manager: EnhancedConfigManager instance for configuration
        """
        # ============================================================
        # STORE REFERENCES TO HARDWARE MODULES
        # ============================================================
        # These modules do the actual hardware operations
        # We coordinate them but don't replace their functionality
        self.bin_capacity = bin_capacity_manager
        self.servo = servo_controller
        self.config_manager = config_manager

        # Create chute state manager
        self.chute_state_manager = create_chute_state_manager(config_manager)

        # ============================================================
        # LOAD CONFIGURATION
        # ============================================================
        # Get the overflow bin number from config
        # This is where pieces go when target bins are full
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)
        self.overflow_bin = sorting_config['overflow_bin']

        # ============================================================
        # STATISTICS TRACKING
        # ============================================================
        # Keep track of sorting operations for monitoring
        self.total_sorted = 0  # Total pieces successfully sorted
        self.overflow_count = 0  # How many went to overflow (target was full)
        self.failed_sorts = 0  # How many sort operations failed

        # ============================================================
        # THREAD SAFETY
        # ============================================================
        # Use a lock to ensure multiple threads can't trigger sorts
        # at the same time (which could cause servo conflicts)
        self.lock = threading.RLock()

        # ============================================================
        # CALLBACK SYSTEM FOR SORT EVENTS
        # ============================================================
        # This list stores callback functions that want to be notified
        # when pieces are physically sorted. The GUI registers its
        # callback here so it can update bin capacity displays in real-time.
        # Each callback should be a function that accepts a bin_number
        self.sort_callbacks: List[Callable[[int], None]] = []

        logger.info("Hardware coordinator initialized with chute state manager")
        logger.info(f"  Overflow bin: {self.overflow_bin}")

    # ========================================================================
    # CALLBACK REGISTRATION AND NOTIFICATION
    # ========================================================================
    # These methods implement the Observer pattern, allowing external systems
    # (like the GUI) to "subscribe" to physical sorting events.

    def register_sort_callback(self, callback_function: Callable[[int], None]):
        """
        Register a callback function to be notified when sorting occurs.

        PURPOSE:
        This allows external systems (especially the GUI) to be notified
        immediately when a piece is physically sorted into a bin. The
        callback is triggered AFTER the servo moves and bin count updates,
        allowing the GUI to update bin capacity displays in real-time.

        WHEN TO USE:
        Call this during initialization to register any listeners that
        need to know about physical sorting events. The sorting_gui calls
        this during its startup to update bin capacity bars.

        CALLBACK SIGNATURE:
        The registered function must accept one argument:
            callback(bin_number: int)

        WHERE CALLBACKS ARE TRIGGERED:
        Callbacks are triggered in notify_piece_exited() after:
        1. Positioned piece exits the ROI
        2. Bin capacity is updated
        3. Statistics are incremented

        OVERFLOW HANDLING:
        If the target bin is full during pre-positioning, the chute is
        redirected to overflow bin (typically bin 0). The callback will
        receive the overflow bin number, not the original target bin.
        This tells the GUI which bin actually received the piece.

        CALLBACK TRIGGERS:
        Callbacks are only called when a piece successfully enters a bin.
        If chute positioning fails or piece tracking is lost, callbacks
        are NOT called.

        THREAD SAFETY:
        Callbacks are called synchronously on whatever thread calls
        notify_piece_exited() (typically the main orchestrator thread).
        The lock is held during callback execution to ensure thread safety.

        Args:
            callback_function: Function to call when sorting occurs.
                              Must accept (bin_number: int)

        Example:
            # In sorting_gui.py __init__:
            hardware.register_sort_callback(self.on_piece_sorted)

            # Later, when servo moves piece:
            # hardware sorts piece into Bin 3
            # GUI's on_piece_sorted(3) is called automatically
            # GUI updates Bin 3's capacity bar
        """
        # Add the callback function to our list
        self.sort_callbacks.append(callback_function)

        # Log that we registered this callback (helpful for debugging)
        logger.debug(
            f"Registered sort callback: {callback_function.__name__}"
        )

    def _notify_sort_callbacks(self, bin_number: int):
        """
        Notify all registered callbacks that a piece has been sorted.

        PURPOSE:
        This is an internal method that calls all registered callbacks
        when physical sorting completes. It handles errors gracefully so
        one failing callback doesn't break the hardware operation.

        WHEN CALLED:
        This is called in notify_piece_exited() after:
        1. Positioned piece successfully exits the ROI
        2. Bin count is incremented via bin_capacity.add_piece()
        3. Statistics are updated

        SUCCESS ONLY:
        This is only called if the piece successfully enters a bin.
        If positioning fails or tracking is lost, this is NOT called.
        This ensures callbacks only receive notifications for pieces
        that actually made it into bins.

        BIN NUMBER:
        The bin_number passed is the ACTUAL bin the piece went to,
        which may be the overflow bin if the original target was full.
        For example, if piece was supposed to go to bin 3 but that was
        full, the callback receives 0 (overflow bin number).

        ERROR HANDLING:
        If a callback raises an exception, it is caught and logged,
        but other callbacks still execute. This prevents one bad
        callback from breaking the hardware operation or preventing
        other systems from being notified.

        THREAD CONTEXT:
        Callbacks execute while holding self.lock, ensuring thread
        safety with hardware operations. Callbacks should complete
        quickly to avoid blocking hardware operations.

        Args:
            bin_number: The actual bin where the piece was sorted
                       (may be overflow bin if original target was full)

        Example Flow:
            # In notify_piece_exited():
            self.bin_capacity.add_piece(bin_number)  # Update count
            self._notify_sort_callbacks(bin_number)  # Tell everyone!

            # This calls each registered callback:
            # 1. sorting_gui.on_piece_sorted(bin_number)
            # 2. any_other_registered_callback(bin_number)
            # 3. etc.
        """
        # Loop through all registered callback functions
        for callback in self.sort_callbacks:
            try:
                # Call the callback with the actual bin number
                # This is where the GUI gets notified!
                callback(bin_number)

            except Exception as e:
                # If a callback crashes, log the error but keep going
                # This ensures one broken callback doesn't stop other callbacks
                # or break the hardware operation
                logger.error(
                    f"Error in sort callback {callback.__name__}: {e}",
                    exc_info=True
                )

    # ========================================================================
    # CORE SORTING OPERATION
    # ========================================================================
    # This is the main functionality - executing the physical sort operation
    # by checking capacity, moving the servo, and updating counts.

    # ========================================================================
    # CHUTE POSITIONING (Delegates to ChuteStateManager)
    # ========================================================================

    def is_chute_available(self) -> bool:
        """Check if chute is available for positioning."""
        return self.chute_state_manager.is_available()

    def get_chute_status(self) -> dict:
        """Get current chute status."""
        status = self.chute_state_manager.get_status()
        return {
            'state': status.state.value,
            'positioned_piece_id': status.positioned_piece_id,
            'positioned_bin': status.positioned_bin,
            'fall_time_remaining': status.fall_time_remaining,
            'available': status.available
        }

    def position_chute_for_piece(self, piece_id: int, bin_number: int) -> bool:
        """
        Position chute for a specific piece (pre-positioning).

        Args:
            piece_id: ID of piece to position for
            bin_number: Target bin number

        Returns:
            True if positioning successful
        """
        # Check if chute is available
        if not self.chute_state_manager.is_available():
            logger.debug(f"Chute not available for piece {piece_id}")
            return False

        logger.info(f"📍 Positioning chute for piece {piece_id} → bin {bin_number}")

        # Check bin capacity
        if not self.bin_capacity.can_accept_piece(bin_number):
            logger.warning(f"Bin {bin_number} full, redirecting to overflow bin 0")
            bin_number = 0

        # Move servo to bin position
        servo_success = self.servo.move_to_bin(bin_number, wait=False)

        if not servo_success:
            logger.error(f"Failed to move servo to bin {bin_number}")
            return False

        # Update state machine
        state_success = self.chute_state_manager.position_for_piece(piece_id, bin_number)

        if not state_success:
            logger.error(f"State machine rejected positioning for piece {piece_id}")
            return False

        logger.info(f"✓ Chute positioned for piece {piece_id} at bin {bin_number}")
        return True

    def notify_piece_exited(self, piece_id: int) -> bool:
        """
        Notify coordinator that positioned piece has exited ROI.

        Starts the fall timer and updates bin capacity.

        Args:
            piece_id: ID of piece that exited

        Returns:
            True if this was the positioned piece
        """
        success = self.chute_state_manager.notify_piece_exited(piece_id)

        if success:
            # Update bin capacity (piece committed to bin)
            bin_number = self.chute_state_manager.get_positioned_bin()
            if bin_number is not None:
                self.bin_capacity.add_piece(bin_number)
                logger.info(f"✓ Piece {piece_id} committed to bin {bin_number}")

                # Notify GUI of bin update
                self._notify_sort_callbacks(bin_number)

                # Update statistics
                with self.lock:
                    self.total_sorted += 1
                    if bin_number == self.overflow_bin:
                        self.overflow_count += 1

        return success

    def update_chute_state(self) -> None:
        """
        Update chute state machine (call every frame).

        Checks if fall timer has expired and transitions state.
        """
        self.chute_state_manager.update()

    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================
    # These methods provide information about sorting performance and status.

    def get_statistics(self) -> Dict:
        """
        Get sorting statistics.

        WHAT IT RETURNS:
        - total_sorted: Number of pieces successfully sorted
        - overflow_redirects: Number of pieces redirected to overflow
        - failed_sorts: Number of sorting failures
        - success_rate: Percentage of successful sorts (0.0 to 1.0)

        THREAD SAFETY:
        Uses the lock to ensure we get consistent statistics even if
        sorting is happening on another thread.

        Returns:
            Dictionary containing sorting statistics

        Example:
            stats = coordinator.get_statistics()
            print(f"Sorted: {stats['total_sorted']}")
            print(f"Success rate: {stats['success_rate']*100:.1f}%")
        """
        # Use lock to ensure thread-safe access to statistics
        with self.lock:
            # Calculate total attempts (successful + failed)
            total_attempts = self.total_sorted + self.failed_sorts

            # Calculate success rate (avoid division by zero)
            success_rate = (self.total_sorted / total_attempts) if total_attempts > 0 else 0.0

            # Return dictionary with all statistics
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
        This provides a nice summary at the end of operation.
        """
        # Get the current statistics
        stats = self.get_statistics()

        # Log a formatted summary
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
        """Clean shutdown of hardware coordinator."""
        logger.info("Releasing hardware coordinator...")

        # Reset chute state
        self.chute_state_manager.reset()

        # Return servo to home position
        if self.servo:
            self.servo.home()

        logger.info("✓ Hardware coordinator released")

# ============================================================================
# FACTORY FUNCTION
# ============================================================================
# This function creates and returns a HardwareCoordinator instance.
# It's called during application startup.

def create_hardware_coordinator(bin_capacity_manager, servo_controller,
                                config_manager: EnhancedConfigManager) -> HardwareCoordinator:
    """
    Factory function to create HardwareCoordinator instance.

    WHY USE A FACTORY FUNCTION?
    - Provides a consistent way to create instances
    - Can add initialization logic in one place
    - Makes testing easier (can mock the factory)
    - Matches the pattern used by other modules

    Args:
        bin_capacity_manager: BinCapacityManager instance for tracking bin fill
        servo_controller: ArduinoServoController instance for servo control
        config_manager: EnhancedConfigManager instance for configuration

    Returns:
        Initialized HardwareCoordinator instance ready to use

    Example:
        # During application startup in LegoSorting008:
        bin_capacity = create_bin_capacity_manager(config_manager)
        servo = create_arduino_servo_controller(config_manager)

        # Create the hardware coordinator
        hardware_coordinator = create_hardware_coordinator(
            bin_capacity,
            servo,
            config_manager
        )

        # Register GUI callback
        hardware_coordinator.register_sort_callback(gui.on_piece_sorted)

        # In main loop when piece identified:
        success = hardware_coordinator.position_chute_for_piece(
            piece_id=123,
            bin_number=5
        )

        # When piece exits ROI:
        hardware_coordinator.notify_piece_exited(piece_id=123)
    """
    return HardwareCoordinator(bin_capacity_manager, servo_controller, config_manager)