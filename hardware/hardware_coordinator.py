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

    # Main orchestrator calls when piece ready to sort:
    success = coordinator.trigger_sort_for_piece(piece_id=123, bin_number=5)

    # Get statistics:
    stats = coordinator.get_statistics()

    # Shutdown:
    coordinator.release()
"""

import logging
import threading
from typing import Dict, List, Callable
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

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

        logger.info("Hardware coordinator initialized")
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
        Callbacks are triggered in trigger_sort_for_piece() after:
        1. Bin capacity check passes (or overflow redirect)
        2. Servo successfully moves to target position
        3. Bin piece count is incremented

        OVERFLOW HANDLING:
        If the target bin is full and the piece is redirected to overflow,
        the callback receives the overflow bin number (typically 0), not
        the original target bin. This tells the GUI which bin actually
        received the piece.

        SERVO FAILURE:
        If servo movement fails, callbacks are NOT called. This ensures
        callbacks only fire for successful physical sorts where the piece
        actually made it into a bin.

        THREAD SAFETY:
        Callbacks are called synchronously on whatever thread calls
        trigger_sort_for_piece() (typically the main orchestrator thread).
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
        This is called in trigger_sort_for_piece() after:
        1. Capacity check completes (with possible overflow redirect)
        2. Servo successfully moves to target bin position
        3. Bin count is incremented via bin_capacity.add_piece()

        SUCCESS ONLY:
        This is only called if the physical sort succeeds. If servo
        movement fails, this is NOT called. This ensures callbacks
        only receive notifications for pieces that actually made it
        into bins.

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
            # In trigger_sort_for_piece():
            self.servo.move_to_bin(actual_bin)  # Move servo
            self.bin_capacity.add_piece(actual_bin)  # Update count
            self._notify_sort_callbacks(actual_bin)  # Tell everyone!

            # This calls each registered callback:
            # 1. sorting_gui.on_piece_sorted(actual_bin)
            # 2. any_other_registered_callback(actual_bin)
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

    def trigger_sort_for_piece(self, piece_id: int, bin_number: int) -> bool:
        """
        Execute physical sorting for a piece.

        This is called by the main orchestrator when a piece reaches the
        exit zone and is ready to be physically sorted. All decision-making
        has already happened - this function just executes the physical
        operation.

        WHAT ALREADY HAPPENED:
        The caller (LegoSorting008) has already determined:
        - The piece is fully processed (IdentifiedPiece.complete = True)
        - The bin assignment (bin_number from bin_assignment_module)
        - The timing is correct (piece in exit zone, rightmost position)

        WHAT THIS FUNCTION DOES:
        This function executes the physical sorting sequence:
        1. Check if target bin has capacity
        2. If full, redirect to overflow bin
        3. Move servo to target bin position
        4. Update bin piece count
        5. Notify registered callbacks (update GUI)
        6. Update statistics

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
        # ============================================================
        # THREAD SAFETY: ACQUIRE LOCK
        # ============================================================
        # Use a lock to ensure only one sort operation happens at a time
        # This prevents multiple threads from trying to move the servo
        # simultaneously, which would cause mechanical conflicts
        with self.lock:
            logger.info(f"Sorting piece {piece_id} to bin {bin_number}")

            # ============================================================
            # STEP 1: CHECK BIN CAPACITY
            # ============================================================
            # Before we move the servo, check if the target bin can
            # accept another piece. If it's full, we'll redirect to
            # the overflow bin instead.

            # Start with the target bin from processing
            actual_target_bin = bin_number

            # Ask bin_capacity if this bin can accept another piece
            if not self.bin_capacity.can_accept_piece(actual_target_bin):
                # Target bin is full! Redirect to overflow
                logger.warning(
                    f"Bin {actual_target_bin} is full! "
                    f"Redirecting to overflow bin {self.overflow_bin}"
                )

                # Change target to overflow bin
                actual_target_bin = self.overflow_bin

                # Track that we had to use overflow
                self.overflow_count += 1

                # Check if overflow bin is also full (disaster scenario!)
                if not self.bin_capacity.can_accept_piece(self.overflow_bin):
                    logger.error(f"Overflow bin {self.overflow_bin} is also full!")
                    logger.error("Cannot sort piece - all capacity exhausted")

                    # Update failure counter
                    self.failed_sorts += 1

                    # Return False to indicate failure
                    # The piece will fall off the conveyor unsorted
                    return False

            # ============================================================
            # STEP 2: MOVE SERVO TO TARGET BIN
            # ============================================================
            # Now that we know which bin to use (either original target
            # or overflow), move the servo to that position.

            # Use wait=False for non-blocking operation
            # This means the servo starts moving immediately, and we
            # continue without waiting for it to reach the position
            # This is important for real-time operation
            success = self.servo.move_to_bin(actual_target_bin, wait=False)

            # Check if servo movement command succeeded
            if not success:
                # Servo failed to move (maybe disconnected, position invalid, etc.)
                logger.error(f"Failed to move servo to bin {actual_target_bin}")

                # Update failure counter
                self.failed_sorts += 1

                # Return False to indicate failure
                # The piece will fall off the conveyor unsorted
                return False

            # ============================================================
            # STEP 3: UPDATE BIN CAPACITY COUNT
            # ============================================================
            # The servo is now moving (or has moved) to the target bin.
            # Update our tracking of how many pieces are in that bin.

            # Note: add_piece() internally checks capacity and triggers
            # warning/full callbacks if thresholds are reached
            self.bin_capacity.add_piece(actual_target_bin)

            # ============================================================
            # STEP 4: NOTIFY CALLBACKS
            # ============================================================
            # Now that the physical sort is complete (servo moved and
            # count updated), notify all registered callbacks.
            # This is where the GUI gets updated!

            # Pass the ACTUAL bin number (which may be overflow if
            # original target was full)
            self._notify_sort_callbacks(actual_target_bin)

            # ============================================================
            # STEP 5: UPDATE STATISTICS
            # ============================================================
            # Track that we successfully sorted a piece
            self.total_sorted += 1

            # ============================================================
            # STEP 6: LOG SUCCESS
            # ============================================================
            logger.info(f"✓ Piece {piece_id} sorted to bin {actual_target_bin}")

            # Log if this was an overflow redirect
            # (helps with debugging and monitoring)
            if actual_target_bin != bin_number:
                logger.info(f"  (Redirected from full bin {bin_number})")

            # Return True to indicate success
            return True

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
        """
        Clean shutdown of hardware coordinator.

        Logs final statistics and returns servo to home position.
        This should be called when shutting down the sorting system.

        WHAT IT DOES:
        1. Logs final statistics for the session
        2. Returns servo to safe home position
        3. Prepares system for shutdown

        WHEN TO CALL:
        Call this when the sorting session is ending, before
        shutting down the application.

        Example:
            # At end of sorting session:
            coordinator.release()
            # Now safe to exit application
        """
        logger.info("Releasing hardware coordinator")

        # Log final statistics
        # This gives us a summary of the entire sorting session
        self._log_final_statistics()

        # Return servo to safe home position
        # This ensures servo is in a known position for next startup
        self.servo.home()

        logger.info("Hardware coordinator released")


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

        # Use when piece needs sorting
        success = hardware_coordinator.trigger_sort_for_piece(
            piece_id=123,
            bin_number=5
        )
    """
    return HardwareCoordinator(bin_capacity_manager, servo_controller, config_manager)