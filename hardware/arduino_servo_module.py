# hardware/arduino_servo_module.py
"""
arduino_servo_module.py - Servo control for bin selection

This module controls the servo motor that directs pieces into different bins.
It uses an ArduinoConnection (shared serial transport) to send commands.

COMMUNICATION MODEL:
- Write-only: Commands are sent to Arduino via the shared ArduinoConnection.
- Command: "A,{angle}\\n"  (e.g. "A,90\\n" for 90 degrees)

RESPONSIBILITIES:
- Bin-to-angle position mapping and validation
- Servo movement commands (move to bin, move to angle, home)
- Auto-calculation of evenly-spaced bin positions
- Position testing and calibration support

DOES NOT:
- Own the serial connection (that's arduino_connection.py)
- Track bin capacity (that's bin_capacity_module)
- Make sorting decisions (that's processing modules)
- Coordinate with other hardware (that's hardware_coordinator)

CONFIGURATION:
All settings stored in 'arduino_servo' section of config.json
"""

import time
import logging
import threading
from typing import Dict, Optional
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig
from hardware.arduino_connection import ArduinoConnection

logger = logging.getLogger(__name__)


class ArduinoServoController:
    """
    Controls the chute servo motor for directing Lego pieces to bins.

    Uses a shared ArduinoConnection for serial communication so the port
    is not opened twice when the motor controller is also active.
    """

    def __init__(self, config_manager: EnhancedConfigManager, connection: ArduinoConnection):
        """
        Initialize servo controller with configuration.

        Args:
            config_manager: Enhanced configuration manager instance
            connection:     Shared ArduinoConnection (serial transport)
        """
        if config_manager is None:
            raise ValueError("config_manager cannot be None")

        self.config_manager = config_manager
        self.connection = connection

        # Load configuration
        arduino_config = config_manager.get_module_config(ModuleConfig.ARDUINO_SERVO.value)
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)

        # =====================================================================
        # SERVO HARDWARE PARAMETERS
        # =====================================================================
        self.min_angle = 0
        self.max_angle = 180
        self.default_position = arduino_config['default_position']
        self.min_bin_separation = arduino_config['min_bin_separation']

        # =====================================================================
        # BIN CONFIGURATION
        # =====================================================================
        self.max_bins = sorting_config['max_bins']
        self.overflow_bin = 0  # Bin 0 always reserved for overflow

        # =====================================================================
        # BIN POSITION MAPPING
        # =====================================================================
        self.bin_positions = {}

        # Load positions from config or auto-calculate defaults
        saved_positions = arduino_config.get('bin_positions', {})
        if saved_positions:
            try:
                logger.info("Loading bin positions from config")
                self.bin_positions = self._load_and_validate_positions(saved_positions)
                logger.info("✓ Bin positions loaded successfully")
            except ValueError as e:
                logger.error(f"Invalid positions in config: {e}")
                logger.info("Auto-calculating valid defaults")
                self.bin_positions = self.auto_calculate_bin_positions(self.max_bins)
                self._save_default_positions_to_config()
        else:
            logger.info("No bin positions found - creating defaults (first run)")
            self.bin_positions = self.auto_calculate_bin_positions(self.max_bins)
            self._save_default_positions_to_config()

        # =====================================================================
        # STATE TRACKING
        # =====================================================================
        self.current_angle = self.default_position
        self.current_bin = None
        self.lock = threading.RLock()  # Thread-safe operations

        # Log initialization status
        self._log_initialization_status()

    # ========================================================================
    # SECTION 1: COMMAND SENDING (delegates to shared ArduinoConnection)
    # ========================================================================

    def _send_command(self, angle: int) -> bool:
        """
        Send servo movement command to Arduino via the shared connection.

        Args:
            angle: Target angle (already validated by caller)

        Returns:
            True if command sent successfully
        """
        return self.connection.send_command('A', angle)

    # ========================================================================
    # SECTION 2: SERVO MOVEMENT CONTROL
    # ========================================================================

    def move_to_angle(self, angle: float, wait: bool = True) -> bool:
        """
        Move servo to specific angle.

        Args:
            angle: Target angle (0-180 degrees)
            wait: If True, block until movement completes

        Returns:
            True if command sent successfully
        """
        with self.lock:
            # Validate angle
            if angle < self.min_angle or angle > self.max_angle:
                logger.error(f"Angle {angle} out of range [{self.min_angle}-{self.max_angle}]")
                return False

            # Check connection
            if not self.connection.is_connected():
                logger.error("Arduino not connected - cannot move servo")
                return False

            try:
                # Send move command
                angle_int = int(angle)
                success = self._send_command(angle_int)

                if not success:
                    logger.error(f"Failed to send move command to {angle}°")
                    return False

                # Wait for movement to complete if requested
                if wait:
                    angle_difference = abs(angle - self.current_angle)
                    # Assume servo speed of 60°/second + small buffer
                    movement_time = (angle_difference / 60.0) + 0.1
                    time.sleep(movement_time)

                # Update current position
                self.current_angle = angle
                logger.debug(f"Servo at {angle}°")
                return True

            except Exception as e:
                logger.error(f"Movement to {angle}° failed: {e}")
                return False

    def move_to_bin(self, bin_number: int, wait: bool = True) -> bool:
        """
        Move servo to position for specified bin.

        Args:
            bin_number: Target bin (0 to max_bins)
            wait: If True, block until movement completes

        Returns:
            True if movement successful
        """
        with self.lock:
            # Validate bin number
            if bin_number not in self.bin_positions:
                logger.error(f"Bin {bin_number} not configured")
                return False

            # Get target angle for this bin
            target_angle = self.bin_positions[bin_number]

            # Execute movement
            success = self.move_to_angle(target_angle, wait)

            if success:
                self.current_bin = bin_number
                logger.info(f"Moved to bin {bin_number} (angle {target_angle}°)")
            else:
                logger.error(f"Failed to move to bin {bin_number}")

            return success

    def home(self) -> bool:
        """
        Return servo to home/default position.

        Used for:
        - Startup: Initialize to known safe position
        - Shutdown: Return to neutral position
        - Testing: Return to reference point
        - Manual reset: User-initiated reset via GUI

        Returns:
            True if movement successful
        """
        logger.info("Returning to home position")
        success = self.move_to_angle(self.default_position, wait=True)
        if success:
            self.current_bin = None
        return success

    # ========================================================================
    # SECTION 3: BIN POSITION MANAGEMENT
    # ========================================================================

    def auto_calculate_bin_positions(self, total_bins: int) -> Dict[int, float]:
        """
        Calculate evenly-spaced positions for ALL bins across servo range.

        Args:
            total_bins: TOTAL number of bins INCLUDING overflow bin 0

        Returns:
            Dictionary mapping bin numbers to angles
        """
        logger.info(f"Auto-calculating positions for {total_bins} TOTAL bins")

        # Define usable range (leave margins at edges)
        edge_margin = 10  # degrees
        usable_min = self.min_angle + edge_margin
        usable_max = self.max_angle - edge_margin
        usable_range = usable_max - usable_min

        positions = {}

        if total_bins == 0:
            logger.error("Cannot calculate positions for 0 bins!")
            raise ValueError("total_bins must be at least 1")
        elif total_bins == 1:
            # Only overflow bin - place at center
            positions[0] = usable_min + (usable_range / 2.0)
        else:
            # Distribute ALL bins evenly across the range
            spacing = usable_range / (total_bins - 1)

            for i in range(total_bins):
                bin_number = i
                angle = usable_min + (i * spacing)
                positions[bin_number] = round(angle, 1)

        logger.info(f"Auto-calculated {len(positions)} total positions: {positions}")
        return positions

    def reload_positions_from_config(self) -> bool:
        """
        Reload bin positions from config file.

        Called by config GUI after updating positions.

        Returns:
            True if positions reloaded successfully
        """
        with self.lock:
            try:
                arduino_config = self.config_manager.get_module_config(
                    ModuleConfig.ARDUINO_SERVO.value
                )

                saved_positions = arduino_config.get('bin_positions', {})

                if not saved_positions:
                    logger.error("No bin positions found in config")
                    return False

                validated = self._load_and_validate_positions(saved_positions)

                # Update in-memory positions
                old_positions = self.bin_positions.copy()
                self.bin_positions = validated

                logger.info("✓ Bin positions reloaded from config")
                logger.debug(f"Old: {old_positions}")
                logger.debug(f"New: {self.bin_positions}")
                return True

            except ValueError as e:
                logger.error(f"Failed to reload positions: {e}")
                return False

    def get_bin_positions(self) -> Dict[int, float]:
        """Get copy of current bin position mappings."""
        with self.lock:
            return self.bin_positions.copy()

    def _load_and_validate_positions(self, saved_positions: Dict) -> Dict[int, float]:
        """
        Load and validate bin positions from config data.

        Validates:
        - All angles within valid range [0-180°]
        - All expected bins have positions
        - Values are numeric

        Args:
            saved_positions: Raw positions from config

        Returns:
            Validated positions dictionary

        Raises:
            ValueError: If positions are invalid
        """
        validated = {}

        # Convert keys to integers and values to floats
        for bin_num, angle in saved_positions.items():
            try:
                bin_int = int(bin_num)
                angle_float = float(angle)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid bin position format: {bin_num}={angle}: {e}")

            # Check angle is in valid range
            if angle_float < self.min_angle or angle_float > self.max_angle:
                raise ValueError(
                    f"Bin {bin_int} angle {angle_float}° out of range "
                    f"[{self.min_angle}-{self.max_angle}]"
                )

            validated[bin_int] = angle_float

        # Check all bins have positions (0 through max_bins)
        expected_bins = set(range(self.max_bins + 1))
        provided_bins = set(validated.keys())
        missing_bins = expected_bins - provided_bins

        if missing_bins:
            raise ValueError(f"Missing positions for bins: {sorted(missing_bins)}")

        # Check separation between adjacent bins (warning only)
        sorted_positions = sorted(validated.items(), key=lambda x: x[1])
        for i in range(len(sorted_positions) - 1):
            bin1, angle1 = sorted_positions[i]
            bin2, angle2 = sorted_positions[i + 1]
            separation = abs(angle2 - angle1)

            if separation < self.min_bin_separation:
                logger.warning(
                    f"Bins {bin1} and {bin2} are close: {separation:.1f}° "
                    f"(recommended minimum: {self.min_bin_separation}°)"
                )

        return validated

    def _save_default_positions_to_config(self):
        """Save bootstrap default positions to config."""
        logger.info("Saving bootstrap default positions to config")
        arduino_config = self.config_manager.get_module_config(
            ModuleConfig.ARDUINO_SERVO.value
        )
        arduino_config['bin_positions'] = self.bin_positions
        self.config_manager.update_module_config(
            ModuleConfig.ARDUINO_SERVO.value,
            arduino_config
        )
        self.config_manager.save_config()
        logger.info("✓ Default positions saved (first run bootstrap)")

    # ========================================================================
    # SECTION 4: TESTING AND CALIBRATION
    # ========================================================================

    def test_all_positions(self, dwell_time: float = 1.0) -> bool:
        """
        Test all bin positions by moving through them sequentially.

        User watches servo physically move to verify positions are correct.

        Args:
            dwell_time: Seconds to pause at each position

        Returns:
            True if all commands sent successfully
        """
        logger.info("=" * 60)
        logger.info("TESTING ALL BIN POSITIONS")
        logger.info("=" * 60)

        sorted_bins = sorted(self.bin_positions.items())
        all_success = True

        for bin_number, angle in sorted_bins:
            logger.info(f"Testing Bin {bin_number} -> {angle}°")
            print(f"Moving to Bin {bin_number} ({angle}°)...")

            success = self.move_to_bin(bin_number, wait=True)

            if not success:
                logger.error(f"✗ Failed to send command for bin {bin_number}")
                print(f"✗ FAILED: Bin {bin_number}")
                all_success = False
            else:
                logger.info(f"✓ Command sent for bin {bin_number}")
                print(f"✓ Bin {bin_number} command sent")

            # Pause at position
            time.sleep(dwell_time)

        # Return to home
        logger.info("Test complete - returning to home position")
        print("\nTest complete - returning home...")
        self.home()

        logger.info("=" * 60)
        if all_success:
            logger.info("✓ ALL POSITION COMMANDS SENT SUCCESSFULLY")
            print("✓ All bin position commands sent successfully!")
        else:
            logger.warning("✗ SOME COMMANDS FAILED")
            print("✗ Some commands failed - check logs")
        logger.info("=" * 60)

        return all_success

    def test_single_bin(self, bin_number: int) -> bool:
        """
        Test a single bin position.

        Moves the servo to the desired bin position immediately without
        hold delay or automatic return to home. Use home() to return to
        neutral position when desired.

        Args:
            bin_number: Bin to test

        Returns:
            True if command sent successfully
        """
        logger.info(f"Testing bin {bin_number}")
        print(f"\nTesting Bin {bin_number}...")

        success = self.move_to_bin(bin_number, wait=True)

        if success:
            angle = self.bin_positions[bin_number]
            logger.info(f"✓ Moved to bin {bin_number} ({angle}°)")
            print(f"✓ Bin {bin_number} moved to {angle}°")
            return True
        else:
            logger.error(f"✗ Failed to send command for bin {bin_number}")
            print(f"✗ Failed to send command for bin {bin_number}")
            return False

    # ========================================================================
    # SECTION 5: STATUS AND INFORMATION
    # ========================================================================

    def get_current_position(self) -> Dict:
        """Get current servo state information."""
        with self.lock:
            conn = self.connection.get_status()
            return {
                'angle': self.current_angle,
                'bin': self.current_bin,
                'initialized': self.connection.is_connected(),
                'simulation': self.connection.is_simulation()
            }

    def is_ready(self) -> bool:
        """True if the shared Arduino connection is up (or in simulation)."""
        return self.connection.is_connected()

    def get_statistics(self) -> Dict:
        """Get comprehensive servo statistics and configuration."""
        conn = self.connection.get_status()
        return {
            'initialized': self.connection.is_connected(),
            'simulation_mode': self.connection.is_simulation(),
            'connection': conn,
            'servo': {
                'current_angle': self.current_angle,
                'current_bin': self.current_bin,
                'default_position': self.default_position,
                'angle_range': (self.min_angle, self.max_angle)
            },
            'bins': {
                'max_bins': self.max_bins,
                'overflow_bin': self.overflow_bin,
                'positions': self.bin_positions.copy()
            }
        }

    def print_status(self):
        """Print formatted status report."""
        print("\n" + "=" * 60)
        print("ARDUINO SERVO CONTROLLER STATUS")
        print("=" * 60)

        stats = self.get_statistics()

        mode = "SIMULATION" if stats['simulation_mode'] else "HARDWARE (WRITE-ONLY)"
        print(f"\nMode: {mode}")
        print(f"Ready: {'YES' if self.is_ready() else 'NO'}")

        if not stats['simulation_mode']:
            print(f"\nConnection:")
            print(f"  Port: {stats['connection']['port']}")
            print(f"  Baud Rate: {stats['connection']['baud_rate']}")
            print(f"  Connected: {stats['connection']['connected']}")

        print(f"\nCurrent Position:")
        print(f"  Angle: {stats['servo']['current_angle']}°")
        print(f"  Bin: {stats['servo']['current_bin'] or 'None (home position)'}")

        print(f"\nBin Configuration:")
        print(f"  Max Bins: {stats['bins']['max_bins']}")
        print(f"  Overflow Bin: {stats['bins']['overflow_bin']}")

        print(f"\nBin Positions:")
        for bin_num in sorted(stats['bins']['positions'].keys()):
            angle = stats['bins']['positions'][bin_num]
            current = " ← CURRENT" if bin_num == self.current_bin else ""
            print(f"  Bin {bin_num}: {angle:5.1f}°{current}")

        print("=" * 60 + "\n")

    def _log_initialization_status(self):
        """Log detailed initialization status."""
        conn = self.connection.get_status()
        logger.info("=" * 60)
        logger.info("ARDUINO SERVO CONTROLLER INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Mode: {'SIMULATION' if conn['simulation_mode'] else 'HARDWARE (WRITE-ONLY)'}")
        logger.info(f"Ready: {self.connection.is_connected()}")

        if not conn['simulation_mode']:
            logger.info(f"Port: {conn['port']}")
            logger.info(f"Baud Rate: {conn['baud_rate']}")

        logger.info(f"Servo Range: {self.min_angle}° - {self.max_angle}°")
        logger.info(f"Default Position: {self.default_position}°")
        logger.info(f"Max Bins: {self.max_bins}")
        logger.info(f"Bin Positions: {self.bin_positions}")
        logger.info("=" * 60)

    # ========================================================================
    # SECTION 6: CLEANUP
    # ========================================================================

    def release(self):
        """
        Clean shutdown of servo controller.

        Returns servo to home position.  The shared ArduinoConnection is
        closed by whoever owns it (typically hardware_coordinator).
        """
        logger.info("Releasing Arduino servo controller")
        self.home()
        logger.info("Arduino servo controller released")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_arduino_servo_controller(
    config_manager: EnhancedConfigManager,
    connection: 'ArduinoConnection'
) -> ArduinoServoController:
    """
    Factory function to create ArduinoServoController instance.

    Args:
        config_manager: Enhanced configuration manager
        connection:     Shared ArduinoConnection instance

    Returns:
        Initialized ArduinoServoController instance
    """
    return ArduinoServoController(config_manager, connection)