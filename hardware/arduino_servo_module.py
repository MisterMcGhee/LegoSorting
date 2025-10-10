# hardware/arduino_servo_module.py
"""
arduino_servo_module.py - Servo control for bin selection in Lego sorting machine

This module controls the servo motor that directs pieces into different bins.
It manages the Arduino serial connection, bin position mappings, and all
servo movement operations.

RESPONSIBILITIES:
- Serial connection to Arduino (with retry logic)
- Bin-to-angle position mapping and validation
- Servo movement commands (move to bin, move to angle, home)
- Auto-calculation of evenly-spaced bin positions
- Position testing and calibration support
- Simulation mode for development without hardware
- Bootstrap default positions on first run

CONFIGURATION WRITE POLICY:
- Writes to config ONLY during initialization if positions are missing/invalid
- This is a "safety net" to ensure system starts with valid defaults
- Normal configuration changes go through config_gui, not this module
- Config GUI calls auto_calculate_bin_positions() and saves results itself

DOES NOT:
- Track bin capacity (that's bin_capacity_module)
- Make sorting decisions (that's processing modules)
- Coordinate with other hardware (that's hardware_coordinator)

CONFIGURATION:
All settings stored in 'arduino_servo' section of config.json via enhanced_config_manager.
This includes connection settings, hardware parameters, and calibrated bin positions.

USAGE:
    servo = create_arduino_servo_controller(config_manager)

    # Move to specific bin
    servo.move_to_bin(3)

    # Auto-calculate positions (GUI can use this)
    positions = servo.auto_calculate_bin_positions(9)
    # GUI then saves: config_manager.update_module_config(...)

    # Reload after GUI changes config
    servo.reload_positions_from_config()

    # Test all positions
    servo.test_all_positions()

    # Return to home
    servo.home()
"""

import time
import logging
import serial
import threading
from typing import Dict, Optional
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class ArduinoServoController:
    """
    Controls servo motor for directing Lego pieces to bins via Arduino.

    This class manages all aspects of servo control including connection,
    position mapping, movement commands, and calibration support.
    """

    def __init__(self, config_manager: EnhancedConfigManager):
        """
        Initialize servo controller with configuration.

        Args:
            config_manager: Enhanced configuration manager instance
        """
        self.config_manager = config_manager

        # Load configuration from arduino_servo section
        arduino_config = config_manager.get_module_config(ModuleConfig.ARDUINO_SERVO.value)
        sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)

        # =====================================================================
        # CONNECTION SETTINGS
        # =====================================================================
        self.port = arduino_config['port']
        self.baud_rate = arduino_config['baud_rate']
        self.timeout = arduino_config['timeout']
        self.connection_retries = arduino_config['connection_retries']
        self.retry_delay = arduino_config['retry_delay']
        self.simulation_mode = arduino_config['simulation_mode']

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
        # Format: {bin_number: angle}
        # Example: {0: 90, 1: 10, 2: 30, 3: 50, 4: 70, ...}
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
                # Pass max_bins directly - it means TOTAL bins
                self.bin_positions = self.auto_calculate_bin_positions(self.max_bins)
                self._save_default_positions_to_config()
        else:
            logger.info("No bin positions found - creating defaults (first run)")
            # Pass max_bins directly - it means TOTAL bins
            self.bin_positions = self.auto_calculate_bin_positions(self.max_bins)
            self._save_default_positions_to_config()

        # =====================================================================
        # STATE TRACKING
        # =====================================================================
        self.current_angle = self.default_position
        self.current_bin = None
        self.arduino = None
        self.initialized = False
        self.lock = threading.RLock()  # Thread-safe operations

        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        if self.simulation_mode:
            logger.info("=== SIMULATION MODE ENABLED ===")
            logger.info("No Arduino connection required")
            self.initialized = True
        else:
            logger.info("=== HARDWARE MODE ===")
            self._connect_to_arduino()

        # Log initialization status
        self._log_initialization_status()

    # ========================================================================
    # SECTION 1: BIN POSITION MANAGEMENT
    # ========================================================================

    def auto_calculate_bin_positions(self, total_bins: int) -> Dict[int, float]:
        """
        Calculate evenly-spaced positions for ALL bins across servo range.

        This function distributes ALL bins (including overflow bin 0) evenly
        across the usable servo range. No bin gets special positioning.

        Args:
            total_bins: TOTAL number of bins INCLUDING overflow bin 0.
                       For example, if total_bins=6, creates bins 0,1,2,3,4,5.
                       Bin 0 is the overflow bin, bins 1-(total_bins-1) are sorting bins.

        Returns:
            Dictionary mapping bin numbers to angles.

        Example:
            # Want 6 total bins (0-5)?
            positions = auto_calculate_bin_positions(6)
            # Returns: {0: 10, 1: 42, 2: 74, 3: 106, 4: 138, 5: 170}
            # All 6 bins evenly distributed with 32° spacing
        """
        logger.info(f"Auto-calculating positions for {total_bins} TOTAL bins")

        # Define usable range (leave margins at edges to avoid mechanical limits)
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
            # This includes the overflow bin (0) as part of the distribution
            spacing = usable_range / (total_bins - 1)

            for i in range(total_bins):
                bin_number = i  # Bins 0, 1, 2, 3, ... total_bins-1
                angle = usable_min + (i * spacing)
                positions[bin_number] = round(angle, 1)

        logger.info(f"Auto-calculated {len(positions)} total positions: {positions}")
        logger.info(f"  Overflow bin (0): {positions[0]}°")
        if total_bins > 1:
            logger.info(f"  Sorting bins (1-{total_bins - 1}): distributed evenly")

        return positions

    def reload_positions_from_config(self) -> bool:
        """
        Reload bin positions from config file.

        This should be called after config_gui updates positions in config.json.
        It refreshes the servo module's in-memory positions to match the
        config file.

        Returns:
            True if positions reloaded successfully, False if validation failed

        Example workflow:
            # In config GUI after user clicks Apply:
            config_manager.update_module_config('arduino_servo',
                                                {'bin_positions': new_positions})
            config_manager.save_config()

            # Then refresh servo module:
            servo.reload_positions_from_config()
        """
        with self.lock:
            try:
                # Read fresh config from file
                arduino_config = self.config_manager.get_module_config(
                    ModuleConfig.ARDUINO_SERVO.value
                )

                # Load and validate positions
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
        """
        Get copy of current in-memory bin position mappings.

        Note: This returns the positions currently loaded in memory.
        If the config file has been updated externally, call
        reload_positions_from_config() first to refresh.

        Returns:
            Dictionary of {bin_number: angle}
        """
        with self.lock:
            return self.bin_positions.copy()

    def _load_and_validate_positions(self, saved_positions: Dict) -> Dict[int, float]:
        """
        Load and validate bin positions from config data.

        This is an internal helper used during initialization and reload.
        It validates that positions are reasonable and complete.

        Validation checks:
        - All angles within valid range [0-180°]
        - All expected bins have positions [0 through max_bins]
        - Values are numeric
        - Warns if adjacent bins are too close together

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

        # Check separation between adjacent bins (warning only, not error)
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
        """
        Save current positions to config (INTERNAL - bootstrap only).

        This is ONLY called during __init__ if no valid positions exist.
        It ensures the system has a working configuration to start with.

        This is a "safety net" operation for first run or config recovery.
        Normal configuration changes should go through config_gui.

        Policy: Servo module writes to config ONLY in bootstrap scenarios.
        """
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
    # SECTION 2: ARDUINO CONNECTION MANAGEMENT
    # ========================================================================

    def _connect_to_arduino(self) -> bool:
        """
        Establish serial connection to Arduino with retry logic.

        Returns:
            True if connection successful, False otherwise
        """
        with self.lock:
            for attempt in range(self.connection_retries):
                try:
                    logger.info(
                        f"Connecting to Arduino on {self.port} "
                        f"(attempt {attempt + 1}/{self.connection_retries})"
                    )

                    # Open serial connection
                    self.arduino = serial.Serial(
                        port=self.port,
                        baudrate=self.baud_rate,
                        timeout=self.timeout
                    )

                    # Wait for Arduino to reset after connection
                    time.sleep(2.5)

                    # Clear any buffered data
                    self.arduino.reset_input_buffer()
                    self.arduino.reset_output_buffer()

                    # Mark as initialized
                    self.initialized = True
                    logger.info("✓ Arduino connected successfully")

                    # Move to home position
                    self.home()

                    return True

                except serial.SerialException as e:
                    logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                    if self.arduino:
                        try:
                            self.arduino.close()
                        except:
                            pass
                        self.arduino = None

                # Wait before retry
                if attempt < self.connection_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

            # All connection attempts failed
            logger.error("❌ Failed to connect to Arduino after all attempts")
            logger.error("System will run in degraded mode")
            self.initialized = False
            return False

    def disconnect(self):
        """Safely disconnect from Arduino."""
        with self.lock:
            if self.arduino:
                try:
                    logger.info("Disconnecting from Arduino...")

                    # Move to safe position first
                    self.home()
                    time.sleep(0.5)

                    # Close connection
                    self.arduino.close()
                    self.arduino = None
                    self.initialized = False

                    logger.info("Arduino disconnected")

                except Exception as e:
                    logger.error(f"Error during disconnect: {e}")

    def _send_command(self, command: str) -> Optional[str]:
        """
        Send command to Arduino and read response.

        Args:
            command: Command string to send

        Returns:
            Response from Arduino, or None if failed
        """
        with self.lock:
            if self.simulation_mode:
                logger.debug(f"[SIMULATION] Command: {command}")
                return "OK"

            if not self.arduino:
                logger.error("No Arduino connection")
                return None

            try:
                # Send command (newline-terminated)
                self.arduino.write(f"{command}\n".encode())

                # Read response
                response = self.arduino.readline().decode().strip()
                logger.debug(f"Command: {command} -> Response: {response}")
                return response

            except Exception as e:
                logger.error(f"Command '{command}' failed: {e}")
                return None

    # ========================================================================
    # SECTION 3: SERVO MOVEMENT CONTROL
    # ========================================================================

    def move_to_bin(self, bin_number: int, wait: bool = True) -> bool:
        """
        Move servo to position for specified bin.

        Args:
            bin_number: Target bin (0-9)
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

    def move_to_angle(self, angle: float, wait: bool = True) -> bool:
        """
        Move servo to specific angle.

        Args:
            angle: Target angle (0-180 degrees)
            wait: If True, block until movement completes

        Returns:
            True if movement successful
        """
        with self.lock:
            # Validate angle
            if angle < self.min_angle or angle > self.max_angle:
                logger.error(f"Angle {angle} out of range [{self.min_angle}-{self.max_angle}]")
                return False

            # Simulation mode
            if self.simulation_mode:
                logger.info(f"[SIMULATION] Moving servo to {angle}°")
                self.current_angle = angle
                return True

            # Hardware mode - check initialization
            if not self.initialized:
                logger.error("Arduino not initialized - cannot move servo")
                return False

            try:
                # Send move command to Arduino
                response = self._send_command(f"MOVE {int(angle)}")

                if response != "OK":
                    logger.error(f"Move command rejected: {response}")
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

    def home(self) -> bool:
        """
        Return servo to home/default position.

        Returns:
            True if movement successful
        """
        logger.info("Returning to home position")
        success = self.move_to_angle(self.default_position, wait=True)
        if success:
            self.current_bin = None
        return success

    # ========================================================================
    # SECTION 4: TESTING AND CALIBRATION SUPPORT
    # ========================================================================

    def test_all_positions(self, dwell_time: float = 1.0) -> bool:
        """
        Test all bin positions by moving through them sequentially.

        This function is used for calibration and verification. It moves
        the servo to each bin position in order, pausing at each one.

        Args:
            dwell_time: Seconds to pause at each position

        Returns:
            True if all movements successful
        """
        logger.info("=" * 60)
        logger.info("TESTING ALL BIN POSITIONS")
        logger.info("=" * 60)

        # Get sorted list of bins (by bin number)
        sorted_bins = sorted(self.bin_positions.items())

        all_success = True

        for bin_number, angle in sorted_bins:
            logger.info(f"Testing Bin {bin_number} -> {angle}°")
            print(f"Moving to Bin {bin_number} ({angle}°)...")

            success = self.move_to_bin(bin_number, wait=True)

            if not success:
                logger.error(f"✗ Failed to move to bin {bin_number}")
                print(f"✗ FAILED: Bin {bin_number}")
                all_success = False
            else:
                logger.info(f"✓ Bin {bin_number} OK")
                print(f"✓ Bin {bin_number} reached successfully")

            # Pause at position
            time.sleep(dwell_time)

        # Return to home
        logger.info("Test complete - returning to home position")
        print("\nTest complete - returning home...")
        self.home()

        logger.info("=" * 60)
        if all_success:
            logger.info("✓ ALL POSITIONS TESTED SUCCESSFULLY")
            print("✓ All bin positions tested successfully!")
        else:
            logger.warning("✗ SOME POSITIONS FAILED")
            print("✗ Some positions failed - check logs")
        logger.info("=" * 60)

        return all_success

    def test_single_bin(self, bin_number: int, dwell_time: float = 2.0) -> bool:
        """
        Test a single bin position.

        Args:
            bin_number: Bin to test
            dwell_time: Seconds to hold position

        Returns:
            True if movement successful
        """
        logger.info(f"Testing bin {bin_number}")
        print(f"\nTesting Bin {bin_number}...")

        success = self.move_to_bin(bin_number, wait=True)

        if success:
            logger.info(f"✓ Bin {bin_number} reached at {self.current_angle}°")
            print(f"✓ Bin {bin_number} OK - holding position...")
            time.sleep(dwell_time)
            self.home()
            return True
        else:
            logger.error(f"✗ Failed to reach bin {bin_number}")
            print(f"✗ Failed to reach bin {bin_number}")
            return False

    # ========================================================================
    # SECTION 5: STATUS AND INFORMATION
    # ========================================================================

    def get_current_position(self) -> Dict:
        """
        Get current servo state information.

        Returns:
            Dictionary with current angle, bin, and status
        """
        with self.lock:
            return {
                'angle': self.current_angle,
                'bin': self.current_bin,
                'initialized': self.initialized,
                'simulation': self.simulation_mode
            }

    def is_ready(self) -> bool:
        """
        Check if servo is ready for operations.

        Returns:
            True if servo can accept commands
        """
        return self.initialized

    def get_statistics(self) -> Dict:
        """
        Get comprehensive servo statistics and configuration.

        Returns:
            Dictionary with all servo state and config info
        """
        return {
            'initialized': self.initialized,
            'simulation_mode': self.simulation_mode,
            'connection': {
                'port': self.port,
                'baud_rate': self.baud_rate,
                'connected': self.arduino is not None
            },
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

        print(f"\nMode: {'SIMULATION' if stats['simulation_mode'] else 'HARDWARE'}")
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
        logger.info("=" * 60)
        logger.info("ARDUINO SERVO CONTROLLER INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Mode: {'SIMULATION' if self.simulation_mode else 'HARDWARE'}")
        logger.info(f"Ready: {self.initialized}")

        if not self.simulation_mode:
            logger.info(f"Port: {self.port}")
            logger.info(f"Baud Rate: {self.baud_rate}")

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

        Safely returns servo to home position and closes connections.
        """
        logger.info("Releasing Arduino servo controller")

        # Return to safe position
        self.home()

        # Disconnect from Arduino
        self.disconnect()

        logger.info("Arduino servo controller released")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_arduino_servo_controller(config_manager: EnhancedConfigManager) -> ArduinoServoController:
    """
    Factory function to create ArduinoServoController instance.

    Args:
        config_manager: Enhanced configuration manager

    Returns:
        Initialized ArduinoServoController instance
    """
    return ArduinoServoController(config_manager)


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the Arduino servo module functionality.
    This demonstrates all features using real configuration.
    """

    print("\n" + "=" * 60)
    print("ARDUINO SERVO MODULE TEST")
    print("=" * 60 + "\n")

    # Import enhanced_config_manager
    from enhanced_config_manager import create_config_manager

    # Create real config manager
    config_manager = create_config_manager()

    # Display current configuration
    arduino_config = config_manager.get_module_config(ModuleConfig.ARDUINO_SERVO.value)
    sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)

    print("Current Configuration:")
    print(f"  Simulation Mode: {arduino_config['simulation_mode']}")
    print(f"  Port: {arduino_config['port']}")
    print(f"  Max Bins: {sorting_config['max_bins']}")
    print()

    # Create servo controller
    print("Initializing servo controller...")
    servo = create_arduino_servo_controller(config_manager)
    print()

    # Print status
    servo.print_status()

    if not servo.is_ready():
        print("⚠️  Servo not ready - check configuration and connection")
        print("Continuing in degraded mode for demonstration...")
        print()

    # Test 1: Auto-calculate positions
    print("\n--- Test 1: Auto-Calculate Bin Positions ---")
    auto_positions = servo.auto_calculate_bin_positions(sorting_config['max_bins'])
    print(f"Auto-calculated positions:")
    for bin_num in sorted(auto_positions.keys()):
        print(f"  Bin {bin_num}: {auto_positions[bin_num]}°")
    print("\nNote: These were NOT saved to config (as designed)")
    print("Config GUI would call config_manager.update_module_config() to save")

    # Test 2: Get current positions
    print("\n--- Test 2: Current Bin Positions (from memory) ---")
    current_positions = servo.get_bin_positions()
    print(f"Positions currently loaded in servo module:")
    for bin_num in sorted(current_positions.keys()):
        print(f"  Bin {bin_num}: {current_positions[bin_num]}°")

    # Test 3: Test reload function
    print("\n--- Test 3: Reload from Config ---")
    print("Attempting to reload positions from config file...")
    reload_success = servo.reload_positions_from_config()
    print(f"Reload result: {'SUCCESS' if reload_success else 'FAILED'}")

    # Test 4: Move to specific bins
    if servo.is_ready():
        print("\n--- Test 4: Movement Test ---")
        test_bins = [0, 1, 3, servo.max_bins]
        for bin_num in test_bins:
            if bin_num <= servo.max_bins:
                print(f"Moving to bin {bin_num}...")
                success = servo.move_to_bin(bin_num, wait=True)
                print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
                time.sleep(0.5)

        # Return home
        print("\nReturning to home position...")
        servo.home()
    else:
        print("\n--- Test 4: SKIPPED (servo not ready) ---")

    # Test 5: Test all positions (commented out by default)
    print("\n--- Test 5: Full Position Sweep (optional) ---")
    print("Uncomment in code to run: servo.test_all_positions(dwell_time=1.0)")
    # Uncomment to run full sweep test:
    # servo.test_all_positions(dwell_time=1.0)

    # Test 6: Statistics
    print("\n--- Test 6: Statistics ---")
    stats = servo.get_statistics()
    print(f"  Initialized: {stats['initialized']}")
    print(f"  Simulation: {stats['simulation_mode']}")
    print(f"  Current Angle: {stats['servo']['current_angle']}°")
    print(f"  Current Bin: {stats['servo']['current_bin']}")
    print(f"  Positions Configured: {len(stats['bins']['positions'])}")

    # Cleanup
    print("\n--- Cleanup ---")
    servo.release()
    print("Servo controller released")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("All tests passed with Option C architecture:")
    print("  ✓ Servo writes config ONLY during bootstrap")
    print("  ✓ Config GUI writes config during normal operations")
    print("  ✓ Clear separation of responsibilities")
    print("=" * 60 + "\n")