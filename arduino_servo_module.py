"""
arduino_servo_module.py - Arduino-based servo control for the Lego sorting application

This module handles communication with an Arduino to control the servo motor
that directs sorted Lego pieces to the appropriate bins. It implements the
same interface as servo_module.py but uses Arduino for hardware control.
"""

import time
import logging
import serial
import threading
from typing import Dict, Any, Optional

# Get module logger
logger = logging.getLogger(__name__)


class ArduinoServoModule:
    """Controls the servo motor for directing Lego pieces to bins via Arduino"""

    def __init__(self, config_manager=None):
        """Initialize servo with configuration

        Args:
            config_manager: Configuration manager object, or None to use defaults
        """
        # Default configuration values
        self.port = "/dev/ttyACM0"  # Default Arduino port on Linux
        self.baud_rate = 9600
        self.timeout = 2.0
        self.connection_retries = 3
        self.retry_delay = 1.0
        self.min_pulse = 500  # Default pulse width for 0 degrees (in microseconds)
        self.max_pulse = 2500  # Default pulse width for 180 degrees (in microseconds)
        self.default_position = 90  # Center position in degrees
        self.calibration_mode = False
        self.simulation_mode = False
        self.min_bin_separation = 20  # Minimum degrees between bins

        # Initialize bin positions dictionary (will be calculated dynamically)
        self.bin_positions = {}
        self.current_bin = None  # Currently selected bin

        # Use config if provided, otherwise use defaults
        if config_manager:
            self.config_manager = config_manager
            arduino_config = config_manager.get_section("arduino_servo")

            if arduino_config:
                self.port = arduino_config.get("port", self.port)
                self.baud_rate = arduino_config.get("baud_rate", self.baud_rate)
                self.timeout = arduino_config.get("timeout", self.timeout)
                self.connection_retries = arduino_config.get("connection_retries", self.connection_retries)
                self.retry_delay = arduino_config.get("retry_delay", self.retry_delay)
                # Get simulation_mode from config
                self.simulation_mode = arduino_config.get("simulation_mode", False)

            servo_config = config_manager.get_section("servo")
            if servo_config:
                self.min_pulse = servo_config.get("min_pulse", self.min_pulse)
                self.max_pulse = servo_config.get("max_pulse", self.max_pulse)
                self.default_position = servo_config.get("default_position", self.default_position)
                self.calibration_mode = servo_config.get("calibration_mode", False)
                self.min_bin_separation = servo_config.get("min_bin_separation", self.min_bin_separation)

            # Load max_bins from sorting config for bin position calculation
            sorting_config = config_manager.get_section("sorting")
            if sorting_config:
                self.max_bins = sorting_config.get("max_bins", 9)
                self.overflow_bin = sorting_config.get("overflow_bin", 0)  # Now using bin 0 as overflow
            else:
                self.max_bins = 9
                self.overflow_bin = 0
        else:
            self.config_manager = None
            self.max_bins = 9
            self.overflow_bin = 0

        # Initialize connection variables
        self.arduino = None
        self.initialized = False
        self.lock = threading.RLock()  # For thread safety

        # Calculate bin positions dynamically
        self._calculate_bin_positions()

        # If simulation mode is enabled in config, skip Arduino connection
        if self.simulation_mode:
            logger.info("Simulation mode enabled in config. Skipping Arduino connection.")
            print("Simulation mode enabled in config. Skipping Arduino connection.")
            self.initialized = True
        else:
            # Connect to Arduino
            self._connect_to_arduino()

        # Run initialization test sequence if connection is successful
        if self.initialized:
            self._run_initialization_test_sequence()

        # If in calibration mode, run calibration
        if self.calibration_mode and self.initialized:
            print("Starting servo calibration mode...")
            self.calibrate()
            # Turn off calibration mode in config after running
            if self.config_manager:
                self.config_manager.set("servo", "calibration_mode", False)
                self.config_manager.save_config()

    def _calculate_bin_positions(self):
        """Calculate bin positions dynamically based on max_bins and minimum bin separation"""
        # Bin 0 (overflow bin) is always at 0 degrees
        self.bin_positions[str(self.overflow_bin)] = 0

        # Distribute remaining bins evenly across 1-180 degrees
        remaining_bins = self.max_bins

        # If overflow bin is included in max_bins count, adjust accordingly
        if self.overflow_bin < self.max_bins:
            remaining_bins -= 1

        # Calculate maximum number of bins possible with minimum separation
        max_possible_bins = int(180 / self.min_bin_separation)

        if remaining_bins > max_possible_bins:
            logger.warning(f"Warning: Requested {remaining_bins} bins, but only {max_possible_bins} " +
                         f"possible with minimum separation of {self.min_bin_separation} degrees")
            print(f"Warning: Requested {remaining_bins} bins, but only {max_possible_bins} " +
                 f"possible with minimum separation of {self.min_bin_separation} degrees")
            remaining_bins = max_possible_bins

        if remaining_bins > 0:
            # Calculate step size based on min_bin_separation or even distribution, whichever is larger
            step = max(self.min_bin_separation, 180 / remaining_bins)

            # Assign positions to each bin
            current_angle = step
            assigned_bins = 0

            for bin_num in range(1, self.max_bins + 1):
                # Skip overflow bin if it's not bin 0
                if bin_num == self.overflow_bin:
                    continue

                # Stop if we've reached the maximum possible bins
                if assigned_bins >= max_possible_bins or current_angle > 180:
                    break

                self.bin_positions[str(bin_num)] = int(current_angle)
                current_angle += step
                assigned_bins += 1

        logger.info(f"Dynamically calculated bin positions with min separation of {self.min_bin_separation} degrees: {self.bin_positions}")
        print(f"Bin positions: {self.bin_positions}")

    def _connect_to_arduino(self):
        """Connect to the Arduino device"""
        with self.lock:
            for attempt in range(self.connection_retries):
                try:
                    logger.info(
                        f"Connecting to Arduino on {self.port}, attempt {attempt + 1}/{self.connection_retries}")
                    print(f"Connecting to Arduino on {self.port}, attempt {attempt + 1}/{self.connection_retries}")

                    self.arduino = serial.Serial(
                        port=self.port,
                        baudrate=self.baud_rate,
                        timeout=self.timeout
                    )

                    # Wait for Arduino to reset
                    time.sleep(3)

                    # Flush input and output buffers
                    self.arduino.flushInput()
                    self.arduino.flushOutput()

                    # Connection established
                    print("Connection established - proceeding with hardware mode")
                    logger.info("Connection established - proceeding with hardware mode")
                    self.initialized = True

                    # Configure servo parameters
                    self._send_command(f"S,{self.min_pulse},{self.max_pulse}")
                    # Don't wait for response as it's not critical

                    # Move to default position
                    self.move_to_angle(self.default_position)
                    return

                except serial.SerialException as e:
                    logger.error(f"Failed to connect to Arduino: {str(e)}")
                    print(f"Failed to connect to Arduino: {str(e)}")

                # Wait before retrying
                time.sleep(self.retry_delay)

            # If we get here, all connection attempts failed
            logger.error("All connection attempts to Arduino failed.")
            print("All connection attempts to Arduino failed.")

            # Prompt user to retry or switch to simulation mode
            while True:
                user_input = input("Would you like to retry connection (r), switch to simulation mode (s), or quit (q)? ").lower()

                if user_input == 'r':
                    print("Retrying Arduino connection...")
                    self._connect_to_arduino()
                    return
                elif user_input == 's':
                    print("Switching to simulation mode.")
                    logger.info("User selected simulation mode after connection failure.")
                    self.simulation_mode = True
                    self.initialized = True  # Mark as initialized for simulation mode

                    # Update config if available
                    if self.config_manager:
                        self.config_manager.set("arduino_servo", "simulation_mode", True)
                        self.config_manager.save_config()
                        logger.info("Updated config: simulation_mode=True")
                    return
                elif user_input == 'q':
                    print("Quitting application due to Arduino connection failure.")
                    logger.error("User quit application due to Arduino connection failure.")
                    # Exit application
                    import sys
                    sys.exit(1)
                else:
                    print("Invalid input. Please enter 'r', 's', or 'q'.")

    def _send_command(self, command):
        """Send a command string to the Arduino

        Args:
            command: Command string to send

        Returns:
            bool: True if command was sent successfully
        """
        with self.lock:
            if not self.initialized or self.simulation_mode:
                return True

            try:
                # Add newline if not present
                if not command.endswith('\n'):
                    command += '\n'

                # Send command
                self.arduino.write(command.encode())
                logger.debug(f"Sent command to Arduino: {command.strip()}")
                return True

            except Exception as e:
                logger.error(f"Error sending command to Arduino: {str(e)}")
                return False

    def _run_initialization_test_sequence(self):
        """Run a sequence to demonstrate the servo is properly connected.

        This moves the servo through all configured bin positions to show
        the connection is working and verify the mechanical setup.
        """
        logger.info("Running servo initialization test sequence")
        print("\n===== ARDUINO SERVO INITIALIZATION TEST =====")
        print("Moving through all configured bin positions...")

        try:
            # First move to the default position
            print(f"Moving to default position ({self.default_position} degrees)")
            self.move_to_angle(self.default_position)
            time.sleep(1.0)

            # Collect bin positions
            positions = []
            for bin_num, angle in self.bin_positions.items():
                positions.append((int(bin_num), angle))

            # Sort by position value to make the sweep smooth
            positions.sort(key=lambda x: x[1])

            # Move to each position
            for bin_num, angle in positions:
                print(f"Moving to bin {bin_num} ({angle} degrees)")
                self.move_to_angle(angle)
                time.sleep(0.8)  # Short delay between positions

            # Return to default position
            print(f"Returning to default position ({self.default_position} degrees)")
            self.move_to_angle(self.default_position)
            print("Initialization test sequence complete!")
            print("========================================\n")

        except Exception as e:
            logger.error(f"Error during initialization test sequence: {str(e)}")
            print(f"Error during test sequence: {str(e)}")

    def _read_response(self, timeout=2.0):
        """Read a response from the Arduino

        Args:
            timeout: Maximum time to wait for a response

        Returns:
            str: Response string or None if timeout occurs
        """
        with self.lock:
            if not self.initialized or self.simulation_mode:
                return None

            try:
                # Set timeout for this read operation
                original_timeout = self.arduino.timeout
                self.arduino.timeout = timeout

                # Read response
                response = self.arduino.readline().decode().strip()

                # Restore original timeout
                self.arduino.timeout = original_timeout

                if response:
                    logger.debug(f"Received from Arduino: {response}")
                    return response

                return None

            except Exception as e:
                logger.error(f"Error reading from Arduino: {str(e)}")
                return None

    def move_to_angle(self, angle: float) -> bool:
        """Move the servo to a specific angle

        Args:
            angle: Target angle in degrees (0-180)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Arduino not initialized")
            return False

        # Ensure angle is within valid range
        angle = max(0, min(180, angle))

        try:
            if self.simulation_mode:
                logger.info(f"Simulation: Moving servo to {angle} degrees")
                print(f"Simulation: Moving servo to {angle} degrees")
                return True

            # Send angle command to Arduino
            command = f"A,{angle}"
            if not self._send_command(command):
                return False

            # Try to read response, but don't depend on it for operation
            response = self._read_response()
            if response and f"Moved to: {angle}" in response:
                logger.debug(f"Confirmed: Moved servo to {angle} degrees")
            else:
                # Continue even if we don't get a response or the expected response
                logger.debug(f"Moved servo to {angle} degrees (no confirmation)")

            return True

        except Exception as e:
            logger.error(f"Error moving servo: {str(e)}")
            print(f"Error moving servo: {str(e)}")
            return False

    def move_to_bin(self, bin_number: int) -> bool:
        """Move the servo to the position for the specified bin

        Args:
            bin_number: Target bin number

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Arduino not initialized")
            return False

        # Convert bin_number to string for dictionary lookup
        bin_key = str(bin_number)

        # If we don't have a position for this bin, use the overflow bin
        if bin_key not in self.bin_positions:
            logger.warning(f"No position defined for bin {bin_number}, using overflow bin")
            bin_key = str(self.overflow_bin)

            # If even overflow bin isn't defined, use default position
            if bin_key not in self.bin_positions:
                logger.warning(f"No position defined for overflow bin, using default position")
                angle = self.default_position
                return self.move_to_angle(angle)

        angle = self.bin_positions[bin_key]
        success = self.move_to_angle(angle)

        if success:
            self.current_bin = bin_number

        return success

    def calibrate(self):
        """Interactive calibration of bin positions using text console interface"""
        if not self.initialized:
            logger.error("Arduino not initialized")
            return False

        print("\n======== SERVO CALIBRATION ========")
        print("For each bin, use these controls:")
        print("  '+': Increase angle by 1 degree")
        print("  '-': Decrease angle by 1 degree")
        print("  '>': Increase angle by 5 degrees")
        print("  '<': Decrease angle by 5 degrees")
        print("  's': Save position and move to next bin")
        print("  'q': Quit calibration")
        print("==================================\n")

        # Start with calculated positions
        positions = self.bin_positions.copy()

        # Calibrate each bin up to max_bins
        for bin_num in range(self.max_bins + 1):
            is_overflow = bin_num == self.overflow_bin
            bin_label = f"OVERFLOW BIN ({bin_num})" if is_overflow else f"Bin {bin_num}"

            # Get current position or default
            current_pos = int(positions.get(str(bin_num), self.default_position))
            self.move_to_angle(current_pos)

            print(f"\nCalibrating position for {bin_label}")
            print(f"Current angle: {current_pos} degrees")
            print("Enter command (+, -, >, <, s, q): ", end="", flush=True)

            while True:
                # Get a single character input without requiring Enter
                try:
                    import msvcrt  # Windows
                    command = msvcrt.getch().decode().lower()
                except (ImportError, AttributeError):
                    try:
                        import tty, termios, sys  # Unix/Linux/MacOS
                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        try:
                            tty.setraw(fd)
                            command = sys.stdin.read(1).lower()
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    except (ImportError, AttributeError):
                        # Fall back to regular input if can't get single char input
                        command = input().lower()[0] if input() else ""

                if command == '+':
                    current_pos += 1
                    print(f"\rCurrent angle: {current_pos} degrees    ", end="", flush=True)
                elif command == '-':
                    current_pos -= 1
                    print(f"\rCurrent angle: {current_pos} degrees    ", end="", flush=True)
                elif command == '>':
                    current_pos += 5
                    print(f"\rCurrent angle: {current_pos} degrees    ", end="", flush=True)
                elif command == '<':
                    current_pos -= 5
                    print(f"\rCurrent angle: {current_pos} degrees    ", end="", flush=True)
                elif command == 's':
                    print(f"\nSaved position for {bin_label}: {current_pos} degrees")
                    break
                elif command == 'q':
                    print("\nCalibration aborted")
                    return False

                # Ensure position stays within valid range
                current_pos = max(0, min(180, current_pos))
                self.move_to_angle(current_pos)

            # Store the confirmed position
            positions[str(bin_num)] = current_pos

        # Store calibrated positions temporarily for this session
        # But they won't be saved to config since we're using dynamic calculation
        self.bin_positions = positions

        print("Calibration complete! Note: These positions are only for this session and won't be saved.")
        return True

    def release(self):
        """Release hardware resources"""
        with self.lock:
            if not self.initialized or self.simulation_mode:
                return

            try:
                # Move to default position before shutting down
                self.move_to_angle(self.default_position)
                time.sleep(0.5)  # Allow time for servo to move

                # Close serial connection
                if self.arduino and self.arduino.is_open:
                    self.arduino.close()
                    self.arduino = None
                    self.initialized = False
                    logger.info("Arduino connection closed")
                    print("Arduino connection closed")
            except Exception as e:
                logger.error(f"Error releasing Arduino resources: {str(e)}")
                print(f"Error releasing Arduino resources: {str(e)}")


# Factory function
def create_arduino_servo_module(config_manager=None):
    """Create an Arduino servo module instance.

    Args:
        config_manager: Configuration manager object

    Returns:
        ArduinoServoModule instance
    """
    return ArduinoServoModule(config_manager)


# Example usage
if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO)

    # Create instance
    servo = create_arduino_servo_module()

    try:
        print("Testing Arduino servo control")

        if not servo.initialized:
            print("Arduino initialization failed, exiting")
            exit(1)

        # Test movement sequence
        print("Moving to various positions...")
        test_positions = [0, 45, 90, 135, 180, 90]

        for pos in test_positions:
            print(f"Moving to {pos} degrees")
            servo.move_to_angle(pos)
            time.sleep(1.5)

        # Test bin movement (if bins are configured)
        bin_count = min(5, len(servo.bin_positions))
        if bin_count > 0:
            print("Testing bin positions...")
            for bin_num in range(bin_count):
                print(f"Moving to bin {bin_num}")
                servo.move_to_bin(bin_num)
                time.sleep(1.5)

        print("Servo test complete, moving to default position")
        servo.move_to_angle(servo.default_position)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    finally:
        # Clean up
        servo.release()