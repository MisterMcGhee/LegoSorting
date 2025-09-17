"""
arduino_servo_module.py - SIMPLIFIED VERSION
Arduino-based servo control that loads saved bin positions from config GUI
"""

import time
import logging
import serial
import threading
from enhanced_config_manager import ModuleConfig

logger = logging.getLogger(__name__)


class ArduinoServoModule:
    """Controls the servo motor for directing Lego pieces to bins via Arduino"""

    def __init__(self, config_manager=None):
        """Initialize servo with configuration

        Args:
            config_manager: Configuration manager object, or None to use defaults
        """
        # Initialize with defaults first
        self.port = "/dev/ttyACM0"
        self.baud_rate = 9600
        self.timeout = 2.0
        self.connection_retries = 3
        self.retry_delay = 1.0
        self.min_pulse = 500
        self.max_pulse = 2500
        self.default_position = 90
        self.simulation_mode = False
        self.min_bin_separation = 20

        # Initialize bin positions dictionary
        self.bin_positions = {}
        self.current_bin = None

        # Use config if provided
        if config_manager:
            self.config_manager = config_manager

            # Get configurations - read from arduino_servo config where GUI saves everything
            arduino_config = config_manager.get_module_config(ModuleConfig.ARDUINO_SERVO.value)
            sorting_config = config_manager.get_module_config(ModuleConfig.SORTING.value)

            # Apply Arduino connection settings
            self.port = arduino_config["port"]
            self.baud_rate = arduino_config["baud_rate"]
            self.timeout = arduino_config["timeout"]
            self.connection_retries = arduino_config["connection_retries"]
            self.retry_delay = arduino_config["retry_delay"]
            self.simulation_mode = arduino_config["simulation_mode"]

            # Apply servo hardware settings from arduino_servo config
            self.min_pulse = arduino_config.get("min_pulse", 500)
            self.max_pulse = arduino_config.get("max_pulse", 2500)
            self.default_position = arduino_config.get("default_position", 90)
            self.min_bin_separation = arduino_config.get("min_bin_separation", 20)

            # Get bin configuration from sorting
            self.max_bins = sorting_config["max_bins"]
            self.overflow_bin = sorting_config["overflow_bin"]

            # Load saved bin positions from config GUI
            saved_positions = arduino_config.get("bin_positions", {})
            if saved_positions:
                self.bin_positions = {str(k): float(v) for k, v in saved_positions.items()}
                logger.info(f"Loaded saved bin positions: {self.bin_positions}")
                print(f"Using saved bin positions: {self.bin_positions}")
            else:
                logger.warning("No bin positions found in config - positions must be set via config GUI")
                print("WARNING: No bin positions configured. Please set positions in config GUI.")
        else:
            self.config_manager = None
            self.max_bins = 9
            self.overflow_bin = 0
            logger.warning("No config manager provided - bin positions must be set manually")

        # Initialize connection variables
        self.arduino = None
        self.initialized = False
        self.lock = threading.RLock()

        # Connect to Arduino or enable simulation mode
        if self.simulation_mode:
            logger.info("Simulation mode enabled. Skipping Arduino connection.")
            print("Simulation mode enabled. Skipping Arduino connection.")
            self.initialized = True
        else:
            self._connect_to_arduino()

    def set_bin_positions(self, positions_dict):
        """Set specific bin positions (used by GUI)

        Args:
            positions_dict: Dictionary of {bin_number: angle} pairs
        """
        self.bin_positions = {str(k): float(v) for k, v in positions_dict.items()}
        logger.info(f"Updated bin positions: {self.bin_positions}")

        # Save to config if available
        if self.config_manager:
            arduino_config = self.config_manager.get_module_config(ModuleConfig.ARDUINO_SERVO.value)
            arduino_config["bin_positions"] = self.bin_positions
            self.config_manager.update_module_config(ModuleConfig.ARDUINO_SERVO.value, arduino_config)
            self.config_manager.save_config()

    def get_bin_positions(self):
        """Get current bin positions dictionary"""
        return self.bin_positions.copy()

    def set_max_bins(self, max_bins: int):
        """Update max_bins setting"""
        self.max_bins = max_bins
        logger.info(f"Updated max_bins to {max_bins}")

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

                    # Move to default position to confirm connection
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
                user_input = input(
                    "Would you like to retry connection (r), switch to simulation mode (s), or quit (q)? ").lower()

                if user_input == 'r':
                    print("Retrying Arduino connection...")
                    self._connect_to_arduino()
                    return
                elif user_input == 's':
                    print("Switching to simulation mode.")
                    logger.info("User selected simulation mode after connection failure.")
                    self.simulation_mode = True
                    self.initialized = True

                    # Update config if available
                    if self.config_manager:
                        arduino_config = self.config_manager.get_module_config(ModuleConfig.ARDUINO_SERVO.value)
                        arduino_config["simulation_mode"] = True
                        self.config_manager.update_module_config(ModuleConfig.ARDUINO_SERVO.value, arduino_config)
                        self.config_manager.save_config()
                        logger.info("Updated config: simulation_mode=True")
                    return
                elif user_input == 'q':
                    print("Quitting application due to Arduino connection failure.")
                    logger.error("User quit application due to Arduino connection failure.")
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

        # Check if we have a position for this bin
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
            logger.info(f"Moved to bin {bin_number} at {angle} degrees")

        return success

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