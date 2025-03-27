"""
servo_module.py - Servo control for the Lego sorting application chute mechanism

This module handles the servo motor that directs sorted Lego pieces to the appropriate bins.
It provides functionality for initializing, calibrating, and controlling the servo based on
bin assignments from the sorting module.
"""

import time
import logging
from typing import Dict, Any, Optional
import cv2

try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo

    HARDWARE_AVAILABLE = True
except (ImportError, NotImplementedError):
    HARDWARE_AVAILABLE = False
    print("Warning: Adafruit libraries not found. Servo will run in simulation mode.")

# Get module logger
logger = logging.getLogger(__name__)


class ServoModule:
    """Controls the servo motor for directing Lego pieces to bins"""

    def __init__(self, config_manager=None):
        """Initialize servo with configuration

        Args:
            config_manager: Configuration manager object, or None to use defaults
        """
        # Default configuration values
        self.channel = 0
        self.frequency = 50
        self.min_pulse = 150  # Typically ~1ms pulse for 0 degrees
        self.max_pulse = 600  # Typically ~2ms pulse for 180 degrees
        self.default_position = 90  # Center position in degrees
        self.bin_positions = {}  # Will store bin_number -> angle mapping
        self.current_bin = None  # Currently selected bin
        self.calibration_mode = False

        # Use config if provided, otherwise use defaults
        if config_manager:
            self.config_manager = config_manager
            servo_config = config_manager.get_section("servo")

            if servo_config:
                self.channel = servo_config.get("channel", self.channel)
                self.frequency = servo_config.get("frequency", self.frequency)
                self.min_pulse = servo_config.get("min_pulse", self.min_pulse)
                self.max_pulse = servo_config.get("max_pulse", self.max_pulse)
                self.default_position = servo_config.get("default_position", self.default_position)
                self.bin_positions = servo_config.get("bin_positions", {})
                self.calibration_mode = servo_config.get("calibration_mode", False)

            # Load max_bins from sorting config for calibration
            sorting_config = config_manager.get_section("sorting")
            if sorting_config:
                self.max_bins = sorting_config.get("max_bins", 9)
                self.overflow_bin = sorting_config.get("overflow_bin", 9)
            else:
                self.max_bins = 9
                self.overflow_bin = 9
        else:
            self.config_manager = None
            self.max_bins = 9
            self.overflow_bin = 9

        # Initialize hardware connection
        self.pca = None
        self.servo_motor = None
        self.initialized = False
        self.simulation_mode = not HARDWARE_AVAILABLE

        self._initialize_hardware()

        # If in calibration mode, run calibration
        if self.calibration_mode and self.initialized:
            self.calibrate()
            # Turn off calibration mode in config after running
            if self.config_manager:
                self.config_manager.set("servo", "calibration_mode", False)
                self.config_manager.save_config()

    def _initialize_hardware(self):
        """Initialize the PCA9685 and servo motor"""
        if self.simulation_mode:
            logger.info("Running in servo simulation mode")
            self.initialized = True
            return

        try:
            # Initialize I2C bus and PCA9685
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = self.frequency

            # Initialize servo
            self.servo_motor = servo.Servo(
                self.pca.channels[self.channel],
                min_pulse=self.min_pulse,
                max_pulse=self.max_pulse,
                actuation_range=180  # Full range in degrees
            )

            # Move to default position
            self.servo_motor.angle = self.default_position
            time.sleep(0.5)  # Allow time for servo to move

            self.initialized = True
            logger.info("Servo hardware initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize servo hardware: {str(e)}")
            self.simulation_mode = True
            self.initialized = True  # Still mark as initialized for simulation mode

    def move_to_angle(self, angle: float) -> bool:
        """Move the servo to a specific angle

        Args:
            angle: Target angle in degrees (0-180)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Servo not initialized")
            return False

        # Ensure angle is within valid range
        angle = max(0, min(180, angle))

        try:
            if self.simulation_mode:
                logger.info(f"Simulation: Moving servo to {angle} degrees")
            else:
                self.servo_motor.angle = angle
                logger.debug(f"Moved servo to {angle} degrees")
            return True
        except Exception as e:
            logger.error(f"Error moving servo: {str(e)}")
            return False

    def move_to_bin(self, bin_number: int) -> bool:
        """Move the servo to the position for the specified bin

        Args:
            bin_number: Target bin number

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Servo not initialized")
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
            logger.error("Servo not initialized")
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

        # Start with current positions or defaults
        positions = self.bin_positions.copy()

        # Calibrate each bin up to max_bins plus overflow bin
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
                        # Fall back to regular input if it can't get single char input
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

        # Save all positions to config
        if self.config_manager:
            self.config_manager.update_section("servo", {"bin_positions": positions})
            self.config_manager.save_config()

        # Update local bin positions
        self.bin_positions = positions

        print("Calibration complete! New positions saved to config.")
        return True

    def release(self):
        """Release hardware resources"""
        if not self.initialized or self.simulation_mode:
            return

        try:
            # Move to default position before shutting down
            self.move_to_angle(self.default_position)
            time.sleep(0.5)  # Allow time for servo to move

            # Deinitialize hardware
            self.pca.deinit()
            logger.info("Servo resources released")
        except Exception as e:
            logger.error(f"Error releasing servo resources: {str(e)}")


# Factory function
def create_servo_module(config_manager=None):
    """Create a servo module instance.

    Args:
        config_manager: Configuration manager object

    Returns:
        ServoModule instance
    """
    return ServoModule(config_manager)


# Example usage
if __name__ == "__main__":
    # This will run in simulation mode if hardware is not available
    servo = create_servo_module()

    try:
        print("Moving to various positions...")

        # Move to a few test positions
        for angle in [0, 45, 90, 135, 180]:
            print(f"Moving to {angle} degrees")
            servo.move_to_angle(angle)
            time.sleep(1)

        # Test bin movement
        for bin_num in range(10):
            print(f"Moving to bin {bin_num}")
            servo.move_to_bin(bin_num)
            time.sleep(1)

    finally:
        # Clean up
        servo.release()
        print("Servo test complete")