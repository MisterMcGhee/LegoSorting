"""
conveyor_motor_module.py - Conveyor belt motor control

FILE LOCATION: hardware/conveyor_motor_module.py

Controls the DC motor that drives the conveyor belt, moving Lego pieces
past the camera and into the sorting chute.

COMMUNICATION MODEL:
- Write-only serial protocol (mirrors arduino_servo_module design)
- Sends: "M,{speed}\\n" where speed is 0-255 (0 = stop)
- Arduino interprets speed as a PWM duty cycle for the motor driver

COMMAND PROTOCOL:
    "M,0\\n"   → Stop motor
    "M,128\\n" → Run at ~50% speed
    "M,255\\n" → Run at full speed

RESPONSIBILITIES:
- Serial connection to motor controller (Arduino or motor shield)
- Speed commands (0-255 PWM equivalent)
- Start / stop operations
- Ramp test for calibration
- Simulation mode for development without hardware

DOES NOT:
- Track piece detection (detector modules handle that)
- Make sorting decisions (processing modules handle that)
- Control the servo chute (arduino_servo_module handles that)

CONFIGURATION:
All settings stored in the 'motor' section of config.json.
"""

import time
import logging
import serial
import threading
from typing import Optional, Dict
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class ConveyorMotorController:
    """
    Controls the conveyor belt DC motor via a write-only serial connection.

    Sends PWM speed commands (0-255) to an Arduino running a compatible
    motor control sketch.  Mirrors the design of ArduinoServoController:
    one serial port, write-only, simulation mode for offline development.
    """

    def __init__(self, config_manager: EnhancedConfigManager):
        """
        Initialize motor controller from configuration.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

        motor_config = config_manager.get_module_config(ModuleConfig.MOTOR.value)

        # =====================================================================
        # CONNECTION SETTINGS
        # =====================================================================
        self.port = motor_config.get('port', '')
        self.baud_rate = motor_config.get('baud_rate', 57600)
        self.timeout = motor_config.get('timeout', 1.0)
        self.connection_retries = motor_config.get('connection_retries', 3)
        self.retry_delay = motor_config.get('retry_delay', 1.0)
        self.simulation_mode = motor_config.get('simulation_mode', False)

        # =====================================================================
        # MOTOR SPEED PARAMETERS
        # =====================================================================
        self.default_speed = motor_config.get('default_speed', 128)
        self.min_speed = motor_config.get('min_speed', 0)
        self.max_speed = motor_config.get('max_speed', 255)
        self.ramp_rate = motor_config.get('ramp_rate', 32)

        # =====================================================================
        # STATE TRACKING
        # =====================================================================
        self.current_speed: int = 0
        self.is_running: bool = False
        self.serial_conn: Optional[serial.Serial] = None
        self.initialized: bool = False
        self.lock = threading.RLock()

        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        if self.simulation_mode:
            logger.info("Motor controller: SIMULATION MODE — no hardware required")
            self.initialized = True
        else:
            logger.info("Motor controller: HARDWARE MODE")
            self._connect()

        self._log_init_status()

    # ========================================================================
    # SECTION 1: SERIAL CONNECTION
    # ========================================================================

    def _connect(self) -> bool:
        """
        Establish serial connection to the motor controller.

        Returns:
            True if connection succeeded, False otherwise
        """
        with self.lock:
            for attempt in range(self.connection_retries):
                try:
                    logger.info(
                        f"Motor connection attempt {attempt + 1}/{self.connection_retries} "
                        f"on {self.port} @ {self.baud_rate}"
                    )
                    self.serial_conn = serial.Serial(
                        port=self.port,
                        baudrate=self.baud_rate,
                        timeout=self.timeout,
                        write_timeout=self.timeout
                    )
                    # Wait for Arduino reset after USB connect
                    time.sleep(2.0)
                    self.serial_conn.reset_input_buffer()
                    self.serial_conn.reset_output_buffer()
                    self.initialized = True
                    logger.info(f"Motor controller connected on {self.port}")
                    return True

                except serial.SerialException as e:
                    logger.warning(f"Motor connection attempt {attempt + 1} failed: {e}")
                    self._close_serial()

                except Exception as e:
                    logger.error(f"Unexpected error connecting motor: {e}", exc_info=True)
                    self._close_serial()

                if attempt < self.connection_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)

            logger.error("Motor controller: all connection attempts failed")
            self.initialized = False
            return False

    def _close_serial(self):
        """Safely close the serial connection."""
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

    def disconnect(self):
        """Stop the motor and close the serial connection."""
        self.stop()
        with self.lock:
            self._close_serial()
            self.initialized = False
        logger.info("Motor controller disconnected")

    # ========================================================================
    # SECTION 2: COMMAND SENDING
    # ========================================================================

    def _send_command(self, speed: int) -> bool:
        """
        Send a speed command to the motor controller.

        Command format: "M,{speed}\\n"

        Args:
            speed: PWM value 0-255 (already validated by caller)

        Returns:
            True if command sent successfully, False otherwise
        """
        with self.lock:
            speed = max(0, min(255, speed))

            if self.simulation_mode:
                logger.debug(f"[SIMULATION] Motor command: M,{speed}")
                return True

            if not self.serial_conn:
                logger.error("Motor: no serial connection")
                return False

            try:
                cmd = f"M,{speed}\n"
                self.serial_conn.write(cmd.encode())
                self.serial_conn.flush()
                logger.debug(f"Motor sent: {cmd.strip()}")
                return True

            except Exception as e:
                logger.error(f"Motor command failed: {e}")
                return False

    # ========================================================================
    # SECTION 3: MOTOR CONTROL API
    # ========================================================================

    def set_speed(self, speed: int) -> bool:
        """
        Set motor speed.  Speed 0 stops the motor.

        Args:
            speed: Target PWM value (0-255).  Clamped to [min_speed, max_speed].

        Returns:
            True if command sent successfully
        """
        speed = max(self.min_speed, min(self.max_speed, speed))
        success = self._send_command(speed)
        if success:
            self.current_speed = speed
            self.is_running = speed > 0
            logger.info(f"Motor speed set to {speed}")
        return success

    def start(self, speed: Optional[int] = None) -> bool:
        """
        Start the conveyor at the specified speed (or default speed).

        Args:
            speed: Speed override (0-255).  Uses default_speed if None.

        Returns:
            True if command sent successfully
        """
        target = speed if speed is not None else self.default_speed
        logger.info(f"Motor starting at speed {target}")
        return self.set_speed(target)

    def stop(self) -> bool:
        """
        Stop the conveyor motor immediately.

        Returns:
            True if stop command sent successfully
        """
        success = self._send_command(0)
        if success:
            self.current_speed = 0
            self.is_running = False
            logger.info("Motor stopped")
        return success

    def ramp_test(self, steps: int = 8, dwell_time: float = 0.5) -> bool:
        """
        Run a ramp test: ramp speed from 0 → max → 0.

        Equivalent to the servo tab's "Test All Positions" sweep.
        Useful for verifying motor control and finding a comfortable
        operating speed.

        Args:
            steps: Number of speed increments in each ramp direction
            dwell_time: Seconds to hold each speed level

        Returns:
            True if the full ramp completed without errors
        """
        if not self.is_ready():
            logger.warning("Motor ramp test: controller not ready")
            return False

        step_size = max(1, self.max_speed // steps)
        logger.info(f"Motor ramp test: {steps} steps, {dwell_time}s dwell, step={step_size}")

        try:
            # Ramp up
            for i in range(steps + 1):
                speed = min(i * step_size, self.max_speed)
                if not self._send_command(speed):
                    return False
                self.current_speed = speed
                self.is_running = speed > 0
                time.sleep(dwell_time)

            # Brief hold at maximum
            time.sleep(dwell_time)

            # Ramp down
            for i in range(steps, -1, -1):
                speed = max(i * step_size, 0)
                if not self._send_command(speed):
                    return False
                self.current_speed = speed
                self.is_running = speed > 0
                time.sleep(dwell_time)

            self.current_speed = 0
            self.is_running = False
            logger.info("Motor ramp test complete")
            return True

        except Exception as e:
            logger.error(f"Motor ramp test failed: {e}", exc_info=True)
            return False

    # ========================================================================
    # SECTION 4: STATUS AND INFORMATION
    # ========================================================================

    def is_ready(self) -> bool:
        """Return True if the motor controller is ready to accept commands."""
        return self.initialized

    def get_status(self) -> Dict:
        """Return a snapshot of the motor controller's current state."""
        with self.lock:
            return {
                'initialized': self.initialized,
                'simulation_mode': self.simulation_mode,
                'port': self.port,
                'baud_rate': self.baud_rate,
                'current_speed': self.current_speed,
                'is_running': self.is_running,
                'default_speed': self.default_speed,
                'speed_range': (self.min_speed, self.max_speed),
            }

    def _log_init_status(self):
        """Log initialization summary."""
        logger.info("=" * 50)
        logger.info("CONVEYOR MOTOR CONTROLLER INITIALIZED")
        logger.info("=" * 50)
        mode = "SIMULATION" if self.simulation_mode else "HARDWARE (WRITE-ONLY)"
        logger.info(f"Mode: {mode}")
        logger.info(f"Ready: {self.initialized}")
        if not self.simulation_mode:
            logger.info(f"Port: {self.port} @ {self.baud_rate} baud")
        logger.info(f"Speed range: {self.min_speed}-{self.max_speed} (default {self.default_speed})")
        logger.info("=" * 50)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_conveyor_motor_controller(
        config_manager: EnhancedConfigManager) -> ConveyorMotorController:
    """
    Factory function to create a ConveyorMotorController.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Initialized ConveyorMotorController
    """
    return ConveyorMotorController(config_manager)
