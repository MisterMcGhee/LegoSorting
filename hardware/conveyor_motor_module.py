"""
conveyor_motor_module.py - Multi-motor conveyor control

FILE LOCATION: hardware/conveyor_motor_module.py

Controls one or more DC motors (conveyor belt, feeder, etc.) via a single
write-only serial connection to an Arduino running a compatible motor sketch.

COMMUNICATION MODEL:
- Write-only serial protocol (mirrors arduino_servo_module design)
- Each motor is addressed by a zero-based integer index
- Sends: "M,{motor_id},{speed}\\n"  e.g. "M,0,128\\n"
- Arduino maps motor_id → motor driver channel and sets PWM duty cycle

COMMAND PROTOCOL:
    "M,0,0\\n"    → Stop motor 0
    "M,0,128\\n"  → Motor 0 at ~50% speed
    "M,1,200\\n"  → Motor 1 at ~78% speed

RESPONSIBILITIES:
- Serial connection to the motor controller Arduino
- Per-motor speed commands (0-255 PWM)
- Independent start / stop per motor
- Stop-all convenience operation
- Ramp test for a single motor
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
from typing import Dict, List, Optional
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class ConveyorMotorController:
    """
    Independently controls N DC motors via a single serial Arduino connection.

    Each motor is addressed by a zero-based index.  Speed is a PWM duty-cycle
    value from 0 (stop) to 255 (full speed).

    Command format sent to Arduino: "M,{motor_id},{speed}\\n"
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
        # MULTI-MOTOR CONFIGURATION
        # =====================================================================
        self.num_motors: int = motor_config.get('num_motors', 2)
        self.motor_names: Dict[int, str] = {
            int(k): v
            for k, v in motor_config.get('motor_names', {}).items()
        }
        # Per-motor default speeds (keyed by int motor_id)
        raw_defaults = motor_config.get('default_speeds', {})
        self.default_speeds: Dict[int, int] = {
            int(k): int(v) for k, v in raw_defaults.items()
        }

        # Shared speed limits
        self.min_speed: int = motor_config.get('min_speed', 0)
        self.max_speed: int = motor_config.get('max_speed', 255)
        self.ramp_rate: int = motor_config.get('ramp_rate', 32)

        # =====================================================================
        # STATE TRACKING  (per motor)
        # =====================================================================
        self._speeds: Dict[int, int] = {i: 0 for i in range(self.num_motors)}
        self._running: Dict[int, bool] = {i: False for i in range(self.num_motors)}

        # =====================================================================
        # SERIAL CONNECTION
        # =====================================================================
        self.serial_conn: Optional[serial.Serial] = None
        self.initialized: bool = False
        self.lock = threading.RLock()

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
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

    def disconnect(self):
        """Stop all motors and close the serial connection."""
        self.stop_all()
        with self.lock:
            self._close_serial()
            self.initialized = False
        logger.info("Motor controller disconnected")

    # ========================================================================
    # SECTION 2: COMMAND SENDING
    # ========================================================================

    def _send_command(self, motor_id: int, speed: int) -> bool:
        """
        Send a speed command for one motor.

        Format: "M,{motor_id},{speed}\\n"

        Args:
            motor_id: Zero-based motor index
            speed:    PWM value 0-255 (already validated by caller)

        Returns:
            True if the command was sent (or simulated) successfully
        """
        with self.lock:
            speed = max(0, min(255, speed))

            if self.simulation_mode:
                logger.debug(f"[SIMULATION] Motor command: M,{motor_id},{speed}")
                return True

            if not self.serial_conn:
                logger.error("Motor: no serial connection")
                return False

            try:
                cmd = f"M,{motor_id},{speed}\n"
                self.serial_conn.write(cmd.encode())
                self.serial_conn.flush()
                logger.debug(f"Motor sent: {cmd.strip()}")
                return True
            except Exception as e:
                logger.error(f"Motor command M,{motor_id},{speed} failed: {e}")
                return False

    # ========================================================================
    # SECTION 3: PER-MOTOR CONTROL API
    # ========================================================================

    def set_motor_speed(self, motor_id: int, speed: int) -> bool:
        """
        Set the speed of one motor independently.

        Args:
            motor_id: Zero-based motor index (0 .. num_motors-1)
            speed:    Target PWM value (0-255); clamped to [min_speed, max_speed]

        Returns:
            True if the command was sent successfully
        """
        if not self._valid_motor(motor_id):
            return False

        speed = max(self.min_speed, min(self.max_speed, speed))
        success = self._send_command(motor_id, speed)
        if success:
            self._speeds[motor_id] = speed
            self._running[motor_id] = speed > 0
            name = self.motor_names.get(motor_id, f"Motor {motor_id}")
            logger.info(f"{name} speed → {speed}")
        return success

    def start_motor(self, motor_id: int, speed: Optional[int] = None) -> bool:
        """
        Start one motor at the specified speed (or its configured default).

        Args:
            motor_id: Zero-based motor index
            speed:    Speed override; uses default_speeds[motor_id] if None

        Returns:
            True if the command was sent successfully
        """
        if not self._valid_motor(motor_id):
            return False
        target = speed if speed is not None else self.default_speeds.get(motor_id, 128)
        return self.set_motor_speed(motor_id, target)

    def stop_motor(self, motor_id: int) -> bool:
        """
        Stop one motor.

        Args:
            motor_id: Zero-based motor index

        Returns:
            True if the stop command was sent successfully
        """
        if not self._valid_motor(motor_id):
            return False
        success = self._send_command(motor_id, 0)
        if success:
            self._speeds[motor_id] = 0
            self._running[motor_id] = False
            name = self.motor_names.get(motor_id, f"Motor {motor_id}")
            logger.info(f"{name} stopped")
        return success

    def stop_all(self) -> bool:
        """
        Stop every motor.

        Returns:
            True if all stop commands were sent successfully
        """
        results = [self.stop_motor(i) for i in range(self.num_motors)]
        return all(results)

    def ramp_test(self, motor_id: int, steps: int = 8,
                  dwell_time: float = 0.5) -> bool:
        """
        Run a ramp test for one motor: 0 → max → 0.

        Args:
            motor_id:   Zero-based motor index
            steps:      Number of increments per direction
            dwell_time: Seconds to hold each speed level

        Returns:
            True if the ramp completed without serial errors
        """
        if not self._valid_motor(motor_id):
            return False
        if not self.is_ready():
            logger.warning("Motor ramp test: controller not ready")
            return False

        name = self.motor_names.get(motor_id, f"Motor {motor_id}")
        step_size = max(1, self.max_speed // steps)
        logger.info(f"Ramp test — {name}: {steps} steps × {step_size} @ {dwell_time}s dwell")

        try:
            # Ramp up
            for i in range(steps + 1):
                speed = min(i * step_size, self.max_speed)
                if not self._send_command(motor_id, speed):
                    return False
                self._speeds[motor_id] = speed
                self._running[motor_id] = speed > 0
                time.sleep(dwell_time)

            time.sleep(dwell_time)  # Hold at max

            # Ramp down
            for i in range(steps, -1, -1):
                speed = max(i * step_size, 0)
                if not self._send_command(motor_id, speed):
                    return False
                self._speeds[motor_id] = speed
                self._running[motor_id] = speed > 0
                time.sleep(dwell_time)

            self._speeds[motor_id] = 0
            self._running[motor_id] = False
            logger.info(f"Ramp test complete — {name}")
            return True

        except Exception as e:
            logger.error(f"Ramp test failed — {name}: {e}", exc_info=True)
            return False

    # ========================================================================
    # SECTION 4: STATUS QUERIES
    # ========================================================================

    def is_ready(self) -> bool:
        """Return True if the controller is ready to accept commands."""
        return self.initialized

    def get_motor_speed(self, motor_id: int) -> int:
        """Return the last-sent speed for a motor (0 if never set)."""
        return self._speeds.get(motor_id, 0)

    def is_motor_running(self, motor_id: int) -> bool:
        """Return True if the motor is currently running (speed > 0)."""
        return self._running.get(motor_id, False)

    def get_status(self) -> Dict:
        """Return a full status snapshot for all motors."""
        with self.lock:
            return {
                'initialized': self.initialized,
                'simulation_mode': self.simulation_mode,
                'port': self.port,
                'num_motors': self.num_motors,
                'motors': {
                    i: {
                        'name': self.motor_names.get(i, f"Motor {i}"),
                        'speed': self._speeds.get(i, 0),
                        'running': self._running.get(i, False),
                    }
                    for i in range(self.num_motors)
                },
            }

    # ========================================================================
    # SECTION 5: HELPERS
    # ========================================================================

    def _valid_motor(self, motor_id: int) -> bool:
        if not (0 <= motor_id < self.num_motors):
            logger.error(f"Invalid motor_id {motor_id} (num_motors={self.num_motors})")
            return False
        return True

    def _log_init_status(self):
        logger.info("=" * 50)
        logger.info("CONVEYOR MOTOR CONTROLLER INITIALIZED")
        logger.info("=" * 50)
        mode = "SIMULATION" if self.simulation_mode else "HARDWARE (WRITE-ONLY)"
        logger.info(f"Mode: {mode}   Ready: {self.initialized}")
        if not self.simulation_mode:
            logger.info(f"Port: {self.port} @ {self.baud_rate} baud")
        logger.info(f"Motors ({self.num_motors}):")
        for i in range(self.num_motors):
            name = self.motor_names.get(i, f"Motor {i}")
            default = self.default_speeds.get(i, 128)
            logger.info(f"  [{i}] {name}  default={default}")
        logger.info(f"Speed range: {self.min_speed}-{self.max_speed}")
        logger.info("=" * 50)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_conveyor_motor_controller(
        config_manager: EnhancedConfigManager) -> ConveyorMotorController:
    """
    Create a ConveyorMotorController from the active configuration.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Initialized ConveyorMotorController
    """
    return ConveyorMotorController(config_manager)
