# hardware/arduino_motor_module.py
"""
arduino_motor_module.py - DC motor control for feeders and conveyor

This module controls the four DC motors connected to the two L298N motor
driver boards.  It is a peer of arduino_servo_module — both use the shared
ArduinoConnection transport to send commands to the same Arduino.

MOTOR LAYOUT (from LegoSorter_MotorControl.ino):
    Letter  Role            L298N   EN pin
    ──────  ─────────────   ─────   ──────
    B       Conveyor belt   #2      EN10
    C       Rotary feeder C #1      EN9
    D       Rotary feeder D #1      EN6
    E       Future feeder   #2      EN5

COMMAND PROTOCOL:
    Sends "LETTER,VALUE\\n" where VALUE is a PWM duty cycle in [-255, 255].
    All sorter motors run forward-only, so values are always 0..255.

SPEED MODEL:
    Speeds are stored and configured as percentages (0–100).
    - 0           → motor stopped (PWM 0)
    - 1–29        → clamped to min_duty_pct (default 30) to prevent stall
    - 30–100      → mapped linearly to PWM 77–255

CONFIGURATION:
    All settings stored in the 'arduino_motor' section of config.json.

RESPONSIBILITIES:
    - Starting / stopping all motors
    - Adjusting individual motor speeds at runtime
    - Enforcing the minimum duty floor to prevent stall
    - Reloading speeds from config (for GUI-driven updates)

DOES NOT:
    - Own the serial connection (that's arduino_connection.py)
    - Make sorting decisions
    - Coordinate other hardware (that's hardware_coordinator)
"""

import logging
import threading
from typing import Dict
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig
from hardware.arduino_connection import ArduinoConnection

logger = logging.getLogger(__name__)

# Map of Arduino motor letters to their config key and display name
MOTORS: Dict[str, tuple] = {
    'B': ('conveyor_speed_pct',  'Conveyor'),
    'C': ('feeder_c_speed_pct',  'Feeder C'),
    'D': ('feeder_d_speed_pct',  'Feeder D'),
    'E': ('feeder_e_speed_pct',  'Feeder E (future)'),
}


class ArduinoMotorController:
    """
    Controls DC motors (feeders and conveyor) via the shared Arduino connection.

    Motor speeds are set once at startup from config and can be adjusted at
    runtime without restarting the application.
    """

    def __init__(self, config_manager: EnhancedConfigManager, connection: ArduinoConnection):
        """
        Initialize motor controller.

        Args:
            config_manager: Enhanced configuration manager instance
            connection:     Shared ArduinoConnection (serial transport)
        """
        self.config_manager = config_manager
        self.connection = connection
        self.lock = threading.RLock()

        # =====================================================================
        # LOAD CONFIGURATION
        # =====================================================================
        motor_config = config_manager.get_module_config(ModuleConfig.ARDUINO_MOTOR.value)

        self.min_duty_pct: int = int(motor_config.get('min_duty_pct', 30))

        # In-memory speed table {letter: pct}
        self._speeds: Dict[str, int] = {}
        for letter, (config_key, _) in MOTORS.items():
            self._speeds[letter] = int(motor_config.get(config_key, 60))

        # =====================================================================
        # STATE
        # =====================================================================
        self._running = False  # True while motors are spinning
        self._initialized = connection.is_connected()

        self._log_initialization_status()

    # -------------------------------------------------------------------------
    # Speed helpers
    # -------------------------------------------------------------------------

    def _clamp(self, pct: int) -> int:
        """
        Apply the stall-prevention floor.

        - 0 is always valid (full stop).
        - Any non-zero value below min_duty_pct is raised to min_duty_pct.
        - Values above 100 are capped at 100.
        """
        if pct <= 0:
            return 0
        clamped = max(pct, self.min_duty_pct)
        if clamped > 100:
            clamped = 100
        return clamped

    def _pct_to_pwm(self, pct: int) -> int:
        """Convert a (clamped) percentage to a PWM duty value 0–255."""
        return int(self._clamp(pct) / 100.0 * 255)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start_all(self) -> bool:
        """
        Send configured speeds to all motors.

        Returns:
            True if all commands were sent successfully.
        """
        with self.lock:
            if not self._initialized:
                logger.error("Motor controller not initialized — cannot start motors")
                return False

            logger.info("=" * 60)
            logger.info("STARTING ALL MOTORS")
            logger.info("=" * 60)

            all_ok = True
            for letter, (config_key, name) in MOTORS.items():
                pct = self._speeds[letter]
                clamped = self._clamp(pct)
                pwm = self._pct_to_pwm(pct)

                if 0 < pct < self.min_duty_pct:
                    logger.warning(
                        f"  {name} ({letter}): requested {pct}% is below floor "
                        f"{self.min_duty_pct}% — clamped to {clamped}% (PWM {pwm})"
                    )

                ok = self.connection.send_command(letter, pwm)
                if ok:
                    logger.info(f"  ✓ {name} ({letter}): {clamped}% → PWM {pwm}")
                else:
                    logger.error(f"  ✗ {name} ({letter}): failed to send command")
                    all_ok = False

            self._running = all_ok
            logger.info("=" * 60)
            return all_ok

    def stop_all(self) -> bool:
        """
        Send stop command (PWM 0) to all motors.

        Returns:
            True if all stop commands were sent successfully.
        """
        with self.lock:
            logger.info("Stopping all motors")
            all_ok = True
            for letter, (_, name) in MOTORS.items():
                ok = self.connection.send_command(letter, 0)
                if ok:
                    logger.info(f"  ✓ {name} ({letter}): stopped")
                else:
                    logger.error(f"  ✗ {name} ({letter}): stop command failed")
                    all_ok = False

            self._running = False
            return all_ok

    def set_motor_speed(self, letter: str, pct: int) -> bool:
        """
        Adjust a single motor speed at runtime.

        Does NOT save to config — use the config GUI for persistent changes.

        Args:
            letter: Motor identifier ('B', 'C', 'D', or 'E')
            pct:    Speed percentage (0 = stop, 30–100 = running)

        Returns:
            True if command sent successfully
        """
        with self.lock:
            letter = letter.upper()
            if letter not in MOTORS:
                logger.error(f"Unknown motor letter '{letter}' — valid: {list(MOTORS)}")
                return False

            if not self._initialized:
                logger.error("Motor controller not initialized")
                return False

            _, name = MOTORS[letter]
            clamped = self._clamp(pct)
            pwm = self._pct_to_pwm(pct)

            if 0 < pct < self.min_duty_pct:
                logger.warning(
                    f"{name} ({letter}): requested {pct}% below floor "
                    f"{self.min_duty_pct}% — clamped to {clamped}%"
                )

            ok = self.connection.send_command(letter, pwm)
            if ok:
                self._speeds[letter] = pct  # record the user-intent value
                logger.info(f"{name} ({letter}): set to {clamped}% (PWM {pwm})")
            else:
                logger.error(f"{name} ({letter}): failed to send speed command")

            return ok

    def reload_speeds_from_config(self) -> bool:
        """
        Re-read motor speeds from config and re-apply to running motors.

        Intended to be called after the user edits speeds in the setup GUI.

        Returns:
            True if speeds reloaded and applied successfully
        """
        with self.lock:
            try:
                motor_config = self.config_manager.get_module_config(
                    ModuleConfig.ARDUINO_MOTOR.value
                )
                old = self._speeds.copy()
                for letter, (config_key, _) in MOTORS.items():
                    self._speeds[letter] = int(motor_config.get(config_key, 60))
                self.min_duty_pct = int(motor_config.get('min_duty_pct', 30))
                logger.info(f"Motor speeds reloaded from config: {old} → {self._speeds}")

                if self._running:
                    return self.start_all()
                return True

            except Exception as e:
                logger.error(f"Failed to reload motor speeds: {e}")
                return False

    # -------------------------------------------------------------------------
    # Status / info
    # -------------------------------------------------------------------------

    def get_motor_speeds(self) -> Dict[str, int]:
        """Return the current in-memory speed percentage for each motor."""
        with self.lock:
            return self._speeds.copy()

    def is_ready(self) -> bool:
        """True if the shared Arduino connection is up."""
        return self._initialized and self.connection.is_connected()

    def get_statistics(self) -> Dict:
        """Return status summary for logging or GUI display."""
        with self.lock:
            speeds_display = {}
            for letter, (_, name) in MOTORS.items():
                pct = self._speeds[letter]
                speeds_display[letter] = {
                    'name': name,
                    'requested_pct': pct,
                    'effective_pct': self._clamp(pct),
                    'pwm': self._pct_to_pwm(pct),
                }
            return {
                'initialized': self._initialized,
                'running': self._running,
                'min_duty_pct': self.min_duty_pct,
                'motors': speeds_display,
                'connection': self.connection.get_status(),
            }

    def _log_initialization_status(self):
        """Log motor controller startup summary."""
        conn = self.connection.get_status()
        logger.info("=" * 60)
        logger.info("ARDUINO MOTOR CONTROLLER INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Mode: {'SIMULATION' if conn['simulation_mode'] else 'HARDWARE'}")
        logger.info(f"Ready: {self._initialized}")
        logger.info(f"Min duty floor: {self.min_duty_pct}%")
        for letter, (_, name) in MOTORS.items():
            pct = self._speeds[letter]
            logger.info(f"  {name} ({letter}): {pct}% configured → PWM {self._pct_to_pwm(pct)}")
        logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def release(self):
        """Stop all motors and mark as not initialized."""
        logger.info("Releasing Arduino motor controller")
        self.stop_all()
        self._initialized = False
        logger.info("Arduino motor controller released")


# =============================================================================
# Factory
# =============================================================================

def create_arduino_motor_controller(
    config_manager: EnhancedConfigManager,
    connection: ArduinoConnection
) -> ArduinoMotorController:
    """
    Factory function to create an ArduinoMotorController instance.

    Args:
        config_manager: Enhanced configuration manager
        connection:     Shared ArduinoConnection instance

    Returns:
        Initialized ArduinoMotorController instance
    """
    return ArduinoMotorController(config_manager, connection)
