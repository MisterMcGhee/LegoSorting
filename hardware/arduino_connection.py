# hardware/arduino_connection.py
"""
arduino_connection.py - Shared serial transport for all Arduino commands

This module owns the single serial connection to the Arduino and exposes a
simple send_command(letter, value) interface.  Both the servo module and the
motor module are peer services that use this shared transport — neither of
them opens the port directly.

COMMUNICATION MODEL:
- Write-only: Commands are sent to the Arduino, no responses are expected.
- Protocol: "LETTER,VALUE\\n"  (e.g. "A,90\\n", "B,153\\n")
- Baud rate must match the Arduino sketch (57600).

CONFIGURATION:
Connection settings are read from the 'arduino_servo' section of config.json
so that existing config files require no migration.
"""

import time
import logging
import threading
import serial
from typing import Dict, Optional
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class ArduinoConnection:
    """
    Shared serial transport layer for all Arduino commands.

    Opens the serial port once and provides send_command() to any module
    that needs to talk to the Arduino.  Thread-safe.
    """

    def __init__(self, config_manager: EnhancedConfigManager):
        arduino_config = config_manager.get_module_config(ModuleConfig.ARDUINO_SERVO.value)

        self.port = arduino_config['port']
        self.baud_rate = arduino_config['baud_rate']
        self.timeout = arduino_config['timeout']
        self.connection_retries = arduino_config['connection_retries']
        self.retry_delay = arduino_config['retry_delay']
        self.simulation_mode = arduino_config['simulation_mode']

        self._serial: Optional[serial.Serial] = None
        self._connected = False
        self.lock = threading.RLock()

        if self.simulation_mode:
            logger.info("=== ARDUINO CONNECTION: SIMULATION MODE ===")
            self._connected = True
        else:
            logger.info("=== ARDUINO CONNECTION: HARDWARE MODE ===")
            self.connect()

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Open the serial port with retry logic.

        Returns:
            True if connection succeeded, False otherwise.
        """
        with self.lock:
            logger.info("=" * 70)
            logger.info("ARDUINO CONNECTION: STARTING")
            logger.info(f"  Port: {self.port}  Baud: {self.baud_rate}  Retries: {self.connection_retries}")
            logger.info("=" * 70)

            for attempt in range(self.connection_retries):
                try:
                    logger.info(f"  Attempt {attempt + 1}/{self.connection_retries}")
                    self._serial = serial.Serial(
                        port=self.port,
                        baudrate=self.baud_rate,
                        timeout=self.timeout,
                        write_timeout=self.timeout
                    )
                    logger.info("  Waiting for Arduino reset (2.5 s)...")
                    time.sleep(2.5)
                    self._serial.reset_input_buffer()
                    self._serial.reset_output_buffer()
                    self._connected = True
                    logger.info("=" * 70)
                    logger.info("ARDUINO CONNECTION: SUCCESS")
                    logger.info("=" * 70)
                    return True

                except serial.SerialException as e:
                    logger.error(f"  Attempt {attempt + 1} failed: {e}")
                    self._close_serial()

                except Exception as e:
                    logger.error(f"  Unexpected error on attempt {attempt + 1}: {e}")
                    self._close_serial()

                if attempt < self.connection_retries - 1:
                    logger.info(f"  Retrying in {self.retry_delay} s...")
                    time.sleep(self.retry_delay)

            logger.error("=" * 70)
            logger.error("ARDUINO CONNECTION: ALL ATTEMPTS FAILED")
            logger.error(f"  Port: {self.port}")
            logger.error("  Troubleshooting:")
            logger.error("    1. Check Arduino is plugged in")
            logger.error("    2. Verify correct port in config")
            logger.error("    3. Close Arduino IDE / other serial programs")
            logger.error("    4. Run: ls -l /dev/cu.*")
            logger.error("=" * 70)
            self._connected = False
            return False

    def disconnect(self):
        """Close the serial port safely."""
        with self.lock:
            if self._serial:
                logger.info("Closing Arduino serial connection...")
                self._close_serial()
                self._connected = False
                logger.info("Arduino connection closed")

    def _close_serial(self):
        """Internal helper — closes serial without acquiring lock."""
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    # -------------------------------------------------------------------------
    # Command sending
    # -------------------------------------------------------------------------

    def send_command(self, letter: str, value: int) -> bool:
        """
        Send a single LETTER,VALUE command to the Arduino.

        Args:
            letter: Single uppercase letter identifying the target device
                    (A=servo, B=conveyor, C=feeder_c, D=feeder_d, E=feeder_e)
            value:  Integer value (angle for servo; PWM -255..255 for motors)

        Returns:
            True if the command was sent (or simulated), False on error.
        """
        with self.lock:
            if self.simulation_mode:
                logger.debug(f"[SIMULATION] Command: {letter},{value}")
                return True

            if not self._serial:
                logger.error("No Arduino serial connection — cannot send command")
                return False

            try:
                command = f"{letter},{value}\n"
                self._serial.write(command.encode())
                self._serial.flush()
                logger.debug(f"Sent: {command.strip()}")
                return True

            except Exception as e:
                logger.error(f"Failed to send command {letter},{value}: {e}")
                return False

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def is_connected(self) -> bool:
        """True if serial port is open (or in simulation mode)."""
        return self._connected

    def is_simulation(self) -> bool:
        """True if running without real hardware."""
        return self.simulation_mode

    def get_status(self) -> Dict:
        """Return a summary dict for logging / GUI display."""
        return {
            'port': self.port,
            'baud_rate': self.baud_rate,
            'connected': self._connected,
            'simulation_mode': self.simulation_mode,
        }


# =============================================================================
# Factory
# =============================================================================

def create_arduino_connection(config_manager: EnhancedConfigManager) -> ArduinoConnection:
    """Create and return a connected ArduinoConnection instance."""
    return ArduinoConnection(config_manager)
