"""
motor_tab.py - Conveyor motor configuration and test tab

FILE LOCATION: GUI/config_tabs/motor_tab.py

Provides connection settings, motor hardware parameters, and an interactive
test panel for the conveyor belt motor.  Mirrors the design language of
hardware_tab.py (servo tab) for a consistent operator experience.

Features:
- Arduino/motor-controller connection settings (port, baud rate, timeouts)
- Simulation mode for development without physical hardware
- Motor hardware parameters (speed range, default speed, ramp rate)
- Interactive speed slider with live PWM value readout
- Start / Stop controls (equivalent to servo Home button)
- Ramp test (equivalent to servo "Test All Positions" sweep)
- Connection status monitoring

Configuration Mapping:
    {
        "motor": {
            "port": "COM4",
            "baud_rate": 57600,
            "timeout": 1.0,
            "connection_retries": 3,
            "retry_delay": 1.0,
            "simulation_mode": false,
            "default_speed": 128,
            "min_speed": 0,
            "max_speed": 255,
            "ramp_rate": 32
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QComboBox, QPushButton, QSpinBox,
                             QLabel, QSlider, QMessageBox, QCheckBox,
                             QDoubleSpinBox, QProgressDialog)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont
from typing import Dict, Any, Optional
import logging
import serial.tools.list_ports

from GUI.config_tabs.base_tab import BaseConfigTab
from enhanced_config_manager import ModuleConfig

logger = logging.getLogger(__name__)


class MotorConfigTab(BaseConfigTab):
    """
    Configuration tab for the conveyor belt motor controller.

    Allows operators to:
    - Configure the serial connection to the motor controller Arduino
    - Enable / disable simulation mode
    - Set motor speed parameters (range, default, ramp rate)
    - Live-test motor speed via an interactive slider
    - Run a ramp test (0 → max → 0) to verify operation
    - Monitor connection status

    Signals:
        motor_status_changed(bool): Emitted when the motor ready state changes
    """

    motor_status_changed = pyqtSignal(bool)  # is_ready

    def __init__(self, config_manager, motor_module=None, parent=None):
        """
        Initialize motor configuration tab.

        Args:
            config_manager: Configuration manager instance
            motor_module: Optional ConveyorMotorController reference for live testing
            parent: Parent widget
        """
        self.motor_module = motor_module

        super().__init__(config_manager, parent)

        self.init_ui()
        self.load_config()

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================

    def get_module_name(self) -> str:
        return ModuleConfig.MOTOR.value

    def init_ui(self):
        """Build the motor configuration UI."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # ── Title ─────────────────────────────────────────────────────────────
        title = QLabel("Motor Configuration")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #2E86AB;
                color: white;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(title)

        # ── Main content ──────────────────────────────────────────────────────
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        # Left panel: connection + motor hardware settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._create_connection_group())
        left_layout.addWidget(self._create_motor_hardware_group())
        left_layout.addStretch()
        content_layout.addWidget(left_panel, stretch=1)

        # Right panel: interactive test controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self._create_test_group(), stretch=1)
        content_layout.addWidget(right_panel, stretch=1)

        # Bottom info strip
        main_layout.addWidget(self._create_info_panel())

        self.logger.info("Motor tab UI initialized")

    # ========================================================================
    # GROUP BUILDERS
    # ========================================================================

    def _create_connection_group(self) -> QGroupBox:
        """Arduino / motor-controller connection settings (mirrors servo tab)."""
        group = QGroupBox("Motor Controller Connection")
        layout = QFormLayout()

        # Simulation mode
        self.simulation_check = QCheckBox("Simulation Mode")
        self.simulation_check.setToolTip(
            "Enable simulation mode for testing without physical motor hardware"
        )
        self.simulation_check.stateChanged.connect(self._on_simulation_mode_changed)
        layout.addRow("Mode:", self.simulation_check)

        # Serial port
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setToolTip("Select serial port for the motor controller")
        self.port_combo.currentTextChanged.connect(self.mark_modified)
        layout.addRow("Serial Port:", self.port_combo)

        scan_btn = QPushButton("🔍 Scan Ports")
        scan_btn.setToolTip("Scan for available serial ports")
        scan_btn.clicked.connect(self._scan_ports)
        layout.addRow("", scan_btn)

        # Baud rate
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_combo.setCurrentText("57600")
        self.baud_combo.setToolTip("Serial communication baud rate")
        self.baud_combo.currentTextChanged.connect(self.mark_modified)
        layout.addRow("Baud Rate:", self.baud_combo)

        # Timeout
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 10.0)
        self.timeout_spin.setSingleStep(0.1)
        self.timeout_spin.setValue(1.0)
        self.timeout_spin.setDecimals(1)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setToolTip("Serial communication timeout")
        self.timeout_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Timeout:", self.timeout_spin)

        # Connection retries
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 10)
        self.retry_spin.setValue(3)
        self.retry_spin.setToolTip("Number of connection retry attempts")
        self.retry_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Retries:", self.retry_spin)

        # Retry delay
        self.retry_delay_spin = QDoubleSpinBox()
        self.retry_delay_spin.setRange(0.1, 5.0)
        self.retry_delay_spin.setSingleStep(0.1)
        self.retry_delay_spin.setValue(1.0)
        self.retry_delay_spin.setDecimals(1)
        self.retry_delay_spin.setSuffix(" s")
        self.retry_delay_spin.setToolTip("Delay between retry attempts")
        self.retry_delay_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Retry Delay:", self.retry_delay_spin)

        # Connection status + test button
        self.connection_status = QLabel("Status: Unknown")
        self.connection_status.setStyleSheet("color: gray; font-weight: bold;")
        layout.addRow("", self.connection_status)

        self.test_connection_btn = QPushButton("🔌 Test Connection")
        self.test_connection_btn.setToolTip("Test motor controller connection")
        self.test_connection_btn.clicked.connect(self._test_connection)
        self.test_connection_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #2980B9; }
        """)
        layout.addRow("", self.test_connection_btn)

        group.setLayout(layout)
        return group

    def _create_motor_hardware_group(self) -> QGroupBox:
        """Motor hardware parameters (mirrors servo hardware group)."""
        group = QGroupBox("Motor Hardware")
        layout = QFormLayout()

        # Default speed
        self.default_speed_spin = QSpinBox()
        self.default_speed_spin.setRange(0, 255)
        self.default_speed_spin.setValue(128)
        self.default_speed_spin.setToolTip(
            "Default running speed (0-255 PWM).\n"
            "Used by Start button and initial belt run."
        )
        self.default_speed_spin.valueChanged.connect(self.mark_modified)
        self.default_speed_spin.valueChanged.connect(self._sync_slider_from_default)
        layout.addRow("Default Speed:", self.default_speed_spin)

        # Min speed
        self.min_speed_spin = QSpinBox()
        self.min_speed_spin.setRange(0, 254)
        self.min_speed_spin.setValue(0)
        self.min_speed_spin.setToolTip("Minimum speed cap (0 = allow full stop)")
        self.min_speed_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Min Speed:", self.min_speed_spin)

        # Max speed
        self.max_speed_spin = QSpinBox()
        self.max_speed_spin.setRange(1, 255)
        self.max_speed_spin.setValue(255)
        self.max_speed_spin.setToolTip("Maximum speed cap (255 = full PWM duty cycle)")
        self.max_speed_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Max Speed:", self.max_speed_spin)

        # Ramp rate (step size per ramp increment)
        self.ramp_rate_spin = QSpinBox()
        self.ramp_rate_spin.setRange(1, 128)
        self.ramp_rate_spin.setValue(32)
        self.ramp_rate_spin.setToolTip(
            "Speed step size for ramp test.\n"
            "Smaller values = smoother ramp, more steps."
        )
        self.ramp_rate_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Ramp Rate:", self.ramp_rate_spin)

        group.setLayout(layout)
        return group

    def _create_test_group(self) -> QGroupBox:
        """
        Interactive motor test controls.

        Equivalent to the servo tab's bin-positions panel: a focused test
        area where the operator can exercise the hardware without leaving the
        configuration screen.
        """
        group = QGroupBox("Motor Test Controls")
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # ── Speed slider row ──────────────────────────────────────────────────
        slider_group = QGroupBox("Speed Control")
        slider_layout = QVBoxLayout(slider_group)

        # Numeric readout above slider
        speed_readout_row = QHBoxLayout()
        speed_readout_row.addWidget(QLabel("Speed:"))
        self.speed_value_label = QLabel("128")
        self.speed_value_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.speed_value_label.setAlignment(Qt.AlignCenter)
        self.speed_value_label.setStyleSheet("""
            QLabel {
                color: #2E86AB;
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px 12px;
                min-width: 60px;
            }
        """)
        speed_readout_row.addWidget(self.speed_value_label)
        speed_readout_row.addStretch()
        slider_layout.addLayout(speed_readout_row)

        # Slider
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 255)
        self.speed_slider.setValue(128)
        self.speed_slider.setTickInterval(32)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setToolTip(
            "Set motor speed (0 = stop, 255 = full speed)\n"
            "If motor is running, speed updates immediately."
        )
        self.speed_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.speed_slider)

        # Tick labels (0 / 64 / 128 / 192 / 255)
        tick_row = QHBoxLayout()
        for label in ["0", "64", "128", "192", "255"]:
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #888; font-size: 9px;")
            lbl.setAlignment(Qt.AlignCenter)
            tick_row.addWidget(lbl)
        slider_layout.addLayout(tick_row)

        layout.addWidget(slider_group)

        # ── Start / Stop row ──────────────────────────────────────────────────
        run_row = QHBoxLayout()

        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setToolTip("Start motor at the current slider speed")
        self.start_btn.clicked.connect(self._start_motor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #229954; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        run_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setToolTip("Stop the motor immediately")
        self.stop_btn.clicked.connect(self._stop_motor)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #C0392B; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        run_row.addWidget(self.stop_btn)

        layout.addLayout(run_row)

        # ── Ramp test row ─────────────────────────────────────────────────────
        ramp_row = QHBoxLayout()

        self.ramp_test_btn = QPushButton("🔄 Ramp Test")
        self.ramp_test_btn.setToolTip(
            "Run a ramp test: gradually ramp speed from 0 → max → 0.\n"
            "Equivalent to the servo 'Test All Positions' sweep.\n"
            "Motor must not be running when this starts."
        )
        self.ramp_test_btn.clicked.connect(self._run_ramp_test)
        self.ramp_test_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #E67E22; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        ramp_row.addWidget(self.ramp_test_btn)
        ramp_row.addStretch()

        layout.addLayout(ramp_row)

        # ── Status indicator ──────────────────────────────────────────────────
        self.motor_status_label = QLabel("Motor: Stopped")
        self.motor_status_label.setStyleSheet(
            "color: #E74C3C; font-weight: bold; font-size: 12px;"
        )
        layout.addWidget(self.motor_status_label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def _create_info_panel(self) -> QWidget:
        """Bottom tip strip (mirrors servo tab)."""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        info = QLabel(
            "💡 Tip: Speed is a PWM value (0-255). "
            "Use Ramp Test to find optimal operating speed before saving."
        )
        info.setStyleSheet("color: #7F8C8D; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)
        return panel

    # ========================================================================
    # SERIAL PORT SCANNING
    # ========================================================================

    def _scan_ports(self):
        """Populate the port combo with currently available serial ports."""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()

        if not ports:
            self.port_combo.addItem("No ports found")
            QMessageBox.information(
                self, "No Ports Found",
                "No serial ports detected.\n\nMake sure the motor controller is connected."
            )
            return

        for port in ports:
            label = port.device
            if port.description and port.description != port.device:
                label += f" - {port.description}"
            self.port_combo.addItem(label)

        QMessageBox.information(
            self, "Ports Scanned",
            f"Found {len(ports)} serial port(s):\n\n" +
            "\n".join([p.device for p in ports])
        )

    # ========================================================================
    # CONNECTION TESTING
    # ========================================================================

    def _test_connection(self):
        """Test the motor controller serial connection."""
        if self.simulation_check.isChecked():
            self.connection_status.setText("Status: Simulation Mode")
            self.connection_status.setStyleSheet("color: #F39C12; font-weight: bold;")
            QMessageBox.information(
                self, "Simulation Mode",
                "Simulation mode is enabled.\nNo hardware connection required."
            )
            return

        if not self.motor_module:
            self.connection_status.setText("Status: No Module")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.warning(
                self, "Module Not Available",
                "Motor module not available for testing.\n\n"
                "Module will be created when configuration is applied."
            )
            return

        if self.motor_module.is_ready():
            self.connection_status.setText("Status: Connected")
            self.connection_status.setStyleSheet("color: #27AE60; font-weight: bold;")
            status = self.motor_module.get_status()
            QMessageBox.information(
                self, "Connection Successful",
                f"✓ Motor controller connected!\n\n"
                f"Port: {status['port']}\n"
                f"Speed range: {status['speed_range'][0]}-{status['speed_range'][1]}"
            )
        else:
            self.connection_status.setText("Status: Not Connected")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.warning(
                self, "Connection Failed",
                "Failed to connect to motor controller.\n\n"
                "Check:\n"
                "• Device is connected\n"
                "• Correct port is selected\n"
                "• No other program is using the port"
            )

    def _on_simulation_mode_changed(self):
        """Enable/disable hardware controls based on simulation mode."""
        is_sim = self.simulation_check.isChecked()
        self.port_combo.setEnabled(not is_sim)
        self.baud_combo.setEnabled(not is_sim)
        self.test_connection_btn.setEnabled(not is_sim)

        if is_sim:
            self.connection_status.setText("Status: Simulation Mode")
            self.connection_status.setStyleSheet("color: #F39C12; font-weight: bold;")
        else:
            self.connection_status.setText("Status: Hardware Mode")
            self.connection_status.setStyleSheet("color: gray; font-weight: bold;")

        self.mark_modified()

    # ========================================================================
    # MOTOR TEST ACTIONS
    # ========================================================================

    def _check_motor_ready(self, action: str = "test") -> bool:
        """Return True if the motor module is available and ready."""
        if not self.motor_module:
            QMessageBox.warning(
                self, "Module Not Available",
                f"Motor module not available for {action}.\n\n"
                "Module will be created when configuration is applied."
            )
            return False

        if not self.motor_module.is_ready():
            QMessageBox.warning(
                self, "Hardware Not Ready",
                "Motor controller is not connected or ready.\n\n"
                "Enable simulation mode or connect the hardware."
            )
            return False

        return True

    def _start_motor(self):
        """Start the motor at the current slider speed."""
        if not self._check_motor_ready("start"):
            return

        speed = self.speed_slider.value()
        success = self.motor_module.set_speed(speed)

        if success:
            self._update_motor_status(running=speed > 0, speed=speed)
        else:
            QMessageBox.warning(
                self, "Start Failed",
                f"Failed to start motor at speed {speed}.\n\nCheck logs for details."
            )

    def _stop_motor(self):
        """Stop the motor."""
        if not self._check_motor_ready("stop"):
            return

        success = self.motor_module.stop()
        if success:
            self._update_motor_status(running=False, speed=0)
        else:
            QMessageBox.warning(
                self, "Stop Failed",
                "Failed to stop motor.\n\nCheck logs for details."
            )

    def _run_ramp_test(self):
        """Run the ramp test (0 → max → 0)."""
        if not self._check_motor_ready("ramp test"):
            return

        reply = QMessageBox.question(
            self,
            "Ramp Test",
            "This will ramp the motor from stopped to full speed, then back to stopped.\n\n"
            "Make sure it is safe to run the conveyor belt.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        steps = max(1, self.max_speed_spin.value() // max(1, self.ramp_rate_spin.value()))

        progress = QProgressDialog("Running ramp test...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Motor Ramp Test")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            success = self.motor_module.ramp_test(steps=steps, dwell_time=0.5)
        finally:
            progress.close()

        self._update_motor_status(running=False, speed=0)

        if success:
            QMessageBox.information(
                self, "Ramp Test Complete",
                "✓ Motor ramp test completed successfully!\n\n"
                "Motor has returned to stopped state."
            )
        else:
            QMessageBox.warning(
                self, "Ramp Test Issues",
                "Some steps during the ramp test encountered errors.\n\n"
                "Check logs for details."
            )

    def _update_motor_status(self, running: bool, speed: int):
        """Refresh the on-screen motor status indicator."""
        if running:
            self.motor_status_label.setText(f"Motor: Running  (speed {speed})")
            self.motor_status_label.setStyleSheet(
                "color: #27AE60; font-weight: bold; font-size: 12px;"
            )
        else:
            self.motor_status_label.setText("Motor: Stopped")
            self.motor_status_label.setStyleSheet(
                "color: #E74C3C; font-weight: bold; font-size: 12px;"
            )

    def _on_slider_changed(self, value: int):
        """Update numeric readout when slider moves; push speed if running."""
        self.speed_value_label.setText(str(value))

        # Live update if motor is already running
        if (self.motor_module and self.motor_module.is_ready()
                and self.motor_module.is_running):
            self.motor_module.set_speed(value)
            self._update_motor_status(running=value > 0, speed=value)

    def _sync_slider_from_default(self, value: int):
        """Keep the slider in sync when the default speed spinbox changes."""
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(value)
        self.speed_slider.blockSignals(False)
        self.speed_value_label.setText(str(value))

    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================

    def load_config(self) -> bool:
        """Load motor configuration and populate all UI fields."""
        try:
            config = self.get_module_config(self.get_module_name())

            if not config:
                self.logger.warning("No motor configuration found, using defaults")
                self.clear_modified()
                return True

            # Block signals during bulk population
            widgets = [
                self.simulation_check, self.port_combo, self.baud_combo,
                self.timeout_spin, self.retry_spin, self.retry_delay_spin,
                self.default_speed_spin, self.min_speed_spin,
                self.max_speed_spin, self.ramp_rate_spin,
            ]
            for w in widgets:
                w.blockSignals(True)

            self.simulation_check.setChecked(config.get("simulation_mode", False))

            port = config.get("port", "")
            if port:
                self.port_combo.setCurrentText(port)

            self.baud_combo.setCurrentText(str(config.get("baud_rate", 57600)))
            self.timeout_spin.setValue(config.get("timeout", 1.0))
            self.retry_spin.setValue(config.get("connection_retries", 3))
            self.retry_delay_spin.setValue(config.get("retry_delay", 1.0))

            default_speed = config.get("default_speed", 128)
            self.default_speed_spin.setValue(default_speed)
            self.min_speed_spin.setValue(config.get("min_speed", 0))
            self.max_speed_spin.setValue(config.get("max_speed", 255))
            self.ramp_rate_spin.setValue(config.get("ramp_rate", 32))

            for w in widgets:
                w.blockSignals(False)

            # Sync slider to loaded default speed
            self.speed_slider.setValue(default_speed)
            self.speed_value_label.setText(str(default_speed))

            # Reflect simulation state in UI
            self._on_simulation_mode_changed()

            self.clear_modified()
            self.logger.info("Motor configuration loaded")
            return True

        except Exception as e:
            self.logger.error(f"Error loading motor config: {e}", exc_info=True)
            return False

    def save_config(self) -> bool:
        """Validate and save motor configuration to config_manager."""
        if not self.validate():
            return False

        self.config_manager.update_module_config(
            self.get_module_name(),
            self.get_config()
        )
        self.clear_modified()
        self.config_changed.emit(self.get_module_name(), self.get_config())
        self.logger.info("Motor configuration saved")
        return True

    def get_config(self) -> Dict[str, Any]:
        """Return the current UI values as a configuration dictionary."""
        port_text = self.port_combo.currentText()
        # Strip description suffix that scan_ports appends (e.g. "COM4 - Arduino")
        port = port_text.split(" - ")[0].strip() if port_text else ""

        return {
            "port": port,
            "baud_rate": int(self.baud_combo.currentText()),
            "timeout": self.timeout_spin.value(),
            "connection_retries": self.retry_spin.value(),
            "retry_delay": self.retry_delay_spin.value(),
            "simulation_mode": self.simulation_check.isChecked(),
            "default_speed": self.default_speed_spin.value(),
            "min_speed": self.min_speed_spin.value(),
            "max_speed": self.max_speed_spin.value(),
            "ramp_rate": self.ramp_rate_spin.value(),
        }

    def validate(self) -> bool:
        """Validate motor configuration values."""
        if self.min_speed_spin.value() >= self.max_speed_spin.value():
            QMessageBox.warning(
                self, "Validation Error",
                "Min Speed must be less than Max Speed."
            )
            self.validation_failed.emit("Min Speed must be less than Max Speed")
            return False

        default = self.default_speed_spin.value()
        lo = self.min_speed_spin.value()
        hi = self.max_speed_spin.value()
        if not (lo <= default <= hi):
            QMessageBox.warning(
                self, "Validation Error",
                f"Default Speed ({default}) must be between "
                f"Min Speed ({lo}) and Max Speed ({hi})."
            )
            self.validation_failed.emit("Default Speed out of range")
            return False

        return True
