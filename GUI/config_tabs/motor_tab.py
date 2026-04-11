"""
motor_tab.py - Multi-motor configuration and test tab

FILE LOCATION: GUI/config_tabs/motor_tab.py

Provides connection settings and an interactive per-motor control panel.
Each motor gets its own row with a synchronized slider and integer spinbox
so the operator can set speeds independently using either the mouse or the
scroll wheel.

Layout:
    Left panel  — connection settings (port, baud, simulation mode)
    Right panel — one row per motor + global action buttons

Per-motor row:
    [Motor Name]  [━━━━━━━━━━━ Slider ━━━━━━━━━━━]  [Spinbox ▲▼]  [■ Stop]
                  0                              255    (0-255)

Global actions (bottom of right panel):
    [■ Stop All]   [🔄 Ramp Test ▾ motor selector]

Configuration Mapping:
    {
        "motor": {
            "port": "COM4",
            "baud_rate": 57600,
            "timeout": 1.0,
            "connection_retries": 3,
            "retry_delay": 1.0,
            "simulation_mode": false,
            "num_motors": 2,
            "motor_names": {"0": "Conveyor Belt", "1": "Feeder"},
            "default_speeds": {"0": 128, "1": 100},
            "min_speed": 0,
            "max_speed": 255,
            "ramp_rate": 32
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QComboBox, QPushButton, QSpinBox,
                             QLabel, QSlider, QMessageBox, QCheckBox,
                             QDoubleSpinBox, QProgressDialog, QScrollArea,
                             QSizePolicy, QFrame)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont
from typing import Dict, Any, List, Optional
import logging
import serial.tools.list_ports

from GUI.config_tabs.base_tab import BaseConfigTab
from enhanced_config_manager import ModuleConfig

logger = logging.getLogger(__name__)

# ── Styling helpers ─────────────────────────────────────────────────────────

def _btn(bg: str, hover: str, text: str = "white", pad: str = "8px") -> str:
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text};
            padding: {pad};
            font-weight: bold;
            border-radius: 4px;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:disabled {{ background-color: #555; color: #888; }}
    """


class MotorConfigTab(BaseConfigTab):
    """
    Configuration tab for independently controlling N conveyor motors.

    Each motor gets its own slider + spinbox row.  Both controls are
    bidirectionally synchronised: dragging the slider updates the spinbox,
    scrolling the spinbox updates the slider.  Pushing a speed to hardware
    happens immediately when the motor is already running.

    Signals:
        motor_status_changed(bool): Emitted when the controller ready state changes
    """

    motor_status_changed = pyqtSignal(bool)

    def __init__(self, config_manager, motor_module=None, parent=None):
        """
        Args:
            config_manager: Configuration manager instance
            motor_module:   Optional live ConveyorMotorController for testing
            parent:         Parent widget
        """
        self.motor_module = motor_module

        # Per-motor widget references, populated in _build_motor_rows()
        # Each entry: {'slider': QSlider, 'spinbox': QSpinBox, 'status': QLabel}
        self._motor_rows: List[Dict] = []

        super().__init__(config_manager, parent)
        self.init_ui()
        self.load_config()

    # ========================================================================
    # ABSTRACT IMPLEMENTATIONS
    # ========================================================================

    def get_module_name(self) -> str:
        return ModuleConfig.MOTOR.value

    def init_ui(self):
        """Build the two-panel layout."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Title
        title = QLabel("Motor Configuration")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px; font-weight: bold; padding: 10px;
                background-color: #2E86AB; color: white; border-radius: 5px;
            }
        """)
        main_layout.addWidget(title)

        # Content
        content = QHBoxLayout()
        main_layout.addLayout(content, stretch=1)

        # Left: connection + motor count settings
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_connection_group())
        left_layout.addWidget(self._build_speed_limits_group())
        left_layout.addStretch()
        content.addWidget(left, stretch=1)

        # Right: per-motor rows + global controls
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self._motor_panel_group = self._build_motor_panel()
        right_layout.addWidget(self._motor_panel_group, stretch=1)
        right_layout.addWidget(self._build_global_controls())
        content.addWidget(right, stretch=2)

        # Info strip
        main_layout.addWidget(self._build_info_panel())
        self.logger.info("Motor tab UI initialised")

    # ========================================================================
    # LEFT PANEL — CONNECTION
    # ========================================================================

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("Motor Controller Connection")
        layout = QFormLayout()

        self.simulation_check = QCheckBox("Simulation Mode")
        self.simulation_check.setToolTip(
            "Simulate motor commands without physical hardware"
        )
        self.simulation_check.stateChanged.connect(self._on_simulation_changed)
        layout.addRow("Mode:", self.simulation_check)

        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setToolTip("Serial port for the motor controller")
        self.port_combo.currentTextChanged.connect(self.mark_modified)
        layout.addRow("Serial Port:", self.port_combo)

        scan_btn = QPushButton("🔍 Scan Ports")
        scan_btn.clicked.connect(self._scan_ports)
        layout.addRow("", scan_btn)

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_combo.setCurrentText("57600")
        self.baud_combo.currentTextChanged.connect(self.mark_modified)
        layout.addRow("Baud Rate:", self.baud_combo)

        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 10.0)
        self.timeout_spin.setSingleStep(0.1)
        self.timeout_spin.setValue(1.0)
        self.timeout_spin.setDecimals(1)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Timeout:", self.timeout_spin)

        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 10)
        self.retry_spin.setValue(3)
        self.retry_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Retries:", self.retry_spin)

        self.retry_delay_spin = QDoubleSpinBox()
        self.retry_delay_spin.setRange(0.1, 5.0)
        self.retry_delay_spin.setSingleStep(0.1)
        self.retry_delay_spin.setValue(1.0)
        self.retry_delay_spin.setDecimals(1)
        self.retry_delay_spin.setSuffix(" s")
        self.retry_delay_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Retry Delay:", self.retry_delay_spin)

        self.connection_status = QLabel("Status: Unknown")
        self.connection_status.setStyleSheet("color: gray; font-weight: bold;")
        layout.addRow("", self.connection_status)

        self.test_conn_btn = QPushButton("🔌 Test Connection")
        self.test_conn_btn.clicked.connect(self._test_connection)
        self.test_conn_btn.setStyleSheet(_btn("#3498DB", "#2980B9"))
        layout.addRow("", self.test_conn_btn)

        group.setLayout(layout)
        return group

    def _build_speed_limits_group(self) -> QGroupBox:
        """Shared speed limits and ramp rate (apply to all motors)."""
        group = QGroupBox("Speed Limits (all motors)")
        layout = QFormLayout()

        self.min_speed_spin = QSpinBox()
        self.min_speed_spin.setRange(0, 254)
        self.min_speed_spin.setValue(0)
        self.min_speed_spin.setToolTip("Minimum speed cap (0 = allow full stop)")
        self.min_speed_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Min Speed:", self.min_speed_spin)

        self.max_speed_spin = QSpinBox()
        self.max_speed_spin.setRange(1, 255)
        self.max_speed_spin.setValue(255)
        self.max_speed_spin.setToolTip("Maximum speed cap (255 = full PWM)")
        self.max_speed_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Max Speed:", self.max_speed_spin)

        self.ramp_rate_spin = QSpinBox()
        self.ramp_rate_spin.setRange(1, 128)
        self.ramp_rate_spin.setValue(32)
        self.ramp_rate_spin.setToolTip("Speed step size used by the ramp test")
        self.ramp_rate_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Ramp Rate:", self.ramp_rate_spin)

        group.setLayout(layout)
        return group

    # ========================================================================
    # RIGHT PANEL — PER-MOTOR ROWS
    # ========================================================================

    def _build_motor_panel(self) -> QGroupBox:
        """Container for per-motor control rows, with a scroll area."""
        group = QGroupBox("Motor Speed Controls")

        outer = QVBoxLayout()

        # Scrollable area so many motors don't squash the layout
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        self._motor_rows_container = QWidget()
        self._motor_rows_layout = QVBoxLayout(self._motor_rows_container)
        self._motor_rows_layout.setSpacing(8)
        self._motor_rows_layout.addStretch()

        scroll.setWidget(self._motor_rows_container)
        outer.addWidget(scroll)
        group.setLayout(outer)
        return group

    def _build_motor_rows(self, num_motors: int,
                          names: Dict[int, str],
                          defaults: Dict[int, int]):
        """
        Populate the motor panel with one row per motor.

        Clears any existing rows first so this can be called on reload.

        Row layout:
            [Name label]  [━━━━ Slider ━━━━]  [Spinbox]  [■ Stop]
                          status label (Running N / Stopped)
        """
        # Remove old rows
        self._motor_rows.clear()
        while self._motor_rows_layout.count() > 1:  # keep the trailing stretch
            item = self._motor_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx in range(num_motors):
            name = names.get(idx, f"Motor {idx}")
            default_speed = defaults.get(idx, 128)

            # ── Row container ────────────────────────────────────────────────
            row_widget = QGroupBox(name)
            row_widget.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    font-size: 12px;
                    border: 1px solid #555;
                    border-radius: 4px;
                    margin-top: 6px;
                    padding-top: 8px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 4px;
                }
            """)
            row_layout = QVBoxLayout(row_widget)
            row_layout.setSpacing(4)

            # ── Controls row: slider + spinbox + stop ────────────────────────
            controls = QHBoxLayout()

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 255)
            slider.setValue(default_speed)
            slider.setTickInterval(32)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setToolTip(
                f"Drag to set {name} speed (0-255).\n"
                "Updates hardware immediately if motor is running."
            )
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            controls.addWidget(slider, stretch=1)

            spinbox = QSpinBox()
            spinbox.setRange(0, 255)
            spinbox.setValue(default_speed)
            spinbox.setFixedWidth(68)
            spinbox.setAlignment(Qt.AlignCenter)
            spinbox.setToolTip(
                f"Exact speed for {name} (0-255).\n"
                "Scroll the mouse wheel to fine-tune."
            )
            spinbox.setFont(QFont("Arial", 11, QFont.Bold))
            spinbox.setStyleSheet("""
                QSpinBox {
                    background-color: #1e1e1e;
                    color: #2E86AB;
                    border: 1px solid #444;
                    border-radius: 3px;
                    padding: 2px 4px;
                }
            """)
            controls.addWidget(spinbox)

            stop_btn = QPushButton("■ Stop")
            stop_btn.setFixedWidth(64)
            stop_btn.setToolTip(f"Stop {name} immediately")
            stop_btn.setStyleSheet(_btn("#E74C3C", "#C0392B", pad="6px 8px"))
            controls.addWidget(stop_btn)

            row_layout.addLayout(controls)

            # ── Tick labels ──────────────────────────────────────────────────
            tick_row = QHBoxLayout()
            tick_row.setContentsMargins(0, 0, 0, 0)
            for lbl_text in ["0", "64", "128", "192", "255"]:
                lbl = QLabel(lbl_text)
                lbl.setStyleSheet("color: #888; font-size: 9px;")
                lbl.setAlignment(Qt.AlignCenter)
                tick_row.addWidget(lbl)
            row_layout.addLayout(tick_row)

            # ── Status label ─────────────────────────────────────────────────
            status_lbl = QLabel("Stopped")
            status_lbl.setStyleSheet("color: #888; font-size: 10px;")
            row_layout.addWidget(status_lbl)

            # ── Wire up signals ──────────────────────────────────────────────
            # Slider → Spinbox (one direction, then push if running)
            slider.valueChanged.connect(
                lambda v, i=idx: self._on_slider_changed(i, v)
            )
            # Spinbox → Slider (other direction, then push if running)
            spinbox.valueChanged.connect(
                lambda v, i=idx: self._on_spinbox_changed(i, v)
            )
            # Individual stop button
            stop_btn.clicked.connect(lambda _, i=idx: self._stop_motor(i))

            # ── Store references ─────────────────────────────────────────────
            self._motor_rows.append({
                'slider': slider,
                'spinbox': spinbox,
                'status': status_lbl,
            })

            # Insert before the trailing stretch
            insert_pos = self._motor_rows_layout.count() - 1
            self._motor_rows_layout.insertWidget(insert_pos, row_widget)

    # ========================================================================
    # GLOBAL CONTROLS (below the motor panel)
    # ========================================================================

    def _build_global_controls(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 0)

        stop_all_btn = QPushButton("■ Stop All Motors")
        stop_all_btn.setToolTip("Send a stop command to every motor")
        stop_all_btn.clicked.connect(self._stop_all)
        stop_all_btn.setStyleSheet(_btn("#E74C3C", "#C0392B"))
        layout.addWidget(stop_all_btn)

        ramp_btn = QPushButton("🔄 Ramp Test…")
        ramp_btn.setToolTip(
            "Select a motor and run a 0 → max → 0 ramp test.\n"
            "Equivalent to the servo 'Test All Positions' sweep."
        )
        ramp_btn.clicked.connect(self._prompt_ramp_test)
        ramp_btn.setStyleSheet(_btn("#F39C12", "#E67E22"))
        layout.addWidget(ramp_btn)

        layout.addStretch()
        return panel

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        info = QLabel(
            "💡 Tip: Speed is a PWM value 0-255.  "
            "Drag the slider or scroll the spinbox to adjust each motor independently.  "
            "Changes push to hardware immediately while the motor is running."
        )
        info.setStyleSheet("color: #7F8C8D; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)
        return panel

    # ========================================================================
    # SIGNAL HANDLERS — Slider ↔ Spinbox sync + live speed push
    # ========================================================================

    def _on_slider_changed(self, motor_id: int, value: int):
        """Slider moved → sync spinbox, push speed if running."""
        row = self._motor_rows[motor_id]
        spinbox = row['spinbox']
        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.blockSignals(False)
        self._push_speed_if_running(motor_id, value)

    def _on_spinbox_changed(self, motor_id: int, value: int):
        """Spinbox changed → sync slider, push speed if running."""
        row = self._motor_rows[motor_id]
        slider = row['slider']
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)
        self._push_speed_if_running(motor_id, value)

    def _push_speed_if_running(self, motor_id: int, speed: int):
        """If the motor is currently running, send the new speed immediately."""
        if (self.motor_module and self.motor_module.is_ready()
                and self.motor_module.is_motor_running(motor_id)):
            self.motor_module.set_motor_speed(motor_id, speed)
            self._refresh_status(motor_id)

    # ========================================================================
    # MOTOR ACTIONS
    # ========================================================================

    def _check_ready(self, action: str = "test") -> bool:
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

    def _stop_motor(self, motor_id: int):
        if not self._check_ready("stop"):
            return
        self.motor_module.stop_motor(motor_id)
        self._refresh_status(motor_id)

    def _stop_all(self):
        if not self._check_ready("stop all"):
            return
        self.motor_module.stop_all()
        for i in range(len(self._motor_rows)):
            self._refresh_status(i)

    def _prompt_ramp_test(self):
        """Ask the operator which motor to ramp-test, then run it."""
        if not self._check_ready("ramp test"):
            return

        num = len(self._motor_rows)
        if num == 0:
            return

        if num == 1:
            self._run_ramp_test(0)
            return

        # Build a quick motor-selection dialog using QMessageBox
        from PyQt5.QtWidgets import QInputDialog
        motor_names = [
            self.motor_module.motor_names.get(i, f"Motor {i}")
            for i in range(num)
        ]
        choice, ok = QInputDialog.getItem(
            self, "Select Motor for Ramp Test",
            "Which motor should be ramp-tested?",
            motor_names, 0, False
        )
        if not ok:
            return
        motor_id = motor_names.index(choice)
        self._run_ramp_test(motor_id)

    def _run_ramp_test(self, motor_id: int):
        name = self.motor_module.motor_names.get(motor_id, f"Motor {motor_id}")

        reply = QMessageBox.question(
            self, "Ramp Test",
            f"This will ramp '{name}' from stopped to full speed, then back to stopped.\n\n"
            "Make sure it is safe to run this motor.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        steps = max(1, self.max_speed_spin.value() // max(1, self.ramp_rate_spin.value()))

        progress = QProgressDialog(
            f"Ramp test: {name}…", "Cancel", 0, 0, self
        )
        progress.setWindowTitle("Motor Ramp Test")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            success = self.motor_module.ramp_test(motor_id, steps=steps, dwell_time=0.5)
        finally:
            progress.close()

        self._refresh_status(motor_id)

        if success:
            QMessageBox.information(
                self, "Ramp Test Complete",
                f"✓ Ramp test for '{name}' completed.\nMotor has returned to stopped."
            )
        else:
            QMessageBox.warning(
                self, "Ramp Test Issues",
                f"Some steps during the ramp test encountered errors.\n\nCheck logs."
            )

    def _refresh_status(self, motor_id: int):
        """Update the status label for one motor from the live module state."""
        if motor_id >= len(self._motor_rows):
            return
        row = self._motor_rows[motor_id]
        if self.motor_module and self.motor_module.is_motor_running(motor_id):
            spd = self.motor_module.get_motor_speed(motor_id)
            row['status'].setText(f"Running — {spd}")
            row['status'].setStyleSheet("color: #27AE60; font-size: 10px; font-weight: bold;")
        else:
            row['status'].setText("Stopped")
            row['status'].setStyleSheet("color: #888; font-size: 10px;")

    # ========================================================================
    # CONNECTION
    # ========================================================================

    def _scan_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.port_combo.addItem("No ports found")
            QMessageBox.information(
                self, "No Ports Found",
                "No serial ports detected.\n\nMake sure the motor controller is connected."
            )
            return
        for p in ports:
            label = p.device
            if p.description and p.description != p.device:
                label += f" - {p.description}"
            self.port_combo.addItem(label)
        QMessageBox.information(
            self, "Ports Scanned",
            f"Found {len(ports)} port(s):\n" + "\n".join(p.device for p in ports)
        )

    def _test_connection(self):
        if self.simulation_check.isChecked():
            self.connection_status.setText("Status: Simulation Mode")
            self.connection_status.setStyleSheet("color: #F39C12; font-weight: bold;")
            QMessageBox.information(self, "Simulation Mode",
                                    "Simulation mode enabled — no hardware needed.")
            return
        if not self.motor_module:
            self.connection_status.setText("Status: No Module")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.warning(self, "Module Not Available",
                                "Motor module not available yet.\n"
                                "It will be created when configuration is applied.")
            return
        if self.motor_module.is_ready():
            self.connection_status.setText("Status: Connected")
            self.connection_status.setStyleSheet("color: #27AE60; font-weight: bold;")
            st = self.motor_module.get_status()
            QMessageBox.information(
                self, "Connection Successful",
                f"✓ Motor controller connected!\n\n"
                f"Port: {st['port']}\n"
                f"Motors: {st['num_motors']}"
            )
        else:
            self.connection_status.setText("Status: Not Connected")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.warning(self, "Connection Failed",
                                "Failed to connect.\n\n"
                                "Check port selection and that no other app is using it.")

    def _on_simulation_changed(self):
        is_sim = self.simulation_check.isChecked()
        self.port_combo.setEnabled(not is_sim)
        self.baud_combo.setEnabled(not is_sim)
        self.test_conn_btn.setEnabled(not is_sim)
        if is_sim:
            self.connection_status.setText("Status: Simulation Mode")
            self.connection_status.setStyleSheet("color: #F39C12; font-weight: bold;")
        else:
            self.connection_status.setText("Status: Hardware Mode")
            self.connection_status.setStyleSheet("color: gray; font-weight: bold;")
        self.mark_modified()

    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================

    def load_config(self) -> bool:
        try:
            config = self.get_module_config(self.get_module_name())
            if not config:
                self.logger.warning("No motor config found — using defaults")
                self.clear_modified()
                return True

            # Block signals on connection widgets
            for w in [self.simulation_check, self.port_combo, self.baud_combo,
                      self.timeout_spin, self.retry_spin, self.retry_delay_spin,
                      self.min_speed_spin, self.max_speed_spin, self.ramp_rate_spin]:
                w.blockSignals(True)

            self.simulation_check.setChecked(config.get("simulation_mode", False))
            port = config.get("port", "")
            if port:
                self.port_combo.setCurrentText(port)
            self.baud_combo.setCurrentText(str(config.get("baud_rate", 57600)))
            self.timeout_spin.setValue(config.get("timeout", 1.0))
            self.retry_spin.setValue(config.get("connection_retries", 3))
            self.retry_delay_spin.setValue(config.get("retry_delay", 1.0))
            self.min_speed_spin.setValue(config.get("min_speed", 0))
            self.max_speed_spin.setValue(config.get("max_speed", 255))
            self.ramp_rate_spin.setValue(config.get("ramp_rate", 32))

            for w in [self.simulation_check, self.port_combo, self.baud_combo,
                      self.timeout_spin, self.retry_spin, self.retry_delay_spin,
                      self.min_speed_spin, self.max_speed_spin, self.ramp_rate_spin]:
                w.blockSignals(False)

            self._on_simulation_changed()

            # (Re)build motor rows from config
            num = config.get("num_motors", 2)
            names = {int(k): v for k, v in config.get("motor_names", {}).items()}
            defaults = {int(k): int(v) for k, v in config.get("default_speeds", {}).items()}
            self._build_motor_rows(num, names, defaults)

            self.clear_modified()
            self.logger.info(f"Motor config loaded — {num} motor(s)")
            return True

        except Exception as e:
            self.logger.error(f"Error loading motor config: {e}", exc_info=True)
            return False

    def save_config(self) -> bool:
        if not self.validate():
            return False
        cfg = self.get_config()
        self.config_manager.update_module_config(self.get_module_name(), cfg)
        self.clear_modified()
        self.config_changed.emit(self.get_module_name(), cfg)
        self.logger.info("Motor configuration saved")
        return True

    def get_config(self) -> Dict[str, Any]:
        port_text = self.port_combo.currentText()
        port = port_text.split(" - ")[0].strip() if port_text else ""

        # Collect per-motor default speeds from current spinbox values
        default_speeds = {}
        motor_names = {}
        for i, row in enumerate(self._motor_rows):
            default_speeds[str(i)] = row['spinbox'].value()
            # Motor names aren't editable in the UI — preserve from loaded config
            cfg = self.get_module_config(self.get_module_name())
            saved_names = cfg.get("motor_names", {}) if cfg else {}
            motor_names = saved_names  # keep existing names

        return {
            "port": port,
            "baud_rate": int(self.baud_combo.currentText()),
            "timeout": self.timeout_spin.value(),
            "connection_retries": self.retry_spin.value(),
            "retry_delay": self.retry_delay_spin.value(),
            "simulation_mode": self.simulation_check.isChecked(),
            "num_motors": len(self._motor_rows),
            "motor_names": motor_names,
            "default_speeds": default_speeds,
            "min_speed": self.min_speed_spin.value(),
            "max_speed": self.max_speed_spin.value(),
            "ramp_rate": self.ramp_rate_spin.value(),
        }

    def validate(self) -> bool:
        lo = self.min_speed_spin.value()
        hi = self.max_speed_spin.value()
        if lo >= hi:
            QMessageBox.warning(self, "Validation Error",
                                "Min Speed must be less than Max Speed.")
            self.validation_failed.emit("Min Speed must be less than Max Speed")
            return False
        return True
