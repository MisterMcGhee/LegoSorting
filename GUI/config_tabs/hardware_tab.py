"""
hardware_tab.py - Hardware configuration tab for Arduino and servo control

FILE LOCATION: GUI/config_tabs/hardware_tab.py

This tab provides Arduino connection settings, servo configuration, and
bin position calibration. It integrates with the arduino_servo_module to
test positions and manage hardware.

Features:
- Arduino connection settings (port, baud rate, timeout)
- Simulation mode for development without hardware
- Servo hardware configuration (pulse widths, positions)
- Bin position table with individual test buttons
- Auto-calculate evenly-spaced positions
- Test all positions (sweep test)
- Manual calibration mode
- Connection status monitoring

Configuration Mapping:
    {
        "arduino_servo": {
            "port": "COM3",
            "baud_rate": 57600,
            "timeout": 1.0,
            "connection_retries": 3,
            "retry_delay": 1.0,
            "simulation_mode": false,
            "min_pulse": 500,
            "max_pulse": 2500,
            "default_position": 90,
            "min_bin_separation": 20,
            "bin_positions": {
                "0": 90,
                "1": 10,
                "2": 30,
                ...
            }
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QComboBox, QPushButton, QSpinBox,
                             QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                             QMessageBox, QCheckBox, QDoubleSpinBox,
                             QAbstractItemView, QProgressDialog)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QColor
from typing import Dict, Any, Optional
import logging
import serial.tools.list_ports

# Import base class
from GUI.config_tabs.base_tab import BaseConfigTab
from enhanced_config_manager import ModuleConfig

# Initialize logger
logger = logging.getLogger(__name__)


class HardwareConfigTab(BaseConfigTab):
    """
    Hardware configuration tab for Arduino and servo control.

    This tab allows users to:
    - Configure Arduino connection (port, baud rate, timeouts)
    - Enable/disable simulation mode
    - Configure servo hardware parameters
    - View and edit bin positions
    - Auto-calculate evenly-spaced positions
    - Test individual bin positions
    - Test all positions (sweep)
    - Monitor connection status

    The tab integrates with arduino_servo_module for hardware control
    and automatically recalculates positions when bin count changes.

    Signals:
        hardware_status_changed(bool): Emitted when hardware status changes
        positions_changed(dict): Emitted when bin positions change
        simulation_mode_changed(bool): Emitted when simulation mode changes
    """

    # Custom signals
    hardware_status_changed = pyqtSignal(bool)  # is_ready
    positions_changed = pyqtSignal(dict)  # bin_positions
    simulation_mode_changed = pyqtSignal(bool)  # simulation_mode

    def __init__(self, config_manager, arduino_module=None, parent=None):
        """
        Initialize hardware configuration tab.

        Args:
            config_manager: Configuration manager instance
            arduino_module: Optional arduino servo module reference
            parent: Parent widget
        """
        # Store arduino module reference
        self.arduino_module = arduino_module

        # Current bin count (will be updated from processing tab)
        self.current_bin_count = 9  # Default

        # Call parent init
        super().__init__(config_manager, parent)

        # Initialize UI and load config
        self.init_ui()
        self.load_config()

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (Required by BaseConfigTab)
    # ========================================================================

    def get_module_name(self) -> str:
        """Return the configuration module name this tab manages."""
        return ModuleConfig.ARDUINO_SERVO.value

    def init_ui(self):
        """Create the hardware configuration user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Title
        title = QLabel("Hardware Configuration")
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

        # Main content in horizontal layout
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        # Left panel: Connection and servo settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        connection_group = self.create_connection_group()
        left_layout.addWidget(connection_group)

        servo_group = self.create_servo_group()
        left_layout.addWidget(servo_group)

        left_layout.addStretch()
        content_layout.addWidget(left_panel, stretch=1)

        # Right panel: Bin positions
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        positions_group = self.create_positions_group()
        right_layout.addWidget(positions_group, stretch=1)

        content_layout.addWidget(right_panel, stretch=1)

        # Info panel at bottom
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel)

        self.logger.info("Hardware tab UI initialized")

    def create_connection_group(self) -> QGroupBox:
        """Create Arduino connection settings group."""
        group = QGroupBox("Arduino Connection")
        layout = QFormLayout()

        # Simulation mode checkbox
        self.simulation_check = QCheckBox("Simulation Mode")
        self.simulation_check.setToolTip(
            "Enable simulation mode for testing without physical Arduino"
        )
        self.simulation_check.stateChanged.connect(self.on_simulation_mode_changed)
        layout.addRow("Mode:", self.simulation_check)

        # Serial port selection
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setToolTip("Select Arduino serial port")
        self.port_combo.currentTextChanged.connect(self.mark_modified)
        layout.addRow("Serial Port:", self.port_combo)

        # Scan ports button
        scan_btn = QPushButton("🔍 Scan Ports")
        scan_btn.setToolTip("Scan for available serial ports")
        scan_btn.clicked.connect(self.scan_ports)
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

        # Connection status
        self.connection_status = QLabel("Status: Unknown")
        self.connection_status.setStyleSheet("color: gray; font-weight: bold;")
        layout.addRow("", self.connection_status)

        # Test connection button
        self.test_connection_btn = QPushButton("🔌 Test Connection")
        self.test_connection_btn.setToolTip("Test Arduino connection")
        self.test_connection_btn.clicked.connect(self.test_connection)
        self.test_connection_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        layout.addRow("", self.test_connection_btn)

        group.setLayout(layout)
        return group

    def create_servo_group(self) -> QGroupBox:
        """Create servo hardware settings group."""
        group = QGroupBox("Servo Hardware")
        layout = QFormLayout()

        # Min pulse width
        self.min_pulse_spin = QSpinBox()
        self.min_pulse_spin.setRange(100, 2000)
        self.min_pulse_spin.setValue(500)
        self.min_pulse_spin.setSuffix(" μs")
        self.min_pulse_spin.setToolTip("Minimum servo pulse width (microseconds)")
        self.min_pulse_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Min Pulse:", self.min_pulse_spin)

        # Max pulse width
        self.max_pulse_spin = QSpinBox()
        self.max_pulse_spin.setRange(1000, 3000)
        self.max_pulse_spin.setValue(2500)
        self.max_pulse_spin.setSuffix(" μs")
        self.max_pulse_spin.setToolTip("Maximum servo pulse width (microseconds)")
        self.max_pulse_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Max Pulse:", self.max_pulse_spin)

        # Default position
        self.default_pos_spin = QSpinBox()
        self.default_pos_spin.setRange(0, 180)
        self.default_pos_spin.setValue(90)
        self.default_pos_spin.setSuffix("°")
        self.default_pos_spin.setToolTip("Default/home servo position")
        self.default_pos_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Default Position:", self.default_pos_spin)

        # Min bin separation
        self.min_separation_spin = QSpinBox()
        self.min_separation_spin.setRange(5, 90)
        self.min_separation_spin.setValue(20)
        self.min_separation_spin.setSuffix("°")
        self.min_separation_spin.setToolTip("Minimum angle between adjacent bins")
        self.min_separation_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Min Separation:", self.min_separation_spin)

        # Home button
        self.home_btn = QPushButton("🏠 Home Position")
        self.home_btn.setToolTip("Move servo to home position")
        self.home_btn.clicked.connect(self.move_to_home)
        layout.addRow("", self.home_btn)

        group.setLayout(layout)
        return group

    def create_positions_group(self) -> QGroupBox:
        """Create bin positions configuration group."""
        group = QGroupBox("Bin Positions")
        layout = QVBoxLayout()

        # Control buttons row
        button_layout = QHBoxLayout()

        self.auto_calc_btn = QPushButton("Auto-Calculate")
        self.auto_calc_btn.setToolTip("Calculate evenly-spaced positions for all bins")
        self.auto_calc_btn.clicked.connect(self.auto_calculate_positions)
        self.auto_calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        button_layout.addWidget(self.auto_calc_btn)

        self.test_all_btn = QPushButton("🔄 Test All")
        self.test_all_btn.setToolTip("Test all bin positions (sweep)")
        self.test_all_btn.clicked.connect(self.test_all_positions)
        self.test_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)
        button_layout.addWidget(self.test_all_btn)

        button_layout.addStretch()

        self.calibration_mode_check = QCheckBox("Calibration Mode")
        self.calibration_mode_check.setToolTip("Enable manual editing of positions")
        self.calibration_mode_check.stateChanged.connect(self.toggle_calibration_mode)
        button_layout.addWidget(self.calibration_mode_check)

        layout.addLayout(button_layout)

        # Bin positions table
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(3)
        self.positions_table.setHorizontalHeaderLabels(["Bin", "Angle (°)", "Test"])
        self.positions_table.horizontalHeader().setStretchLastSection(False)
        self.positions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.positions_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.positions_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.positions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.positions_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.positions_table.setMaximumHeight(300)

        # Connect table editing
        self.positions_table.itemChanged.connect(self.on_position_edited)

        layout.addWidget(self.positions_table)

        # Status label
        self.positions_status = QLabel("No positions configured")
        self.positions_status.setStyleSheet("color: #7F8C8D; font-style: italic;")
        layout.addWidget(self.positions_status)

        group.setLayout(layout)
        return group

    def create_info_panel(self) -> QWidget:
        """Create information panel."""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        info = QLabel("💡 Tip: Positions auto-calculate when bin count changes. Use 'Test All' to verify.")
        info.setStyleSheet("color: #7F8C8D; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)

        return panel

    # ========================================================================
    # SERIAL PORT MANAGEMENT
    # ========================================================================

    def scan_ports(self):
        """Scan for available serial ports."""
        self.logger.info("Scanning for serial ports...")

        # Clear current items
        self.port_combo.clear()

        # Get available ports
        ports = serial.tools.list_ports.comports()

        if not ports:
            self.port_combo.addItem("No ports found")
            QMessageBox.information(
                self,
                "No Ports Found",
                "No serial ports detected.\n\n"
                "Make sure your Arduino is connected."
            )
            return

        # Add ports to combo
        for port in ports:
            # Format: "COM3 - Arduino Uno"
            port_str = f"{port.device}"
            if port.description and port.description != port.device:
                port_str += f" - {port.description}"
            self.port_combo.addItem(port_str)

        self.logger.info(f"Found {len(ports)} serial port(s)")
        QMessageBox.information(
            self,
            "Ports Scanned",
            f"Found {len(ports)} serial port(s):\n\n" +
            "\n".join([p.device for p in ports])
        )

    # ========================================================================
    # CONNECTION TESTING
    # ========================================================================

    def test_connection(self):
        """Test Arduino connection."""
        self.logger.info("Testing Arduino connection...")

        if self.simulation_check.isChecked():
            self.connection_status.setText("Status: Simulation Mode")
            self.connection_status.setStyleSheet("color: #F39C12; font-weight: bold;")
            QMessageBox.information(
                self,
                "Simulation Mode",
                "Simulation mode is enabled.\n\n"
                "No hardware connection is required."
            )
            return

        if not self.arduino_module:
            self.connection_status.setText("Status: No Module")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.warning(
                self,
                "Module Not Available",
                "Arduino module not available for testing.\n\n"
                "Module will be created when configuration is applied."
            )
            return

        # Check if module is ready
        if self.arduino_module.is_ready():
            self.connection_status.setText("Status: Connected")
            self.connection_status.setStyleSheet("color: #27AE60; font-weight: bold;")

            # Get statistics
            stats = self.arduino_module.get_statistics()
            port = stats.get('arduino', {}).get('port', 'Unknown')

            QMessageBox.information(
                self,
                "Connection Successful",
                f"✓ Arduino connected successfully!\n\n"
                f"Port: {port}\n"
                f"Status: Ready"
            )
        else:
            self.connection_status.setText("Status: Not Connected")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.warning(
                self,
                "Connection Failed",
                "Failed to connect to Arduino.\n\n"
                "Check:\n"
                "• Arduino is connected\n"
                "• Correct port is selected\n"
                "• No other program is using the port"
            )

    def on_simulation_mode_changed(self):
        """Handle simulation mode checkbox change."""
        is_simulation = self.simulation_check.isChecked()

        # Enable/disable connection settings
        self.port_combo.setEnabled(not is_simulation)
        self.baud_combo.setEnabled(not is_simulation)
        self.test_connection_btn.setEnabled(not is_simulation)

        # Update status
        if is_simulation:
            self.connection_status.setText("Status: Simulation Mode")
            self.connection_status.setStyleSheet("color: #F39C12; font-weight: bold;")
        else:
            self.connection_status.setText("Status: Hardware Mode")
            self.connection_status.setStyleSheet("color: gray; font-weight: bold;")

        self.mark_modified()
        self.simulation_mode_changed.emit(is_simulation)

    # ========================================================================
    # BIN POSITION MANAGEMENT
    # ========================================================================

    def populate_positions_table(self, positions: Dict[int, float]):
        """
        Populate the positions table with bin data.

        Args:
            positions: Dictionary of {bin_number: angle}
        """
        # Block signals while populating
        self.positions_table.blockSignals(True)

        # Clear existing rows
        self.positions_table.setRowCount(0)

        # Sort bins by number
        sorted_bins = sorted(positions.items())

        for bin_num, angle in sorted_bins:
            row = self.positions_table.rowCount()
            self.positions_table.insertRow(row)

            # Bin number (never editable)
            bin_item = QTableWidgetItem(str(bin_num))
            bin_item.setTextAlignment(Qt.AlignCenter)
            bin_item.setFlags(bin_item.flags() & ~Qt.ItemIsEditable)

            # Highlight overflow bin
            if bin_num == 0:
                bin_item.setBackground(QColor(255, 243, 205))  # Light yellow

            self.positions_table.setItem(row, 0, bin_item)

            # Angle (editable in calibration mode)
            angle_item = QTableWidgetItem(f"{angle:.1f}")
            angle_item.setTextAlignment(Qt.AlignCenter)
            angle_item.setData(Qt.UserRole, bin_num)  # Store bin number for editing!

            # Set editability based on current calibration mode
            if self.calibration_mode_check.isChecked():
                angle_item.setFlags(angle_item.flags() | Qt.ItemIsEditable)
                angle_item.setBackground(QColor(255, 255, 200))  # Light yellow for editing
            else:
                angle_item.setFlags(angle_item.flags() & ~Qt.ItemIsEditable)

            self.positions_table.setItem(row, 1, angle_item)

            # Test button
            test_btn = QPushButton("▶ Test")
            test_btn.setToolTip(f"Test bin {bin_num} at current angle")
            test_btn.clicked.connect(lambda checked, b=bin_num: self.test_single_position(b))
            test_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
            """)
            self.positions_table.setCellWidget(row, 2, test_btn)

        # Restore signals
        self.positions_table.blockSignals(False)

        # Update status
        count = len(positions)
        if self.calibration_mode_check.isChecked():
            self.positions_status.setText(f"⚠️ CALIBRATION MODE: {count} bins • Double-click to edit")
        else:
            self.positions_status.setText(f"{count} bin position(s) configured")

    def auto_calculate_positions(self):
        """Auto-calculate evenly-spaced bin positions."""
        self.logger.info("Auto-calculating bin positions...")

        if not self.arduino_module:
            QMessageBox.warning(
                self,
                "Module Not Available",
                "Arduino module not available.\n\n"
                "Positions will be calculated when hardware is initialized."
            )
            return

        try:
            # Pass current_bin_count directly - it means TOTAL bins
            positions = self.arduino_module.auto_calculate_bin_positions(self.current_bin_count)

            self.logger.info(f"Calculated positions: {positions}")

            # Update table
            self.populate_positions_table(positions)

            # Mark as modified
            self.mark_modified()

            # Emit signal
            self.positions_changed.emit(positions)

            sorting_bins = self.current_bin_count - 1
            QMessageBox.information(
                self,
                "Positions Calculated",
                f"✓ Auto-calculated positions for {self.current_bin_count} bins.\n\n"
                f"• Bin 0: Overflow\n"
                f"• Bins 1-{sorting_bins}: Sorting bins\n\n"
                "Positions are evenly spaced across the servo range.\n"
                "Click 'Save Configuration' to apply these positions."
            )

        except Exception as e:
            self.logger.error(f"Failed to auto-calculate positions: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Calculation Failed",
                f"Failed to calculate positions:\n{str(e)}"
            )

    def test_single_position(self, bin_num: int):
        """
        Test a single bin position using the current angle in the table.

        Moves the servo to the desired position immediately without dialogue,
        hold delay, or automatic return to home. Use the Home button to return
        to neutral position when desired.

        Args:
            bin_num: Bin number to test
        """
        self.logger.info(f"Testing bin {bin_num}...")

        if not self.arduino_module:
            QMessageBox.warning(
                self,
                "Module Not Available",
                "Arduino module not available for testing."
            )
            return

        if not self.arduino_module.is_ready():
            QMessageBox.warning(
                self,
                "Hardware Not Ready",
                "Arduino is not connected or ready.\n\n"
                "Enable simulation mode or connect hardware."
            )
            return

        # Get the current angle from the table (not from arduino_module!)
        target_angle = None
        for row in range(self.positions_table.rowCount()):
            bin_item = self.positions_table.item(row, 0)
            angle_item = self.positions_table.item(row, 1)

            if bin_item and angle_item:
                if int(bin_item.text()) == bin_num:
                    try:
                        target_angle = float(angle_item.text())
                        break
                    except ValueError:
                        self.logger.error(f"Invalid angle in table for bin {bin_num}")

        if target_angle is None:
            QMessageBox.warning(
                self,
                "Bin Not Found",
                f"Could not find bin {bin_num} in position table."
            )
            return

        try:
            # Move to the angle specified in the table
            success = self.arduino_module.move_to_angle(target_angle, wait=True)

            if success:
                self.logger.info(f"✓ Bin {bin_num} moved to {target_angle}°")
            else:
                QMessageBox.warning(
                    self,
                    "Test Failed",
                    f"Failed to move to {target_angle}° for bin {bin_num}.\n\n"
                    "Check logs for details."
                )

        except Exception as e:
            self.logger.error(f"Error testing bin {bin_num}: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Test Error",
                f"Error testing bin {bin_num}:\n{str(e)}"
            )

    def test_all_positions(self):
        """Test all bin positions (sweep test)."""
        self.logger.info("Testing all positions...")

        if not self.arduino_module:
            QMessageBox.warning(
                self,
                "Module Not Available",
                "Arduino module not available for testing."
            )
            return

        if not self.arduino_module.is_ready():
            QMessageBox.warning(
                self,
                "Hardware Not Ready",
                "Arduino is not connected or ready.\n\n"
                "Enable simulation mode or connect hardware."
            )
            return

        # Confirm action
        reply = QMessageBox.question(
            self,
            "Test All Positions",
            "This will move the servo through all bin positions.\n\n"
            "Make sure the area is clear and it's safe to move the servo.\n\n"
            "Continue with sweep test?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            # Create progress dialog
            progress = QProgressDialog(
                "Testing all bin positions...",
                "Cancel",
                0,
                0,
                self
            )
            progress.setWindowTitle("Position Sweep Test")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Run test
            success = self.arduino_module.test_all_positions(dwell_time=1.0)

            progress.close()

            if success:
                QMessageBox.information(
                    self,
                    "Test Complete",
                    "✓ All bin positions tested successfully!\n\n"
                    "All positions are reachable."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Test Issues",
                    "Some positions failed during testing.\n\n"
                    "Check logs for details."
                )

        except Exception as e:
            self.logger.error(f"Error testing all positions: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Test Error",
                f"Error during sweep test:\n{str(e)}"
            )

    def move_to_home(self):
        """Move servo to home position."""
        self.logger.info("Moving to home position...")

        if not self.arduino_module:
            QMessageBox.warning(
                self,
                "Module Not Available",
                "Arduino module not available."
            )
            return

        if not self.arduino_module.is_ready():
            QMessageBox.warning(
                self,
                "Hardware Not Ready",
                "Arduino is not connected or ready.\n\n"
                "Enable simulation mode or connect hardware."
            )
            return

        try:
            success = self.arduino_module.home()

            if success:
                home_pos = self.default_pos_spin.value()
                QMessageBox.information(
                    self,
                    "Home Position",
                    f"✓ Servo moved to home position ({home_pos}°)"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Movement Failed",
                    "Failed to move to home position.\n\n"
                    "Check logs for details."
                )

        except Exception as e:
            self.logger.error(f"Error moving to home: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Movement Error",
                f"Error moving to home:\n{str(e)}"
            )

    def toggle_calibration_mode(self, state):
        """Toggle calibration mode for manual position editing."""
        is_calibration = bool(state)

        if is_calibration:
            self.logger.info("Calibration mode enabled")

            # Enable editing in table with single-click on selected item
            self.positions_table.setEditTriggers(
                QAbstractItemView.SelectedClicked | QAbstractItemView.EditKeyPressed
            )

            # Enable editing for angle items and highlight them
            for row in range(self.positions_table.rowCount()):
                angle_item = self.positions_table.item(row, 1)
                if angle_item:
                    angle_item.setFlags(angle_item.flags() | Qt.ItemIsEditable)
                    # Light yellow background to show it's editable
                    angle_item.setBackground(QColor(255, 255, 200))

            self.positions_status.setText(
                "⚠️ CALIBRATION MODE: Click angle to edit • Test after changes • Save when done")
            self.positions_status.setStyleSheet("color: #E67E22; font-weight: bold; font-size: 11px;")

        else:
            self.logger.info("Calibration mode disabled")

            # Disable editing in table
            self.positions_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

            # Disable editing for angle items and remove highlight
            for row in range(self.positions_table.rowCount()):
                angle_item = self.positions_table.item(row, 1)
                if angle_item:
                    angle_item.setFlags(angle_item.flags() & ~Qt.ItemIsEditable)
                    angle_item.setBackground(QColor(255, 255, 255))  # White

            count = self.positions_table.rowCount()
            self.positions_status.setText(f"{count} bin position(s) configured")
            self.positions_status.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 11px;")

    def on_position_edited(self, item):
        """Handle manual editing of position angle."""
        if item.column() != 1:  # Only angle column
            return

        if not self.calibration_mode_check.isChecked():
            return

        # Block signals temporarily to prevent recursion
        self.positions_table.blockSignals(True)

        try:
            # Get edited angle
            new_angle_text = item.text().strip()
            new_angle = float(new_angle_text)

            # Get bin number
            bin_num = item.data(Qt.UserRole)
            if bin_num is None:
                # Try to get from row
                row = item.row()
                bin_item = self.positions_table.item(row, 0)
                if bin_item:
                    bin_num = int(bin_item.text())

            # Validate range
            if new_angle < 0 or new_angle > 180:
                self.positions_table.blockSignals(False)
                QMessageBox.warning(
                    self,
                    "Invalid Angle",
                    f"Angle must be between 0° and 180°.\n\n"
                    f"You entered: {new_angle}°"
                )
                # Revert - but we need to know what it was before
                # For now, just highlight the cell red
                item.setBackground(QColor(255, 200, 200))  # Light red
                return

            # Valid angle - format it nicely and mark as modified
            item.setText(f"{new_angle:.1f}")
            item.setBackground(QColor(200, 255, 200))  # Light green to show it's changed

            self.mark_modified()
            self.logger.info(f"Position edited: Bin {bin_num} -> {new_angle}°")

            # Update status to remind user to test and save
            self.positions_status.setText(
                f"⚠️ Bin {bin_num} changed to {new_angle}° • Click 'Test' to verify • Save when done"
            )

        except ValueError:
            self.positions_table.blockSignals(False)
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"Please enter a valid number for the angle.\n\n"
                f"You entered: '{item.text()}'"
            )
            # Highlight red
            item.setBackground(QColor(255, 200, 200))
            return

        finally:
            self.positions_table.blockSignals(False)

    # ========================================================================
    # INTER-TAB COMMUNICATION
    # ========================================================================

    def set_bin_count(self, bin_count: int):
        """
        Update bin count from processing tab.

        Args:
            bin_count: New TOTAL bin count (including overflow bin 0)
        """
        if bin_count == self.current_bin_count:
            return

        self.logger.info(f"Bin count changed: {self.current_bin_count} -> {bin_count}")
        self.current_bin_count = bin_count

        # Auto-calculate new positions
        if self.arduino_module:
            try:
                # Pass bin_count directly - it means TOTAL bins
                positions = self.arduino_module.auto_calculate_bin_positions(bin_count)
                self.populate_positions_table(positions)
                self.mark_modified()

                self.logger.info(f"Auto-calculated positions for {bin_count} total bins")
            except Exception as e:
                self.logger.error(f"Failed to auto-calculate positions: {e}")

    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================

    def load_config(self) -> bool:
        """Load configuration from config_manager and populate UI."""
        try:
            config = self.get_module_config(self.get_module_name())

            # Also get sorting config for bin count
            sorting_config = self.get_module_config(ModuleConfig.SORTING.value)
            self.current_bin_count = sorting_config.get('max_bins', 9)

            self.logger.info(f"Loaded bin count: {self.current_bin_count}")

            if not config:
                self.logger.warning("No configuration found, using defaults")
                self.clear_modified()
                return True

            # Block signals during loading
            self.simulation_check.blockSignals(True)
            self.port_combo.blockSignals(True)
            self.baud_combo.blockSignals(True)
            self.timeout_spin.blockSignals(True)
            self.retry_spin.blockSignals(True)
            self.retry_delay_spin.blockSignals(True)
            self.min_pulse_spin.blockSignals(True)
            self.max_pulse_spin.blockSignals(True)
            self.default_pos_spin.blockSignals(True)
            self.min_separation_spin.blockSignals(True)

            # Load connection settings
            self.simulation_check.setChecked(config.get("simulation_mode", False))

            port = config.get("port", "")
            if port:
                self.port_combo.setCurrentText(port)

            self.baud_combo.setCurrentText(str(config.get("baud_rate", 57600)))
            self.timeout_spin.setValue(config.get("timeout", 1.0))
            self.retry_spin.setValue(config.get("connection_retries", 3))
            self.retry_delay_spin.setValue(config.get("retry_delay", 1.0))

            # Load servo settings
            self.min_pulse_spin.setValue(config.get("min_pulse", 500))
            self.max_pulse_spin.setValue(config.get("max_pulse", 2500))
            self.default_pos_spin.setValue(config.get("default_position", 90))
            self.min_separation_spin.setValue(config.get("min_bin_separation", 20))

            # Restore signals
            self.simulation_check.blockSignals(False)
            self.port_combo.blockSignals(False)
            self.baud_combo.blockSignals(False)
            self.timeout_spin.blockSignals(False)
            self.retry_spin.blockSignals(False)
            self.retry_delay_spin.blockSignals(False)
            self.min_pulse_spin.blockSignals(False)
            self.max_pulse_spin.blockSignals(False)
            self.default_pos_spin.blockSignals(False)
            self.min_separation_spin.blockSignals(False)

            # Update UI state based on simulation mode
            self.on_simulation_mode_changed()

            # Load bin positions
            bin_positions = config.get("bin_positions", {})
            if bin_positions:
                # Convert string keys to int, values to float
                positions = {int(k): float(v) for k, v in bin_positions.items()}
                self.populate_positions_table(positions)
            else:
                # No positions - will auto-calculate when module is available
                self.positions_status.setText("No positions configured - will auto-calculate")

            # Clear modified flag
            self.clear_modified()

            self.logger.info("Configuration loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            return False

    def save_config(self) -> bool:
        """Save current UI values to config_manager."""
        if not self.validate():
            return False

        try:
            config = self.get_config()

            self.config_manager.update_module_config(
                self.get_module_name(),
                config
            )

            # Reload arduino module positions if available
            if self.arduino_module:
                self.arduino_module.reload_positions_from_config()

            self.clear_modified()
            self.logger.info("Configuration saved successfully")

            # Emit signal
            self.config_changed.emit(self.get_module_name(), config)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}", exc_info=True)
            return False

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration as a dictionary."""
        # Get bin positions from table (this is the source of truth!)
        bin_positions = {}
        for row in range(self.positions_table.rowCount()):
            bin_item = self.positions_table.item(row, 0)
            angle_item = self.positions_table.item(row, 1)

            if bin_item and angle_item:
                try:
                    bin_num = int(bin_item.text())
                    angle = float(angle_item.text())
                    bin_positions[str(bin_num)] = angle  # JSON requires string keys
                except ValueError as e:
                    self.logger.error(f"Invalid bin position in table row {row}: {e}")
                    # Skip invalid rows

        # Extract port (may have description attached)
        port_text = self.port_combo.currentText()
        port = port_text.split(" ")[0] if " " in port_text else port_text

        config = {
            "port": port,
            "baud_rate": int(self.baud_combo.currentText()),
            "timeout": self.timeout_spin.value(),
            "connection_retries": self.retry_spin.value(),
            "retry_delay": self.retry_delay_spin.value(),
            "simulation_mode": self.simulation_check.isChecked(),
            "min_pulse": self.min_pulse_spin.value(),
            "max_pulse": self.max_pulse_spin.value(),
            "default_position": self.default_pos_spin.value(),
            "min_bin_separation": self.min_separation_spin.value(),
            "bin_positions": bin_positions
        }

        return config

    def validate(self) -> bool:
        """Validate current configuration."""
        # Validate pulse widths
        min_pulse = self.min_pulse_spin.value()
        max_pulse = self.max_pulse_spin.value()

        if min_pulse >= max_pulse:
            self.show_validation_error(
                f"Min pulse ({min_pulse}) must be less than max pulse ({max_pulse})"
            )
            return False

        # Validate default position
        default_pos = self.default_pos_spin.value()
        if default_pos < 0 or default_pos > 180:
            self.show_validation_error(
                f"Default position ({default_pos}°) must be between 0° and 180°"
            )
            return False

        # Validate bin positions if any exist
        for row in range(self.positions_table.rowCount()):
            angle_item = self.positions_table.item(row, 1)
            if angle_item:
                try:
                    angle = float(angle_item.text())
                    if angle < 0 or angle > 180:
                        bin_item = self.positions_table.item(row, 0)
                        bin_num = bin_item.text() if bin_item else "?"
                        self.show_validation_error(
                            f"Bin {bin_num} angle ({angle}°) must be between 0° and 180°"
                        )
                        return False
                except ValueError:
                    self.show_validation_error(
                        f"Invalid angle value in bin positions table"
                    )
                    return False

        # Hardware mode requires port selection
        if not self.simulation_check.isChecked():
            port = self.port_combo.currentText()
            if not port or port == "No ports found":
                self.show_validation_error(
                    "Hardware mode requires a valid serial port selection"
                )
                return False

        return True

    def reset_to_defaults(self):
        """Reset all values to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Reset all hardware settings to default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Reset connection settings
            self.simulation_check.setChecked(False)
            self.baud_combo.setCurrentText("57600")
            self.timeout_spin.setValue(1.0)
            self.retry_spin.setValue(3)
            self.retry_delay_spin.setValue(1.0)

            # Reset servo settings
            self.min_pulse_spin.setValue(500)
            self.max_pulse_spin.setValue(2500)
            self.default_pos_spin.setValue(90)
            self.min_separation_spin.setValue(20)

            # Clear positions
            self.positions_table.setRowCount(0)
            self.positions_status.setText("No positions configured")

            self.mark_modified()
            self.logger.info("Settings reset to defaults")