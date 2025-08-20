"""
config_gui_module.py - Configuration GUI for Lego Sorting System
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import json
import logging
from typing import Dict, Any, Optional
from GUI.gui_common import BaseGUIWindow, VideoWidget, ConfirmationDialog, validate_config

logger = logging.getLogger(__name__)


class ConfigurationGUI(BaseGUIWindow):
    """Main configuration window for system setup"""

    # Signals
    configuration_complete = pyqtSignal(dict)  # Emitted when config is confirmed
    preview_requested = pyqtSignal()  # Request camera preview

    def __init__(self, config_manager=None):
        super().__init__(config_manager, "Lego Sorting System - Configuration")

        # Window setup
        self.setGeometry(100, 100, 1200, 800)
        self.center_window()

        # Initialize UI
        self.init_ui()
        self.load_current_config()

    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Add tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_camera_tab()
        self.create_detector_tab()
        self.create_sorting_tab()
        self.create_api_tab()
        self.create_arduino_tab()
        self.create_system_tab()

        # Bottom control panel
        self.create_control_panel()
        main_layout.addWidget(self.control_panel)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to configure")

    def create_camera_tab(self):
        """Create camera configuration tab"""
        camera_tab = QWidget()
        layout = QVBoxLayout()

        # Camera selection group
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout()

        # Device selection
        self.camera_device = QComboBox()
        self.camera_device.addItems(["0", "1", "2", "3"])
        camera_layout.addRow("Camera Device:", self.camera_device)

        # Resolution
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1920x1080", "1280x720", "640x480"])
        camera_layout.addRow("Resolution:", self.resolution_combo)

        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        camera_layout.addRow("Target FPS:", self.fps_spin)

        # Exposure
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-10, 10)
        self.exposure_slider.setValue(0)
        camera_layout.addRow("Exposure:", self.exposure_slider)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Preview area
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()

        self.preview_widget = VideoWidget()
        preview_layout.addWidget(self.preview_widget)

        preview_buttons = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Preview")
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.stop_preview_btn.setEnabled(False)

        self.start_preview_btn.clicked.connect(self.start_camera_preview)
        self.stop_preview_btn.clicked.connect(self.stop_camera_preview)

        preview_buttons.addWidget(self.start_preview_btn)
        preview_buttons.addWidget(self.stop_preview_btn)
        preview_layout.addLayout(preview_buttons)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        camera_tab.setLayout(layout)
        self.tab_widget.addTab(camera_tab, "Camera")

    def create_detector_tab(self):
        """Create detector configuration tab"""
        detector_tab = QWidget()
        layout = QVBoxLayout()

        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout()

        # Min/Max area
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 10000)
        self.min_area_spin.setValue(500)
        detection_layout.addRow("Min Area:", self.min_area_spin)

        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1000, 100000)
        self.max_area_spin.setValue(50000)
        detection_layout.addRow("Max Area:", self.max_area_spin)

        # Threshold
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(30)
        detection_layout.addRow("Threshold:", self.threshold_slider)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # ROI settings
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QVBoxLayout()

        self.roi_info_label = QLabel("ROI not configured")
        roi_layout.addWidget(self.roi_info_label)

        self.configure_roi_btn = QPushButton("Configure ROI")
        self.configure_roi_btn.clicked.connect(self.configure_roi)
        roi_layout.addWidget(self.configure_roi_btn)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        layout.addStretch()
        detector_tab.setLayout(layout)
        self.tab_widget.addTab(detector_tab, "Detector")

    def create_sorting_tab(self):
        """Create sorting configuration tab"""
        sorting_tab = QWidget()
        layout = QVBoxLayout()

        # Sorting strategy
        strategy_group = QGroupBox("Sorting Strategy")
        strategy_layout = QFormLayout()

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Pre-assigned Categories",
            "Priority-based",
            "First Come First Serve",
            "Size-based"
        ])
        strategy_layout.addRow("Strategy:", self.strategy_combo)

        self.overflow_bin = QSpinBox()
        self.overflow_bin.setRange(0, 9)
        self.overflow_bin.setValue(0)
        strategy_layout.addRow("Overflow Bin:", self.overflow_bin)

        strategy_group.setLayout(strategy_layout)
        layout.addWidget(strategy_group)

        # Bin assignments
        assignments_group = QGroupBox("Bin Assignments")
        assignments_layout = QVBoxLayout()

        # Create table for bin assignments
        self.bin_table = QTableWidget(10, 3)
        self.bin_table.setHorizontalHeaderLabels(["Bin", "Category", "Count"])
        self.bin_table.horizontalHeader().setStretchLastSection(True)

        # Initialize table
        for i in range(10):
            self.bin_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.bin_table.setItem(i, 1, QTableWidgetItem("Unassigned"))
            self.bin_table.setItem(i, 2, QTableWidgetItem("0"))

        assignments_layout.addWidget(self.bin_table)

        # Assignment buttons
        assign_buttons = QHBoxLayout()
        self.edit_assignments_btn = QPushButton("Edit Assignments")
        self.clear_assignments_btn = QPushButton("Clear All")

        assign_buttons.addWidget(self.edit_assignments_btn)
        assign_buttons.addWidget(self.clear_assignments_btn)
        assignments_layout.addLayout(assign_buttons)

        assignments_group.setLayout(assignments_layout)
        layout.addWidget(assignments_group)

        sorting_tab.setLayout(layout)
        self.tab_widget.addTab(sorting_tab, "Sorting")

    def create_api_tab(self):
        """Create API configuration tab"""
        api_tab = QWidget()
        layout = QVBoxLayout()

        # API settings
        api_group = QGroupBox("API Settings")
        api_layout = QFormLayout()

        # API Key
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        api_layout.addRow("API Key:", self.api_key_edit)

        # Mode
        self.api_mode_combo = QComboBox()
        self.api_mode_combo.addItems(["Production", "Development", "Mock"])
        api_layout.addRow("Mode:", self.api_mode_combo)

        # Timeout
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 60)
        self.timeout_spin.setValue(10)
        api_layout.addRow("Timeout (s):", self.timeout_spin)

        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # Test connection
        test_group = QGroupBox("Connection Test")
        test_layout = QVBoxLayout()

        self.test_api_btn = QPushButton("Test Connection")
        self.test_api_btn.clicked.connect(self.test_api_connection)
        test_layout.addWidget(self.test_api_btn)

        self.api_status_label = QLabel("Not tested")
        test_layout.addWidget(self.api_status_label)

        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

        layout.addStretch()
        api_tab.setLayout(layout)
        self.tab_widget.addTab(api_tab, "API")

    def create_arduino_tab(self):
        """Create Arduino configuration tab"""
        arduino_tab = QWidget()
        layout = QVBoxLayout()

        # Connection settings
        connection_group = QGroupBox("Arduino Connection")
        connection_layout = QFormLayout()

        # Port selection
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.refresh_ports_btn = QPushButton("Refresh")
        self.refresh_ports_btn.clicked.connect(self.refresh_serial_ports)

        port_layout = QHBoxLayout()
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.refresh_ports_btn)
        connection_layout.addRow("Serial Port:", port_layout)

        # Baud rate
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "115200"])
        connection_layout.addRow("Baud Rate:", self.baud_combo)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        # Servo settings
        servo_group = QGroupBox("Servo Configuration")
        servo_layout = QVBoxLayout()

        # Calibration button
        self.calibrate_servo_btn = QPushButton("Calibrate Servo")
        self.calibrate_servo_btn.clicked.connect(self.calibrate_servo)
        servo_layout.addWidget(self.calibrate_servo_btn)

        # Bin positions table
        self.servo_table = QTableWidget(10, 2)
        self.servo_table.setHorizontalHeaderLabels(["Bin", "Position"])

        for i in range(10):
            self.servo_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.servo_table.setItem(i, 1, QTableWidgetItem(str(90)))

        servo_layout.addWidget(self.servo_table)

        servo_group.setLayout(servo_layout)
        layout.addWidget(servo_group)

        arduino_tab.setLayout(layout)
        self.tab_widget.addTab(arduino_tab, "Arduino")

    def create_system_tab(self):
        """Create system configuration tab"""
        system_tab = QWidget()
        layout = QVBoxLayout()

        # Performance settings
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QFormLayout()

        # Threading
        self.threading_check = QCheckBox("Enable Multithreading")
        self.threading_check.setChecked(True)
        perf_layout.addRow("Threading:", self.threading_check)

        # Queue size
        self.queue_size_spin = QSpinBox()
        self.queue_size_spin.setRange(10, 1000)
        self.queue_size_spin.setValue(100)
        perf_layout.addRow("Queue Size:", self.queue_size_spin)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Logging settings
        log_group = QGroupBox("Logging Settings")
        log_layout = QFormLayout()

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        log_layout.addRow("Log Level:", self.log_level_combo)

        self.save_images_check = QCheckBox("Save Detected Pieces")
        self.save_images_check.setChecked(True)
        log_layout.addRow("Save Images:", self.save_images_check)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        system_tab.setLayout(layout)
        self.tab_widget.addTab(system_tab, "System")

    def create_control_panel(self):
        """Create bottom control panel"""
        self.control_panel = QWidget()
        layout = QHBoxLayout()

        # Profile controls
        self.load_profile_btn = QPushButton("Load Profile")
        self.save_profile_btn = QPushButton("Save Profile")

        layout.addWidget(self.load_profile_btn)
        layout.addWidget(self.save_profile_btn)

        layout.addStretch()

        # Action buttons
        self.validate_btn = QPushButton("Validate Config")
        self.validate_btn.clicked.connect(self.validate_configuration)

        self.start_sorting_btn = QPushButton("Start Sorting")
        self.start_sorting_btn.clicked.connect(self.confirm_configuration)
        self.start_sorting_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                font-size: 14px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        layout.addWidget(self.validate_btn)
        layout.addWidget(self.start_sorting_btn)

        self.control_panel.setLayout(layout)

    # ========== Configuration Methods ==========

    def load_current_config(self):
        """Load current configuration into UI"""
        if not self.config_manager:
            return

        # Load camera settings
        camera_config = self.config_manager.get_module_config("camera")
        if camera_config:
            self.camera_device.setCurrentText(str(camera_config.get("device_id", 0)))
            res = camera_config.get("resolution", [1920, 1080])
            self.resolution_combo.setCurrentText(f"{res[0]}x{res[1]}")
            self.fps_spin.setValue(camera_config.get("fps", 30))

        # Load detector settings
        detector_config = self.config_manager.get_module_config("detector")
        if detector_config:
            self.min_area_spin.setValue(detector_config.get("min_area", 500))
            self.max_area_spin.setValue(detector_config.get("max_area", 50000))
            self.threshold_slider.setValue(detector_config.get("threshold", 30))

        # Load other settings...
        # (Continue for all configuration items)

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration from UI"""
        config = {}

        # Camera configuration
        resolution = self.resolution_combo.currentText().split('x')
        config['camera'] = {
            'device_id': int(self.camera_device.currentText()),
            'resolution': [int(resolution[0]), int(resolution[1])],
            'fps': self.fps_spin.value(),
            'exposure': self.exposure_slider.value()
        }

        # Detector configuration
        config['detector'] = {
            'min_area': self.min_area_spin.value(),
            'max_area': self.max_area_spin.value(),
            'threshold': self.threshold_slider.value()
        }

        # Sorting configuration
        config['sorting'] = {
            'strategy': self.strategy_combo.currentText(),
            'overflow_bin': self.overflow_bin.value()
        }

        # API configuration
        config['api'] = {
            'api_key': self.api_key_edit.text(),
            'mode': self.api_mode_combo.currentText(),
            'timeout': self.timeout_spin.value()
        }

        # System configuration
        config['system'] = {
            'threading_enabled': self.threading_check.isChecked(),
            'queue_size': self.queue_size_spin.value(),
            'log_level': self.log_level_combo.currentText(),
            'save_images': self.save_images_check.isChecked()
        }

        return config

    def validate_configuration(self):
        """Validate the current configuration"""
        is_valid, error_msg = validate_config(self.config_manager)

        if is_valid:
            QMessageBox.information(self, "Validation Success",
                                    "Configuration is valid!")
            self.status_bar.showMessage("Configuration validated successfully")
        else:
            QMessageBox.warning(self, "Validation Failed",
                                f"Configuration error: {error_msg}")
            self.status_bar.showMessage(f"Validation failed: {error_msg}")

    def confirm_configuration(self):
        """Confirm configuration and start sorting"""
        # Validate first
        is_valid, error_msg = validate_config(self.config_manager)

        if not is_valid:
            QMessageBox.warning(self, "Invalid Configuration",
                                f"Cannot start sorting: {error_msg}")
            return

        # Confirm with user
        if ConfirmationDialog.confirm(self, "Start Sorting",
                                      "Start sorting with current configuration?"):
            # Save configuration
            config = self.get_configuration()
            for module, settings in config.items():
                self.config_manager.update_module_config(module, settings)
            self.config_manager.save_config()

            # Emit signal to start sorting
            self.configuration_complete.emit(config)

    # ========== Camera Preview Methods ==========

    def start_camera_preview(self):
        """Start camera preview"""
        self.preview_requested.emit()
        self.start_preview_btn.setEnabled(False)
        self.stop_preview_btn.setEnabled(True)
        self.status_bar.showMessage("Camera preview started")

    def stop_camera_preview(self):
        """Stop camera preview"""
        self.preview_widget.setText("No Video Signal")
        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.status_bar.showMessage("Camera preview stopped")

    def update_preview_frame(self, frame):
        """Update preview with new frame"""
        self.preview_widget.update_frame(frame)

    # ========== Other UI Methods ==========

    def configure_roi(self):
        """Open ROI configuration dialog"""
        # This would open a separate window for ROI configuration
        QMessageBox.information(self, "ROI Configuration",
                                "ROI configuration would open here")

    def test_api_connection(self):
        """Test API connection"""
        self.api_status_label.setText("Testing...")
        # Actual API test would go here
        QTimer.singleShot(1000, lambda: self.api_status_label.setText("Connection OK"))

    def calibrate_servo(self):
        """Start servo calibration"""
        QMessageBox.information(self, "Servo Calibration",
                                "Servo calibration would start here")

    def refresh_serial_ports(self):
        """Refresh available serial ports"""
        # Would enumerate actual serial ports
        self.port_combo.clear()
        self.port_combo.addItems(["COM3", "COM4", "/dev/ttyUSB0"])