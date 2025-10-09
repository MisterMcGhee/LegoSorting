"""
camera_tab.py - Camera configuration tab

FILE LOCATION: GUI/config_tabs/camera_tab.py

This tab provides camera configuration with device detection, testing, and live preview.
Users can select camera devices, adjust settings, and verify functionality before saving.

Features:
- Automatic camera detection (indices 0-2)
- Camera testing and validation
- Live camera preview using CameraViewRaw
- Resolution, FPS, and exposure configuration
- Real-time camera switching

Configuration Mapping:
    {
        "camera": {
            "device_id": 0,
            "buffer_size": 1,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "auto_exposure": true
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QComboBox, QPushButton, QSpinBox,
                             QCheckBox, QLabel, QSlider, QApplication)
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
import cv2
import logging
from typing import Dict, Any, Optional

# Import base class and widgets
from GUI.config_tabs.base_tab import BaseConfigTab
from GUI.widgets.camera_view import CameraViewRaw
from camera_module import create_camera
from enhanced_config_manager import ModuleConfig

# Initialize logger
logger = logging.getLogger(__name__)


class CameraConfigTab(BaseConfigTab):
    """
    Camera configuration tab with device detection and live preview.

    This tab allows users to:
    - Detect and test available cameras
    - Configure camera settings (resolution, FPS, exposure)
    - View live camera preview
    - Save configuration

    Signals:
        camera_changed(int): Emitted when camera device changes
        preview_started(): Emitted when preview begins
        preview_stopped(): Emitted when preview stops
    """

    # Custom signals
    camera_changed = pyqtSignal(int)  # Device ID
    preview_started = pyqtSignal()
    preview_stopped = pyqtSignal()

    def __init__(self, config_manager, camera=None, parent=None):
        """
        Initialize camera configuration tab.

    Args:
        config_manager: Configuration manager instance
        camera: Shared camera module reference
        parent: Parent widget

        """
        # Initialize camera-related attributes before calling super().__init__
        self.detected_cameras = {}  # {device_id: camera_info}
        self.camera = camera  # Use shared camera reference
        self.preview_active = False

        # Call parent init
        super().__init__(config_manager, parent)

        # Initialize UI and load config
        self.init_ui()
        self.load_config()

        # Auto-detect cameras on startup
        QTimer.singleShot(100, self.detect_cameras)

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (Required by BaseConfigTab)
    # ========================================================================

    def get_module_name(self) -> str:
        """Return the configuration module name this tab manages."""
        return ModuleConfig.CAMERA.value

    def init_ui(self):
        """Create the camera configuration user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create sections
        self.create_detection_section(main_layout)
        self.create_settings_section(main_layout)
        self.create_preview_section(main_layout)

        # Add stretch at bottom
        main_layout.addStretch()

        self.log_info("Camera tab UI initialized")

    def load_config(self) -> bool:
        """Load camera configuration and populate UI."""
        try:
            config = self.get_module_config(self.get_module_name())

            if config:
                # Set device ID (will be set after camera detection)
                device_id = config.get("device_id", 0)

                # Set resolution
                width = config.get("width", 1920)
                height = config.get("height", 1080)
                resolution_str = f"{width}x{height}"
                index = self.resolution_combo.findText(resolution_str)
                if index >= 0:
                    self.resolution_combo.setCurrentIndex(index)

                # Set FPS
                self.fps_spin.setValue(config.get("fps", 30))

                # Set buffer size
                self.buffer_spin.setValue(config.get("buffer_size", 1))

                # Set auto exposure
                self.auto_exposure_check.setChecked(config.get("auto_exposure", True))

                self.log_info(f"Configuration loaded: Device {device_id}")
                self.clear_modified()
                return True
            else:
                self.log_warning("No camera configuration found, using defaults")
                return False

        except Exception as e:
            self.log_error(f"Error loading configuration: {e}")
            return False

    def save_config(self) -> bool:
        """Save camera configuration."""
        if not self.validate():
            return False

        try:
            config = self.get_config()
            self.config_manager.update_module_config(
                self.get_module_name(),
                config
            )

            self.log_info("Camera configuration saved")
            self.clear_modified()
            return True

        except Exception as e:
            self.log_error(f"Error saving configuration: {e}")
            self.show_validation_error(f"Failed to save: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """Return current camera configuration as dictionary."""
        # Parse resolution
        resolution_str = self.resolution_combo.currentText()
        width, height = map(int, resolution_str.split('x'))

        # Get device ID
        device_id = self.device_combo.currentData()
        if device_id is None:
            device_id = 0

        return {
            "device_id": device_id,
            "buffer_size": self.buffer_spin.value(),
            "width": width,
            "height": height,
            "fps": self.fps_spin.value(),
            "auto_exposure": self.auto_exposure_check.isChecked()
        }

    def validate(self) -> bool:
        """Validate camera configuration."""
        # Check if a camera device is selected
        if self.device_combo.count() == 0:
            self.show_validation_error("No camera device selected")
            return False

        device_id = self.device_combo.currentData()
        if device_id is None:
            self.show_validation_error("Invalid camera device")
            return False

        # Warn if camera hasn't been tested
        if device_id not in self.detected_cameras:
            self.log_warning(f"Camera {device_id} has not been tested")

        return True

    # ========================================================================
    # UI CREATION METHODS
    # ========================================================================

    def create_detection_section(self, parent_layout):
        """Create camera detection and selection section."""
        detection_group = QGroupBox("Camera Detection")
        detection_layout = QFormLayout()

        # Device selection row
        device_row = QHBoxLayout()

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(250)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        device_row.addWidget(self.device_combo)

        self.detect_btn = QPushButton("🔄 Detect Cameras")
        self.detect_btn.clicked.connect(self.detect_cameras)
        self.detect_btn.setToolTip("Scan for available camera devices")
        device_row.addWidget(self.detect_btn)

        self.test_btn = QPushButton("Test Camera")
        self.test_btn.clicked.connect(self.test_camera)
        self.test_btn.setToolTip("Verify selected camera is working")
        device_row.addWidget(self.test_btn)

        device_row.addStretch()
        detection_layout.addRow("Camera Device:", device_row)

        # Status label
        self.status_label = QLabel("Status: Not detected")
        self.status_label.setStyleSheet("color: gray; font-weight: bold;")
        detection_layout.addRow("Status:", self.status_label)

        # Camera info label
        self.info_label = QLabel("No camera information")
        self.info_label.setStyleSheet("color: gray; font-style: italic;")
        detection_layout.addRow("Info:", self.info_label)

        detection_group.setLayout(detection_layout)
        parent_layout.addWidget(detection_group)

    def create_settings_section(self, parent_layout):
        """Create camera settings section."""
        settings_group = QGroupBox("Camera Settings")
        settings_layout = QFormLayout()

        # Resolution
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "1920x1080",  # Full HD
            "1280x720",  # HD
            "640x480"  # VGA
        ])
        self.resolution_combo.currentIndexChanged.connect(self.mark_modified)
        settings_layout.addRow("Resolution:", self.resolution_combo)

        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self.mark_modified)
        settings_layout.addRow("Target FPS:", self.fps_spin)

        # Buffer size
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(1, 10)
        self.buffer_spin.setValue(1)
        self.buffer_spin.setToolTip("Number of frames to buffer (usually 1)")
        self.buffer_spin.valueChanged.connect(self.mark_modified)
        settings_layout.addRow("Buffer Size:", self.buffer_spin)

        # Auto exposure
        self.auto_exposure_check = QCheckBox("Enable auto-exposure")
        self.auto_exposure_check.setChecked(True)
        self.auto_exposure_check.stateChanged.connect(self.mark_modified)
        settings_layout.addRow("Exposure:", self.auto_exposure_check)

        settings_group.setLayout(settings_layout)
        parent_layout.addWidget(settings_group)

    def create_preview_section(self, parent_layout):
        """Create camera preview section."""
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()

        # Camera view widget
        self.preview_widget = CameraViewRaw()
        self.preview_widget.set_fps_visible(True)
        self.preview_widget.setMinimumHeight(400)
        preview_layout.addWidget(self.preview_widget)

        # Preview controls
        controls_layout = QHBoxLayout()

        self.start_preview_btn = QPushButton("▶ Start Preview")
        self.start_preview_btn.clicked.connect(self.start_preview)
        self.start_preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        controls_layout.addWidget(self.start_preview_btn)

        self.stop_preview_btn = QPushButton("⏹ Stop Preview")
        self.stop_preview_btn.clicked.connect(self.stop_preview)
        self.stop_preview_btn.setEnabled(False)
        self.stop_preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        controls_layout.addWidget(self.stop_preview_btn)

        controls_layout.addStretch()

        self.preview_status_label = QLabel("Preview inactive")
        self.preview_status_label.setStyleSheet("color: gray;")
        controls_layout.addWidget(self.preview_status_label)

        preview_layout.addLayout(controls_layout)

        preview_group.setLayout(preview_layout)
        parent_layout.addWidget(preview_group)

    # ========================================================================
    # CAMERA DETECTION AND TESTING
    # ========================================================================

    def detect_cameras(self):
        """Detect available cameras by testing indices 0-2."""
        self.log_info("Detecting cameras...")
        self.status_label.setText("Status: Detecting...")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.detect_btn.setEnabled(False)
        QApplication.processEvents()

        # Clear previous detections
        self.detected_cameras.clear()
        self.device_combo.clear()

        # Test camera indices 0-2
        for device_id in range(3):
            self.log_debug(f"Testing camera index {device_id}...")

            try:
                cap = cv2.VideoCapture(device_id)

                if cap.isOpened():
                    ret, frame = cap.read()

                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))

                        self.detected_cameras[device_id] = {
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'tested': True
                        }

                        # Add to combo box
                        self.device_combo.addItem(
                            f"Camera {device_id} ({width}x{height} @ {fps}fps)",
                            userData=device_id
                        )

                        self.log_info(f"✓ Found camera {device_id}: {width}x{height}")

                    cap.release()

            except Exception as e:
                self.log_debug(f"Camera {device_id} not available: {e}")

        # Update UI based on results
        if self.detected_cameras:
            count = len(self.detected_cameras)
            self.status_label.setText(f"Status: Found {count} camera(s)")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.log_info(f"Detection complete: {count} camera(s) found")

            # Try to restore previous device selection
            config = self.get_module_config(self.get_module_name())
            if config:
                device_id = config.get("device_id", 0)
                for i in range(self.device_combo.count()):
                    if self.device_combo.itemData(i) == device_id:
                        self.device_combo.setCurrentIndex(i)
                        break
        else:
            self.status_label.setText("Status: No cameras found")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.log_warning("No cameras detected")

            # Add fallback options
            for i in range(3):
                self.device_combo.addItem(f"Camera {i} (not tested)", userData=i)

        self.detect_btn.setEnabled(True)

    def test_camera(self):
        """Test the currently selected camera."""
        device_id = self.device_combo.currentData()

        if device_id is None:
            self.show_validation_error("No camera device selected")
            return

        self.log_info(f"Testing camera {device_id}...")
        self.status_label.setText(f"Status: Testing camera {device_id}...")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.test_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            cap = cv2.VideoCapture(device_id)

            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    self.status_label.setText(f"Status: Camera {device_id} works!")
                    self.status_label.setStyleSheet("color: green; font-weight: bold;")
                    self.log_info(f"✓ Camera {device_id} test successful")
                else:
                    self.status_label.setText(f"Status: Camera {device_id} failed to capture")
                    self.status_label.setStyleSheet("color: red; font-weight: bold;")
                    self.log_error(f"Camera {device_id} opened but failed to capture frame")
            else:
                self.status_label.setText(f"Status: Cannot open camera {device_id}")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                self.log_error(f"Cannot open camera {device_id}")

        except Exception as e:
            self.status_label.setText(f"Status: Test failed - {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.log_error(f"Camera test error: {e}")

        finally:
            self.test_btn.setEnabled(True)

    def on_device_changed(self, index):
        """Handle camera device selection change."""
        device_id = self.device_combo.currentData()

        if device_id is None:
            return

        self.log_info(f"Camera device changed to {device_id}")

        # Update info label
        if device_id in self.detected_cameras:
            info = self.detected_cameras[device_id]
            self.info_label.setText(
                f"{info['width']}x{info['height']} @ {info['fps']}fps"
            )
            self.info_label.setStyleSheet("color: green; font-style: italic;")
        else:
            self.info_label.setText("Camera not tested")
            self.info_label.setStyleSheet("color: orange; font-style: italic;")

        # Stop preview if active (camera changed)
        if self.preview_active:
            self.stop_preview()

        # Mark as modified and emit signal
        self.mark_modified()
        self.camera_changed.emit(device_id)

    # ========================================================================
    # PREVIEW CONTROL
    # ========================================================================

    def start_preview(self):
        """Start live camera preview by registering widget as consumer."""
        device_id = self.device_combo.currentData()

        if device_id is None:
            self.show_validation_error("No camera device selected")
            return

        self.log_info(f"Starting camera preview for device {device_id}...")

        try:
            # Verify we have shared camera reference
            if not self.camera:
                raise Exception("No shared camera reference available")

            # Unregister first if already registered
            try:
                self.camera.unregister_consumer("camera_config_preview")
            except:
                pass  # Ignore if not registered

            # Register camera tab's CameraViewRaw widget as a consumer
            # This widget is SEPARATE from detector tab's CameraViewROI widget
            # Both receive frames from the SAME camera_module
            success = self.camera.register_consumer(
                name="camera_config_preview",
                callback=self.preview_widget.receive_frame,
                processing_type="async",
                priority=20
            )

            if not success:
                raise Exception("Failed to register preview consumer")

            # Update UI
            self.preview_active = True
            self.start_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(True)
            self.preview_status_label.setText("Preview active")
            self.preview_status_label.setStyleSheet("color: green; font-weight: bold;")

            self.log_info("✓ Camera preview started successfully")
            self.preview_started.emit()

        except Exception as e:
            self.log_error(f"Failed to start preview: {e}")
            self.show_validation_error(f"Preview failed: {e}")
            self.stop_preview()
    def stop_preview(self):
        """Stop live camera preview."""
        self.log_info("Stopping camera preview...")

        try:
            if self.camera and self.preview_active:
                # Only unregister THIS tab's consumer
                # DON'T stop or release - other tabs may be using the camera!
                self.camera.unregister_consumer("camera_config_preview")

            # Clear preview widget display
            if self.preview_widget:
                self.preview_widget.clear_display()

            # Update UI
            self.preview_active = False
            self.start_preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(False)
            self.preview_status_label.setText("Preview stopped")
            self.preview_status_label.setStyleSheet("color: gray;")

            self.log_info("✓ Camera preview stopped")
            self.preview_stopped.emit()

        except Exception as e:
            self.log_error(f"Error stopping preview: {e}")

    def cleanup(self):
        """Clean up resources when tab is closed."""
        self.log_info("Cleaning up camera tab...")

        # Stop preview (only unregisters consumer)
        if self.preview_active:
            self.stop_preview()

        # Note: We do NOT release self.camera because it's shared!
        # configuration_gui.py owns the camera lifecycle

        self.log_info("✓ Camera tab cleanup complete")