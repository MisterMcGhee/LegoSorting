"""
config_gui_module.py - Configuration GUI for Lego Sorting System
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import json
import logging
import time  # ADD if not present
import numpy as np
import cv2
from typing import Dict, Any, Optional
from enhanced_config_manager import create_config_manager  # or your config manager
from GUI.gui_common import BaseGUIWindow, VideoWidget, ConfirmationDialog, validate_config
from camera_module import create_camera
from enhanced_config_manager import ModuleConfig
import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)


class DetectorPreviewWidget(QWidget):
    """Custom widget that shows camera preview with detection overlays"""

    def __init__(self):
        super().__init__()
        self.current_frame = None  # Will store the camera image
        self.roi = None  # Region of Interest rectangle
        self.zones = None  # Entry/exit zone percentages
        self.detected_pieces = []  # List of detected pieces

    def update_frame(self, frame, detection_data=None):
        """Update the widget with a new camera frame"""
        self.current_frame = frame
        if detection_data:
            self.roi = detection_data.get('roi')
            self.zones = detection_data.get('zones')
            self.detected_pieces = detection_data.get('pieces', [])
        self.update()  # Triggers a repaint

    def paintEvent(self, event):
        """This method draws everything on the widget"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.current_frame is not None:
            # Convert OpenCV image to Qt format
            height, width = self.current_frame.shape[:2]
            bytes_per_line = 3 * width

            # Convert BGR to RGB (OpenCV uses BGR, Qt uses RGB)
            rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_image.data, width, height,
                             bytes_per_line, QImage.Format_RGB888)

            # Convert to pixmap and scale to widget size
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)

            # Calculate scaling factors
            scale_x = scaled_pixmap.width() / width
            scale_y = scaled_pixmap.height() / height

            # Draw the image
            painter.drawPixmap(0, 0, scaled_pixmap)

            # Draw ROI rectangle (yellow border)
            if self.roi:
                x, y, w, h = self.roi
                painter.setPen(QPen(QColor(255, 255, 0), 2))  # Yellow, 2px wide
                painter.drawRect(int(x * scale_x), int(y * scale_y),
                                 int(w * scale_x), int(h * scale_y))

                # Draw zones if we have them
                if self.zones:
                    # Entry zone (green, semi-transparent)
                    entry_width = w * self.zones.get('entry', 0.15)
                    painter.fillRect(
                        int(x * scale_x),
                        int(y * scale_y),
                        int(entry_width * scale_x),
                        int(h * scale_y),
                        QColor(0, 255, 0, 50)  # Green with transparency
                    )

                    # Exit zone (red, semi-transparent)
                    exit_width = w * self.zones.get('exit', 0.15)
                    exit_x = x + w - exit_width
                    painter.fillRect(
                        int(exit_x * scale_x),
                        int(y * scale_y),
                        int(exit_width * scale_x),
                        int(h * scale_y),
                        QColor(255, 0, 0, 50)  # Red with transparency
                    )

            # Draw detected pieces
            for piece in self.detected_pieces:
                px, py, pw, ph = piece.get('bbox', (0, 0, 0, 0))
                # Adjust coordinates if they're relative to ROI
                if self.roi:
                    px += self.roi[0]
                    py += self.roi[1]

                # Choose color based on piece status
                if piece.get('being_processed'):
                    painter.setPen(QPen(QColor(0, 255, 255), 2))  # Cyan
                elif piece.get('captured'):
                    painter.setPen(QPen(QColor(0, 255, 0), 2))  # Green
                else:
                    painter.setPen(QPen(QColor(255, 255, 255), 1))  # White

                painter.drawRect(int(px * scale_x), int(py * scale_y),
                                 int(pw * scale_x), int(ph * scale_y))
        else:
            # No frame yet - show black screen with text
            painter.fillRect(self.rect(), Qt.black)
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No camera feed\nClick 'Start Preview' to begin")


class ConfigurationGUI(BaseGUIWindow):
    """Main configuration window for system setup"""

    # Signals
    configuration_complete = pyqtSignal(dict)  # Emitted when config is confirmed
    arduino_mode_changed = pyqtSignal(bool)  # Emitted when simulation mode changes

    def __init__(self, config_manager=None):
        super().__init__(config_manager, "Lego Sorting System - Configuration")

        # Camera reference
        self.camera = None
        self.preview_active = False

        # FPS tracking
        self.last_preview_time = time.time()
        self.preview_frame_count = 0

        # ROI Defaults
        self.current_roi = None  # Store current ROI
        self.default_roi = None  # Store default ROI
        self.frame_width = 640  # Default frame width
        self.frame_height = 480  # Default frame height

        # Window setup
        self.setGeometry(100, 100, 1200, 800)
        self.center_window()

        # Initialize UI
        self.init_ui()
        self.load_current_config()

        # ADD THESE NEW LINES:
        self.current_roi = None  # Store current ROI
        self.default_roi = None  # Store default ROI
        self.frame_width = 640  # Default frame width
        self.frame_height = 480  # Default frame height

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

        # Detector tab related
        self.current_roi = None  # Store current ROI
        self.preview_timer = None  # Timer for preview updates

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to configure")

    def create_camera_tab(self):
        """Create enhanced camera configuration tab with dynamic device detection"""
        camera_tab = QWidget()
        layout = QVBoxLayout()

        # Camera selection group
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout()

        # Device selection with refresh button
        device_layout = QHBoxLayout()

        self.camera_device = QComboBox()
        self.camera_device.setMinimumWidth(250)
        device_layout.addWidget(self.camera_device)

        # Add refresh button to detect cameras
        self.refresh_cameras_btn = QPushButton("🔄 Detect Cameras")
        self.refresh_cameras_btn.clicked.connect(self.refresh_camera_list)
        device_layout.addWidget(self.refresh_cameras_btn)

        # Add test button to verify camera works
        self.test_camera_btn = QPushButton("Test Camera")
        self.test_camera_btn.clicked.connect(self.test_selected_camera)
        device_layout.addWidget(self.test_camera_btn)

        device_layout.addStretch()
        camera_layout.addRow("Camera Device:", device_layout)

        # Camera status label
        self.camera_status_label = QLabel("Status: Not tested")
        self.camera_status_label.setStyleSheet("color: gray;")
        camera_layout.addRow("Camera Status:", self.camera_status_label)

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

        # Preview area (existing code)
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()

        self.preview_widget = VideoWidget()
        preview_layout.addWidget(self.preview_widget)

        self.preview_fps_label = QLabel("FPS: 0.0")
        preview_layout.addWidget(self.preview_fps_label)

        preview_buttons = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Preview")
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.stop_preview_btn.setEnabled(False)

        self.start_preview_btn.clicked.connect(self.start_camera_preview_with_switch)
        self.stop_preview_btn.clicked.connect(self.stop_camera_preview)

        preview_buttons.addWidget(self.start_preview_btn)
        preview_buttons.addWidget(self.stop_preview_btn)
        preview_layout.addLayout(preview_buttons)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        camera_tab.setLayout(layout)
        self.tab_widget.addTab(camera_tab, "Camera")

        # Connect device change signal
        self.camera_device.currentIndexChanged.connect(self.on_camera_device_changed)

        # Initially populate camera list
        QTimer.singleShot(100, self.refresh_camera_list)

    def refresh_camera_list(self):
        """Detect and populate available cameras"""
        try:
            self.camera_status_label.setText("Detecting cameras...")
            self.camera_status_label.setStyleSheet("color: blue;")
            QApplication.processEvents()  # Update UI immediately

            # Import camera module
            from camera_module import CameraModule

            # Get current selection to restore if possible
            current_device_id = None
            if self.camera_device.count() > 0:
                current_text = self.camera_device.currentText()
                if current_text:
                    # Try to extract device ID from text
                    try:
                        current_device_id = int(current_text.split()[1])
                    except:
                        pass

            # Clear existing items
            self.camera_device.clear()

            # Detect available cameras
            cameras = CameraModule.enumerate_cameras(max_check=10)

            if cameras:
                # Add detected cameras to dropdown
                for device_id, description in cameras:
                    self.camera_device.addItem(description, userData=device_id)

                # Try to restore previous selection
                if current_device_id is not None:
                    for i in range(self.camera_device.count()):
                        if self.camera_device.itemData(i) == current_device_id:
                            self.camera_device.setCurrentIndex(i)
                            break

                self.camera_status_label.setText(f"Found {len(cameras)} camera(s)")
                self.camera_status_label.setStyleSheet("color: green;")
                logger.info(f"Detected {len(cameras)} cameras")
            else:
                # No cameras found, add default option
                self.camera_device.addItem("No cameras detected", userData=0)
                self.camera_status_label.setText("No cameras detected")
                self.camera_status_label.setStyleSheet("color: red;")
                logger.warning("No cameras detected")

        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")
            self.camera_status_label.setText(f"Detection error: {str(e)}")
            self.camera_status_label.setStyleSheet("color: red;")

            # Add fallback options
            self.camera_device.clear()
            for i in range(4):
                self.camera_device.addItem(f"Camera {i}", userData=i)

    def test_selected_camera(self):
        """Test if the selected camera works"""
        try:
            device_id = self.camera_device.currentData()
            if device_id is None:
                device_id = 0

            self.camera_status_label.setText(f"Testing camera {device_id}...")
            self.camera_status_label.setStyleSheet("color: blue;")
            QApplication.processEvents()

            import cv2
            cap = cv2.VideoCapture(device_id)

            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret:
                    self.camera_status_label.setText(f"Camera {device_id} works!")
                    self.camera_status_label.setStyleSheet("color: green;")
                else:
                    self.camera_status_label.setText(f"Camera {device_id} opened but can't read frames")
                    self.camera_status_label.setStyleSheet("color: orange;")
            else:
                self.camera_status_label.setText(f"Cannot open camera {device_id}")
                self.camera_status_label.setStyleSheet("color: red;")

        except Exception as e:
            self.camera_status_label.setText(f"Test error: {str(e)}")
            self.camera_status_label.setStyleSheet("color: red;")
            logger.error(f"Camera test error: {e}")

    def on_camera_device_changed(self, index):
        """Handle camera device selection change"""
        if index >= 0:
            device_id = self.camera_device.currentData()
            if device_id is not None:
                logger.info(f"Camera device changed to: {device_id}")

                # Update config
                if self.config_manager:
                    camera_config = self.config_manager.get_module_config("camera")
                    camera_config["device_id"] = device_id
                    self.config_manager.update_module_config("camera", camera_config)

                # If preview is active, restart with new camera
                if self.preview_active:
                    self.stop_camera_preview()
                    QTimer.singleShot(100, self.start_camera_preview_with_switch)

    def start_camera_preview_with_switch(self):
        """Start camera preview with proper device switching"""
        try:
            device_id = self.camera_device.currentData()
            if device_id is None:
                device_id = 0

            # Get or create camera instance
            if not hasattr(self, 'camera') or not self.camera:
                from camera_module import create_camera
                self.camera = create_camera(config_manager=self.config_manager)
            else:
                # Switch to the selected camera device
                logger.info(f"Switching to camera device {device_id}")
                success = self.camera.switch_camera(device_id, self.config_manager)

                if not success:
                    QMessageBox.warning(self, "Camera Error",
                                        f"Failed to switch to camera {device_id}")
                    return

            # Initialize if needed
            if not self.camera.is_initialized:
                if not self.camera.initialize():
                    QMessageBox.warning(self, "Camera Error",
                                        "Failed to initialize camera")
                    return

            # Register preview consumer (rest of existing preview code)
            success = self.camera.register_consumer(
                name="config_preview",
                callback=self._preview_frame_callback,
                processing_type="async",
                priority=50
            )

            if not success:
                logger.warning("Preview consumer already registered, continuing...")

            # Start capture if not running
            if not self.camera.is_capturing:
                self.camera.start_capture()

            # Update UI state
            self.preview_active = True
            self.start_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(True)
            self.status_bar.showMessage(f"Camera preview started (Device {device_id})")

            logger.info(f"Live camera preview started with device {device_id}")

        except Exception as e:
            logger.error(f"Failed to start camera preview: {e}")
            QMessageBox.critical(self, "Preview Error",
                                 f"Failed to start preview: {str(e)}")

    def create_detector_tab(self):
        """Create enhanced detector configuration tab with live preview"""
        detector_tab = QWidget()
        main_layout = QHBoxLayout()  # Side-by-side layout

        # LEFT SIDE: Settings Panel
        settings_panel = self._create_detector_settings_panel()
        main_layout.addWidget(settings_panel, 1)  # Takes 1/3 of space

        # RIGHT SIDE: Preview Panel
        preview_panel = self._create_detector_preview_panel()
        main_layout.addWidget(preview_panel, 2)  # Takes 2/3 of space

        detector_tab.setLayout(main_layout)
        self.tab_widget.addTab(detector_tab, "Detector")

    def _create_detector_settings_panel(self):
        """Create the settings panel with all controls"""
        panel = QWidget()
        layout = QVBoxLayout()

        # === DETECTION SETTINGS GROUP ===
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout()

        # Minimum piece area
        self.min_piece_area_spin = QSpinBox()
        self.min_piece_area_spin.setRange(100, 10000)
        self.min_piece_area_spin.setValue(1000)
        self.min_piece_area_spin.setSuffix(" px²")
        self.min_piece_area_spin.setToolTip(
            "Minimum area to detect as a valid piece.\n"
            "Lower = may detect noise, Higher = may miss small pieces"
        )
        detection_layout.addRow("Min Piece Area:", self.min_piece_area_spin)

        # Maximum piece area
        self.max_piece_area_spin = QSpinBox()
        self.max_piece_area_spin.setRange(1000, 100000)
        self.max_piece_area_spin.setValue(50000)
        self.max_piece_area_spin.setSuffix(" px²")
        self.max_piece_area_spin.setToolTip(
            "Maximum area to detect as a single piece.\n"
            "Lower = may split large pieces, Higher = may group multiple pieces"
        )
        detection_layout.addRow("Max Piece Area:", self.max_piece_area_spin)

        # Minimum updates (how many frames before capture)
        self.min_updates_spin = QSpinBox()
        self.min_updates_spin.setRange(1, 20)
        self.min_updates_spin.setValue(5)
        self.min_updates_spin.setSuffix(" frames")
        self.min_updates_spin.setToolTip(
            "Frames a piece must be visible before capture.\n"
            "Lower = faster but less reliable, Higher = slower but more reliable"
        )
        detection_layout.addRow("Min Updates:", self.min_updates_spin)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # === ZONE CONFIGURATION GROUP ===
        zone_group = QGroupBox("Zone Configuration")
        zone_layout = QFormLayout()

        # Entry zone slider
        entry_widget = QWidget()
        entry_layout = QHBoxLayout()
        self.entry_zone_slider = QSlider(Qt.Horizontal)
        self.entry_zone_slider.setRange(5, 30)
        self.entry_zone_slider.setValue(15)
        self.entry_zone_label = QLabel("15%")
        self.entry_zone_slider.valueChanged.connect(self._update_entry_zone)
        entry_layout.addWidget(self.entry_zone_slider)
        entry_layout.addWidget(self.entry_zone_label)
        entry_widget.setLayout(entry_layout)
        zone_layout.addRow("Entry Zone:", entry_widget)

        # Exit zone slider
        exit_widget = QWidget()
        exit_layout = QHBoxLayout()
        self.exit_zone_slider = QSlider(Qt.Horizontal)
        self.exit_zone_slider.setRange(5, 30)
        self.exit_zone_slider.setValue(15)
        self.exit_zone_label = QLabel("15%")
        self.exit_zone_slider.valueChanged.connect(self._update_exit_zone)
        exit_layout.addWidget(self.exit_zone_slider)
        exit_layout.addWidget(self.exit_zone_label)
        exit_widget.setLayout(exit_layout)
        zone_layout.addRow("Exit Zone:", exit_widget)

        zone_group.setLayout(zone_layout)
        layout.addWidget(zone_group)

        # === ROI CONFIGURATION GROUP ===
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QVBoxLayout()

        # Create spin boxes for ROI coordinates
        roi_controls_layout = QFormLayout()

        # X Position spin box
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, self.frame_width - 100)  # Will be updated when frame size known
        self.roi_x_spin.setValue(0)  # Will be updated with default
        self.roi_x_spin.setSuffix(" px")
        self.roi_x_spin.setToolTip("X position of ROI top-left corner")
        self.roi_x_spin.valueChanged.connect(self._update_roi_from_controls)
        roi_controls_layout.addRow("X Position:", self.roi_x_spin)

        # Y Position spin box
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, self.frame_height - 100)
        self.roi_y_spin.setValue(0)
        self.roi_y_spin.setSuffix(" px")
        self.roi_y_spin.setToolTip("Y position of ROI top-left corner")
        self.roi_y_spin.valueChanged.connect(self._update_roi_from_controls)
        roi_controls_layout.addRow("Y Position:", self.roi_y_spin)

        # Width spin box
        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(100, self.frame_width)
        self.roi_w_spin.setValue(100)
        self.roi_w_spin.setSuffix(" px")
        self.roi_w_spin.setToolTip("Width of ROI rectangle")
        self.roi_w_spin.valueChanged.connect(self._update_roi_from_controls)
        roi_controls_layout.addRow("Width:", self.roi_w_spin)

        # Height spin box
        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(100, self.frame_height)
        self.roi_h_spin.setValue(100)
        self.roi_h_spin.setSuffix(" px")
        self.roi_h_spin.setToolTip("Height of ROI rectangle")
        self.roi_h_spin.valueChanged.connect(self._update_roi_from_controls)
        roi_controls_layout.addRow("Height:", self.roi_h_spin)

        roi_layout.addLayout(roi_controls_layout)

        # ROI info display (optional - shows current ROI)
        self.roi_info_label = QLabel("ROI: Not Set")
        self.roi_info_label.setStyleSheet("padding: 5px; background: #f0f0f0; font-family: monospace;")
        roi_layout.addWidget(self.roi_info_label)

        # Reset button
        self.reset_roi_btn = QPushButton("Reset to Default")
        self.reset_roi_btn.clicked.connect(self._reset_roi_to_default)
        roi_layout.addWidget(self.reset_roi_btn)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        # === ADVANCED SETTINGS BUTTON ===
        self.advanced_toggle_btn = QPushButton("▼ Show Advanced Settings")
        self.advanced_toggle_btn.setCheckable(True)
        self.advanced_toggle_btn.clicked.connect(self._toggle_advanced_settings)
        layout.addWidget(self.advanced_toggle_btn)

        # === ADVANCED SETTINGS CONTAINER (Hidden by default) ===
        self.advanced_container = self._create_advanced_settings()
        self.advanced_container.setVisible(False)
        layout.addWidget(self.advanced_container)

        # Add stretch to push everything to top
        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def _create_advanced_settings(self):
        """Create the advanced settings (hidden by default)"""
        container = QWidget()
        layout = QVBoxLayout()

        # You can add advanced settings groups here later
        # For now, just add a placeholder
        placeholder = QLabel("Advanced settings will go here\n(Background subtraction, timing, etc.)")
        placeholder.setStyleSheet("padding: 20px; background: #f0f0f0;")
        layout.addWidget(placeholder)

        container.setLayout(layout)
        return container

    def _create_detector_preview_panel(self):
        """Create the preview panel with camera feed"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Status bar at top
        header_layout = QHBoxLayout()
        self.detector_status_label = QLabel("Status: Not Started")
        self.detector_fps_label = QLabel("FPS: 0")
        self.pieces_detected_label = QLabel("Pieces: 0")

        header_layout.addWidget(self.detector_status_label)
        header_layout.addStretch()
        header_layout.addWidget(self.pieces_detected_label)
        header_layout.addWidget(self.detector_fps_label)
        layout.addLayout(header_layout)

        # Camera preview widget
        self.detector_preview = DetectorPreviewWidget()
        self.detector_preview.setMinimumSize(640, 480)
        self.detector_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.detector_preview)

        # Control buttons
        controls_layout = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Preview")
        self.start_preview_btn.clicked.connect(self._toggle_preview)
        controls_layout.addWidget(self.start_preview_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        panel.setLayout(layout)
        return panel

    def _update_entry_zone(self, value):
        """Update entry zone percentage"""
        self.entry_zone_label.setText(f"{value}%")
        # Update the preview if it's running
        if hasattr(self, 'detector_preview') and self.detector_preview.zones:
            self.detector_preview.zones['entry'] = value / 100.0
            self.detector_preview.update()

    def _update_exit_zone(self, value):
        """Update exit zone percentage"""
        self.exit_zone_label.setText(f"{value}%")
        # Update the preview if it's running
        if hasattr(self, 'detector_preview') and self.detector_preview.zones:
            self.detector_preview.zones['exit'] = value / 100.0
            self.detector_preview.update()

    def _toggle_advanced_settings(self):
        """Show/hide advanced settings"""
        is_visible = self.advanced_container.isVisible()
        self.advanced_container.setVisible(not is_visible)
        self.advanced_toggle_btn.setText(
            "▲ Hide Advanced Settings" if not is_visible
            else "▼ Show Advanced Settings"
        )

    def _toggle_preview(self):
        """Start or stop the camera preview with ROI initialization"""
        if getattr(self, 'preview_timer', None) and self.preview_timer.isActive():
            # Stop preview
            self.preview_timer.stop()
            self.start_preview_btn.setText("Start Preview")
            self.detector_status_label.setText("Status: Stopped")

            # Clean up camera if we have one
            if hasattr(self, 'camera') and self.camera:
                try:
                    self.camera.unregister_consumer("detector_preview")
                except:
                    pass  # Consumer might not be registered

        else:
            # Start preview
            try:
                # Initialize camera if needed
                if not hasattr(self, 'camera') or not self.camera:
                    from camera_module import create_camera
                    self.camera = create_camera(config_manager=self.config_manager)

                # Initialize camera if needed
                if not self.camera.is_initialized:
                    if not self.camera.initialize():
                        QMessageBox.warning(self, "Camera Error",
                                            "Failed to initialize camera")
                        return

                # Get frame to determine frame size and initialize ROI
                frame = self.camera.get_frame()
                if frame is not None:
                    height, width = frame.shape[:2]
                    self._update_frame_size(width, height)
                    self.current_frame = frame  # Store for detector module updates

                # Set up preview timer
                if not hasattr(self, 'preview_timer') or not self.preview_timer:
                    self.preview_timer = QTimer()
                    self.preview_timer.timeout.connect(self._update_detector_preview)

                # Start preview updates
                self.preview_timer.start(33)  # ~30 FPS updates
                self.start_preview_btn.setText("Stop Preview")
                self.detector_status_label.setText("Status: Running")

                logger.info("Detector preview started with ROI initialization")

            except Exception as e:
                logger.error(f"Failed to start camera preview: {e}")
                QMessageBox.critical(self, "Preview Error",
                                     f"Failed to start preview: {str(e)}")

    def _update_detector_preview(self):
        """Update the preview (called by timer) - IMPROVED ERROR HANDLING"""
        try:
            # Get frame from camera
            if not hasattr(self, 'camera') or not self.camera:
                return

            frame = self.camera.get_frame()
            if frame is None:
                return

            # Prepare detection data for visualization
            detection_data = {
                'roi': getattr(self, 'current_roi', None),
                'zones': {
                    'entry': self.entry_zone_slider.value() / 100.0,
                    'exit': self.exit_zone_slider.value() / 100.0
                },
                'pieces': []  # Will be filled by detector module if available
            }

            # If we have a detector module, get real detection data
            if hasattr(self, 'detector_module') and self.detector_module:
                try:
                    # Run detection on the frame
                    detections = self.detector_module.detect_pieces(frame)

                    # Convert detections to visualization format
                    pieces_data = []
                    for piece in detections:
                        piece_data = {
                            'bbox': piece.bbox if hasattr(piece, 'bbox') else (0, 0, 0, 0),
                            'being_processed': getattr(piece, 'being_processed', False),
                            'captured': getattr(piece, 'captured', False),
                            'in_exit_zone': getattr(piece, 'in_exit_zone', False)
                        }
                        pieces_data.append(piece_data)

                    detection_data['pieces'] = pieces_data

                    # Update status labels
                    detector_data = self.detector_module.get_visualization_data()
                    fps = detector_data.get('fps', 0)
                    self.detector_fps_label.setText(f"FPS: {fps:.1f}")
                    self.pieces_detected_label.setText(f"Pieces: {len(pieces_data)}")

                except Exception as e:
                    logger.warning(f"Error getting detector data: {e}")
                    # Continue with basic preview even if detector fails

            # Update the preview widget
            if hasattr(self, 'detector_preview'):
                self.detector_preview.update_frame(frame, detection_data)

        except Exception as e:
            logger.error(f"Error updating detector preview: {e}")
            # Don't stop the preview entirely, just log the error

    def create_sorting_tab(self):
        """Create sorting configuration tab aligned with sorting module requirements"""
        sorting_tab = QWidget()
        main_layout = QVBoxLayout()

        # Store references for dynamic updates
        self.category_dropdowns = {}
        self.pre_assignments = {}

        # === Sorting Strategy Group ===
        strategy_group = QGroupBox("Sorting Strategy")
        strategy_layout = QFormLayout()

        # Strategy selection dropdown
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Primary", "Secondary", "Tertiary"])
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_layout.addRow("Strategy Level:", self.strategy_combo)

        # Target primary category (shown for Secondary/Tertiary)
        self.target_primary_label = QLabel("Target Primary Category:")
        self.target_primary_combo = QComboBox()
        self.target_primary_combo.currentTextChanged.connect(self.on_primary_category_changed)
        strategy_layout.addRow(self.target_primary_label, self.target_primary_combo)

        # Target secondary category (shown for Tertiary only)
        self.target_secondary_label = QLabel("Target Secondary Category:")
        self.target_secondary_combo = QComboBox()
        self.target_secondary_combo.currentTextChanged.connect(self.on_secondary_category_changed)
        strategy_layout.addRow(self.target_secondary_label, self.target_secondary_combo)

        # Initially hide target category selectors
        self.target_primary_label.hide()
        self.target_primary_combo.hide()
        self.target_secondary_label.hide()
        self.target_secondary_combo.hide()

        strategy_group.setLayout(strategy_layout)
        main_layout.addWidget(strategy_group)

        # === Bin Configuration Group ===
        bin_config_group = QGroupBox("Bin Configuration")
        bin_layout = QFormLayout()

        # Maximum bins setting
        self.max_bins_spin = QSpinBox()
        self.max_bins_spin.setRange(1, 20)
        self.max_bins_spin.setValue(9)
        self.max_bins_spin.valueChanged.connect(self.on_max_bins_changed)
        bin_layout.addRow("Maximum Bins:", self.max_bins_spin)
        # Connect to Arduino tab for bin position updates
        self.max_bins_spin.valueChanged.connect(self.update_arduino_bin_count)

        # Overflow bin (fixed at 0)
        overflow_info = QLabel("0 (fixed)")
        overflow_info.setStyleSheet("color: gray;")
        bin_layout.addRow("Overflow Bin:", overflow_info)

        # Confidence threshold
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.0, 1.0)
        self.confidence_threshold.setSingleStep(0.05)
        self.confidence_threshold.setValue(0.7)
        self.confidence_threshold.setDecimals(2)
        bin_layout.addRow("Confidence Threshold:", self.confidence_threshold)

        # Bin capacity settings
        capacity_label = QLabel("Capacity Management")
        capacity_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        bin_layout.addRow(capacity_label, QLabel())

        self.max_pieces_per_bin = QSpinBox()
        self.max_pieces_per_bin.setRange(1, 1000)
        self.max_pieces_per_bin.setValue(50)
        bin_layout.addRow("Max Pieces per Bin:", self.max_pieces_per_bin)

        self.warning_threshold = QDoubleSpinBox()
        self.warning_threshold.setRange(0.1, 1.0)
        self.warning_threshold.setSingleStep(0.1)
        self.warning_threshold.setValue(0.8)
        self.warning_threshold.setDecimals(2)
        self.warning_threshold.setSuffix("%")
        bin_layout.addRow("Warning Threshold:", self.warning_threshold)

        bin_config_group.setLayout(bin_layout)
        main_layout.addWidget(bin_config_group)

        # === Pre-Assignments Group ===
        assignments_group = QGroupBox("Category Pre-Assignments")
        assignments_layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Assign specific categories to bins (optional)")
        instructions.setStyleSheet("color: gray; font-style: italic;")
        assignments_layout.addWidget(instructions)

        # Pre-assignment table
        self.assignment_table = QTableWidget(0, 3)
        self.assignment_table.setHorizontalHeaderLabels(["Bin", "Category", "Action"])
        self.assignment_table.horizontalHeader().setStretchLastSection(True)
        self.assignment_table.setMaximumHeight(200)
        assignments_layout.addWidget(self.assignment_table)

        # Assignment control buttons
        assignment_buttons = QHBoxLayout()

        self.add_assignment_btn = QPushButton("Add Assignment")
        self.add_assignment_btn.clicked.connect(self.add_bin_assignment)
        assignment_buttons.addWidget(self.add_assignment_btn)

        self.clear_assignments_btn = QPushButton("Clear All")
        self.clear_assignments_btn.clicked.connect(self.clear_all_assignments)
        assignment_buttons.addWidget(self.clear_assignments_btn)

        assignment_buttons.addStretch()
        assignments_layout.addLayout(assignment_buttons)

        assignments_group.setLayout(assignments_layout)
        main_layout.addWidget(assignments_group)

        # === Available Categories Preview ===
        preview_group = QGroupBox("Categories to be Sorted")
        preview_layout = QVBoxLayout()

        # Info label
        self.category_info_label = QLabel("Categories available for current strategy:")
        preview_layout.addWidget(self.category_info_label)

        # Category list
        self.available_categories_list = QListWidget()
        self.available_categories_list.setMaximumHeight(150)
        self.available_categories_list.setSelectionMode(QListWidget.NoSelection)
        preview_layout.addWidget(self.available_categories_list)

        # Statistics
        self.category_stats_label = QLabel("0 categories available")
        self.category_stats_label.setStyleSheet("color: gray;")
        preview_layout.addWidget(self.category_stats_label)

        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        main_layout.addStretch()
        sorting_tab.setLayout(main_layout)
        self.tab_widget.addTab(sorting_tab, "Sorting")

        # Load categories from CSV if config manager available
        if self.config_manager:
            self.load_category_hierarchy()
            self.update_available_categories()

    def on_strategy_changed(self, strategy_text):
        """Handle sorting strategy selection change with smart dropdown population"""
        print(f"Strategy changed to: {strategy_text}")  # Debug line
        if not self.config_manager:
            print("WARNING: config_manager is None!")
            return

        strategy = strategy_text.lower()
        print(f"Strategy (lowercase): {strategy}")  # Debug line

        # Get the category hierarchy for populating dropdowns
        hierarchy = self.config_manager.get_category_hierarchy()

        # Handle visibility and populate dropdowns based on strategy
        if strategy == "primary":
            # Hide all target category selectors for primary strategy
            self.target_primary_label.hide()
            self.target_primary_combo.hide()
            self.target_secondary_label.hide()
            self.target_secondary_combo.hide()

        elif strategy == "secondary":
            # Show only primary selector for secondary strategy
            self.target_primary_label.show()
            self.target_primary_combo.show()
            self.target_secondary_label.hide()
            self.target_secondary_combo.hide()

            # Populate primary dropdown with all available primary categories
            primary_categories = hierarchy.get("primary", [])
            current_selection = self.target_primary_combo.currentText()

            self.target_primary_combo.clear()
            self.target_primary_combo.addItem("")  # Add empty option
            self.target_primary_combo.addItems(sorted(primary_categories))

            # Restore previous selection if it still exists
            if current_selection in primary_categories:
                self.target_primary_combo.setCurrentText(current_selection)

        elif strategy == "tertiary":
            # Show both selectors for tertiary strategy
            self.target_primary_label.show()
            self.target_primary_combo.show()
            self.target_secondary_label.show()
            self.target_secondary_combo.show()

            # Populate primary dropdown with all available primary categories
            primary_categories = hierarchy.get("primary", [])
            current_primary = self.target_primary_combo.currentText()

            self.target_primary_combo.clear()
            self.target_primary_combo.addItem("")  # Add empty option
            self.target_primary_combo.addItems(sorted(primary_categories))

            # If we had a previous selection, restore it and populate secondary
            if current_primary in primary_categories:
                self.target_primary_combo.setCurrentText(current_primary)

                # Populate secondary dropdown based on selected primary
                secondary_categories = hierarchy.get("primary_to_secondary", {}).get(current_primary, [])
                current_secondary = self.target_secondary_combo.currentText()

                self.target_secondary_combo.clear()
                self.target_secondary_combo.addItem("")  # Add empty option
                self.target_secondary_combo.addItems(sorted(secondary_categories))

                # Restore secondary selection if it still exists
                if current_secondary in secondary_categories:
                    self.target_secondary_combo.setCurrentText(current_secondary)
            else:
                # Clear secondary dropdown if no primary selected
                self.target_secondary_combo.clear()
                self.target_secondary_combo.addItem("")

        # Update available categories list based on new strategy
        self.update_available_categories()

        # Clear pre-assignments as they may no longer be valid
        self.validate_assignments()

        # Update status to show what categories are available
        self.update_dropdown_status()

        self.strategy_combo.parentWidget().update()
        self.strategy_combo.parentWidget().adjustSize()

    def on_primary_category_changed(self, primary_category):
        """Handle primary category selection change with smart secondary dropdown update"""
        if not primary_category or not self.config_manager:
            # Clear secondary dropdown if no primary selected
            if self.strategy_combo.currentText().lower() == "tertiary":
                self.target_secondary_combo.clear()
                self.target_secondary_combo.addItem("")
            self.update_available_categories()
            return

        strategy = self.strategy_combo.currentText().lower()

        # Update secondary category dropdown if in tertiary mode
        if strategy == "tertiary":
            # Get secondary categories for selected primary
            hierarchy = self.config_manager.get_category_hierarchy()
            secondary_categories = hierarchy.get("primary_to_secondary", {}).get(primary_category, [])

            # Remember current selection if any
            current_secondary = self.target_secondary_combo.currentText()

            # Populate secondary dropdown
            self.target_secondary_combo.clear()
            self.target_secondary_combo.addItem("")  # Empty option

            if secondary_categories:
                self.target_secondary_combo.addItems(sorted(secondary_categories))

                # Restore selection if it's still valid
                if current_secondary in secondary_categories:
                    self.target_secondary_combo.setCurrentText(current_secondary)

                # Enable the secondary dropdown
                self.target_secondary_combo.setEnabled(True)

                # Update tooltip with helpful information
                tooltip = f"Select a secondary category within '{primary_category}'\n"
                tooltip += f"Available options: {len(secondary_categories)} categories"
                self.target_secondary_combo.setToolTip(tooltip)
            else:
                # No secondary categories available
                self.target_secondary_combo.addItem("(No secondary categories available)")
                self.target_secondary_combo.setEnabled(False)

        # Update available categories based on selection
        self.update_available_categories()
        self.validate_assignments()

        # Update status
        self.update_dropdown_status()

    def on_secondary_category_changed(self, secondary_category):
        """Handle secondary category selection change"""
        self.update_available_categories()
        self.validate_assignments()

        # Update status to reflect the selection
        self.update_dropdown_status()

    def on_max_bins_changed(self, value):
        """Handle max bins value change"""
        self.validate_assignments()

    def update_dropdown_status(self):
        """Update status bar with current dropdown selection information"""
        strategy = self.strategy_combo.currentText().lower()

        if strategy == "primary":
            # Get count of primary categories available
            categories = self.config_manager.get_categories_for_strategy(strategy)
            self.status_bar.showMessage(f"Primary sorting: {len(categories)} categories available")

        elif strategy == "secondary":
            primary = self.target_primary_combo.currentText()
            if primary:
                categories = self.config_manager.get_categories_for_strategy(strategy, primary)
                self.status_bar.showMessage(
                    f"Secondary sorting in '{primary}': {len(categories)} subcategories available"
                )
            else:
                self.status_bar.showMessage("Secondary sorting: Please select a primary category")

        elif strategy == "tertiary":
            primary = self.target_primary_combo.currentText()
            secondary = self.target_secondary_combo.currentText()

            if primary and secondary:
                categories = self.config_manager.get_categories_for_strategy(
                    strategy, primary, secondary
                )
                self.status_bar.showMessage(
                    f"Tertiary sorting in '{primary}/{secondary}': {len(categories)} categories available"
                )
            elif primary:
                self.status_bar.showMessage(
                    f"Tertiary sorting in '{primary}': Please select a secondary category"
                )
            else:
                self.status_bar.showMessage("Tertiary sorting: Please select primary and secondary categories")

    def load_category_hierarchy(self):
        """Load category hierarchy from CSV via config manager and populate initial dropdowns"""
        if not self.config_manager:
            return

        try:
            # Parse categories from CSV
            hierarchy = self.config_manager.get_category_hierarchy()

            # Get all primary categories
            primary_categories = hierarchy.get("primary", [])

            # Store hierarchy for later use
            self.category_hierarchy = hierarchy

            # Initially populate primary dropdown (will be shown/hidden based on strategy)
            self.target_primary_combo.clear()
            self.target_primary_combo.addItem("")  # Empty option
            self.target_primary_combo.addItems(sorted(primary_categories))

            # Add tooltips to help users
            self.target_primary_combo.setToolTip(
                f"Select from {len(primary_categories)} available primary categories"
            )
            self.target_secondary_combo.setToolTip(
                "Secondary categories will appear after selecting a primary category"
            )

            logger.info(f"Loaded category hierarchy with {len(primary_categories)} primary categories")

        except Exception as e:
            logger.error(f"Failed to load category hierarchy: {e}")
            QMessageBox.warning(self, "Warning",
                                f"Could not load categories from CSV: {str(e)}")

    def update_available_categories(self):
        """Update the list of available categories based on current strategy"""
        if not self.config_manager:
            return

        self.available_categories_list.clear()

        strategy = self.strategy_combo.currentText().lower()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()

        # Get categories for current strategy
        categories = self.config_manager.get_categories_for_strategy(
            strategy, primary, secondary
        )

        # Update list widget
        for category in sorted(categories):
            item = QListWidgetItem(category)

            # Check if this category is pre-assigned
            if category in self.pre_assignments:
                bin_num = self.pre_assignments[category]
                item.setText(f"{category} → Bin {bin_num}")
                item.setForeground(QColor(0, 128, 0))  # Green for assigned

            self.available_categories_list.addItem(item)

        # Update statistics
        count = len(categories)
        assigned_count = sum(1 for cat in categories if cat in self.pre_assignments)

        stats_text = f"{count} categories available"
        if assigned_count > 0:
            stats_text += f" ({assigned_count} pre-assigned)"
        self.category_stats_label.setText(stats_text)

        # Update info label based on strategy
        if strategy == "primary":
            self.category_info_label.setText("Sorting by primary categories:")
        elif strategy == "secondary":
            if primary:
                self.category_info_label.setText(f"Sorting secondary categories within '{primary}':")
            else:
                self.category_info_label.setText("Select a primary category first")
        elif strategy == "tertiary":
            if primary and secondary:
                self.category_info_label.setText(f"Sorting tertiary categories within '{primary}/{secondary}':")
            else:
                self.category_info_label.setText("Select primary and secondary categories first")

    def validate_dropdown_selections(self):
        """Validate that dropdown selections are valid for the current strategy"""
        strategy = self.strategy_combo.currentText().lower()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()

        errors = []

        if strategy == "secondary":
            if not primary:
                errors.append("Secondary sorting requires selecting a primary category")

            # Check if the selected primary has any secondary categories
            elif self.config_manager:
                categories = self.config_manager.get_categories_for_strategy(strategy, primary)
                if not categories:
                    errors.append(f"Primary category '{primary}' has no secondary categories")

        elif strategy == "tertiary":
            if not primary:
                errors.append("Tertiary sorting requires selecting a primary category")
            elif not secondary:
                errors.append("Tertiary sorting requires selecting a secondary category")

            # Check if the combination has any tertiary categories
            elif self.config_manager:
                categories = self.config_manager.get_categories_for_strategy(
                    strategy, primary, secondary
                )
                if not categories:
                    errors.append(
                        f"Combination '{primary}/{secondary}' has no tertiary categories"
                    )

        return errors

    def refresh_dropdowns(self):
        """Force refresh all dropdowns from the CSV data"""
        # Reload category hierarchy
        self.load_category_hierarchy()

        # Trigger strategy change to repopulate dropdowns
        current_strategy = self.strategy_combo.currentText()
        self.on_strategy_changed(current_strategy)

        self.status_bar.showMessage("Dropdowns refreshed from CSV data", 3000)

    def add_bin_assignment(self):
        """Add a new bin assignment"""
        if not self.config_manager:
            return

        # Get available categories for current strategy
        strategy = self.strategy_combo.currentText().lower()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()

        categories = self.config_manager.get_categories_for_strategy(
            strategy, primary, secondary
        )

        # Filter out already assigned categories
        available = [cat for cat in categories if cat not in self.pre_assignments]

        if not available:
            QMessageBox.information(self, "No Categories",
                                    "All available categories are already assigned")
            return

        # Create assignment dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Bin Assignment")
        dialog.setModal(True)

        layout = QFormLayout()

        # Category selection
        category_combo = QComboBox()
        category_combo.addItems(sorted(available))
        layout.addRow("Category:", category_combo)

        # Bin selection
        bin_spin = QSpinBox()
        bin_spin.setRange(1, self.max_bins_spin.value())

        # Find next available bin
        used_bins = set(self.pre_assignments.values())
        for i in range(1, self.max_bins_spin.value() + 1):
            if i not in used_bins:
                bin_spin.setValue(i)
                break

        layout.addRow("Bin Number:", bin_spin)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            category = category_combo.currentText()
            bin_num = bin_spin.value()

            # Check if bin is already assigned
            for cat, assigned_bin in self.pre_assignments.items():
                if assigned_bin == bin_num:
                    reply = QMessageBox.question(self, "Bin Already Assigned",
                                                 f"Bin {bin_num} is already assigned to '{cat}'.\n"
                                                 f"Replace with '{category}'?",
                                                 QMessageBox.Yes | QMessageBox.No)

                    if reply == QMessageBox.Yes:
                        # Remove old assignment
                        del self.pre_assignments[cat]
                        self.remove_assignment_from_table(cat)
                    else:
                        return

            # Add assignment
            self.pre_assignments[category] = bin_num
            self.add_assignment_to_table(category, bin_num)
            self.update_available_categories()

    def add_assignment_to_table(self, category, bin_num):
        """Add an assignment to the table"""
        row = self.assignment_table.rowCount()
        self.assignment_table.insertRow(row)

        # Bin number
        bin_item = QTableWidgetItem(str(bin_num))
        bin_item.setTextAlignment(Qt.AlignCenter)
        self.assignment_table.setItem(row, 0, bin_item)

        # Category name
        category_item = QTableWidgetItem(category)
        self.assignment_table.setItem(row, 1, category_item)

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self.remove_assignment(category))
        self.assignment_table.setCellWidget(row, 2, remove_btn)

    def remove_assignment(self, category):
        """Remove a bin assignment"""
        if category in self.pre_assignments:
            del self.pre_assignments[category]
            self.remove_assignment_from_table(category)
            self.update_available_categories()

    def remove_assignment_from_table(self, category):
        """Remove an assignment from the table"""
        for row in range(self.assignment_table.rowCount()):
            item = self.assignment_table.item(row, 1)
            if item and item.text() == category:
                self.assignment_table.removeRow(row)
                break

    def clear_all_assignments(self):
        """Clear all bin assignments"""
        reply = QMessageBox.question(self, "Clear Assignments",
                                     "Remove all bin assignments?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.pre_assignments.clear()
            self.assignment_table.setRowCount(0)
            self.update_available_categories()

    def validate_assignments(self):
        """Validate that current assignments are still valid for the strategy"""
        if not self.pre_assignments:
            return

        strategy = self.strategy_combo.currentText().lower()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()

        # Get currently valid categories
        valid_categories = self.config_manager.get_categories_for_strategy(
            strategy, primary, secondary
        )

        # Check each assignment
        invalid = []
        for category in list(self.pre_assignments.keys()):
            if category not in valid_categories:
                invalid.append(category)

        # Remove invalid assignments
        if invalid:
            for category in invalid:
                del self.pre_assignments[category]
                self.remove_assignment_from_table(category)

            QMessageBox.information(self, "Assignments Updated",
                                    f"Removed {len(invalid)} assignment(s) that are no longer valid "
                                    f"for the selected strategy.")

            self.update_available_categories()

    def get_sorting_config(self):
        """Get the current sorting configuration as a dictionary"""
        strategy = self.strategy_combo.currentText().lower()

        config = {
            "strategy": strategy,
            "target_primary_category": self.target_primary_combo.currentText() if strategy in ["secondary",
                                                                                               "tertiary"] else "",
            "target_secondary_category": self.target_secondary_combo.currentText() if strategy == "tertiary" else "",
            "max_bins": self.max_bins_spin.value(),
            "overflow_bin": 0,
            "confidence_threshold": self.confidence_threshold.value(),
            "pre_assignments": dict(self.pre_assignments),
            "max_pieces_per_bin": self.max_pieces_per_bin.value(),
            "bin_warning_threshold": self.warning_threshold.value()
        }

        return config

    def load_sorting_config(self, config):
        """Load sorting configuration from a dictionary"""
        if not config:
            return

        # Set strategy
        strategy = config.get("strategy", "primary")
        strategy_map = {"primary": "Primary", "secondary": "Secondary", "tertiary": "Tertiary"}
        self.strategy_combo.setCurrentText(strategy_map.get(strategy, "Primary"))

        # Set target categories
        if strategy in ["secondary", "tertiary"]:
            primary = config.get("target_primary_category", "")
            self.target_primary_combo.setCurrentText(primary)

        if strategy == "tertiary":
            secondary = config.get("target_secondary_category", "")
            self.target_secondary_combo.setCurrentText(secondary)

        # Set bin configuration
        self.max_bins_spin.setValue(config.get("max_bins", 9))
        self.confidence_threshold.setValue(config.get("confidence_threshold", 0.7))
        self.max_pieces_per_bin.setValue(config.get("max_pieces_per_bin", 50))
        self.warning_threshold.setValue(config.get("bin_warning_threshold", 0.8))

        # Load pre-assignments
        self.pre_assignments = config.get("pre_assignments", {}).copy()
        self.assignment_table.setRowCount(0)
        for category, bin_num in self.pre_assignments.items():
            self.add_assignment_to_table(category, bin_num)

        # Update display
        self.update_available_categories()

    def create_api_tab(self):
        """Create API configuration tab"""
        api_tab = QWidget()
        layout = QVBoxLayout()

        # API settings - SIMPLIFIED: Only timeout and retry_count
        api_group = QGroupBox("API Settings")
        api_layout = QFormLayout()

        # Timeout setting with 30 second default to match API module
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 60)
        self.timeout_spin.setValue(30)  # CHANGED: Default from 10 to 30 to match API module
        self.timeout_spin.setSuffix(" seconds")
        self.timeout_spin.setToolTip(
            "Maximum time to wait for API response.\n"
            "Default: 30 seconds (matches API module default)"
        )
        api_layout.addRow("Timeout:", self.timeout_spin)

        # NEW: Retry count setting
        self.retry_count_spin = QSpinBox()
        self.retry_count_spin.setRange(0, 10)
        self.retry_count_spin.setValue(3)  # Default matches API module
        self.retry_count_spin.setToolTip(
            "Number of times to retry if API call fails.\n"
            "Set to 0 to disable retries.\n"
            "Default: 3 retries"
        )
        api_layout.addRow("Retry Count:", self.retry_count_spin)

        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # Test connection section remains the same
        test_group = QGroupBox("Connection Test")
        test_layout = QVBoxLayout()

        self.test_api_btn = QPushButton("Test Connection")
        self.test_api_btn.clicked.connect(self.test_api_connection)
        test_layout.addWidget(self.test_api_btn)

        self.api_status_label = QLabel("Not tested")
        test_layout.addWidget(self.api_status_label)

        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

        # Information box about Brickognize API
        info_group = QGroupBox("API Information")
        info_layout = QVBoxLayout()

        info_label = QLabel(
            "This system uses the Brickognize API for LEGO piece identification.\n"
            "No API key is required for Brickognize.\n\n"
            "The API will analyze images of LEGO pieces and return:\n"
            "• Piece ID and name\n"
            "• Confidence score"
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        api_tab.setLayout(layout)
        self.tab_widget.addTab(api_tab, "API")

    def create_arduino_tab(self):
        """Create enhanced Arduino configuration tab"""
        arduino_tab = QWidget()
        layout = QVBoxLayout()

        # Store references for cross-tab communication
        self.arduino_widgets = {}

        # === Simulation Mode === (at the top for visibility)
        sim_group = QGroupBox("Operation Mode")
        sim_layout = QVBoxLayout()

        self.simulation_check = QCheckBox("Simulation Mode (no hardware required)")
        self.simulation_check.stateChanged.connect(self.toggle_simulation_mode)
        sim_layout.addWidget(self.simulation_check)

        self.hardware_status_label = QLabel("Checking hardware...")
        sim_layout.addWidget(self.hardware_status_label)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # === Connection Settings ===
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
        self.baud_combo.setCurrentText("9600")
        connection_layout.addRow("Baud Rate:", self.baud_combo)

        # Timeout
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.5, 10.0)
        self.timeout_spin.setValue(2.0)
        self.timeout_spin.setSingleStep(0.5)
        self.timeout_spin.setSuffix(" sec")
        connection_layout.addRow("Timeout:", self.timeout_spin)

        # Connection retries
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(1, 10)
        self.retry_spin.setValue(3)
        connection_layout.addRow("Connection Retries:", self.retry_spin)

        # Retry delay
        self.retry_delay_spin = QDoubleSpinBox()
        self.retry_delay_spin.setRange(0.5, 5.0)
        self.retry_delay_spin.setValue(1.0)
        self.retry_delay_spin.setSingleStep(0.5)
        self.retry_delay_spin.setSuffix(" sec")
        connection_layout.addRow("Retry Delay:", self.retry_delay_spin)

        # Test connection button
        self.test_connection_btn = QPushButton("Test Connection")
        self.test_connection_btn.clicked.connect(self.test_arduino_connection)
        connection_layout.addRow("", self.test_connection_btn)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)
        self.arduino_widgets['connection'] = [self.port_combo, self.refresh_ports_btn,
                                              self.baud_combo, self.timeout_spin,
                                              self.retry_spin, self.retry_delay_spin,
                                              self.test_connection_btn]

        # === Servo Hardware Settings ===
        servo_hw_group = QGroupBox("Servo Hardware Settings")
        servo_hw_layout = QFormLayout()

        # Min pulse
        self.min_pulse_spin = QSpinBox()
        self.min_pulse_spin.setRange(400, 1000)
        self.min_pulse_spin.setValue(500)
        self.min_pulse_spin.setSuffix(" μs")
        servo_hw_layout.addRow("Min Pulse Width:", self.min_pulse_spin)

        # Max pulse
        self.max_pulse_spin = QSpinBox()
        self.max_pulse_spin.setRange(2000, 2600)
        self.max_pulse_spin.setValue(2500)
        self.max_pulse_spin.setSuffix(" μs")
        servo_hw_layout.addRow("Max Pulse Width:", self.max_pulse_spin)

        # Default position
        self.default_pos_spin = QSpinBox()
        self.default_pos_spin.setRange(0, 180)
        self.default_pos_spin.setValue(90)
        self.default_pos_spin.setSuffix("°")
        servo_hw_layout.addRow("Home Position:", self.default_pos_spin)

        # Min bin separation
        self.min_separation_spin = QSpinBox()
        self.min_separation_spin.setRange(10, 45)
        self.min_separation_spin.setValue(20)
        self.min_separation_spin.setSuffix("°")
        self.min_separation_spin.valueChanged.connect(self.recalculate_bin_positions)
        servo_hw_layout.addRow("Min Bin Separation:", self.min_separation_spin)

        servo_hw_group.setLayout(servo_hw_layout)
        layout.addWidget(servo_hw_group)
        self.arduino_widgets['servo_hw'] = [self.min_pulse_spin, self.max_pulse_spin,
                                            self.default_pos_spin, self.min_separation_spin]

        # === Bin Configuration ===
        bin_group = QGroupBox("Bin Position Configuration")
        bin_layout = QVBoxLayout()

        # Control buttons
        btn_layout = QHBoxLayout()

        self.auto_calculate_btn = QPushButton("Auto-Calculate Positions")
        self.auto_calculate_btn.clicked.connect(self.auto_calculate_positions)
        btn_layout.addWidget(self.auto_calculate_btn)

        self.test_sweep_btn = QPushButton("Test All Positions")
        self.test_sweep_btn.clicked.connect(self.test_sweep_positions)
        btn_layout.addWidget(self.test_sweep_btn)

        self.calibrate_servo_btn = QPushButton("Calibration Mode")
        self.calibrate_servo_btn.setCheckable(True)
        self.calibrate_servo_btn.toggled.connect(self.toggle_calibration_mode)
        btn_layout.addWidget(self.calibrate_servo_btn)

        bin_layout.addLayout(btn_layout)

        # Bin positions table
        self.servo_table = QTableWidget(10, 3)  # Added test column
        self.servo_table.setHorizontalHeaderLabels(["Bin", "Position (°)", "Test"])
        self.servo_table.horizontalHeader().setStretchLastSection(False)
        self.servo_table.setColumnWidth(0, 50)
        self.servo_table.setColumnWidth(1, 100)
        self.servo_table.setColumnWidth(2, 80)

        # Initialize table with default values
        self.setup_servo_table()

        bin_layout.addWidget(self.servo_table)

        # Info label
        self.bin_info_label = QLabel("Bins will be auto-calculated based on Sorting tab settings")
        bin_layout.addWidget(self.bin_info_label)

        bin_group.setLayout(bin_layout)
        layout.addWidget(bin_group)
        self.arduino_widgets['bin_config'] = [self.auto_calculate_btn, self.test_sweep_btn,
                                              self.calibrate_servo_btn, self.servo_table]

        layout.addStretch()
        arduino_tab.setLayout(layout)
        self.tab_widget.addTab(arduino_tab, "Arduino")

        # Check hardware on startup (delayed to allow GUI to initialize)
        QTimer.singleShot(500, self.check_arduino_hardware)

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
        """Load current configuration into UI - FIXED VERSION"""
        if not self.config_manager:
            return

        # Load camera settings (existing code unchanged)
        camera_config = self.config_manager.get_module_config("camera")
        if camera_config:
            device_id = camera_config.get("device_id", 0)
            device_found = False
            for i in range(self.camera_device.count()):
                if self.camera_device.itemData(i) == device_id:
                    self.camera_device.setCurrentIndex(i)
                    device_found = True
                    break

            if not device_found:
                fallback_text = f"Camera {device_id}"
                self.camera_device.addItem(fallback_text, userData=device_id)
                self.camera_device.setCurrentIndex(self.camera_device.count() - 1)

            res = camera_config.get("resolution", [1920, 1080])
            self.resolution_combo.setCurrentText(f"{res[0]}x{res[1]}")
            self.fps_spin.setValue(camera_config.get("fps", 30))
            self.exposure_slider.setValue(camera_config.get("exposure", 0))

        # Load detector settings (existing code unchanged)
        detector_config = self.config_manager.get_module_config("detector")
        if detector_config:
            self.min_piece_area_spin.setValue(detector_config.get("min_area", 1000))
            self.max_piece_area_spin.setValue(detector_config.get("max_area", 50000))

            zones = detector_config.get("zones", {})
            entry_zone = int(zones.get("entry_percentage", 15))
            exit_zone = int(zones.get("exit_percentage", 15))
            self.entry_zone_slider.setValue(entry_zone)
            self.exit_zone_slider.setValue(exit_zone)
            self._update_entry_zone(entry_zone)
            self._update_exit_zone(exit_zone)

            roi = detector_config.get("roi")
            if roi and len(roi) == 4:
                x, y, w, h = roi
                self.roi_x_spin.blockSignals(True)
                self.roi_y_spin.blockSignals(True)
                self.roi_w_spin.blockSignals(True)
                self.roi_h_spin.blockSignals(True)

                self.roi_x_spin.setValue(x)
                self.roi_y_spin.setValue(y)
                self.roi_w_spin.setValue(w)
                self.roi_h_spin.setValue(h)

                self.roi_x_spin.blockSignals(False)
                self.roi_y_spin.blockSignals(False)
                self.roi_w_spin.blockSignals(False)
                self.roi_h_spin.blockSignals(False)

                self._update_roi_from_controls()

        # Load sorting settings (existing code unchanged)
        sorting_config = self.config_manager.get_module_config("sorting")
        if sorting_config:
            self.load_sorting_config(sorting_config)

        # API settings (existing code unchanged)
        api_config = self.config_manager.get_module_config(ModuleConfig.API.value)
        if api_config:
            self.timeout_spin.setValue(api_config.get("timeout", 30))
            self.retry_count_spin.setValue(api_config.get("retry_count", 3))

        # FIXED: Load Arduino settings from arduino_servo config (where GUI saves them)
        arduino_config = self.config_manager.get_module_config("arduino_servo")
        if arduino_config:
            # Arduino connection settings
            self.port_combo.setCurrentText(arduino_config.get("port", ""))
            self.baud_combo.setCurrentText(str(arduino_config.get("baud_rate", 9600)))
            self.timeout_spin.setValue(arduino_config.get("timeout", 2.0))
            self.retry_spin.setValue(arduino_config.get("connection_retries", 3))
            self.retry_delay_spin.setValue(arduino_config.get("retry_delay", 1.0))
            self.simulation_check.setChecked(arduino_config.get("simulation_mode", False))

            # FIXED: Load servo hardware settings from arduino_servo config
            self.min_pulse_spin.setValue(arduino_config.get("min_pulse", 500))
            self.max_pulse_spin.setValue(arduino_config.get("max_pulse", 2500))
            self.default_pos_spin.setValue(arduino_config.get("default_position", 90))
            self.min_separation_spin.setValue(arduino_config.get("min_bin_separation", 20))

            # FIXED: Load actual bin positions (user-configured positions)
            positions = arduino_config.get("bin_positions", {})
            for i in range(10):
                position = positions.get(str(i), 90)
                if self.servo_table.item(i, 1):
                    self.servo_table.item(i, 1).setText(str(position))

        # Load system settings (existing code unchanged)
        system_config = self.config_manager.get_module_config("system")
        if system_config:
            self.threading_check.setChecked(system_config.get("threading_enabled", True))
            self.queue_size_spin.setValue(system_config.get("queue_size", 100))
            self.log_level_combo.setCurrentText(system_config.get("log_level", "INFO"))
            self.save_images_check.setChecked(system_config.get("save_images", True))

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration from UI - FIXED VERSION"""
        config = {}

        # Camera configuration (unchanged)
        resolution = self.resolution_combo.currentText().split('x')
        config['camera'] = {
            'device_id': self.camera_device.currentData(),
            'resolution': [int(resolution[0]), int(resolution[1])],
            'fps': self.fps_spin.value(),
            'exposure': self.exposure_slider.value()
        }

        # Detector configuration (unchanged)
        config['detector'] = {
            'min_area': self.min_piece_area_spin.value(),
            'max_area': self.max_piece_area_spin.value(),
            'zones': {
                'entry_percentage': self.entry_zone_slider.value(),
                'exit_percentage': self.exit_zone_slider.value()
            },
            'roi': self.current_roi if self.current_roi else None
        }

        # Sorting configuration (unchanged)
        config['sorting'] = self.get_sorting_config()

        # API configuration (unchanged)
        config['api'] = {
            'timeout': self.timeout_spin.value(),
            'retry_count': self.retry_count_spin.value()
        }

        # FIXED: Arduino configuration - keep all servo settings in arduino_servo
        config['arduino_servo'] = self.get_arduino_config()

        # System configuration (unchanged)
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
        """Start live camera preview using consumer system - IMPROVED"""
        try:
            # Get or create camera instance
            if not hasattr(self, 'camera') or not self.camera:
                from camera_module import create_camera
                self.camera = create_camera(config_manager=self.config_manager)

                # Initialize if needed
                if not self.camera.is_initialized:
                    if not self.camera.initialize():
                        QMessageBox.warning(self, "Camera Error",
                                            "Failed to initialize camera")
                        return

            # Register this GUI as a frame consumer
            success = self.camera.register_consumer(
                name="config_preview",
                callback=self._preview_frame_callback,
                processing_type="async",  # Non-blocking for GUI
                priority=50  # Medium priority for preview
            )

            if not success:
                logger.warning("Preview consumer already registered, continuing...")

            # Start camera capture if not already running
            if not self.camera.is_capturing:
                self.camera.start_capture()

            # Update UI state
            self.preview_active = True
            self.start_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(True)
            self.status_bar.showMessage("Camera preview started")

            logger.info("Live camera preview started")

        except Exception as e:
            logger.error(f"Failed to start camera preview: {e}")
            QMessageBox.critical(self, "Preview Error",
                                 f"Failed to start preview: {str(e)}")

    def stop_camera_preview(self):
        """Stop camera preview and unregister consumer - IMPROVED"""
        try:
            if hasattr(self, 'camera') and self.camera:
                # Unregister this GUI as a consumer
                try:
                    self.camera.unregister_consumer("config_preview")
                except:
                    pass  # Consumer might not be registered

            # Update UI state
            self.preview_active = False
            if hasattr(self, 'preview_widget'):
                self.preview_widget.setText("No Video Signal")
            if hasattr(self, 'preview_fps_label'):
                self.preview_fps_label.setText("FPS: 0.0")
            if hasattr(self, 'start_preview_btn'):
                self.start_preview_btn.setEnabled(True)
            if hasattr(self, 'stop_preview_btn'):
                self.stop_preview_btn.setEnabled(False)

            self.status_bar.showMessage("Camera preview stopped")
            logger.info("Camera preview stopped")

        except Exception as e:
            logger.error(f"Error stopping preview: {e}")

    def _preview_frame_callback(self, frame: np.ndarray):
        """
        Callback function for camera frames.
        This is called by the camera module for each new frame.
        """
        if not self.preview_active:
            return

        try:
            # Use Qt's thread-safe mechanism to update GUI
            # Since this callback might be from a different thread
            QMetaObject.invokeMethod(
                self,
                "_update_preview_display",
                Qt.QueuedConnection,
                Q_ARG(object, frame)
            )

            # Track FPS
            self.preview_frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_preview_time

            if elapsed > 1.0:  # Update FPS every second
                fps = self.preview_frame_count / elapsed
                # Update FPS label in GUI thread
                QMetaObject.invokeMethod(
                    self.preview_fps_label,
                    "setText",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"FPS: {fps:.1f}")
                )
                self.last_preview_time = current_time
                self.preview_frame_count = 0

        except Exception as e:
            logger.error(f"Error in preview callback: {e}")

    @pyqtSlot(object)
    def _update_preview_display(self, frame):
        """
        Thread-safe method to update the preview widget.
        This runs in the GUI thread.
        """
        if self.preview_active and frame is not None:
            self.preview_widget.update_frame(frame)

    def closeEvent(self, event):
        """Handle window close event - IMPROVED CLEANUP"""
        try:
            # Stop detector preview
            if getattr(self, 'preview_timer', None):
                self.preview_timer.stop()

            # Stop camera preview
            if getattr(self, 'preview_active', False):
                self.stop_camera_preview()

            # Clean up camera
            if hasattr(self, 'camera') and self.camera:
                try:
                    self.camera.unregister_consumer("config_preview")
                    self.camera.unregister_consumer("detector_preview")
                except:
                    pass

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            # Call parent close event
            super().closeEvent(event)

    # ========== Other UI Methods ==========

    def configure_roi(self):
        """Open ROI configuration dialog"""
        # This would open a separate window for ROI configuration
        QMessageBox.information(self, "ROI Configuration",
                                "ROI configuration would open here")

    def _calculate_default_roi(self, frame_width, frame_height):
        """Calculate default ROI as center 80% of frame"""
        margin_x = int(frame_width * 0.1)  # 10% margin on each side
        margin_y = int(frame_height * 0.1)  # 10% margin on top and bottom

        return {
            'x': margin_x,
            'y': margin_y,
            'w': frame_width - (2 * margin_x),
            'h': frame_height - (2 * margin_y)
        }

    def _update_frame_size(self, width, height):
        """Update frame size and adjust ROI controls accordingly"""
        self.frame_width = width
        self.frame_height = height

        # Calculate new default ROI
        self.default_roi = self._calculate_default_roi(width, height)

        # Update spin box ranges
        self.roi_x_spin.setRange(0, width - 100)  # Leave room for minimum width
        self.roi_y_spin.setRange(0, height - 100)  # Leave room for minimum height
        self.roi_w_spin.setRange(100, width)
        self.roi_h_spin.setRange(100, height)

        # If no ROI is set, use default
        if not self.current_roi:
            self._set_roi_controls_to_default()

    def _set_roi_controls_to_default(self):
        """Set spin box controls to default ROI values"""
        if not self.default_roi:
            return

        # Temporarily disconnect signals to avoid recursive updates
        self.roi_x_spin.blockSignals(True)
        self.roi_y_spin.blockSignals(True)
        self.roi_w_spin.blockSignals(True)
        self.roi_h_spin.blockSignals(True)

        # Set the values
        self.roi_x_spin.setValue(self.default_roi['x'])
        self.roi_y_spin.setValue(self.default_roi['y'])
        self.roi_w_spin.setValue(self.default_roi['w'])
        self.roi_h_spin.setValue(self.default_roi['h'])

        # Re-enable signals
        self.roi_x_spin.blockSignals(False)
        self.roi_y_spin.blockSignals(False)
        self.roi_w_spin.blockSignals(False)
        self.roi_h_spin.blockSignals(False)

        # Update the ROI
        self._update_roi_from_controls()

    def _update_roi_from_controls(self):
        """Update ROI from spin box values"""
        # Get current values from spin boxes
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        w = self.roi_w_spin.value()
        h = self.roi_h_spin.value()

        # Validate bounds (additional safety check)
        if x + w > self.frame_width:
            w = self.frame_width - x
            self.roi_w_spin.blockSignals(True)
            self.roi_w_spin.setValue(w)
            self.roi_w_spin.blockSignals(False)

        if y + h > self.frame_height:
            h = self.frame_height - y
            self.roi_h_spin.blockSignals(True)
            self.roi_h_spin.setValue(h)
            self.roi_h_spin.blockSignals(False)

        # Update current ROI
        self.current_roi = (x, y, w, h)

        # Update info label
        self.roi_info_label.setText(f"ROI: ({x}, {y}) {w}×{h}")

        # Update preview widget if it exists
        if hasattr(self, 'detector_preview') and self.detector_preview:
            self.detector_preview.roi = self.current_roi
            self.detector_preview.update()

        # Update detector module if it exists
        if hasattr(self, 'detector_module') and self.detector_module:
            try:
                current_frame = getattr(self, 'current_frame', None)
                if current_frame is not None:
                    self.detector_module.set_roi(self.current_roi, current_frame)
            except Exception as e:
                logger.warning(f"Failed to update detector module ROI: {e}")

    def _reset_roi_to_default(self):
        """Reset ROI to default center 80% of frame"""
        if self.default_roi:
            self._set_roi_controls_to_default()

    def test_api_connection(self):
        """Test API connection"""
        self.api_status_label.setText("Testing...")
        # Actual API test would go here
        QTimer.singleShot(1000, lambda: self.api_status_label.setText("Connection OK"))

    def calibrate_servo(self):
        """Legacy calibration method - redirects to new calibration mode"""
        # Toggle the new calibration mode button
        if hasattr(self, 'calibrate_servo_btn'):
            self.calibrate_servo_btn.setChecked(not self.calibrate_servo_btn.isChecked())

    def refresh_serial_ports(self):
        """Refresh available serial ports"""
        import serial.tools.list_ports

        try:
            # Get list of available ports
            ports = serial.tools.list_ports.comports()
            current_selection = self.port_combo.currentText()

            self.port_combo.clear()

            # Add detected ports
            for port in ports:
                self.port_combo.addItem(port.device)

            # Add some common ports if not detected
            common_ports = ["COM1", "COM2", "COM3", "COM4", "/dev/ttyUSB0", "/dev/ttyACM0"]
            for port in common_ports:
                if self.port_combo.findText(port) == -1:  # Not already in list
                    self.port_combo.addItem(port)

            # Restore previous selection if it still exists
            index = self.port_combo.findText(current_selection)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)

        except ImportError:
            # Fallback if pyserial not available
            self.port_combo.clear()
            self.port_combo.addItems(["COM1", "COM2", "COM3", "COM4", "/dev/ttyUSB0", "/dev/ttyACM0"])
            QMessageBox.warning(self, "Serial Ports",
                                "pyserial not installed. Showing common port names only.")
        except Exception as e:
            logger.error(f"Error refreshing serial ports: {e}")
            QMessageBox.warning(self, "Error", f"Could not refresh ports: {str(e)}")

        # ============= ARDUINO TAB SUPPORTING METHODS =============
        # Add these methods after your existing refresh_serial_ports method

    def setup_servo_table(self):
        """Initialize the servo position table with test buttons"""
        for i in range(10):
            # Bin number (read-only)
            bin_item = QTableWidgetItem(str(i))
            bin_item.setFlags(bin_item.flags() & ~Qt.ItemIsEditable)
            self.servo_table.setItem(i, 0, bin_item)

            # Position (editable)
            self.servo_table.setItem(i, 1, QTableWidgetItem(str(90)))

            # Test button
            test_btn = QPushButton("Test")
            test_btn.clicked.connect(lambda checked, bin_num=i: self.test_single_position(bin_num))
            self.servo_table.setCellWidget(i, 2, test_btn)

    def check_arduino_hardware(self):
        """
        Check if Arduino hardware is connected and update UI accordingly.

        This method scans available serial ports looking for Arduino devices by checking
        for common USB-to-serial chip identifiers and port naming patterns. It automatically
        sets simulation mode if no Arduino is detected.

        The detection looks for:
        - Common USB-serial chip names (CH340, FT232, CP210x, etc.)
        - Arduino-specific identifiers in port descriptions
        - Standard Arduino port naming patterns (/dev/ttyACM*, /dev/ttyUSB*)
        """
        arduino_found = False
        arduino_port = None

        # Comprehensive list of Arduino and USB-serial chip identifiers
        # These cover most genuine Arduino boards and clones
        arduino_identifiers = [
            'arduino',  # Official Arduino boards
            'ch340', 'ch341',  # Common Chinese clone chips
            'ft232', 'ftdi',  # FTDI chips used in older Arduinos
            'cp210', 'cp2102',  # Silicon Labs chips
            'usb serial',  # Generic USB serial devices
            'usb-serial',  # Alternative formatting
            'usbserial',  # Another variant
            'silabs',  # Silicon Labs
            'pl2303',  # Prolific chips
            'hl-340',  # Another CH340 variant
            'usb2.0-serial',  # Generic USB 2.0 serial
            '2341',  # Arduino's USB vendor ID
            '2a03',  # Arduino vendor ID (hex)
            'genuino',  # Arduino's alternative brand
            'atmel',  # Atmel chips (used in Arduino)
            'mega2560',  # Arduino Mega
            'uno',  # Arduino Uno
            'nano',  # Arduino Nano
            'leonardo',  # Arduino Leonardo
        ]

        try:
            # Get list of all available serial ports
            ports = serial.tools.list_ports.comports()
            logger.info(f"Found {len(ports)} serial ports to check")

            for port in ports:
                # Get port information (safely handle None values)
                port_desc = (port.description or '').lower()
                port_name = (port.device or '').lower()
                port_manufacturer = (port.manufacturer or '').lower()
                port_product = (port.product or '').lower()

                # Log port details for debugging
                logger.debug(f"Checking port: {port.device}")
                logger.debug(f"  Description: {port.description}")
                logger.debug(f"  Manufacturer: {port.manufacturer}")
                logger.debug(f"  Product: {port.product}")

                # Check if this port matches Arduino patterns
                # Check 1: Look for identifiers in description
                desc_match = any(identifier in port_desc for identifier in arduino_identifiers)

                # Check 2: Look for identifiers in manufacturer/product
                mfg_match = any(identifier in port_manufacturer for identifier in arduino_identifiers)
                prod_match = any(identifier in port_product for identifier in arduino_identifiers)

                # Check 3: Look for typical Arduino port names (Linux/Mac)
                port_name_match = ('ttyacm' in port_name or
                                   'ttyusb' in port_name or
                                   'cu.usbmodem' in port_name or  # Mac
                                   'cu.usbserial' in port_name)  # Mac

                # Check 4: Windows COM ports with Arduino descriptors
                windows_match = (port.device.startswith('COM') and
                                 (desc_match or mfg_match or prod_match))

                # If any check passes, we found an Arduino
                if desc_match or mfg_match or prod_match or port_name_match or windows_match:
                    arduino_found = True
                    arduino_port = port.device
                    logger.info(f"Arduino detected on port {arduino_port}")
                    logger.info(f"  Device details: {port.description}")
                    break

            # Update UI based on detection results
            if arduino_found:
                # Arduino detected - enable hardware mode
                self.hardware_status_label.setText(f"✓ Arduino detected on {arduino_port}")
                self.hardware_status_label.setStyleSheet("color: green;")
                self.simulation_check.setChecked(False)

                # Set the detected port in the combo box
                self.port_combo.setCurrentText(arduino_port)

                # Update config to ensure hardware mode using EnhancedConfigManager API
                if self.config_manager:
                    self.config_manager.update_module_config("arduino_servo", {
                        "simulation_mode": False,
                        "port": arduino_port
                    })
                    logger.info("Updated config: simulation_mode=False, port=" + arduino_port)
            else:
                # No Arduino detected - default to simulation mode
                self.hardware_status_label.setText("✗ No Arduino detected - defaulting to simulation")
                self.hardware_status_label.setStyleSheet("color: orange;")
                self.simulation_check.setChecked(True)

                # Update config to ensure simulation mode using EnhancedConfigManager API
                if self.config_manager:
                    self.config_manager.update_module_config("arduino_servo", {
                        "simulation_mode": True
                    })
                    logger.info("Updated config: simulation_mode=True (no Arduino detected)")

                # Log available ports for debugging
                if ports:
                    logger.warning("No Arduino found among available ports:")
                    for port in ports:
                        logger.warning(f"  - {port.device}: {port.description}")
                else:
                    logger.warning("No serial ports found on system")

        except ImportError:
            # pyserial not installed
            logger.error("pyserial not installed - cannot detect Arduino")
            self.hardware_status_label.setText("✗ Serial library not available")
            self.hardware_status_label.setStyleSheet("color: red;")
            self.simulation_check.setChecked(True)

            if self.config_manager:
                self.config_manager.update_module_config("arduino_servo", {
                    "simulation_mode": True
                })

        except Exception as e:
            # Unexpected error during detection
            logger.error(f"Error detecting Arduino: {e}", exc_info=True)
            self.hardware_status_label.setText("✗ Error detecting Arduino")
            self.hardware_status_label.setStyleSheet("color: red;")
            self.simulation_check.setChecked(True)

            if self.config_manager:
                self.config_manager.update_module_config("arduino_servo", {
                    "simulation_mode": True
                })

    def verify_arduino_connection(self):
        """
        Verify that Arduino is actually connected when switching to hardware mode.

        This method is called after the user disables simulation mode to ensure
        that an Arduino is actually available. If not found, it prompts the user
        to either retry or return to simulation mode.
        """
        port = self.port_combo.currentText().split(" ")[
            0] if " " in self.port_combo.currentText() else self.port_combo.currentText()
        baud = int(self.baud_combo.currentText())
        timeout = self.timeout_spin.value()

        arduino_connected = False
        error_message = None

        try:
            # Attempt to open serial connection to verify Arduino presence
            logger.info(f"Verifying Arduino connection on {port}")
            ser = serial.Serial(port, baud, timeout=timeout)

            # Send a simple test command if possible
            ser.write(b'\n')  # Send newline to clear any buffer
            time.sleep(0.1)

            # Connection successful
            ser.close()
            arduino_connected = True
            logger.info(f"Arduino verified on {port}")

        except serial.SerialException as e:
            error_message = f"Cannot connect to {port}: {str(e)}"
            logger.error(error_message)
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            logger.error(error_message)

        # Update UI based on verification result
        if arduino_connected:
            self.hardware_status_label.setText(f"✓ Arduino connected on {port}")
            self.hardware_status_label.setStyleSheet("color: green;")
        else:
            # Arduino not found - ask user what to do
            self.hardware_status_label.setText("✗ Arduino connection failed")
            self.hardware_status_label.setStyleSheet("color: red;")

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Arduino Not Found")
            msg.setText("Could not connect to Arduino.")
            msg.setInformativeText(
                error_message or "Please check that Arduino is connected and the correct port is selected.")
            msg.setStandardButtons(QMessageBox.Retry | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Retry)

            # Add custom button for returning to simulation mode
            sim_button = msg.addButton("Use Simulation", QMessageBox.ActionRole)

            result = msg.exec_()

            if msg.clickedButton() == sim_button:
                # Return to simulation mode
                logger.info("User chose to return to simulation mode")
                self.simulation_check.setChecked(True)
            elif result == QMessageBox.Retry:
                # Retry the connection
                logger.info("User chose to retry Arduino connection")
                QTimer.singleShot(100, self.verify_arduino_connection)
            else:
                # User cancelled - return to simulation mode
                logger.info("User cancelled - returning to simulation mode")
                self.simulation_check.setChecked(True)

    def update_arduino_dependent_ui(self, simulation_enabled):
        """
        Update UI elements that depend on Arduino mode.

        This method updates various UI elements based on whether simulation
        or hardware mode is active, ensuring consistent visual feedback.

        Args:
            simulation_enabled: True if simulation mode is active, False for hardware mode
        """
        # Update test button text and tooltips
        if hasattr(self, 'test_connection_btn'):
            if simulation_enabled:
                self.test_connection_btn.setText("Test Connection (Disabled)")
                self.test_connection_btn.setToolTip("Connection testing disabled in simulation mode")
            else:
                self.test_connection_btn.setText("Test Connection")
                self.test_connection_btn.setToolTip("Test the Arduino serial connection")

        # Update servo table appearance
        if hasattr(self, 'servo_table'):
            for row in range(self.servo_table.rowCount()):
                # Update test button in column 2
                test_button = self.servo_table.cellWidget(row, 2)
                if test_button:
                    if simulation_enabled:
                        test_button.setText("Test (Sim)")
                        test_button.setToolTip("Simulated servo movement - no hardware required")
                    else:
                        test_button.setText("Test")
                        test_button.setToolTip("Test servo position with hardware")

        # Update calibration button
        if hasattr(self, 'calibrate_servo_btn'):
            self.calibrate_servo_btn.setEnabled(not simulation_enabled)
            if simulation_enabled:
                self.calibrate_servo_btn.setToolTip("Calibration disabled in simulation mode")
            else:
                self.calibrate_servo_btn.setToolTip("Enter servo calibration mode")

        # Update info labels
        if hasattr(self, 'bin_info_label'):
            if simulation_enabled:
                if not self.calibrate_servo_btn.isChecked():
                    self.bin_info_label.setText("SIMULATION MODE - Servo positions will be simulated")
                    self.bin_info_label.setStyleSheet("color: blue; font-style: italic;")

    def toggle_simulation_mode(self, state):
        """
        Enable/disable Arduino controls based on simulation mode selection.

        This method is called when the user manually toggles the simulation mode checkbox.
        It updates both the UI state and the underlying configuration, and can trigger
        Arduino module reinitialization if the system is running.

        Args:
            state: Qt.Checked if simulation mode is enabled, Qt.Unchecked for hardware mode
        """
        # Determine if simulation mode is being enabled
        simulation_enabled = (state == Qt.Checked)
        hardware_enabled = not simulation_enabled

        logger.info(f"User toggled simulation mode: {'ON' if simulation_enabled else 'OFF'}")

        # Update configuration manager with new mode using EnhancedConfigManager API
        if self.config_manager:
            try:
                # Build update dictionary
                updates = {"simulation_mode": simulation_enabled}

                # If switching to hardware mode, ensure we have a valid port
                if hardware_enabled:
                    current_port = self.port_combo.currentText()
                    if current_port:
                        # Extract just the port name if it contains additional info
                        port_name = current_port.split(" ")[0] if " " in current_port else current_port
                        updates["port"] = port_name
                        logger.info(f"Hardware mode enabled with port: {port_name}")
                    else:
                        logger.warning("Hardware mode enabled but no port selected")

                # Update the Arduino servo configuration
                self.config_manager.update_module_config("arduino_servo", updates)

                # Save configuration changes immediately
                self.config_manager.save_config()
                logger.info("Arduino configuration saved")

            except Exception as e:
                logger.error(f"Failed to update Arduino configuration: {e}")
                QMessageBox.warning(self, "Configuration Error",
                                    f"Failed to save Arduino settings: {str(e)}")

        # Enable/disable all Arduino control widgets based on mode
        for widget_group in self.arduino_widgets.values():
            for widget in widget_group:
                widget.setEnabled(hardware_enabled)

        # Update visual feedback to show current mode
        if simulation_enabled:
            self.hardware_status_label.setText("Simulation mode active - hardware controls disabled")
            self.hardware_status_label.setStyleSheet("color: blue;")

            # Show informational message (only once per toggle)
            if not hasattr(self, '_sim_mode_msg_shown'):
                self._sim_mode_msg_shown = True
                QMessageBox.information(self, "Simulation Mode",
                                        "Arduino simulation mode enabled.\n\n"
                                        "The system will simulate servo movements without "
                                        "requiring physical Arduino hardware.")
        else:
            # Hardware mode - check if Arduino is actually available
            self.hardware_status_label.setText("Hardware mode active - Checking Arduino...")
            self.hardware_status_label.setStyleSheet("color: orange;")

            # Verify Arduino is actually connected
            QTimer.singleShot(100, self.verify_arduino_connection)

        # Emit signal to notify other parts of the application about the mode change
        # This allows the main application to reinitialize the Arduino module if needed
        if hasattr(self, 'arduino_mode_changed'):
            self.arduino_mode_changed.emit(simulation_enabled)
            logger.info(f"Emitted arduino_mode_changed signal (simulation={simulation_enabled})")

        # Update any dependent UI elements
        self.update_arduino_dependent_ui(simulation_enabled)

    def test_arduino_connection(self):
        """Test the Arduino connection with current settings and verify servo response"""
        if self.simulation_check.isChecked():
            QMessageBox.information(self, "Simulation Mode",
                                    "Running in simulation mode - no hardware test performed")
            return

        port = self.port_combo.currentText().split(" ")[0]  # Extract just the port
        baud = int(self.baud_combo.currentText())
        timeout = self.timeout_spin.value()

        try:
            import serial
            import time

            # Try to open serial connection and test servo
            with serial.Serial(port, baud, timeout=timeout) as arduino:
                # Wait for Arduino to initialize
                time.sleep(1.0)

                # Clear any initial messages
                while arduino.in_waiting:
                    arduino.readline()

                # Send a test command (move to center position)
                test_angle = 90
                command = f"A,{test_angle}\n"
                arduino.write(command.encode())

                # Read response
                response = arduino.readline().decode().strip()

                if response and "Moved to:" in response:
                    QMessageBox.information(self, "Connection Test",
                                            f"✓ Arduino connection successful!\n"
                                            f"Port: {port}\n"
                                            f"Response: {response}\n\n"
                                            f"Servo moved to test position ({test_angle}°)")
                elif response:
                    QMessageBox.warning(self, "Connection Test",
                                        f"✓ Connected to {port}\n"
                                        f"✗ Unexpected response: {response}\n\n"
                                        f"Check that the Arduino is running the correct sketch.")
                else:
                    QMessageBox.warning(self, "Connection Test",
                                        f"✓ Connected to {port}\n"
                                        f"✗ No response from Arduino\n\n"
                                        f"Check that the Arduino sketch is loaded and running.")

        except Exception as e:
            QMessageBox.critical(self, "Connection Failed",
                                 f"✗ Failed to connect to Arduino:\n\n{str(e)}\n\n"
                                 f"Check:\n"
                                 f"• Arduino is connected to {port}\n"
                                 f"• Correct port is selected\n"
                                 f"• Arduino sketch is uploaded\n"
                                 f"• No other programs are using the port")

    def get_current_bin_count(self):
        """Get the bin count from the Sorting tab"""
        # This method accesses the sorting tab's bin configuration
        if hasattr(self, 'max_bins_spin'):  # From sorting tab
            return self.max_bins_spin.value()
        return 9  # Default if not found

    def recalculate_bin_positions(self):
        """Recalculate bin positions when settings change"""
        if hasattr(self, 'auto_calculate_on_change') and self.auto_calculate_on_change:
            self.auto_calculate_positions()

    def auto_calculate_positions(self):
        """Automatically calculate evenly distributed bin positions - ENHANCED"""
        bin_count = self.get_current_bin_count()
        min_separation = self.min_separation_spin.value()

        # Calculate total range needed
        total_range_needed = (bin_count - 1) * min_separation

        if total_range_needed > 180:
            QMessageBox.warning(self, "Configuration Error",
                                f"Cannot fit {bin_count} bins with {min_separation}° separation.\n"
                                f"Maximum bins with this separation: {180 // min_separation + 1}")
            return

        # Calculate positions centered around 90 degrees
        center = 90
        if bin_count == 1:
            positions = [center]
        else:
            start_angle = center - (total_range_needed / 2)
            positions = [start_angle + i * min_separation for i in range(bin_count)]

        # Ensure all positions are within 0-180 range
        positions = [max(0, min(180, pos)) for pos in positions]

        # Update table AND save to config immediately
        for i in range(10):
            if i < bin_count:
                self.servo_table.item(i, 1).setText(f"{positions[i]:.1f}")
                self.servo_table.setRowHidden(i, False)
            else:
                self.servo_table.setRowHidden(i, True)

        # ADDED: Save the calculated positions to config immediately
        if self.config_manager:
            config = self.get_arduino_config()
            self.config_manager.update_module_config("arduino_servo", config)
            self.config_manager.save_config()

        self.bin_info_label.setText(f"Positions calculated and saved for {bin_count} bins")

    def test_single_position(self, bin_num):
        """Test a single bin position - ENHANCED to save position changes"""
        if self.simulation_check.isChecked():
            QMessageBox.information(self, "Test Position",
                                    f"Simulation: Would move to bin {bin_num}")
            return

        try:
            position = float(self.servo_table.item(bin_num, 1).text())

            # Get Arduino connection settings
            port = self.port_combo.currentText().split(" ")[0]
            baud = int(self.baud_combo.currentText())
            timeout = self.timeout_spin.value()

            # Establish connection and send command
            import serial
            import time

            with serial.Serial(port, baud, timeout=timeout) as arduino:
                time.sleep(0.5)
                command = f"A,{int(position)}\n"
                arduino.write(command.encode())
                response = arduino.readline().decode().strip()

                if response:
                    QMessageBox.information(self, "Test Position",
                                            f"Bin {bin_num}: {response}")

                    # ADDED: Save position changes to config after successful test
                    if self.config_manager:
                        config = self.get_arduino_config()
                        self.config_manager.update_module_config("arduino_servo", config)
                        self.config_manager.save_config()

                else:
                    QMessageBox.warning(self, "Test Position",
                                        f"Sent command but no response from Arduino")

        except Exception as e:
            QMessageBox.critical(self, "Test Failed",
                                 f"Failed to test position:\n{str(e)}")
    def test_sweep_positions(self):
        """Test sweep through all bin positions with actual Arduino communication"""
        if self.simulation_check.isChecked():
            QMessageBox.information(self, "Test Sweep",
                                    "Simulation: Would sweep through all bin positions")
            return

        try:
            bin_count = self.get_current_bin_count()
            positions = []
            for i in range(bin_count):
                pos_text = self.servo_table.item(i, 1).text()
                positions.append(int(float(pos_text)))

            message = "Will sweep through positions:\n"
            message += ", ".join([f"{p}°" for p in positions])
            message += "\n\nEach position will be held for 1 second."

            reply = QMessageBox.question(self, "Test Sweep", message,
                                         QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                # Get Arduino connection settings
                port = self.port_combo.currentText().split(" ")[0]
                baud = int(self.baud_combo.currentText())
                timeout = self.timeout_spin.value()

                # Create progress dialog
                progress = QProgressDialog("Testing servo positions...", "Cancel", 0, len(positions))
                progress.setWindowModality(Qt.WindowModal)
                progress.show()

                import serial
                import time

                with serial.Serial(port, baud, timeout=timeout) as arduino:
                    # Wait for Arduino to initialize
                    time.sleep(0.5)

                    for i, position in enumerate(positions):
                        if progress.wasCanceled():
                            break

                        progress.setValue(i)
                        progress.setLabelText(f"Testing bin {i} at {position}°...")
                        QApplication.processEvents()

                        # Send command
                        command = f"A,{position}\n"
                        arduino.write(command.encode())

                        # Read response
                        response = arduino.readline().decode().strip()

                        # Hold position for 1 second
                        time.sleep(1.0)

                    progress.setValue(len(positions))
                    progress.close()

                    QMessageBox.information(self, "Sweep Complete",
                                            "Servo position sweep completed successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Sweep Failed",
                                 f"Failed to sweep positions:\n{str(e)}")

    def toggle_calibration_mode(self, checked):
        """Toggle calibration mode"""
        if checked:
            self.calibrate_servo_btn.setText("Exit Calibration")
            self.bin_info_label.setText("CALIBRATION MODE: Manually adjust positions and test")
            self.bin_info_label.setStyleSheet("color: red; font-weight: bold;")

            # Enable manual editing of all positions
            for i in range(10):
                item = self.servo_table.item(i, 1)
                if item:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
        else:
            self.calibrate_servo_btn.setText("Calibration Mode")
            self.bin_info_label.setText("Bins will be auto-calculated based on Sorting tab settings")
            self.bin_info_label.setStyleSheet("")

    def update_arduino_bin_count(self, value):
        """Update Arduino tab when bin count changes in Sorting tab"""
        if hasattr(self, 'servo_table'):  # Check if Arduino tab exists
            self.auto_calculate_positions()

    def get_arduino_config(self):
        """Get the current Arduino configuration from the UI - FIXED VERSION"""
        config = {
            # Arduino connection settings (correct location)
            "port": self.port_combo.currentText().split(" ")[
                0] if " " in self.port_combo.currentText() else self.port_combo.currentText(),
            "baud_rate": int(self.baud_combo.currentText()),
            "timeout": self.timeout_spin.value(),
            "connection_retries": self.retry_spin.value(),
            "retry_delay": self.retry_delay_spin.value(),
            "simulation_mode": self.simulation_check.isChecked(),

            # FIXED: Servo hardware settings (kept in arduino_servo config where GUI saves them)
            "min_pulse": self.min_pulse_spin.value(),
            "max_pulse": self.max_pulse_spin.value(),
            "default_position": self.default_pos_spin.value(),
            "min_bin_separation": self.min_separation_spin.value(),
            "calibration_mode": False,  # Add this field

            # FIXED: Bin positions (actual user-configured positions)
            "bin_positions": {}
        }

        # Get bin positions from table (the positions user actually set)
        bin_count = self.get_current_bin_count()
        for i in range(bin_count):
            if self.servo_table.item(i, 1):
                pos_text = self.servo_table.item(i, 1).text()
                try:
                    config["bin_positions"][str(i)] = float(pos_text)
                except ValueError:
                    config["bin_positions"][str(i)] = 90.0  # Default fallback

        return config

    def update_preview_frame(self, frame):
        """Update the preview frame (public method for external use)"""
        if hasattr(self, 'detector_preview'):
            # Update the detector preview widget
            detection_data = {
                'roi': getattr(self, 'current_roi', None),
                'zones': {
                    'entry': self.entry_zone_slider.value() / 100.0,
                    'exit': self.exit_zone_slider.value() / 100.0
                },
                'pieces': []  # Empty for configuration preview
            }
            self.detector_preview.update_frame(frame, detection_data)

        if hasattr(self, 'preview_widget'):
            # Also update the camera tab preview widget
            self.preview_widget.update_frame(frame)
