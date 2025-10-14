"""
test_camera_views.py - Test script for camera view widgets

FILE LOCATION: tests/test_camera_views.py (or project root for quick testing)

This script tests the CameraViewRaw and CameraViewROI widgets to verify:
1. Camera frame reception and display
2. FPS counter functionality
3. ROI overlay rendering
4. Thread-safe frame updates

Usage:
    python test_camera_views.py

Controls:
    - Toggle FPS: Press 'F' key
    - Switch between Raw/ROI: Use tabs
    - Close: Press 'Q' or close window
"""

import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QTabWidget,
                             QGroupBox, QSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, QTimer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from camera_module import create_camera
    from enhanced_config_manager import create_config_manager
    from GUI.widgets.camera_view import CameraViewRaw, CameraViewROI

    logger.info("✓ All imports successful")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)


class CameraViewTestWindow(QMainWindow):
    """
    Test window for camera view widgets.

    Features:
    - Side-by-side display of Raw and ROI views
    - FPS toggle controls
    - ROI configuration controls
    - Camera status display
    """

    def __init__(self):
        super().__init__()

        # Initialize components
        self.camera = None
        self.raw_view = None
        self.roi_view = None

        # Setup UI
        self.setWindowTitle("Camera View Widget Test")
        self.setGeometry(100, 100, 1400, 800)

        self.init_ui()
        self.init_camera()

        logger.info("Test window initialized")

    def init_ui(self):
        """Initialize the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("Camera View Widget Test")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #2E86AB;
                color: white;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(title)

        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create Raw view tab
        self.create_raw_view_tab()

        # Create ROI view tab
        self.create_roi_view_tab()

        # Control panel
        self.create_control_panel(main_layout)

        # Status bar
        self.statusBar().showMessage("Ready - Initializing camera...")

    def create_raw_view_tab(self):
        """Create the Raw Camera View tab."""
        raw_tab = QWidget()
        layout = QVBoxLayout(raw_tab)

        # Title
        title = QLabel("Raw Camera View (No Overlays)")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)

        # Camera view widget
        self.raw_view = CameraViewRaw()
        self.raw_view.set_fps_visible(True)
        layout.addWidget(self.raw_view)

        # Controls
        controls = QHBoxLayout()

        self.raw_fps_btn = QPushButton("Toggle FPS")
        self.raw_fps_btn.clicked.connect(self.toggle_raw_fps)
        controls.addWidget(self.raw_fps_btn)

        controls.addStretch()

        status_label = QLabel("Status: Waiting for frames...")
        status_label.setStyleSheet("color: orange;")
        controls.addWidget(status_label)
        self.raw_status_label = status_label

        layout.addLayout(controls)

        self.tab_widget.addTab(raw_tab, "Raw View")

    def create_roi_view_tab(self):
        """Create the ROI Camera View tab."""
        roi_tab = QWidget()
        layout = QVBoxLayout(roi_tab)

        # Title
        title = QLabel("ROI Camera View (With Zones) - Real-time Configuration")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)

        # Main content layout (camera + controls side by side)
        content_layout = QHBoxLayout()

        # Camera view widget
        self.roi_view = CameraViewROI()
        self.roi_view.set_fps_visible(True)
        content_layout.addWidget(self.roi_view, stretch=3)

        # ROI configuration panel
        roi_config = self.create_roi_config_panel()
        content_layout.addWidget(roi_config, stretch=1)

        layout.addLayout(content_layout)

        # Controls
        controls = QHBoxLayout()

        self.roi_fps_btn = QPushButton("Toggle FPS")
        self.roi_fps_btn.clicked.connect(self.toggle_roi_fps)
        controls.addWidget(self.roi_fps_btn)

        controls.addStretch()

        status_label = QLabel("Status: Waiting for frames...")
        status_label.setStyleSheet("color: orange;")
        controls.addWidget(status_label)
        self.roi_status_label = status_label

        layout.addLayout(controls)

        self.tab_widget.addTab(roi_tab, "ROI View")

    def create_roi_config_panel(self):
        """Create ROI configuration panel."""
        config_group = QGroupBox("ROI Configuration")
        config_layout = QFormLayout()

        # ROI coordinates
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 1920)
        self.roi_x_spin.setValue(100)
        self.roi_x_spin.valueChanged.connect(self.update_roi)
        config_layout.addRow("ROI X:", self.roi_x_spin)

        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 1080)
        self.roi_y_spin.setValue(50)
        self.roi_y_spin.valueChanged.connect(self.update_roi)
        config_layout.addRow("ROI Y:", self.roi_y_spin)

        self.roi_width_spin = QSpinBox()
        self.roi_width_spin.setRange(100, 1920)
        self.roi_width_spin.setValue(1720)
        self.roi_width_spin.valueChanged.connect(self.update_roi)
        config_layout.addRow("ROI Width:", self.roi_width_spin)

        self.roi_height_spin = QSpinBox()
        self.roi_height_spin.setRange(100, 1080)
        self.roi_height_spin.setValue(980)
        self.roi_height_spin.valueChanged.connect(self.update_roi)
        config_layout.addRow("ROI Height:", self.roi_height_spin)

        # Zone percentages
        self.entry_zone_spin = QSpinBox()
        self.entry_zone_spin.setRange(5, 50)
        self.entry_zone_spin.setValue(15)
        self.entry_zone_spin.setSuffix("%")
        self.entry_zone_spin.valueChanged.connect(self.update_zones)
        config_layout.addRow("Entry Zone:", self.entry_zone_spin)

        self.exit_zone_spin = QSpinBox()
        self.exit_zone_spin.setRange(5, 50)
        self.exit_zone_spin.setValue(15)
        self.exit_zone_spin.setSuffix("%")
        self.exit_zone_spin.valueChanged.connect(self.update_zones)
        config_layout.addRow("Exit Zone:", self.exit_zone_spin)

        # Apply button (now optional since real-time update is enabled)
        apply_btn = QPushButton("Apply ROI")
        apply_btn.clicked.connect(lambda: self.apply_roi_config(show_message=True))
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        apply_btn.setToolTip("ROI updates automatically as you adjust values")
        config_layout.addRow(apply_btn)

        config_group.setLayout(config_layout)
        return config_group

    def create_control_panel(self, parent_layout):
        """Create bottom control panel."""
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Info label
        info_label = QLabel("ROI updates in real-time | Press 'F' to toggle FPS | Press 'Q' to quit")
        info_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        control_layout.addWidget(info_label)

        control_layout.addStretch()

        # Camera info
        self.camera_info_label = QLabel("Camera: Not initialized")
        self.camera_info_label.setStyleSheet("color: #E67E22; font-weight: bold;")
        control_layout.addWidget(self.camera_info_label)

        # Close button
        close_btn = QPushButton("Close Test")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        control_layout.addWidget(close_btn)

        parent_layout.addWidget(control_panel)

    def init_camera(self):
        """Initialize camera and register consumers."""
        try:
            logger.info("Initializing camera...")

            # Create config manager and camera
            config_manager = create_config_manager()
            self.camera = create_camera("webcam", config_manager)

            # Initialize camera hardware
            logger.info("Initializing camera hardware...")
            if not self.camera.initialize():
                raise Exception("Failed to initialize camera hardware")

            logger.info("Camera hardware initialized successfully")

            # Get camera info from statistics
            stats = self.camera.get_statistics()
            camera_info = stats.get('camera', {})

            # Extract device info
            device_id = camera_info.get('device_id', 'N/A')
            resolution = camera_info.get('resolution', 'unknown')
            fps = camera_info.get('fps_requested', 0)

            camera_text = f"Camera: Device {device_id} | {resolution} | {fps} FPS"
            self.camera_info_label.setText(camera_text)
            self.camera_info_label.setStyleSheet("color: #27AE60; font-weight: bold;")

            # Register Raw view as consumer
            logger.info("Registering Raw view as camera consumer...")
            success_raw = self.camera.register_consumer(
                name="raw_view_test",
                callback=self.raw_view.receive_frame,
                processing_type="async",
                priority=20
            )

            if success_raw:
                logger.info("✓ Raw view registered successfully")
                self.raw_status_label.setText("Status: Registered")
                self.raw_status_label.setStyleSheet("color: orange;")
            else:
                logger.error("✗ Failed to register Raw view")
                self.raw_status_label.setText("Status: Registration Failed")
                self.raw_status_label.setStyleSheet("color: red;")
                raise Exception("Failed to register raw view consumer")

            # Register ROI view as consumer
            logger.info("Registering ROI view as camera consumer...")
            success_roi = self.camera.register_consumer(
                name="roi_view_test",
                callback=self.roi_view.receive_frame,
                processing_type="async",
                priority=20
            )

            if success_roi:
                logger.info("✓ ROI view registered successfully")
                self.roi_status_label.setText("Status: Registered")
                self.roi_status_label.setStyleSheet("color: orange;")

                # Set initial ROI configuration (without status message)
                self.apply_roi_config(show_message=False)
            else:
                logger.error("✗ Failed to register ROI view")
                self.roi_status_label.setText("Status: Registration Failed")
                self.roi_status_label.setStyleSheet("color: red;")
                raise Exception("Failed to register ROI view consumer")

            # Start camera capture
            logger.info("Starting camera capture...")
            if not self.camera.start_capture():
                raise Exception("Failed to start camera capture")

            logger.info("✓ Camera capture started successfully")

            # Update status labels after successful capture start
            self.raw_status_label.setText("Status: Active")
            self.raw_status_label.setStyleSheet("color: green;")
            self.roi_status_label.setText("Status: Active")
            self.roi_status_label.setStyleSheet("color: green;")

            self.statusBar().showMessage("Camera active - Receiving frames", 3000)
            logger.info("✓ Camera test initialization complete")

            # Start a timer to check if frames are being received
            QTimer.singleShot(2000, self.check_frame_reception)

        except Exception as e:
            logger.error(f"Error initializing camera: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {str(e)}")
            self.camera_info_label.setText(f"Camera Error: {str(e)}")
            self.camera_info_label.setStyleSheet("color: red; font-weight: bold;")

    def check_frame_reception(self):
        """Check if frames are actually being received after 2 seconds."""
        if self.raw_view.fps_value == 0.0 and self.roi_view.fps_value == 0.0:
            logger.warning("⚠ No frames received after 2 seconds!")
            logger.warning("Camera may not be capturing or distributing frames")
            self.statusBar().showMessage("Warning: No frames received - check camera", 5000)

            # Try to get camera statistics
            try:
                stats = self.camera.get_statistics()
                logger.info(f"Camera statistics: {stats}")
            except Exception as e:
                logger.error(f"Could not get camera statistics: {e}")
        else:
            logger.info(
                f"✓ Frames are being received! Raw FPS: {self.raw_view.fps_value:.1f}, ROI FPS: {self.roi_view.fps_value:.1f}")

    def toggle_raw_fps(self):
        """Toggle FPS display on raw view."""
        current = self.raw_view.fps_visible
        self.raw_view.set_fps_visible(not current)
        logger.info(f"Raw view FPS toggled: {not current}")

    def toggle_roi_fps(self):
        """Toggle FPS display on ROI view."""
        current = self.roi_view.fps_visible
        self.roi_view.set_fps_visible(not current)
        logger.info(f"ROI view FPS toggled: {not current}")

    def update_roi(self):
        """Update ROI in real-time when spinbox values change."""
        self.apply_roi_config(show_message=False)

    def update_zones(self):
        """Update zones in real-time when spinbox values change."""
        self.apply_roi_config(show_message=False)

    def apply_roi_config(self, show_message=True):
        """Apply current ROI configuration to the ROI view."""
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        width = self.roi_width_spin.value()
        height = self.roi_height_spin.value()

        entry_pct = self.entry_zone_spin.value() / 100.0
        exit_pct = self.exit_zone_spin.value() / 100.0

        self.roi_view.set_roi(x, y, width, height)
        self.roi_view.set_zones(entry_pct, exit_pct)

        logger.debug(f"ROI applied: ({x}, {y}, {width}, {height})")
        logger.debug(f"Zones applied: entry={entry_pct}, exit={exit_pct}")

        # Only show status message if explicitly requested (e.g., from button click)
        if show_message:
            self.statusBar().showMessage(f"ROI updated: ({x}, {y}, {width}, {height})", 2000)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_F:
            # Toggle FPS on current tab
            current_index = self.tab_widget.currentIndex()
            if current_index == 0:
                self.toggle_raw_fps()
            else:
                self.toggle_roi_fps()
        elif event.key() == Qt.Key_Q:
            self.close()

    def closeEvent(self, event):
        """Clean up when closing the window."""
        logger.info("Closing test window...")

        if self.camera:
            try:
                logger.info("Unregistering consumers...")
                self.camera.unregister_consumer("raw_view_test")
                self.camera.unregister_consumer("roi_view_test")

                logger.info("Stopping camera...")
                self.camera.stop_capture()

                logger.info("Releasing camera...")
                self.camera.release()

            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        logger.info("✓ Cleanup complete")
        event.accept()


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("Camera View Widget Test Starting")
    logger.info("=" * 60)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create and show test window
    window = CameraViewTestWindow()
    window.show()

    logger.info("Test window displayed - Check both tabs!")
    logger.info("Keyboard shortcuts: 'F' = Toggle FPS, 'Q' = Quit")

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()