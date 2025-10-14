"""
demo_detector_tab.py - Demo script for DetectorConfigTab

FILE LOCATION: tests/demo_detector_tab.py (or project root for quick testing)

This script demonstrates and tests the DetectorConfigTab to verify:
1. ROI configuration with automatic frame boundary enforcement
2. Real-time preview updates
3. Zone percentage controls
4. Detection parameter configuration
5. Preset button functionality
6. Configuration save/load
7. Validation logic

Usage:
    python demo_detector_tab.py

Controls:
    - Adjust ROI: Use spinboxes in left panel
    - Change Zones: Use sliders
    - Presets: Click "Full Frame" or "Center 80%" buttons
    - Toggle FPS: Click "Toggle FPS" button
    - Save Config: Click "Save Configuration" button
    - Reset: Click "Reset to Defaults" button
    - Close: Press 'Q' or close window

Features to Test:
    - ROI automatically constrained to frame boundaries
    - Real-time preview updates as you adjust values
    - Preset buttons set ROI correctly
    - Validation prevents invalid configurations
    - Save/load preserves settings
"""

import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QMessageBox,
                             QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

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
    from GUI.config_tabs.detector_tab import DetectorConfigTab

    logger.info("✓ All imports successful")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)


class DetectorTabDemoWindow(QMainWindow):
    """
    Demo window for DetectorConfigTab testing.

    Features:
    - Full detector tab with live preview
    - Camera initialization and management
    - Configuration save/load testing
    - Validation testing
    - Status monitoring
    - Test controls
    """

    def __init__(self):
        super().__init__()

        # Initialize components
        self.camera = None
        self.config_manager = None
        self.detector_tab = None

        # Setup UI
        self.setWindowTitle("Detector Tab Demo - Configuration Testing")
        self.setGeometry(100, 100, 1600, 900)

        self.init_ui()
        self.init_camera_and_tab()

        logger.info("Demo window initialized")

    def init_ui(self):
        """Initialize the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("Detector Configuration Tab - Demo & Testing")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                background-color: #2E86AB;
                color: white;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(title)

        # Status panel at top
        status_panel = self.create_status_panel()
        main_layout.addWidget(status_panel)

        # Detector tab will be added here after initialization
        self.tab_placeholder = QWidget()
        self.tab_layout = QVBoxLayout(self.tab_placeholder)
        main_layout.addWidget(self.tab_placeholder, stretch=1)

        # Control panel at bottom
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Status bar
        self.statusBar().showMessage("Ready - Initializing...")

    def create_status_panel(self) -> QWidget:
        """Create status monitoring panel."""
        panel = QGroupBox("System Status")
        layout = QHBoxLayout(panel)

        # Camera status
        self.camera_status = QLabel("Camera: Not initialized")
        self.camera_status.setStyleSheet("color: orange; font-weight: bold;")
        layout.addWidget(self.camera_status)

        layout.addStretch()

        # Tab status
        self.tab_status = QLabel("Tab: Not created")
        self.tab_status.setStyleSheet("color: orange; font-weight: bold;")
        layout.addWidget(self.tab_status)

        layout.addStretch()

        # Modified indicator
        self.modified_indicator = QLabel("Modified: No")
        self.modified_indicator.setStyleSheet("color: green;")
        layout.addWidget(self.modified_indicator)

        layout.addStretch()

        # Frame dimensions
        self.frame_dims_label = QLabel("Frame: Unknown")
        self.frame_dims_label.setStyleSheet("color: gray;")
        layout.addWidget(self.frame_dims_label)

        return panel

    def create_control_panel(self) -> QWidget:
        """Create control panel with test buttons."""
        panel = QGroupBox("Test Controls")
        layout = QVBoxLayout(panel)

        # Button row 1: Configuration actions
        row1 = QHBoxLayout()

        save_btn = QPushButton("💾 Save Configuration")
        save_btn.setToolTip("Save current settings to config manager")
        save_btn.clicked.connect(self.test_save_config)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        row1.addWidget(save_btn)

        load_btn = QPushButton("📂 Reload Configuration")
        load_btn.setToolTip("Reload settings from config manager")
        load_btn.clicked.connect(self.test_load_config)
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        row1.addWidget(load_btn)

        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setToolTip("Reset all values to defaults")
        reset_btn.clicked.connect(self.test_reset_defaults)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)
        row1.addWidget(reset_btn)

        layout.addLayout(row1)

        # Button row 2: Validation and testing
        row2 = QHBoxLayout()

        validate_btn = QPushButton("✓ Validate Configuration")
        validate_btn.setToolTip("Test validation logic")
        validate_btn.clicked.connect(self.test_validation)
        validate_btn.setStyleSheet("""
            QPushButton {
                background-color: #16A085;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138D75;
            }
        """)
        row2.addWidget(validate_btn)

        get_config_btn = QPushButton("📋 Get Current Config")
        get_config_btn.setToolTip("Display current configuration as dictionary")
        get_config_btn.clicked.connect(self.test_get_config)
        get_config_btn.setStyleSheet("""
            QPushButton {
                background-color: #8E44AD;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7D3C98;
            }
        """)
        row2.addWidget(get_config_btn)

        test_bounds_btn = QPushButton("⚠️ Test Boundary Enforcement")
        test_bounds_btn.setToolTip("Test ROI boundary constraint logic")
        test_bounds_btn.clicked.connect(self.test_boundary_enforcement)
        test_bounds_btn.setStyleSheet("""
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
        row2.addWidget(test_bounds_btn)

        layout.addLayout(row2)

        # Info label
        info = QLabel("💡 Tip: Adjust ROI values and watch them auto-constrain to frame boundaries")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; padding: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        return panel

    def init_camera_and_tab(self):
        """Initialize camera and detector tab."""
        try:
            logger.info("Initializing config manager...")
            self.config_manager = create_config_manager()
            logger.info("✓ Config manager created")

            logger.info("Initializing camera...")
            self.camera = create_camera("webcam", self.config_manager)

            # Initialize camera hardware
            if not self.camera.initialize():
                raise Exception("Failed to initialize camera hardware")

            logger.info("✓ Camera hardware initialized")
            self.camera_status.setText("Camera: Initialized")
            self.camera_status.setStyleSheet("color: orange; font-weight: bold;")

            # Get camera info
            stats = self.camera.get_statistics()
            camera_info = stats.get('camera', {})
            resolution = camera_info.get('resolution', 'unknown')

            # Update frame dimensions label
            if 'x' in resolution:
                parts = resolution.split('x')
                if len(parts) == 2:
                    width, height = int(parts[0]), int(parts[1])
                    self.frame_dims_label.setText(f"Frame: {width}×{height}")
                    self.frame_dims_label.setStyleSheet("color: #27AE60; font-weight: bold;")

            # Create detector tab
            logger.info("Creating detector tab...")
            self.detector_tab = DetectorConfigTab(
                self.config_manager,
                camera=self.camera,
                parent=self
            )

            # Add tab to layout
            self.tab_layout.addWidget(self.detector_tab)

            logger.info("✓ Detector tab created")
            self.tab_status.setText("Tab: Created")
            self.tab_status.setStyleSheet("color: orange; font-weight: bold;")

            # Connect tab signals
            self.detector_tab.modification_changed.connect(self.on_modification_changed)
            self.detector_tab.validation_failed.connect(self.on_validation_failed)
            self.detector_tab.roi_changed.connect(self.on_roi_changed)
            self.detector_tab.zones_changed.connect(self.on_zones_changed)

            # Register preview
            logger.info("Registering preview consumer...")
            if self.detector_tab.register_preview():
                logger.info("✓ Preview registered")
            else:
                logger.warning("⚠ Preview registration failed")

            # Start camera capture
            logger.info("Starting camera capture...")
            if not self.camera.start_capture():
                raise Exception("Failed to start camera capture")

            logger.info("✓ Camera capture started")
            self.camera_status.setText("Camera: Active")
            self.camera_status.setStyleSheet("color: #27AE60; font-weight: bold;")
            self.tab_status.setText("Tab: Active")
            self.tab_status.setStyleSheet("color: #27AE60; font-weight: bold;")

            self.statusBar().showMessage("✓ All systems active - Test away!", 3000)

            # Start monitoring timer
            QTimer.singleShot(2000, self.check_preview_status)

        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {str(e)}")
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize:\n{str(e)}\n\nCheck the logs for details."
            )

    def check_preview_status(self):
        """Check if preview is receiving frames."""
        if not self.detector_tab or not self.detector_tab.preview_widget:
            return

        fps = self.detector_tab.preview_widget.fps_value
        if fps > 0:
            logger.info(f"✓ Preview receiving frames at {fps:.1f} FPS")
            self.statusBar().showMessage(f"✓ Preview active: {fps:.1f} FPS", 3000)
        else:
            logger.warning("⚠ Preview not receiving frames")
            self.statusBar().showMessage("⚠ Preview not receiving frames - check camera", 5000)

    # ========================================================================
    # SIGNAL HANDLERS
    # ========================================================================

    def on_modification_changed(self, is_modified: bool):
        """Handle modification status changes."""
        if is_modified:
            self.modified_indicator.setText("Modified: YES")
            self.modified_indicator.setStyleSheet("color: #E67E22; font-weight: bold;")
        else:
            self.modified_indicator.setText("Modified: No")
            self.modified_indicator.setStyleSheet("color: #27AE60;")

    def on_validation_failed(self, error_message: str):
        """Handle validation failures."""
        logger.warning(f"Validation failed: {error_message}")
        QMessageBox.warning(self, "Validation Failed", error_message)

    def on_roi_changed(self, roi: tuple):
        """Handle ROI changes."""
        x, y, w, h = roi
        logger.debug(f"ROI changed: ({x}, {y}, {w}, {h})")
        self.statusBar().showMessage(f"ROI: ({x}, {y}, {w}, {h})", 2000)

    def on_zones_changed(self, zones: tuple):
        """Handle zone changes."""
        entry, exit = zones
        logger.debug(f"Zones changed: entry={entry:.2f}, exit={exit:.2f}")

    # ========================================================================
    # TEST METHODS
    # ========================================================================

    def test_save_config(self):
        """Test saving configuration."""
        logger.info("=" * 60)
        logger.info("TEST: Save Configuration")
        logger.info("=" * 60)

        if not self.detector_tab:
            QMessageBox.warning(self, "Error", "Detector tab not initialized")
            return

        success = self.detector_tab.save_config()

        if success:
            logger.info("✓ Configuration saved successfully")
            QMessageBox.information(
                self,
                "Save Successful",
                "Configuration has been saved to config manager."
            )
        else:
            logger.error("✗ Configuration save failed")
            QMessageBox.warning(
                self,
                "Save Failed",
                "Failed to save configuration. Check logs for details."
            )

    def test_load_config(self):
        """Test loading configuration."""
        logger.info("=" * 60)
        logger.info("TEST: Load Configuration")
        logger.info("=" * 60)

        if not self.detector_tab:
            QMessageBox.warning(self, "Error", "Detector tab not initialized")
            return

        success = self.detector_tab.load_config()

        if success:
            logger.info("✓ Configuration loaded successfully")
            QMessageBox.information(
                self,
                "Load Successful",
                "Configuration has been reloaded from config manager."
            )
        else:
            logger.error("✗ Configuration load failed")
            QMessageBox.warning(
                self,
                "Load Failed",
                "Failed to load configuration. Check logs for details."
            )

    def test_reset_defaults(self):
        """Test reset to defaults."""
        logger.info("=" * 60)
        logger.info("TEST: Reset to Defaults")
        logger.info("=" * 60)

        if not self.detector_tab:
            QMessageBox.warning(self, "Error", "Detector tab not initialized")
            return

        self.detector_tab.reset_to_defaults()
        logger.info("Reset to defaults triggered")

    def test_validation(self):
        """Test validation logic."""
        logger.info("=" * 60)
        logger.info("TEST: Validation")
        logger.info("=" * 60)

        if not self.detector_tab:
            QMessageBox.warning(self, "Error", "Detector tab not initialized")
            return

        is_valid = self.detector_tab.validate()

        if is_valid:
            logger.info("✓ Current configuration is valid")
            QMessageBox.information(
                self,
                "Validation Passed",
                "✓ Current configuration is valid!"
            )
        else:
            logger.warning("✗ Current configuration is invalid")
            # Error message already shown by validation_failed signal

    def test_get_config(self):
        """Test getting current configuration."""
        logger.info("=" * 60)
        logger.info("TEST: Get Current Configuration")
        logger.info("=" * 60)

        if not self.detector_tab:
            QMessageBox.warning(self, "Error", "Detector tab not initialized")
            return

        config = self.detector_tab.get_config()

        # Log configuration
        logger.info("Current configuration:")
        logger.info(f"  Detector: {config.get('detector', {})}")
        logger.info(f"  ROI: {config.get('detector_roi', {})}")

        # Display in dialog
        import json
        config_text = json.dumps(config, indent=2)

        msg = QMessageBox(self)
        msg.setWindowTitle("Current Configuration")
        msg.setText("Current configuration values:")
        msg.setDetailedText(config_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def test_boundary_enforcement(self):
        """Test ROI boundary enforcement."""
        logger.info("=" * 60)
        logger.info("TEST: Boundary Enforcement")
        logger.info("=" * 60)

        if not self.detector_tab:
            QMessageBox.warning(self, "Error", "Detector tab not initialized")
            return

        # Get current frame dimensions
        frame_w = self.detector_tab.frame_width
        frame_h = self.detector_tab.frame_height

        logger.info(f"Frame dimensions: {frame_w}×{frame_h}")

        # Test 1: Try to set ROI exceeding width
        logger.info("\nTest 1: Setting ROI width exceeding frame width...")
        original_x = self.detector_tab.roi_x_spin.value()

        self.detector_tab.roi_x_spin.setValue(100)
        self.detector_tab.roi_width_spin.setValue(frame_w)  # This should be reduced

        actual_width = self.detector_tab.roi_width_spin.value()
        logger.info(f"  Set width to {frame_w}, actual width: {actual_width}")

        if actual_width < frame_w:
            logger.info("  ✓ Width was constrained correctly")
        else:
            logger.warning("  ✗ Width was NOT constrained")

        # Test 2: Try to set ROI exceeding height
        logger.info("\nTest 2: Setting ROI height exceeding frame height...")

        self.detector_tab.roi_y_spin.setValue(50)
        self.detector_tab.roi_height_spin.setValue(frame_h)  # This should be reduced

        actual_height = self.detector_tab.roi_height_spin.value()
        logger.info(f"  Set height to {frame_h}, actual height: {actual_height}")

        if actual_height < frame_h:
            logger.info("  ✓ Height was constrained correctly")
        else:
            logger.warning("  ✗ Height was NOT constrained")

        # Test 3: Try to move ROI beyond boundaries
        logger.info("\nTest 3: Moving ROI beyond frame boundaries...")

        self.detector_tab.roi_width_spin.setValue(500)
        self.detector_tab.roi_height_spin.setValue(400)
        self.detector_tab.roi_x_spin.setValue(frame_w - 100)  # Should be constrained

        actual_x = self.detector_tab.roi_x_spin.value()
        logger.info(f"  Tried to set X to {frame_w - 100}, actual X: {actual_x}")

        if actual_x + 500 <= frame_w:
            logger.info("  ✓ X position was constrained correctly")
        else:
            logger.warning("  ✗ X position was NOT constrained")

        # Show summary
        QMessageBox.information(
            self,
            "Boundary Enforcement Test",
            "Boundary enforcement tests completed.\n\n"
            "Check the console logs for detailed results.\n\n"
            "The ROI should have been automatically constrained "
            "to stay within the frame boundaries."
        )

    # ========================================================================
    # KEYBOARD SHORTCUTS
    # ========================================================================

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            # Ctrl+S: Save
            self.test_save_config()
        elif event.key() == Qt.Key_L and event.modifiers() == Qt.ControlModifier:
            # Ctrl+L: Load
            self.test_load_config()
        elif event.key() == Qt.Key_R and event.modifiers() == Qt.ControlModifier:
            # Ctrl+R: Reset
            self.test_reset_defaults()
        elif event.key() == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
            # Ctrl+V: Validate
            self.test_validation()
        elif event.key() == Qt.Key_Q:
            # Q: Quit
            self.close()

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def closeEvent(self, event):
        """Clean up when closing the window."""
        logger.info("Closing demo window...")

        if self.detector_tab:
            try:
                logger.info("Cleaning up detector tab...")
                self.detector_tab.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up tab: {e}")

        if self.camera:
            try:
                logger.info("Unregistering preview...")
                self.camera.unregister_consumer("detector_preview")

                logger.info("Stopping camera...")
                self.camera.stop_capture()

                logger.info("Releasing camera...")
                self.camera.release()

            except Exception as e:
                logger.error(f"Error during camera cleanup: {e}")

        logger.info("✓ Cleanup complete")
        event.accept()


def main():
    """Main demo function."""
    logger.info("=" * 70)
    logger.info("DETECTOR TAB DEMO - Configuration Testing")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This demo tests the DetectorConfigTab with live camera preview.")
    logger.info("")
    logger.info("Features to test:")
    logger.info("  1. ROI configuration with automatic boundary enforcement")
    logger.info("  2. Real-time preview updates")
    logger.info("  3. Zone percentage controls")
    logger.info("  4. Preset buttons (Full Frame, Center 80%)")
    logger.info("  5. Configuration save/load")
    logger.info("  6. Validation logic")
    logger.info("")
    logger.info("Keyboard shortcuts:")
    logger.info("  Ctrl+S - Save configuration")
    logger.info("  Ctrl+L - Load configuration")
    logger.info("  Ctrl+R - Reset to defaults")
    logger.info("  Ctrl+V - Validate configuration")
    logger.info("  Q      - Quit")
    logger.info("")
    logger.info("=" * 70)

    # Create Qt application
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show demo window
    window = DetectorTabDemoWindow()
    window.show()

    logger.info("Demo window displayed - Start testing!")

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()