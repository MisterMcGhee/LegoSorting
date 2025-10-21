"""
demo_camera_tab.py - Interactive demo for the camera configuration tab

FILE LOCATION: GUI/demos/demo_camera_tab.py (or project root)

This script demonstrates the camera configuration tab in a standalone window.
Run this to see and interact with the camera tab UI.

Usage:
    python demo_camera_tab.py
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from enhanced_config_manager import create_config_manager
    from GUI.config_tabs.camera_tab import CameraConfigTab

    logger.info("✓ All imports successful")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)


class CameraTabDemoWindow(QMainWindow):
    """
    Demo window that displays the camera configuration tab.

    This window allows you to interact with the camera tab and test all features:
    - Camera detection
    - Settings configuration
    - Live preview
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Configuration Tab Demo")
        self.setGeometry(100, 100, 1000, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Add title
        from PyQt5.QtWidgets import QLabel
        title = QLabel("Camera Configuration Tab Demo")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #2E86AB;
                color: white;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(title)

        # Create config manager
        logger.info("Creating configuration manager...")
        self.config_manager = create_config_manager()

        # Create camera tab
        logger.info("Creating camera configuration tab...")
        self.camera_tab = CameraConfigTab(self.config_manager)
        main_layout.addWidget(self.camera_tab)

        # Add control buttons at bottom
        self.create_control_buttons(main_layout)

        # Status bar
        self.statusBar().showMessage("Demo ready - Interact with the camera tab above")

        logger.info("✓ Demo window initialized successfully")

    def create_control_buttons(self, parent_layout):
        """Create control buttons at the bottom."""
        button_layout = QHBoxLayout()

        # Info label
        info_label = QLabel("Demo Controls:")
        info_label.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(info_label)

        # Save config button
        save_btn = QPushButton("💾 Save Configuration")
        save_btn.clicked.connect(self.save_configuration)
        save_btn.setStyleSheet("""
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
        button_layout.addWidget(save_btn)

        # Get config button
        get_config_btn = QPushButton("📋 Print Current Config")
        get_config_btn.clicked.connect(self.print_current_config)
        button_layout.addWidget(get_config_btn)

        button_layout.addStretch()

        # Close button
        close_btn = QPushButton("❌ Close Demo")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
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
        """)
        button_layout.addWidget(close_btn)

        parent_layout.addLayout(button_layout)

    def save_configuration(self):
        """Save the camera configuration."""
        logger.info("Saving camera configuration...")

        if self.camera_tab.save_config():
            self.statusBar().showMessage("✓ Configuration saved successfully!", 3000)
            logger.info("✓ Configuration saved")
        else:
            self.statusBar().showMessage("✗ Failed to save configuration", 3000)
            logger.error("✗ Failed to save configuration")

    def print_current_config(self):
        """Print the current configuration to console."""
        logger.info("=" * 60)
        logger.info("CURRENT CAMERA CONFIGURATION")
        logger.info("=" * 60)

        config = self.camera_tab.get_config()
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        logger.info("=" * 60)
        self.statusBar().showMessage("Configuration printed to console", 2000)

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Closing demo window...")

        # Stop camera preview if active
        if self.camera_tab.preview_active:
            logger.info("Stopping camera preview...")
            self.camera_tab.stop_preview()

        logger.info("✓ Demo closed successfully")
        event.accept()


def main():
    """Main entry point for the camera tab demo."""
    logger.info("=" * 60)
    logger.info("Camera Configuration Tab Demo Starting")
    logger.info("=" * 60)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create and show demo window
    window = CameraTabDemoWindow()
    window.show()

    logger.info("Demo window displayed!")
    logger.info("You can now:")
    logger.info("  1. Click 'Detect Cameras' to find available cameras")
    logger.info("  2. Adjust camera settings")
    logger.info("  3. Click 'Start Preview' to see live feed")
    logger.info("  4. Click 'Save Configuration' to save settings")
    logger.info("=" * 60)

    # Run application event loop
    exit_code = app.exec_()

    logger.info("Demo application exited")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()