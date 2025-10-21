"""
demo_api_tab.py - Interactive demo for the API configuration tab

FILE LOCATION: GUI/demos/demo_api_tab.py (or project root)

This script demonstrates the API configuration tab in a standalone window.
Run this to see and interact with the API tab UI.

Usage:
    python demo_api_tab.py
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
    from GUI.config_tabs.api_tab import APIConfigTab

    logger.info("✓ All imports successful")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)


class APITabDemoWindow(QMainWindow):
    """
    Demo window that displays the API configuration tab.

    This window allows you to interact with the API tab and test all features:
    - View API information
    - Configure connection settings
    - Enable/disable caching
    - Test API connectivity
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("API Configuration Tab Demo")
        self.setGeometry(100, 100, 800, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Add title
        title = QLabel("API Configuration Tab Demo")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #3498DB;
                color: white;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(title)

        # Create config manager
        logger.info("Creating configuration manager...")
        self.config_manager = create_config_manager()

        # Create API tab
        logger.info("Creating API configuration tab...")
        self.api_tab = APIConfigTab(self.config_manager)

        # Connect signals to log changes
        self.api_tab.api_settings_changed.connect(self.on_settings_changed)
        self.api_tab.connection_test_complete.connect(self.on_test_complete)

        main_layout.addWidget(self.api_tab)

        # Add control buttons at bottom
        self.create_control_buttons(main_layout)

        # Status bar
        self.statusBar().showMessage("Demo ready - Configure API settings and test connection")

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

        # Load config button
        load_btn = QPushButton("📂 Reload Configuration")
        load_btn.clicked.connect(self.reload_configuration)
        button_layout.addWidget(load_btn)

        # Get config button
        get_config_btn = QPushButton("📋 Print Current Config")
        get_config_btn.clicked.connect(self.print_current_config)
        button_layout.addWidget(get_config_btn)

        button_layout.addStretch()

        # Modified indicator
        self.modified_label = QLabel("Unsaved changes")
        self.modified_label.setStyleSheet("color: #E67E22; font-weight: bold;")
        self.modified_label.setVisible(False)
        button_layout.addWidget(self.modified_label)

        # Connect modification signal
        self.api_tab.modification_changed.connect(self.on_modified_changed)

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
        """Save the API configuration."""
        logger.info("Saving API configuration...")

        if self.api_tab.save_config():
            self.statusBar().showMessage("✓ Configuration saved successfully!", 3000)
            logger.info("✓ Configuration saved")
        else:
            self.statusBar().showMessage("✗ Failed to save configuration", 3000)
            logger.error("✗ Failed to save configuration")

    def reload_configuration(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration...")

        if self.api_tab.load_config():
            self.statusBar().showMessage("✓ Configuration reloaded from file", 3000)
            logger.info("✓ Configuration reloaded")
        else:
            self.statusBar().showMessage("⚠ No configuration found, using defaults", 3000)
            logger.warning("⚠ No configuration found")

    def print_current_config(self):
        """Print the current configuration to console."""
        logger.info("=" * 60)
        logger.info("CURRENT API CONFIGURATION")
        logger.info("=" * 60)

        config = self.api_tab.get_config()
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        logger.info("=" * 60)
        self.statusBar().showMessage("Configuration printed to console", 2000)

    def on_settings_changed(self):
        """Log when API settings change."""
        logger.debug("API settings changed")
        self.statusBar().showMessage("API settings modified", 1000)

    def on_test_complete(self, success):
        """Handle test completion."""
        if success:
            logger.info("✓ API connection test: SUCCESS")
            logger.info("  Test image was successfully identified by the API")
            self.statusBar().showMessage("✓ API test successful - see results above!", 5000)
        else:
            logger.warning("✗ API connection test: FAILED")
            self.statusBar().showMessage("✗ API connection test failed - check error message", 5000)

    def on_modified_changed(self, is_modified):
        """Update modified indicator."""
        self.modified_label.setVisible(is_modified)
        if is_modified:
            logger.debug("Configuration has unsaved changes")


def main():
    """Main entry point for the API tab demo."""
    logger.info("=" * 60)
    logger.info("API Configuration Tab Demo Starting")
    logger.info("=" * 60)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create and show demo window
    window = APITabDemoWindow()
    window.show()

    logger.info("Demo window displayed!")
    logger.info("You can now:")
    logger.info("  1. View the Brickognize API information")
    logger.info("  2. Adjust timeout, retry count, and rate limit")
    logger.info("  3. Enable/disable caching")
    logger.info("  4. Click 'Test API Connection' to send test_piece.jpg")
    logger.info("     (Make sure LegoPictures/test_piece.jpg exists!)")
    logger.info("  5. Click 'Clear Cache' to remove cached responses")
    logger.info("  6. Click 'Save Configuration' to save settings")
    logger.info("=" * 60)

    # Run application event loop
    exit_code = app.exec_()

    logger.info("Demo application exited")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()