"""
demo_processing_tab.py - Demo script for ProcessingConfigTab

FILE LOCATION: tests/demo_processing_tab.py (or project root for quick testing)

This script demonstrates and tests the ProcessingConfigTab to verify:
1. Sorting strategy selection (Primary, Secondary, Tertiary)
2. Dynamic dropdown updates based on strategy and selections
3. Pre-assignment table management
4. Bin configuration
5. Configuration save/load
6. Validation logic
7. Integration with category_hierarchy_service

Usage:
    python demo_processing_tab.py

Features to Test:
    - Switch strategies and watch UI adapt
    - Select categories and see dropdowns update
    - Add/remove pre-assignments
    - Save and reload configuration
    - Test validation with invalid configurations
    - Verify category service integration

Controls:
    - Use radio buttons to switch strategies
    - Use dropdowns to select categories
    - Click "Add Assignment" to create pre-assignments
    - Click test buttons to verify functionality
    - Close: Press 'Q' or close window
"""

import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QMessageBox,
                             QGroupBox, QTextEdit, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from enhanced_config_manager import create_config_manager
    from processing.category_hierarchy_service import create_category_hierarchy_service
    from GUI.config_tabs.processing_tab import ProcessingConfigTab

    logger.info("✓ All imports successful")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)


class ProcessingTabDemoWindow(QMainWindow):
    """
    Demo window for ProcessingConfigTab testing.

    Features:
    - Full processing tab with all controls
    - Category hierarchy service integration
    - Configuration save/load testing
    - Validation testing
    - Status monitoring
    - Test controls
    """

    def __init__(self):
        super().__init__()

        # Initialize components
        self.config_manager = None
        self.category_service = None
        self.processing_tab = None

        # Setup UI
        self.setWindowTitle("Processing Tab Demo - Sorting Strategy Configuration")
        self.setGeometry(100, 100, 1100, 650)

        self.init_ui()
        self.init_services_and_tab()

        logger.info("Demo window initialized")

    def init_ui(self):
        """Initialize the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("Processing Configuration Tab - Demo & Testing")
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

        # Create splitter for tab and log
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, stretch=1)

        # Top: Status + Tab
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # Status panel
        status_panel = self.create_status_panel()
        top_layout.addWidget(status_panel)

        # Processing tab placeholder
        self.tab_placeholder = QWidget()
        self.tab_layout = QVBoxLayout(self.tab_placeholder)
        top_layout.addWidget(self.tab_placeholder, stretch=1)

        splitter.addWidget(top_widget)

        # Bottom: Event log
        log_panel = self.create_log_panel()
        splitter.addWidget(log_panel)

        # Set initial splitter sizes (80% tab, 20% log)
        splitter.setSizes([520, 130])

        # Control panel at bottom
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Status bar
        self.statusBar().showMessage("Ready - Initializing...")

    def create_status_panel(self) -> QWidget:
        """Create status monitoring panel."""
        panel = QGroupBox("System Status")
        layout = QHBoxLayout(panel)

        # Service status
        self.service_status = QLabel("Category Service: Not initialized")
        self.service_status.setStyleSheet("color: orange; font-weight: bold;")
        layout.addWidget(self.service_status)

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

        # Current strategy
        self.strategy_label = QLabel("Strategy: Unknown")
        self.strategy_label.setStyleSheet("color: gray;")
        layout.addWidget(self.strategy_label)

        return panel

    def create_log_panel(self) -> QGroupBox:
        """Create event log panel."""
        panel = QGroupBox("Event Log")
        layout = QVBoxLayout(panel)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        font = QFont("Courier", 9)
        self.log_text.setFont(font)
        layout.addWidget(self.log_text)

        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

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
        save_btn.setStyleSheet(self.get_button_style("#27AE60", "#229954"))
        row1.addWidget(save_btn)

        load_btn = QPushButton("📂 Reload Configuration")
        load_btn.setToolTip("Reload settings from config manager")
        load_btn.clicked.connect(self.test_load_config)
        load_btn.setStyleSheet(self.get_button_style("#3498DB", "#2980B9"))
        row1.addWidget(load_btn)

        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setToolTip("Reset all values to defaults")
        reset_btn.clicked.connect(self.test_reset_defaults)
        reset_btn.setStyleSheet(self.get_button_style("#F39C12", "#E67E22"))
        row1.addWidget(reset_btn)

        layout.addLayout(row1)

        # Button row 2: Testing
        row2 = QHBoxLayout()

        validate_btn = QPushButton("✓ Validate Configuration")
        validate_btn.setToolTip("Test validation logic")
        validate_btn.clicked.connect(self.test_validation)
        validate_btn.setStyleSheet(self.get_button_style("#16A085", "#138D75"))
        row2.addWidget(validate_btn)

        get_config_btn = QPushButton("📋 Get Current Config")
        get_config_btn.setToolTip("Display current configuration as dictionary")
        get_config_btn.clicked.connect(self.test_get_config)
        get_config_btn.setStyleSheet(self.get_button_style("#8E44AD", "#7D3C98"))
        row2.addWidget(get_config_btn)

        test_categories_btn = QPushButton("🏷️ Test Category Service")
        test_categories_btn.setToolTip("Test category hierarchy queries")
        test_categories_btn.clicked.connect(self.test_category_service)
        test_categories_btn.setStyleSheet(self.get_button_style("#E74C3C", "#C0392B"))
        row2.addWidget(test_categories_btn)

        layout.addLayout(row2)

        # Info label
        info = QLabel("💡 Tip: Switch strategies and watch the dropdowns update dynamically")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; padding: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        return panel

    def get_button_style(self, bg_color: str, hover_color: str) -> str:
        """Get button stylesheet with colors."""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """

    def init_services_and_tab(self):
        """Initialize config manager, category service, and processing tab."""
        try:
            logger.info("Initializing config manager...")
            self.config_manager = create_config_manager()
            logger.info("✓ Config manager created")

            logger.info("Initializing category hierarchy service...")
            self.category_service = create_category_hierarchy_service(self.config_manager)
            logger.info("✓ Category service created")

            # Test category service
            primaries = self.category_service.get_primary_categories()
            logger.info(f"✓ Loaded {len(primaries)} primary categories")

            self.service_status.setText(f"Category Service: Ready ({len(primaries)} primaries)")
            self.service_status.setStyleSheet("color: #27AE60; font-weight: bold;")

            self.log_event(f"Category service initialized with {len(primaries)} primary categories")

            # Create processing tab
            logger.info("Creating processing tab...")
            self.processing_tab = ProcessingConfigTab(
                self.config_manager,
                category_service=self.category_service,
                parent=self
            )

            # Add tab to layout
            self.tab_layout.addWidget(self.processing_tab)

            logger.info("✓ Processing tab created")
            self.tab_status.setText("Tab: Active")
            self.tab_status.setStyleSheet("color: #27AE60; font-weight: bold;")

            # Connect tab signals
            self.processing_tab.modification_changed.connect(self.on_modification_changed)
            self.processing_tab.validation_failed.connect(self.on_validation_failed)
            self.processing_tab.strategy_changed.connect(self.on_strategy_changed)
            self.processing_tab.bins_changed.connect(self.on_bins_changed)
            self.processing_tab.target_categories_changed.connect(self.on_target_categories_changed)

            # Update initial strategy display
            self.update_strategy_display()

            self.statusBar().showMessage("✓ All systems ready - Test away!", 3000)
            self.log_event("Processing tab initialized successfully")

        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {str(e)}")
            self.log_event(f"ERROR: {str(e)}", error=True)
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize:\n{str(e)}\n\nCheck the logs for details."
            )

    # ========================================================================
    # EVENT LOGGING
    # ========================================================================

    def log_event(self, message: str, error: bool = False):
        """Add event to log panel."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        if error:
            html = f'<span style="color: red;">[{timestamp}] ERROR: {message}</span>'
        else:
            html = f'<span style="color: #555;">[{timestamp}] {message}</span>'

        self.log_text.append(html)

        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    # ========================================================================
    # SIGNAL HANDLERS
    # ========================================================================

    def on_modification_changed(self, is_modified: bool):
        """Handle modification status changes."""
        if is_modified:
            self.modified_indicator.setText("Modified: YES")
            self.modified_indicator.setStyleSheet("color: #E67E22; font-weight: bold;")
            self.log_event("Configuration modified")
        else:
            self.modified_indicator.setText("Modified: No")
            self.modified_indicator.setStyleSheet("color: #27AE60;")

    def on_validation_failed(self, error_message: str):
        """Handle validation failures."""
        logger.warning(f"Validation failed: {error_message}")
        self.log_event(f"Validation failed: {error_message}", error=True)
        QMessageBox.warning(self, "Validation Failed", error_message)

    def on_strategy_changed(self, strategy: str):
        """Handle strategy changes."""
        logger.info(f"Strategy changed to: {strategy}")
        self.log_event(f"Strategy changed to: {strategy}")
        self.log_event("⚠️  Pre-assignments cleared (strategy changed)", error=False)
        self.update_strategy_display()

    def on_bins_changed(self, max_bins: int):
        """Handle max bins changes."""
        logger.info(f"Max bins changed to: {max_bins}")
        self.log_event(f"Max bins changed to: {max_bins}")

    def on_target_categories_changed(self, primary: str, secondary: str):
        """Handle target category changes."""
        logger.info(f"Target categories: primary='{primary}', secondary='{secondary}'")
        if primary and secondary:
            self.log_event(f"Target categories: {primary} → {secondary}")
        elif primary:
            self.log_event(f"Target primary: {primary}")

        # Note: Pre-assignments are cleared when targets change
        if self.processing_tab and self.processing_tab.get_current_strategy() in ["secondary", "tertiary"]:
            self.log_event("⚠️  Pre-assignments cleared (target changed)", error=False)

    def update_strategy_display(self):
        """Update strategy display label."""
        if not self.processing_tab:
            return

        strategy = self.processing_tab.get_current_strategy()
        self.strategy_label.setText(f"Strategy: {strategy.capitalize()}")
        self.strategy_label.setStyleSheet("color: #2E86AB; font-weight: bold;")

    # ========================================================================
    # TEST METHODS
    # ========================================================================

    def test_save_config(self):
        """Test saving configuration."""
        logger.info("=" * 60)
        logger.info("TEST: Save Configuration")
        logger.info("=" * 60)

        if not self.processing_tab:
            QMessageBox.warning(self, "Error", "Processing tab not initialized")
            return

        self.log_event("=== Testing Save Configuration ===")
        success = self.processing_tab.save_config()

        if success:
            logger.info("✓ Configuration saved successfully")
            self.log_event("✓ Configuration saved successfully")
            QMessageBox.information(
                self,
                "Save Successful",
                "Configuration has been saved to config manager."
            )
        else:
            logger.error("✗ Configuration save failed")
            self.log_event("✗ Configuration save failed", error=True)
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

        if not self.processing_tab:
            QMessageBox.warning(self, "Error", "Processing tab not initialized")
            return

        self.log_event("=== Testing Load Configuration ===")
        success = self.processing_tab.load_config()

        if success:
            logger.info("✓ Configuration loaded successfully")
            self.log_event("✓ Configuration loaded successfully")
            self.update_strategy_display()
            QMessageBox.information(
                self,
                "Load Successful",
                "Configuration has been reloaded from config manager."
            )
        else:
            logger.error("✗ Configuration load failed")
            self.log_event("✗ Configuration load failed", error=True)
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

        if not self.processing_tab:
            QMessageBox.warning(self, "Error", "Processing tab not initialized")
            return

        self.log_event("=== Testing Reset to Defaults ===")
        self.processing_tab.reset_to_defaults()
        logger.info("Reset to defaults triggered")
        self.log_event("Reset to defaults triggered")

    def test_validation(self):
        """Test validation logic."""
        logger.info("=" * 60)
        logger.info("TEST: Validation")
        logger.info("=" * 60)

        if not self.processing_tab:
            QMessageBox.warning(self, "Error", "Processing tab not initialized")
            return

        self.log_event("=== Testing Validation ===")
        is_valid = self.processing_tab.validate()

        if is_valid:
            logger.info("✓ Current configuration is valid")
            self.log_event("✓ Current configuration is valid")
            QMessageBox.information(
                self,
                "Validation Passed",
                "✓ Current configuration is valid!"
            )
        else:
            logger.warning("✗ Current configuration is invalid")
            self.log_event("✗ Current configuration is invalid", error=True)

    def test_get_config(self):
        """Test getting current configuration."""
        logger.info("=" * 60)
        logger.info("TEST: Get Current Configuration")
        logger.info("=" * 60)

        if not self.processing_tab:
            QMessageBox.warning(self, "Error", "Processing tab not initialized")
            return

        self.log_event("=== Testing Get Configuration ===")
        config = self.processing_tab.get_config()

        # Log configuration
        logger.info("Current configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
            self.log_event(f"  {key}: {value}")

        # Display in dialog
        import json
        config_text = json.dumps(config, indent=2)

        msg = QMessageBox(self)
        msg.setWindowTitle("Current Configuration")
        msg.setText("Current configuration values:")
        msg.setDetailedText(config_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def test_category_service(self):
        """Test category hierarchy service queries."""
        logger.info("=" * 60)
        logger.info("TEST: Category Hierarchy Service")
        logger.info("=" * 60)

        if not self.category_service:
            QMessageBox.warning(self, "Error", "Category service not initialized")
            return

        self.log_event("=== Testing Category Service ===")

        try:
            # Test 1: Get primary categories
            primaries = self.category_service.get_primary_categories()
            logger.info(f"Primary categories ({len(primaries)}): {primaries}")
            self.log_event(f"Primary categories: {len(primaries)} found")

            # Test 2: Get secondary categories
            if primaries:
                test_primary = primaries[0]
                secondaries = self.category_service.get_secondary_categories(test_primary)
                logger.info(f"Secondary under '{test_primary}' ({len(secondaries)}): {secondaries}")
                self.log_event(f"Secondary under '{test_primary}': {len(secondaries)} found")

                # Test 3: Get tertiary categories
                if secondaries:
                    test_secondary = secondaries[0]
                    tertiaries = self.category_service.get_tertiary_categories(
                        test_primary, test_secondary
                    )
                    logger.info(
                        f"Tertiary under '{test_primary}' → '{test_secondary}' "
                        f"({len(tertiaries)}): {tertiaries}"
                    )
                    self.log_event(
                        f"Tertiary under '{test_primary}' → '{test_secondary}': "
                        f"{len(tertiaries)} found"
                    )

            QMessageBox.information(
                self,
                "Category Service Test",
                f"Category service test completed!\n\n"
                f"Primary categories: {len(primaries)}\n"
                f"Check the log for full details."
            )

        except Exception as e:
            logger.error(f"Category service test failed: {e}", exc_info=True)
            self.log_event(f"Category service test failed: {e}", error=True)
            QMessageBox.critical(
                self,
                "Test Failed",
                f"Category service test failed:\n{str(e)}"
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
        self.log_event("Closing demo window...")
        logger.info("✓ Cleanup complete")
        event.accept()


def main():
    """Main demo function."""
    logger.info("=" * 70)
    logger.info("PROCESSING TAB DEMO - Sorting Strategy Configuration")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This demo tests the ProcessingConfigTab with category hierarchy.")
    logger.info("")
    logger.info("Features to test:")
    logger.info("  1. Sorting strategy selection (Primary, Secondary, Tertiary)")
    logger.info("  2. Dynamic category dropdowns")
    logger.info("  3. Pre-assignment table management")
    logger.info("  4. Bin configuration")
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
    window = ProcessingTabDemoWindow()
    window.show()

    logger.info("Demo window displayed - Start testing!")

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()