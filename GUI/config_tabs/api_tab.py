"""
api_tab.py - API configuration tab

FILE LOCATION: GUI/config_tabs/api_tab.py

This tab provides API configuration for the Brickognize piece identification service.
While the API endpoint is fixed, users can configure timeout, retry, and caching settings.

Features:
- API endpoint display (fixed)
- Timeout configuration
- Retry count configuration
- Cache settings
- Test connection functionality
- API status display

Configuration Mapping:
    {
        "api": {
            "api_type": "rebrickable",
            "api_key": "",
            "cache_enabled": true,
            "cache_dir": "api_cache",
            "timeout": 30.0,
            "retry_count": 3,
            "rate_limit": 5.0
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QComboBox, QPushButton, QSpinBox,
                             QCheckBox, QLabel, QLineEdit, QTextEdit,
                             QDoubleSpinBox, QProgressBar, QApplication)
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtGui import QFont
import logging
import os
from typing import Dict, Any, Optional

# Import base class
from GUI.config_tabs.base_tab import BaseConfigTab
from enhanced_config_manager import ModuleConfig

# Import the actual API handler
from processing.identification_api_handler import IdentificationAPIHandler

# Initialize logger
logger = logging.getLogger(__name__)


class APITestThread(QThread):
    """Worker thread for testing API connection without blocking UI."""

    # Signals
    test_complete = pyqtSignal(bool, str, dict)  # (success, message, result_data)

    def __init__(self, config_manager, test_image_path):
        super().__init__()
        self.config_manager = config_manager
        self.test_image_path = test_image_path

    def run(self):
        """Test API connection by sending a test image using the real API handler."""
        try:
            # Check if test image exists
            if not os.path.exists(self.test_image_path):
                self.test_complete.emit(
                    False,
                    f"Test image not found: {os.path.basename(self.test_image_path)}",
                    {}
                )
                return

            # Use the actual IdentificationAPIHandler
            api_handler = IdentificationAPIHandler(self.config_manager)

            # Send test image to API (this uses the production code path)
            result = api_handler.identify_piece(self.test_image_path)

            # Format the results
            result_data = {
                'element_id': result.element_id,
                'name': result.name,
                'confidence': result.confidence
            }

            message = (
                f"API Connection Successful!\n"
                f"Identified: {result.name}\n"
                f"Element ID: {result.element_id}\n"
                f"Confidence: {result.confidence:.1%}"
            )

            self.test_complete.emit(True, message, result_data)

        except FileNotFoundError as e:
            self.test_complete.emit(False, f"File error: {str(e)}", {})
        except ValueError as e:
            self.test_complete.emit(False, f"Validation error: {str(e)}", {})
        except Exception as e:
            # This will catch requests.RequestException and other errors
            self.test_complete.emit(False, f"Error: {str(e)}", {})


class APIConfigTab(BaseConfigTab):
    """
    API configuration tab for Brickognize identification service.

    This tab allows users to:
    - View API endpoint information
    - Configure timeout settings
    - Configure retry behavior
    - Enable/disable caching
    - Test API connectivity

    Signals:
        api_settings_changed(): Emitted when any API setting changes
        connection_test_complete(bool): Emitted when connection test finishes
    """

    # Custom signals
    api_settings_changed = pyqtSignal()
    connection_test_complete = pyqtSignal(bool)

    def __init__(self, config_manager, parent=None):
        """
        Initialize API configuration tab.

        Args:
            config_manager: Configuration manager instance
            parent: Parent widget
        """
        # API test thread
        self.test_thread = None

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
        return ModuleConfig.API.value

    def init_ui(self):
        """Create the API configuration user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create sections
        self.create_api_info_section(main_layout)
        self.create_connection_settings_section(main_layout)
        self.create_cache_settings_section(main_layout)
        self.create_test_section(main_layout)

        # Add stretch at bottom
        main_layout.addStretch()

        self.log_info("API tab UI initialized")

    def load_config(self) -> bool:
        """Load API configuration and populate UI."""
        try:
            config = self.get_module_config(self.get_module_name())

            if config:
                # Connection settings
                self.timeout_spin.setValue(config.get("timeout", 30.0))
                self.retry_spin.setValue(config.get("retry_count", 3))
                self.rate_limit_spin.setValue(config.get("rate_limit", 5.0))

                # Cache settings
                self.cache_enabled_check.setChecked(config.get("cache_enabled", True))
                self.cache_dir_edit.setText(config.get("cache_dir", "api_cache"))

                self.log_info(f"Configuration loaded: Timeout={config.get('timeout')}s, "
                            f"Retries={config.get('retry_count')}")
                self.clear_modified()
                return True
            else:
                self.log_warning("No API configuration found, using defaults")
                return False

        except Exception as e:
            self.log_error(f"Error loading configuration: {e}")
            return False

    def save_config(self) -> bool:
        """Save API configuration."""
        if not self.validate():
            return False

        try:
            config = self.get_config()
            self.config_manager.update_module_config(
                self.get_module_name(),
                config
            )

            self.log_info("API configuration saved")
            self.clear_modified()
            return True

        except Exception as e:
            self.log_error(f"Error saving configuration: {e}")
            self.show_validation_error(f"Failed to save: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """Return current API configuration as dictionary."""
        return {
            "api_type": "rebrickable",  # Fixed
            "api_key": "",  # Not required for Brickognize
            "cache_enabled": self.cache_enabled_check.isChecked(),
            "cache_dir": self.cache_dir_edit.text(),
            "timeout": self.timeout_spin.value(),
            "retry_count": self.retry_spin.value(),
            "rate_limit": self.rate_limit_spin.value()
        }

    def validate(self) -> bool:
        """Validate API configuration."""
        # Validate timeout
        timeout = self.timeout_spin.value()
        if timeout < 5:
            self.show_validation_error("Timeout should be at least 5 seconds")
            return False

        # Validate cache directory if caching enabled
        if self.cache_enabled_check.isChecked():
            cache_dir = self.cache_dir_edit.text().strip()
            if not cache_dir:
                self.show_validation_error("Cache directory cannot be empty when caching is enabled")
                return False

        return True

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.log_info("Resetting API configuration to defaults")

        # Connection defaults
        self.timeout_spin.setValue(30.0)
        self.retry_spin.setValue(3)
        self.rate_limit_spin.setValue(5.0)

        # Cache defaults
        self.cache_enabled_check.setChecked(True)
        self.cache_dir_edit.setText("api_cache")

        self.mark_modified()

    # ========================================================================
    # UI CREATION METHODS
    # ========================================================================

    def create_api_info_section(self, parent_layout):
        """Create API information section."""
        info_group = QGroupBox("Brickognize API Information")
        info_layout = QVBoxLayout()

        # API description
        description = QLabel(
            "This system uses the Brickognize API for LEGO piece identification.\n"
            "The API analyzes images and returns piece identification with confidence scores."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #34495E; padding: 5px;")
        info_layout.addWidget(description)

        # API endpoint (read-only)
        endpoint_layout = QFormLayout()

        self.api_url_edit = QLineEdit()
        self.api_url_edit.setText("https://api.brickognize.com/predict/")
        self.api_url_edit.setReadOnly(True)
        self.api_url_edit.setMinimumWidth(400)  # Make wider to show full URL
        self.api_url_edit.setStyleSheet("background-color: #ECF0F1; color: #2C3E50;")
        endpoint_layout.addRow("API Endpoint:", self.api_url_edit)

        info_layout.addLayout(endpoint_layout)

        # Key features
        features_label = QLabel(
            "✓ No API key required\n"
            "✓ Free to use\n"
            "✓ Returns piece ID, name, and confidence score\n"
            "✓ Supports multiple image formats (JPG, PNG, GIF)"
        )
        features_label.setStyleSheet("color: #27AE60; font-weight: bold; padding: 10px;")
        info_layout.addWidget(features_label)

        info_group.setLayout(info_layout)
        parent_layout.addWidget(info_group)

    def create_connection_settings_section(self, parent_layout):
        """Create connection settings section."""
        connection_group = QGroupBox("Connection Settings")
        connection_layout = QFormLayout()

        # Timeout
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(5.0, 120.0)
        self.timeout_spin.setValue(30.0)
        self.timeout_spin.setSingleStep(5.0)
        self.timeout_spin.setSuffix(" seconds")
        self.timeout_spin.setToolTip(
            "Maximum time to wait for API response.\n"
            "Recommended: 30 seconds\n"
            "Increase if you have a slow connection."
        )
        self.timeout_spin.valueChanged.connect(self.on_setting_changed)
        connection_layout.addRow("Timeout:", self.timeout_spin)

        # Retry count
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 10)
        self.retry_spin.setValue(3)
        self.retry_spin.setToolTip(
            "Number of times to retry if API call fails.\n"
            "Recommended: 3 retries\n"
            "Set to 0 to disable retries."
        )
        self.retry_spin.valueChanged.connect(self.on_setting_changed)
        connection_layout.addRow("Retry Count:", self.retry_spin)

        # Rate limit
        self.rate_limit_spin = QDoubleSpinBox()
        self.rate_limit_spin.setRange(0.1, 20.0)
        self.rate_limit_spin.setValue(5.0)
        self.rate_limit_spin.setSingleStep(0.5)
        self.rate_limit_spin.setSuffix(" req/sec")
        self.rate_limit_spin.setToolTip(
            "Maximum API requests per second.\n"
            "Recommended: 5 requests/second\n"
            "Prevents overwhelming the API."
        )
        self.rate_limit_spin.valueChanged.connect(self.on_setting_changed)
        connection_layout.addRow("Rate Limit:", self.rate_limit_spin)

        connection_group.setLayout(connection_layout)
        parent_layout.addWidget(connection_group)

    def create_cache_settings_section(self, parent_layout):
        """Create cache settings section."""
        cache_group = QGroupBox("Caching Settings")
        cache_layout = QFormLayout()

        # Cache enabled
        self.cache_enabled_check = QCheckBox("Enable API response caching")
        self.cache_enabled_check.setChecked(True)
        self.cache_enabled_check.setToolTip(
            "Cache API responses to avoid redundant calls.\n"
            "Improves performance and reduces API load."
        )
        self.cache_enabled_check.stateChanged.connect(self.on_cache_enabled_changed)
        cache_layout.addRow("Caching:", self.cache_enabled_check)

        # Cache directory
        cache_dir_layout = QHBoxLayout()

        self.cache_dir_edit = QLineEdit()
        self.cache_dir_edit.setText("api_cache")
        self.cache_dir_edit.setToolTip("Directory where cached responses will be stored")
        self.cache_dir_edit.textChanged.connect(self.mark_modified)
        cache_dir_layout.addWidget(self.cache_dir_edit)

        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        self.clear_cache_btn.setToolTip("Delete all cached API responses")
        cache_dir_layout.addWidget(self.clear_cache_btn)

        cache_layout.addRow("Cache Directory:", cache_dir_layout)

        # Cache info
        self.cache_info_label = QLabel("")
        self.cache_info_label.setWordWrap(True)
        self.cache_info_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        cache_layout.addRow("", self.cache_info_label)
        self.update_cache_info()

        cache_group.setLayout(cache_layout)
        parent_layout.addWidget(cache_group)

    def create_test_section(self, parent_layout):
        """Create API connection test section."""
        test_group = QGroupBox("Connection Test")
        test_layout = QVBoxLayout()

        # Test image path info
        test_info_layout = QHBoxLayout()
        test_info_label = QLabel("Test Image:")
        test_info_label.setStyleSheet("font-weight: bold;")
        test_info_layout.addWidget(test_info_label)

        self.test_image_label = QLabel("LegoPictures/test_piece.jpg")
        self.test_image_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        test_info_layout.addWidget(self.test_image_label)
        test_info_layout.addStretch()

        test_layout.addLayout(test_info_layout)

        # Test button
        self.test_btn = QPushButton("🔌 Test API Connection")
        self.test_btn.clicked.connect(self.test_connection)
        self.test_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        test_layout.addWidget(self.test_btn)

        # Progress bar
        self.test_progress = QProgressBar()
        self.test_progress.setVisible(False)
        self.test_progress.setRange(0, 0)  # Indeterminate
        test_layout.addWidget(self.test_progress)

        # Status label
        self.test_status_label = QLabel("Status: Not tested")
        self.test_status_label.setAlignment(Qt.AlignCenter)
        self.test_status_label.setWordWrap(True)
        self.test_status_label.setMinimumHeight(80)
        self.test_status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                background-color: #ECF0F1;
                color: #7F8C8D;
                font-weight: bold;
            }
        """)
        test_layout.addWidget(self.test_status_label)

        # Test info
        test_help = QLabel(
            "This will send test_piece.jpg to the Brickognize API to verify:\n"
            "• API is reachable\n"
            "• Authentication works\n"
            "• Identification results are returned\n\n"
            "Make sure test_piece.jpg exists in the LegoPictures directory."
        )
        test_help.setWordWrap(True)
        test_help.setStyleSheet("color: #7F8C8D; font-style: italic; padding: 5px;")
        test_layout.addWidget(test_help)

        test_group.setLayout(test_layout)
        parent_layout.addWidget(test_group)

    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================

    def on_setting_changed(self):
        """Handle any connection setting change."""
        self.mark_modified()
        self.api_settings_changed.emit()

    def on_cache_enabled_changed(self, state):
        """Handle cache enabled checkbox change."""
        enabled = (state == Qt.Checked)
        self.cache_dir_edit.setEnabled(enabled)
        self.clear_cache_btn.setEnabled(enabled)
        self.mark_modified()

    def update_cache_info(self):
        """Update cache information label."""
        cache_dir = self.cache_dir_edit.text()

        if os.path.exists(cache_dir):
            try:
                file_count = len([f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))])
                self.cache_info_label.setText(f"Cache contains {file_count} files")
                self.cache_info_label.setStyleSheet("color: #27AE60; font-style: italic;")
            except Exception as e:
                self.cache_info_label.setText(f"Error reading cache: {e}")
                self.cache_info_label.setStyleSheet("color: #E74C3C; font-style: italic;")
        else:
            self.cache_info_label.setText("Cache directory does not exist (will be created)")
            self.cache_info_label.setStyleSheet("color: #E67E22; font-style: italic;")

    def clear_cache(self):
        """Clear all cached API responses."""
        cache_dir = self.cache_dir_edit.text()

        if not os.path.exists(cache_dir):
            self.show_validation_error("Cache directory does not exist")
            return

        try:
            import shutil
            file_count = len([f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))])

            if file_count == 0:
                self.show_validation_error("Cache is already empty")
                return

            # Confirm before deleting
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Clear Cache",
                f"Are you sure you want to delete {file_count} cached files?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                for filename in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                self.log_info(f"Cleared {file_count} files from cache")
                self.update_cache_info()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully cleared {file_count} cached files"
                )
        except Exception as e:
            self.log_error(f"Failed to clear cache: {e}")
            self.show_validation_error(f"Failed to clear cache: {e}")

    def test_connection(self):
        """Test API connection in background thread."""
        self.log_info("Testing API connection with test image...")

        # Check if test image exists
        test_image_path = os.path.join("LegoPictures", "test_piece.jpg")
        if not os.path.exists(test_image_path):
            self.show_validation_error(
                f"Test image not found: {test_image_path}\n\n"
                "Please ensure test_piece.jpg exists in the LegoPictures directory."
            )
            return

        # Disable button and show progress
        self.test_btn.setEnabled(False)
        self.test_progress.setVisible(True)
        self.test_status_label.setText("Status: Testing connection...\nSending test image to API...")
        self.test_status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                border: 2px solid #3498DB;
                border-radius: 5px;
                background-color: #EBF5FB;
                color: #3498DB;
                font-weight: bold;
            }
        """)
        QApplication.processEvents()

        # Create and start test thread with config manager
        self.test_thread = APITestThread(self.config_manager, test_image_path)
        self.test_thread.test_complete.connect(self.on_test_complete)
        self.test_thread.start()

    def on_test_complete(self, success, message, result_data):
        """Handle test completion."""
        # Hide progress
        self.test_progress.setVisible(False)
        self.test_btn.setEnabled(True)

        if success:
            # Format success message with results
            display_message = f"✓ {message}"

            self.test_status_label.setText(display_message)
            self.test_status_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    border: 2px solid #27AE60;
                    border-radius: 5px;
                    background-color: #D5F4E6;
                    color: #27AE60;
                    font-weight: bold;
                }
            """)

            # Log detailed results
            self.log_info(f"API connection test successful!")
            if result_data:
                self.log_info(f"  Element ID: {result_data.get('element_id', 'N/A')}")
                self.log_info(f"  Name: {result_data.get('name', 'N/A')}")
                self.log_info(f"  Confidence: {result_data.get('confidence', 0):.1%}")
        else:
            self.test_status_label.setText(f"✗ Test Failed\n{message}")
            self.test_status_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    border: 2px solid #E74C3C;
                    border-radius: 5px;
                    background-color: #FADBD8;
                    color: #E74C3C;
                    font-weight: bold;
                }
            """)
            self.log_error(f"API connection test failed: {message}")

        self.connection_test_complete.emit(success)