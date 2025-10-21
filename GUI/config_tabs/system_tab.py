"""
system_tab.py - System configuration tab

FILE LOCATION: GUI/config_tabs/system_tab.py

This tab provides system-wide configuration including threading and logging.
These settings affect the overall behavior of the Lego Sorting System.

Features:
- Threading configuration (enable/disable)
- Logging level selection

Configuration Mapping:
    {
        "system": {
            "threading_enabled": true,
            "log_level": "INFO"
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QComboBox, QPushButton, QSpinBox,
                             QCheckBox, QLabel, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt
import logging
from typing import Dict, Any, Optional

# Import base class
from GUI.config_tabs.base_tab import BaseConfigTab
from enhanced_config_manager import ModuleConfig

# Initialize logger
logger = logging.getLogger(__name__)


class SystemConfigTab(BaseConfigTab):
    """
    System configuration tab for application-wide settings.

    This tab allows users to:
    - Enable/disable multithreading
    - Set logging level

    Signals:
        threading_changed(bool): Emitted when threading setting changes
        log_level_changed(str): Emitted when log level changes
    """

    # Custom signals
    threading_changed = pyqtSignal(bool)
    log_level_changed = pyqtSignal(str)

    def __init__(self, config_manager, parent=None):
        """
        Initialize system configuration tab.

        Args:
            config_manager: Configuration manager instance
            parent: Parent widget
        """
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
        return ModuleConfig.SYSTEM.value

    def init_ui(self):
        """Create the system configuration user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create sections
        self.create_threading_section(main_layout)
        self.create_logging_section(main_layout)

        # Add stretch at bottom
        main_layout.addStretch()

        # Add reset button at bottom
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.confirm_reset)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
        """)
        reset_layout.addWidget(self.reset_btn)

        main_layout.addLayout(reset_layout)

        self.log_info("System tab UI initialized")

    def load_config(self) -> bool:
        """Load system configuration and populate UI."""
        try:
            config = self.get_module_config(self.get_module_name())

            if config:
                # Threading settings
                self.threading_check.setChecked(config.get("threading_enabled", True))

                # Logging settings
                log_level = config.get("log_level", "INFO")
                index = self.log_level_combo.findText(log_level)
                if index >= 0:
                    self.log_level_combo.setCurrentIndex(index)

                self.log_info(f"Configuration loaded: Threading={config.get('threading_enabled')}, "
                              f"Log Level={log_level}")
                self.clear_modified()
                return True
            else:
                self.log_warning("No system configuration found, using defaults")
                return False

        except Exception as e:
            self.log_error(f"Error loading configuration: {e}")
            return False

    def save_config(self) -> bool:
        """Save system configuration."""
        if not self.validate():
            return False

        try:
            config = self.get_config()
            self.config_manager.update_module_config(
                self.get_module_name(),
                config
            )

            self.log_info("System configuration saved")
            self.clear_modified()
            return True

        except Exception as e:
            self.log_error(f"Error saving configuration: {e}")
            self.show_validation_error(f"Failed to save: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """Return current system configuration as dictionary."""
        return {
            "threading_enabled": self.threading_check.isChecked(),
            "log_level": self.log_level_combo.currentText()  # Fixed: return log_level instead of queue_size
        }

    def validate(self) -> bool:
        """Validate system configuration."""
        # No validation needed for simple boolean and string values
        # Log level is constrained by combo box
        return True

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.log_info("Resetting system configuration to defaults")

        # Threading defaults
        self.threading_check.setChecked(True)

        # Logging defaults
        self.log_level_combo.setCurrentText("INFO")

        self.mark_modified()

    # ========================================================================
    # UI CREATION METHODS
    # ========================================================================

    def create_threading_section(self, parent_layout):
        """Create threading configuration section."""
        threading_group = QGroupBox("Threading Configuration")
        threading_layout = QFormLayout()

        # Threading enabled checkbox
        self.threading_check = QCheckBox("Enable Multithreading")
        self.threading_check.setChecked(True)
        self.threading_check.setToolTip(
            "Enable parallel processing for better performance.\n"
            "Disable for debugging or troubleshooting."
        )
        self.threading_check.stateChanged.connect(self.on_threading_changed)
        threading_layout.addRow("Multithreading:", self.threading_check)

        # Threading info label
        self.threading_info_label = QLabel(
            "Multithreading allows parallel processing of camera frames, "
            "API calls, and sorting operations for improved performance."
        )
        self.threading_info_label.setWordWrap(True)
        self.threading_info_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        threading_layout.addRow("", self.threading_info_label)

        threading_group.setLayout(threading_layout)
        parent_layout.addWidget(threading_group)

    def create_logging_section(self, parent_layout):
        """Create logging configuration section."""
        logging_group = QGroupBox("Logging Configuration")
        logging_layout = QFormLayout()

        # Log level selection
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.setToolTip(
            "DEBUG: Detailed information for debugging\n"
            "INFO: General information about system operation\n"
            "WARNING: Warning messages for potential issues\n"
            "ERROR: Error messages for failures\n"
            "CRITICAL: Critical errors that may cause system failure"
        )
        self.log_level_combo.currentTextChanged.connect(self.on_log_level_changed)
        logging_layout.addRow("Log Level:", self.log_level_combo)

        # Log level description
        self.log_level_desc_label = QLabel(self._get_log_level_description("INFO"))
        self.log_level_desc_label.setWordWrap(True)
        self.log_level_desc_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        logging_layout.addRow("", self.log_level_desc_label)

        logging_group.setLayout(logging_layout)
        parent_layout.addWidget(logging_group)

    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================

    def on_threading_changed(self, state):
        """Handle threading checkbox state change."""
        enabled = (state == Qt.Checked)
        self.log_info(f"Threading {'enabled' if enabled else 'disabled'}")

        # Update info label
        if enabled:
            self.threading_info_label.setStyleSheet("color: #27AE60; font-style: italic;")
        else:
            self.threading_info_label.setStyleSheet("color: #E67E22; font-style: italic;")

        self.mark_modified()
        self.threading_changed.emit(enabled)

    def on_log_level_changed(self, level):
        """Handle log level change."""
        self.log_info(f"Log level changed to: {level}")

        # Update description
        self.log_level_desc_label.setText(self._get_log_level_description(level))

        # Optionally apply immediately to root logger
        try:
            logging.getLogger().setLevel(getattr(logging, level))
            self.log_info(f"Applied log level {level} to root logger")
        except Exception as e:
            self.log_warning(f"Could not apply log level immediately: {e}")

        self.mark_modified()
        self.log_level_changed.emit(level)

    def confirm_reset(self):
        """Confirm before resetting to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all system settings to their default values?\n\n"
            "This will overwrite your current configuration.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.reset_to_defaults()
            self.log_info("System configuration reset to defaults")

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_log_level_description(self, level: str) -> str:
        """Get description for log level."""
        descriptions = {
            "DEBUG": "Shows all messages including detailed debugging information. Use for troubleshooting.",
            "INFO": "Shows informational messages about system operation. Recommended for normal use.",
            "WARNING": "Shows only warnings and errors. Use when you want minimal logging.",
            "ERROR": "Shows only error messages. Use in production for critical issues only.",
            "CRITICAL": "Shows only critical failures. Minimal logging."
        }
        return descriptions.get(level, "")