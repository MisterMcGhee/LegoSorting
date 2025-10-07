"""
system_tab.py - System configuration tab

FILE LOCATION: GUI/config_tabs/system_tab.py

This tab provides system-wide configuration including threading, queue management,
logging, and performance settings. These settings affect the overall behavior of
the Lego Sorting System.

Features:
- Threading configuration (enable/disable)
- Queue size management for processing pipeline
- Logging level selection
- Image saving preferences
- Performance tuning options

Configuration Mapping:
    {
        "system": {
            "threading_enabled": true,
            "queue_size": 100,
            "log_level": "INFO",
            "save_images": true,
            "image_save_path": "LegoPictures"
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QComboBox, QPushButton, QSpinBox,
                             QCheckBox, QLabel, QLineEdit, QFileDialog,
                             QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt
import logging
import os
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
    - Configure queue sizes for processing pipeline
    - Set logging level
    - Configure image saving
    - Adjust performance settings

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
        self.create_performance_section(main_layout)
        self.create_storage_section(main_layout)

        # Add stretch at bottom
        main_layout.addStretch()

        self.log_info("System tab UI initialized")

    def load_config(self) -> bool:
        """Load system configuration and populate UI."""
        try:
            config = self.get_module_config(self.get_module_name())

            if config:
                # Threading settings
                self.threading_check.setChecked(config.get("threading_enabled", True))

                # Queue settings
                self.queue_size_spin.setValue(config.get("queue_size", 100))

                # Logging settings
                log_level = config.get("log_level", "INFO")
                index = self.log_level_combo.findText(log_level)
                if index >= 0:
                    self.log_level_combo.setCurrentIndex(index)

                # Storage settings
                self.save_images_check.setChecked(config.get("save_images", True))
                self.image_path_edit.setText(config.get("image_save_path", "LegoPictures"))

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
            "queue_size": self.queue_size_spin.value(),
            "log_level": self.log_level_combo.currentText(),
            "save_images": self.save_images_check.isChecked(),
            "image_save_path": self.image_path_edit.text()
        }

    def validate(self) -> bool:
        """Validate system configuration."""
        # Validate queue size
        queue_size = self.queue_size_spin.value()
        if queue_size < 10:
            self.show_validation_error("Queue size must be at least 10")
            return False

        # Validate image path if saving is enabled
        if self.save_images_check.isChecked():
            image_path = self.image_path_edit.text().strip()
            if not image_path:
                self.show_validation_error("Image save path cannot be empty when image saving is enabled")
                return False

        return True

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.log_info("Resetting system configuration to defaults")

        # Threading defaults
        self.threading_check.setChecked(True)

        # Queue defaults
        self.queue_size_spin.setValue(100)

        # Logging defaults
        self.log_level_combo.setCurrentText("INFO")

        # Storage defaults
        self.save_images_check.setChecked(True)
        self.image_path_edit.setText("LegoPictures")

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

    def create_performance_section(self, parent_layout):
        """Create performance tuning section."""
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QFormLayout()

        # Queue size
        self.queue_size_spin = QSpinBox()
        self.queue_size_spin.setRange(10, 1000)
        self.queue_size_spin.setValue(100)
        self.queue_size_spin.setSuffix(" items")
        self.queue_size_spin.setToolTip(
            "Maximum number of pieces queued for processing.\n"
            "Larger values use more memory but handle bursts better.\n"
            "Recommended: 100-200 for normal operation"
        )
        self.queue_size_spin.valueChanged.connect(self.mark_modified)
        performance_layout.addRow("Queue Size:", self.queue_size_spin)

        # Queue size warning
        self.queue_warning_label = QLabel("")
        self.queue_warning_label.setWordWrap(True)
        self.queue_warning_label.setStyleSheet("color: #E67E22; font-weight: bold;")
        performance_layout.addRow("", self.queue_warning_label)
        self.queue_size_spin.valueChanged.connect(self.update_queue_warning)

        performance_group.setLayout(performance_layout)
        parent_layout.addWidget(performance_group)

    def create_storage_section(self, parent_layout):
        """Create image storage configuration section."""
        storage_group = QGroupBox("Image Storage")
        storage_layout = QFormLayout()

        # Save images checkbox
        self.save_images_check = QCheckBox("Save detected piece images")
        self.save_images_check.setChecked(True)
        self.save_images_check.setToolTip(
            "Save images of detected pieces for review and debugging.\n"
            "Images are saved with sequential numbering."
        )
        self.save_images_check.stateChanged.connect(self.on_save_images_changed)
        storage_layout.addRow("Save Images:", self.save_images_check)

        # Image save path
        path_layout = QHBoxLayout()

        self.image_path_edit = QLineEdit()
        self.image_path_edit.setText("LegoPictures")
        self.image_path_edit.setToolTip("Directory where piece images will be saved")
        self.image_path_edit.textChanged.connect(self.mark_modified)
        path_layout.addWidget(self.image_path_edit)

        self.browse_path_btn = QPushButton("Browse...")
        self.browse_path_btn.clicked.connect(self.browse_image_path)
        self.browse_path_btn.setToolTip("Select image save directory")
        path_layout.addWidget(self.browse_path_btn)

        self.create_path_btn = QPushButton("Create")
        self.create_path_btn.clicked.connect(self.create_image_directory)
        self.create_path_btn.setToolTip("Create the directory if it doesn't exist")
        path_layout.addWidget(self.create_path_btn)

        storage_layout.addRow("Save Path:", path_layout)

        # Path status
        self.path_status_label = QLabel("")
        self.path_status_label.setWordWrap(True)
        storage_layout.addRow("", self.path_status_label)
        self.image_path_edit.textChanged.connect(self.update_path_status)

        # Update initial status
        self.update_path_status()

        storage_group.setLayout(storage_layout)
        parent_layout.addWidget(storage_group)

        # Reset to defaults button
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

        parent_layout.addLayout(reset_layout)

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

        self.mark_modified()
        self.log_level_changed.emit(level)

    def on_save_images_changed(self, state):
        """Handle save images checkbox state change."""
        enabled = (state == Qt.Checked)
        self.log_info(f"Image saving {'enabled' if enabled else 'disabled'}")

        # Enable/disable path controls
        self.image_path_edit.setEnabled(enabled)
        self.browse_path_btn.setEnabled(enabled)
        self.create_path_btn.setEnabled(enabled)

        self.mark_modified()

    def update_queue_warning(self, value):
        """Update queue size warning label."""
        if value < 50:
            self.queue_warning_label.setText(
                "⚠ Small queue size may cause dropped pieces during bursts"
            )
            self.queue_warning_label.setStyleSheet("color: #E67E22; font-weight: bold;")
        elif value > 500:
            self.queue_warning_label.setText(
                "⚠ Large queue size will use more memory"
            )
            self.queue_warning_label.setStyleSheet("color: #E67E22; font-weight: bold;")
        else:
            self.queue_warning_label.setText("")

    def update_path_status(self):
        """Update image path status label."""
        path = self.image_path_edit.text().strip()

        if not path:
            self.path_status_label.setText("Path is empty")
            self.path_status_label.setStyleSheet("color: #E74C3C;")
            return

        if os.path.exists(path):
            if os.path.isdir(path):
                self.path_status_label.setText("✓ Directory exists")
                self.path_status_label.setStyleSheet("color: #27AE60;")
            else:
                self.path_status_label.setText("✗ Path exists but is not a directory")
                self.path_status_label.setStyleSheet("color: #E74C3C;")
        else:
            self.path_status_label.setText("Directory does not exist (will be created)")
            self.path_status_label.setStyleSheet("color: #E67E22;")

    def browse_image_path(self):
        """Open directory browser dialog."""
        current_path = self.image_path_edit.text()

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Image Save Directory",
            current_path if current_path else os.getcwd()
        )

        if directory:
            self.image_path_edit.setText(directory)
            self.log_info(f"Image save path set to: {directory}")

    def create_image_directory(self):
        """Create the image save directory if it doesn't exist."""
        path = self.image_path_edit.text().strip()

        if not path:
            self.show_validation_error("Please enter a directory path")
            return

        try:
            os.makedirs(path, exist_ok=True)
            self.log_info(f"Created directory: {path}")
            self.update_path_status()

            QMessageBox.information(
                self,
                "Success",
                f"Directory created successfully:\n{path}"
            )
        except Exception as e:
            self.log_error(f"Failed to create directory: {e}")
            self.show_validation_error(f"Failed to create directory: {e}")

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