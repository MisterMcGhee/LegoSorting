"""
base_tab.py - Abstract base class for all configuration tabs

FILE LOCATION: GUI/config_tabs/base_tab.py

This module provides the BaseConfigTab abstract base class that all configuration
tabs must inherit from. It establishes a consistent interface for configuration
management, validation, and inter-tab communication.

USAGE BY DEVELOPERS:
1. Inherit from BaseConfigTab for each new configuration tab
2. Implement all abstract methods (init_ui, get_module_name, load_config, save_config, get_config)
3. Use self.config_manager to access/modify configuration
4. Call mark_modified() when UI values change
5. Emit config_changed signal when configuration updates
6. Override validate() for custom validation logic
7. Use get_module_config() to access specific module configurations

EXAMPLE IMPLEMENTATION:
    class CameraConfigTab(BaseConfigTab):
        def __init__(self, config_manager, parent=None):
            super().__init__(config_manager, parent)
            self.init_ui()
            self.load_config()

        def get_module_name(self) -> str:
            return "camera"

        def init_ui(self):
            # Create widgets
            self.device_combo = QComboBox()
            self.device_combo.currentIndexChanged.connect(self.mark_modified)
            # ... more UI setup

        def load_config(self) -> bool:
            config = self.get_module_config(self.get_module_name())
            if config:
                self.device_combo.setCurrentIndex(config.get("device_id", 0))
            return True

        def save_config(self) -> bool:
            if not self.validate():
                return False
            self.config_manager.update_module_config(
                self.get_module_name(),
                self.get_config()
            )
            self.clear_modified()
            return True

        def get_config(self) -> Dict[str, Any]:
            return {
                "device_id": self.device_combo.currentIndex(),
                # ... more config values
            }
"""

from abc import ABC, abstractmethod, ABCMeta
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal
from typing import Dict, Any, Optional
import logging


# ============================================================================
# METACLASS RESOLUTION
# ============================================================================

class CombinedMeta(type(QWidget), ABCMeta):
    """
    Combined metaclass to resolve conflict between QWidget and ABC.

    QWidget uses Qt's metaclass (accessed via type(QWidget)),
    and ABC uses ABCMeta. This combined metaclass allows us to inherit from both.

    This approach is compatible across different PyQt5 versions.
    """
    pass


class BaseConfigTab(QWidget, metaclass=CombinedMeta):
    """
    Abstract base class for all configuration tabs.

    This class provides the foundation for configuration tab implementations,
    ensuring consistent behavior across all tabs in the configuration GUI.
    It handles configuration loading/saving, modification tracking, validation,
    and inter-tab communication via signals.

    Signals:
        config_changed(str, dict): Emitted when configuration changes
            - str: module name
            - dict: configuration dictionary

        validation_failed(str): Emitted when validation fails
            - str: error message describing the validation failure

        modification_changed(bool): Emitted when modification status changes
            - bool: True if modified, False if not modified

    Attributes:
        config_manager: Reference to the configuration manager
        logger: Logger instance for this tab (named by class name)
        _modified: Boolean flag tracking if tab has unsaved changes
    """

    # Common signals for all configuration tabs
    config_changed = pyqtSignal(str, dict)  # (module_name, config_dict)
    validation_failed = pyqtSignal(str)  # (error_message)
    modification_changed = pyqtSignal(bool)  # (is_modified)

    def __init__(self, config_manager, parent=None):
        """
        Initialize the base configuration tab.

        Args:
            config_manager: Configuration manager instance for loading/saving config
            parent: Parent widget (typically None or the main window)
        """
        super().__init__(parent)

        # Store configuration manager reference
        self.config_manager = config_manager

        # Setup logger with class-specific name
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize modification tracking
        self._modified = False

        # Log tab initialization
        self.logger.debug(f"{self.__class__.__name__} initialized")

    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def init_ui(self):
        """
        Create the tab's user interface.

        This method must be implemented by subclasses to create all widgets,
        layouts, and connections specific to the tab's functionality.

        Implementation should:
        - Create all necessary widgets
        - Set up layouts
        - Connect signals to appropriate slots
        - Call mark_modified() when values change

        Example:
            def init_ui(self):
                layout = QVBoxLayout()
                self.spinbox = QSpinBox()
                self.spinbox.valueChanged.connect(self.mark_modified)
                layout.addWidget(self.spinbox)
                self.setLayout(layout)
        """
        pass

    @abstractmethod
    def get_module_name(self) -> str:
        """
        Return the configuration module name this tab manages.

        This name is used to identify which section of the configuration
        this tab is responsible for managing.

        Returns:
            str: Module name (e.g., "camera", "detector", "sorting")

        Example:
            def get_module_name(self) -> str:
                return "camera"
        """
        pass

    @abstractmethod
    def load_config(self) -> bool:
        """
        Load configuration from config_manager and populate UI.

        This method should:
        1. Retrieve configuration using get_module_config()
        2. Populate all UI widgets with values from config
        3. Handle missing or None values gracefully
        4. Clear the modified flag after loading

        Returns:
            bool: True if configuration loaded successfully, False otherwise

        Example:
            def load_config(self) -> bool:
                try:
                    config = self.get_module_config(self.get_module_name())
                    if config:
                        self.spinbox.setValue(config.get("value", 0))
                    self.clear_modified()
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to load config: {e}")
                    return False
        """
        pass

    @abstractmethod
    def save_config(self) -> bool:
        """
        Save current UI values to config_manager.

        This method should:
        1. Validate configuration using validate()
        2. Get current config using get_config()
        3. Update config_manager with new values
        4. Clear the modified flag on success

        Returns:
            bool: True if configuration saved successfully, False otherwise

        Example:
            def save_config(self) -> bool:
                if not self.validate():
                    return False
                try:
                    self.config_manager.update_module_config(
                        self.get_module_name(),
                        self.get_config()
                    )
                    self.clear_modified()
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to save config: {e}")
                    return False
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return current configuration as a dictionary.

        This method should read all UI widget values and return them
        as a dictionary that matches the expected configuration structure.

        Returns:
            Dict[str, Any]: Configuration dictionary with current UI values

        Example:
            def get_config(self) -> Dict[str, Any]:
                return {
                    "device_id": self.device_combo.currentIndex(),
                    "width": self.width_spin.value(),
                    "height": self.height_spin.value()
                }
        """
        pass

    # ========================================================================
    # VIRTUAL METHODS - Can be overridden by subclasses
    # ========================================================================

    def validate(self) -> bool:
        """
        Validate current configuration.

        Default implementation returns True (no validation).
        Override this method to implement custom validation logic.

        When validation fails:
        1. Call show_validation_error() with a descriptive message
        2. Return False

        Returns:
            bool: True if configuration is valid, False otherwise

        Example:
            def validate(self) -> bool:
                if self.width_spin.value() <= 0:
                    self.show_validation_error("Width must be positive")
                    return False
                return True
        """
        return True

    def reset_to_defaults(self):
        """
        Reset all UI values to their defaults.

        Default implementation does nothing.
        Override this method to implement reset functionality.

        Example:
            def reset_to_defaults(self):
                self.device_combo.setCurrentIndex(0)
                self.width_spin.setValue(1920)
                self.height_spin.setValue(1080)
                self.mark_modified()
        """
        pass

    # ========================================================================
    # CONCRETE HELPER METHODS - Available to all subclasses
    # ========================================================================

    def get_module_config(self, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific module.

        This is a convenience method that wraps config_manager access
        and handles errors gracefully.

        Args:
            module_name: Name of the module to get configuration for

        Returns:
            Dictionary containing module configuration, or None if not found

        Example:
            camera_config = self.get_module_config("camera")
            if camera_config:
                device_id = camera_config.get("device_id", 0)
        """
        try:
            config = self.config_manager.get_module_config(module_name)
            if config is None:
                self.logger.warning(f"No configuration found for module: {module_name}")
            return config
        except Exception as e:
            self.logger.error(f"Error getting config for {module_name}: {e}")
            return None

    def mark_modified(self):
        """
        Mark the tab as having unsaved changes.

        This should be called whenever a UI value changes.
        It sets the _modified flag and emits the modification_changed signal.

        Typically connected to widget signals:
            self.spinbox.valueChanged.connect(self.mark_modified)
        """
        if not self._modified:
            self._modified = True
            self.modification_changed.emit(True)
            self.logger.debug(f"{self.__class__.__name__} marked as modified")

    def clear_modified(self):
        """
        Clear the modified flag.

        This should be called after successfully saving configuration.
        It resets the _modified flag and emits the modification_changed signal.
        """
        if self._modified:
            self._modified = False
            self.modification_changed.emit(False)
            self.logger.debug(f"{self.__class__.__name__} cleared modified flag")

    def is_modified(self) -> bool:
        """
        Check if the tab has unsaved changes.

        Returns:
            bool: True if tab has been modified, False otherwise
        """
        return self._modified

    def emit_config_changed(self):
        """
        Emit the config_changed signal with current configuration.

        This is useful for notifying other tabs or the main window
        that configuration has changed. The signal includes both the
        module name and the current configuration dictionary.

        Example:
            def on_strategy_changed(self):
                self.update_ui_visibility()
                self.emit_config_changed()  # Notify other tabs
        """
        try:
            module_name = self.get_module_name()
            config = self.get_config()
            self.config_changed.emit(module_name, config)
            self.logger.debug(f"Emitted config_changed for {module_name}")
        except Exception as e:
            self.logger.error(f"Error emitting config_changed: {e}")

    def show_validation_error(self, message: str):
        """
        Report a validation error.

        This logs the error and emits the validation_failed signal,
        allowing the main window or other components to display the
        error to the user.

        Args:
            message: Descriptive error message

        Example:
            if self.port_combo.currentText() == "":
                self.show_validation_error("Please select a valid port")
        """
        self.logger.error(f"Validation failed: {message}")
        self.validation_failed.emit(message)

    # ========================================================================
    # LOGGING CONVENIENCE METHODS
    # ========================================================================

    def log_info(self, message: str):
        """
        Log an info-level message.

        Args:
            message: Message to log
        """
        self.logger.info(message)

    def log_error(self, message: str):
        """
        Log an error-level message.

        Args:
            message: Message to log
        """
        self.logger.error(message)

    def log_debug(self, message: str):
        """
        Log a debug-level message.

        Args:
            message: Message to log
        """
        self.logger.debug(message)

    def log_warning(self, message: str):
        """
        Log a warning-level message.

        Args:
            message: Message to log
        """
        self.logger.warning(message)