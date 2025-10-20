"""
configuration_gui.py - Main configuration GUI coordinator

FILE LOCATION: GUI/configuration_gui.py

This is the main coordinator for the modular configuration GUI. It orchestrates
all configuration tabs, manages inter-tab communication, and provides a unified
interface for system configuration.

Features:
- Creates and manages all configuration tabs
- Handles inter-tab signal connections
- Provides save/load/validate all functionality
- Tracks modification status across tabs
- Manages module references (camera, arduino, category service)
- Unified apply and close workflow

Usage:
    from enhanced_config_manager import create_config_manager
    from GUI.configuration_gui import ConfigurationGUI

    config_manager = create_config_manager()
    gui = ConfigurationGUI(config_manager)
    gui.show()
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QLabel, QMessageBox,
                             QGroupBox, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QCloseEvent
import logging
from typing import Dict, Optional, Any

# Import configuration tabs
from GUI.config_tabs.camera_tab import CameraConfigTab
from GUI.config_tabs.system_tab import SystemConfigTab
from GUI.config_tabs.api_tab import APIConfigTab
from GUI.config_tabs.detector_tab import DetectorConfigTab
from GUI.config_tabs.processing_tab import ProcessingConfigTab
from GUI.config_tabs.hardware_tab import HardwareConfigTab

# Import modules
from enhanced_config_manager import EnhancedConfigManager

# Initialize logger
logger = logging.getLogger(__name__)


class ConfigurationGUI(QMainWindow):
    """
    Main configuration GUI coordinator.

    This class orchestrates all configuration tabs and provides a unified
    interface for system configuration. It manages:
    - Tab creation and initialization
    - Inter-tab signal connections
    - Module references (camera, arduino, category service)
    - Configuration save/load/validate workflows
    - Modification tracking
    - Apply and close operations

    Signals:
        configuration_complete(dict): Emitted when configuration is applied
        configuration_cancelled(): Emitted when user cancels
    """

    # Main signals
    configuration_complete = pyqtSignal(dict)
    configuration_cancelled = pyqtSignal()

    def __init__(self, config_manager: EnhancedConfigManager,
                 camera=None, arduino=None, category_service=None,
                 parent=None):
        """
        Initialize the configuration GUI.

        Args:
            config_manager: Enhanced configuration manager instance
            camera: Optional camera module reference
            arduino: Optional arduino servo module reference
            category_service: Optional category hierarchy service reference
            parent: Parent widget
        """
        super().__init__(parent)

        # Store references
        self.config_manager = config_manager
        self.camera = camera
        self.arduino = arduino
        self.category_service = category_service

        # Log what we received
        logger.info("=" * 60)
        logger.info("ConfigurationGUI __init__ - Module References:")
        logger.info(f"  config_manager: {config_manager is not None}")
        logger.info(f"  camera: {camera is not None}")
        logger.info(f"  arduino: {arduino is not None}")
        logger.info(f"  category_service: {category_service is not None}")
        if category_service is not None:
            logger.info(f"  category_service type: {type(category_service)}")
        logger.info("=" * 60)

        # Tab tracking
        self.tabs = {}  # Dict[str, BaseConfigTab]
        self.modified_tabs = set()  # Set of tab names that have modifications

        # Setup UI
        self.setWindowTitle("System Configuration")

        # Set window size based on available screen resolution
        self.set_optimal_window_size()

        self.init_ui()
        self.create_tabs()
        self.setup_inter_tab_connections()
        self.load_all_configs()

        # Initialize camera previews if camera is available
        # Delay slightly to ensure camera is fully ready
        if self.camera:
            QTimer.singleShot(500, self.init_camera_previews)

        logger.info("Configuration GUI initialized")

    # ========================================================================
    # WINDOW SIZING
    # ========================================================================

    def set_optimal_window_size(self):
        """
        Set window size based on available screen resolution.

        This ensures the window fits properly on any screen size by:
        - Getting available screen geometry
        - Using 90% of width and 85% of height
        - Leaving room for taskbar and window decorations
        - Centering the window on screen
        """
        try:
            # Get the screen that contains the cursor
            screen = QApplication.primaryScreen()
            if screen is None:
                logger.warning("Could not get primary screen, using defaults")
                self.setGeometry(70, 50, 1200, 650)
                return

            # Get available geometry (excludes taskbar, menu bar, etc.)
            available_geom = screen.availableGeometry()

            logger.info(f"Screen resolution: {screen.size().width()}×{screen.size().height()}")
            logger.info(f"Available geometry: {available_geom.width()}×{available_geom.height()}")

            # Calculate window size (90% width, 85% height of available space)
            window_width = int(available_geom.width() * 0.90)
            window_height = int(available_geom.height() * 0.85)

            # Ensure minimum size
            min_width = 1000
            min_height = 600
            window_width = max(window_width, min_width)
            window_height = max(window_height, min_height)

            # Ensure maximum size (don't exceed available)
            window_width = min(window_width, available_geom.width() - 20)
            window_height = min(window_height, available_geom.height() - 20)

            # Calculate centered position
            x = available_geom.x() + (available_geom.width() - window_width) // 2
            y = available_geom.y() + (available_geom.height() - window_height) // 2

            # Set geometry
            self.setGeometry(x, y, window_width, window_height)

            logger.info(f"Window sized: {window_width}×{window_height} at ({x}, {y})")

        except Exception as e:
            logger.error(f"Error setting optimal window size: {e}", exc_info=True)
            # Fallback to safe default
            self.setGeometry(70, 50, 1200, 650)

    # ========================================================================
    # UI INITIALIZATION
    # ========================================================================

    def init_ui(self):
        """Initialize the main user interface."""
        # Central widget with scroll area
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout for scroll area
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Scrollable content widget
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setSpacing(5)  # Reduced spacing

        # Title (more compact)
        title = QLabel("System Configuration")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 8px;
                background-color: #2E86AB;
                color: white;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(title)

        # Status panel (more compact)
        status_panel = self.create_status_panel()
        main_layout.addWidget(status_panel)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tab_widget, stretch=1)

        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Set the scrollable content
        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_status_panel(self) -> QWidget:
        """Create the status monitoring panel."""
        panel = QGroupBox("Status")
        panel.setMaximumHeight(60)  # Limit height
        layout = QHBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)

        # Camera status
        self.camera_status = QLabel("Camera: Unknown")
        self.camera_status.setStyleSheet("color: gray; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.camera_status)

        layout.addStretch()

        # Hardware status
        self.hardware_status = QLabel("Hardware: Unknown")
        self.hardware_status.setStyleSheet("color: gray; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.hardware_status)

        layout.addStretch()

        # Category service status
        self.category_status = QLabel("Categories: Unknown")
        self.category_status.setStyleSheet("color: gray; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.category_status)

        layout.addStretch()

        # Modified indicator
        self.modified_indicator = QLabel("Modified: No")
        self.modified_indicator.setStyleSheet("color: green; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.modified_indicator)

        return panel

    def create_control_panel(self) -> QWidget:
        """Create the control button panel."""
        panel = QGroupBox("Configuration Controls")
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)  # Reduced spacing
        layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins

        # Row 1: Primary actions
        row1 = QHBoxLayout()

        self.save_all_btn = QPushButton("💾 Save All")
        self.save_all_btn.setToolTip("Save all configuration tabs")
        self.save_all_btn.clicked.connect(self.save_all_configs)
        self.save_all_btn.setStyleSheet(self.get_compact_button_style("#27AE60", "#229954"))
        row1.addWidget(self.save_all_btn)

        self.load_all_btn = QPushButton("📂 Reload All")
        self.load_all_btn.setToolTip("Reload all configurations and reinitialize hardware")
        self.load_all_btn.clicked.connect(self.reload_all_with_hardware_reinit)
        self.load_all_btn.setStyleSheet(self.get_compact_button_style("#3498DB", "#2980B9"))
        row1.addWidget(self.load_all_btn)

        self.validate_all_btn = QPushButton("✓ Validate All")
        self.validate_all_btn.setToolTip("Validate all configuration tabs")
        self.validate_all_btn.clicked.connect(self.validate_all_configs)
        self.validate_all_btn.setStyleSheet(self.get_compact_button_style("#16A085", "#138D75"))
        row1.addWidget(self.validate_all_btn)

        self.reset_all_btn = QPushButton("🔄 Reset All")
        self.reset_all_btn.setToolTip("Reset all tabs to defaults")
        self.reset_all_btn.clicked.connect(self.reset_all_configs)
        self.reset_all_btn.setStyleSheet(self.get_compact_button_style("#F39C12", "#E67E22"))
        row1.addWidget(self.reset_all_btn)

        layout.addLayout(row1)

        # Row 2: Apply and close
        row2 = QHBoxLayout()

        info = QLabel("💡 Changes take effect when you click 'Apply & Close'")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        row2.addWidget(info)

        row2.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setToolTip("Close without applying changes")
        cancel_btn.clicked.connect(self.cancel_configuration)
        cancel_btn.setStyleSheet(self.get_compact_button_style("#95A5A6", "#7F8C8D"))
        row2.addWidget(cancel_btn)

        self.apply_btn = QPushButton("✓ Apply & Close")
        self.apply_btn.setToolTip("Save all changes and close")
        self.apply_btn.clicked.connect(self.apply_and_close)
        self.apply_btn.setStyleSheet(self.get_compact_button_style("#27AE60", "#229954"))
        self.apply_btn.setMinimumWidth(120)
        row2.addWidget(self.apply_btn)

        layout.addLayout(row2)

        return panel

    def get_compact_button_style(self, bg_color: str, hover_color: str) -> str:
        """Get compact button stylesheet with colors."""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """

    def get_compact_button_style(self, bg_color: str, hover_color: str) -> str:
        """Get compact button stylesheet with colors."""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """

    # ========================================================================
    # TAB MANAGEMENT
    # ========================================================================

    def create_tabs(self):
        """Create all configuration tabs in proper order."""
        logger.info("Creating configuration tabs...")
        logger.info(f"  Camera available: {self.camera is not None}")
        logger.info(f"  Arduino available: {self.arduino is not None}")
        logger.info(f"  Category service available: {self.category_service is not None}")

        # Create tabs in dependency order
        # Phase 1: Independent tabs
        self.add_tab("System", SystemConfigTab, self.config_manager)
        self.add_tab("API", APIConfigTab, self.config_manager)

        # Phase 2: Camera-dependent tabs
        self.add_tab("Camera", CameraConfigTab, self.config_manager,
                     camera=self.camera)
        # Phase 3: Complex tabs with dependencies
        self.add_tab("Detector", DetectorConfigTab, self.config_manager,
                     camera=self.camera)

        # IMPORTANT: Check if category service is available
        if self.category_service is None:
            logger.warning("⚠️  Category service is None - Processing tab will have limited functionality")
            logger.warning("    Target category dropdowns will not populate")

        self.add_tab("Processing", ProcessingConfigTab, self.config_manager,
                     category_service=self.category_service)

        self.add_tab("Hardware", HardwareConfigTab, self.config_manager,
                     arduino_module=self.arduino)

        logger.info(f"✓ Created {len(self.tabs)} configuration tabs")

        # Update initial status
        QTimer.singleShot(500, self.update_status_indicators)

    def add_tab(self, name: str, tab_class, *args, **kwargs):
        """
        Create and add a configuration tab.

        Args:
            name: Display name for the tab
            tab_class: Tab class to instantiate
            *args: Positional arguments for tab constructor
            **kwargs: Keyword arguments for tab constructor
        """
        try:
            # Log what we're passing
            logger.info(f"Creating {name} tab...")
            if kwargs:
                logger.info(f"  Keyword args: {list(kwargs.keys())}")
                for key, value in kwargs.items():
                    logger.info(
                        f"    {key}: {value is not None} (type: {type(value).__name__ if value is not None else 'None'})")

            # Create tab instance
            tab = tab_class(*args, **kwargs)

            # Store reference
            self.tabs[name] = tab

            # Add to tab widget
            self.tab_widget.addTab(tab, name)

            # Connect common signals
            tab.modification_changed.connect(
                lambda modified, tab_name=name: self.on_tab_modified(tab_name, modified)
            )
            tab.validation_failed.connect(self.on_validation_failed)
            tab.config_changed.connect(self.on_config_changed)

            logger.info(f"✓ Created {name} tab")

        except Exception as e:
            logger.error(f"✗ Failed to create {name} tab: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Tab Creation Failed",
                f"Failed to create {name} tab:\n{str(e)}\n\n"
                "The GUI will continue without this tab."
            )

    def setup_inter_tab_connections(self):
        """Set up signal connections between tabs."""
        logger.info("Setting up inter-tab connections...")

        # Processing → Hardware: Bin count changes
        if "Processing" in self.tabs and "Hardware" in self.tabs:
            self.tabs["Processing"].bins_changed.connect(
                self.tabs["Hardware"].set_bin_count
            )
            logger.info("✓ Connected Processing → Hardware (bin count)")

        # Camera → Detector: Camera device changes
        if "Camera" in self.tabs and "Detector" in self.tabs:
            # Note: This would need a camera_changed signal in camera tab
            # For now, detector tab handles camera reference directly
            pass

        # Add more inter-tab connections as needed

        logger.info("✓ Inter-tab connections established")

    def init_camera_previews(self):
        """Initialize camera preview widgets as consumers."""
        logger.info("Initializing camera previews...")

        if not self.camera:
            logger.warning("No camera available for previews")
            return

        # Check if camera is initialized
        try:
            if hasattr(self.camera, 'is_initialized') and not self.camera.is_initialized():
                logger.warning("Camera not initialized yet")
                # Try to initialize it
                if hasattr(self.camera, 'initialize'):
                    logger.info("Attempting to initialize camera...")
                    if not self.camera.initialize():
                        logger.error("Failed to initialize camera")
                        return
        except Exception as e:
            logger.error(f"Error checking camera status: {e}")
            return

        # Start camera capture if not already running
        try:
            if hasattr(self.camera, 'start_capture'):
                logger.info("Starting camera capture...")
                if not self.camera.start_capture():
                    logger.error("Failed to start camera capture")
                    return
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return

        # Register detector tab preview
        if "Detector" in self.tabs:
            try:
                logger.info("Registering detector preview...")
                detector_tab = self.tabs["Detector"]

                if hasattr(detector_tab, 'register_preview'):
                    success = detector_tab.register_preview()
                    if success:
                        logger.info("✓ Detector preview registered")
                    else:
                        logger.warning("✗ Detector preview registration failed")
                else:
                    logger.warning("Detector tab has no register_preview method")
            except Exception as e:
                logger.error(f"Error registering detector preview: {e}", exc_info=True)

        logger.info("✓ Camera preview initialization complete")

    # ========================================================================
    # CONFIGURATION OPERATIONS
    # ========================================================================

    def load_all_configs(self):
        """Load configuration for all tabs (without hardware reinitialization)."""
        logger.info("=" * 60)
        logger.info("Loading all configurations...")
        logger.info("=" * 60)

        success_count = 0
        fail_count = 0

        for name, tab in self.tabs.items():
            try:
                if tab.load_config():
                    logger.info(f"✓ {name} tab loaded")
                    success_count += 1
                else:
                    logger.warning(f"⚠ {name} tab load returned False")
                    fail_count += 1
            except Exception as e:
                logger.error(f"✗ {name} tab load failed: {e}", exc_info=True)
                fail_count += 1

        # Update status
        if fail_count == 0:
            self.statusBar().showMessage(f"✓ All configurations loaded", 3000)
            logger.info(f"✓ All {success_count} tabs loaded successfully")
        else:
            self.statusBar().showMessage(
                f"⚠ {success_count} loaded, {fail_count} failed", 5000
            )
            logger.warning(f"⚠ {success_count} succeeded, {fail_count} failed")

        # Clear modified flags
        self.modified_tabs.clear()
        self.update_modified_indicator()

    def reload_all_with_hardware_reinit(self):
        """
        Reload all configurations AND reinitialize hardware modules.

        This is called when the user clicks the "Reload All" button.
        It performs a full reload of configuration from disk and
        reinitializes hardware modules with the new settings.

        This is different from load_all_configs() which is called
        during startup and should NOT reinitialize already-initialized hardware.
        """
        logger.info("=" * 70)
        logger.info("RELOAD ALL WITH HARDWARE REINITIALIZATION")
        logger.info("=" * 70)

        # Step 1: Load all tab configurations from disk
        logger.info("Step 1: Loading configurations from disk...")
        success_count = 0
        fail_count = 0

        for name, tab in self.tabs.items():
            try:
                if tab.load_config():
                    logger.info(f"✓ {name} tab loaded")
                    success_count += 1
                else:
                    logger.warning(f"⚠ {name} tab load returned False")
                    fail_count += 1
            except Exception as e:
                logger.error(f"✗ {name} tab load failed: {e}", exc_info=True)
                fail_count += 1

        # Step 2: Reinitialize hardware modules with new configuration
        logger.info("=" * 70)
        logger.info("Step 2: Reinitializing hardware modules...")
        logger.info("=" * 70)

        hardware_reinit_status = []

        # Reinitialize Arduino servo controller
        if self.arduino:
            try:
                logger.info("Reinitializing Arduino servo controller...")

                # Get the updated configuration
                arduino_config = self.config_manager.get_module_config("arduino_servo")

                # Disconnect existing connection
                logger.info("Disconnecting existing Arduino connection...")
                self.arduino.disconnect()

                # Update Arduino internal configuration with new values
                self.arduino.port = arduino_config['port']
                self.arduino.baud_rate = arduino_config['baud_rate']
                self.arduino.timeout = arduino_config['timeout']
                self.arduino.connection_retries = arduino_config['connection_retries']
                self.arduino.retry_delay = arduino_config['retry_delay']
                self.arduino.simulation_mode = arduino_config['simulation_mode']
                self.arduino.default_position = arduino_config['default_position']
                self.arduino.min_bin_separation = arduino_config['min_bin_separation']

                logger.info(f"  Port: {self.arduino.port}")
                logger.info(f"  Baud Rate: {self.arduino.baud_rate}")
                logger.info(f"  Simulation Mode: {self.arduino.simulation_mode}")

                # Reconnect with new settings
                if not self.arduino.simulation_mode:
                    logger.info(f"Reconnecting to Arduino on port: {self.arduino.port}")

                    if self.arduino._connect_to_arduino():
                        logger.info("✓ Arduino reconnected successfully")
                        hardware_reinit_status.append(("Arduino", True, f"Connected to {self.arduino.port}"))
                    else:
                        logger.error("✗ Failed to reconnect to Arduino")
                        hardware_reinit_status.append(("Arduino", False, "Connection failed"))
                else:
                    logger.info("✓ Arduino in simulation mode - no hardware connection needed")
                    self.arduino.initialized = True
                    hardware_reinit_status.append(("Arduino", True, "Simulation mode enabled"))

                # Reload bin positions from updated config
                logger.info("Reloading bin positions...")
                if self.arduino.reload_positions_from_config():
                    logger.info("✓ Bin positions reloaded")
                else:
                    logger.warning("⚠ Bin positions reload returned False")

            except Exception as e:
                logger.error(f"✗ Failed to reinitialize Arduino: {e}", exc_info=True)
                hardware_reinit_status.append(("Arduino", False, str(e)))

        # Reinitialize Camera
        if self.camera:
            try:
                logger.info("Reinitializing camera...")

                # Get new camera config
                camera_config = self.config_manager.get_module_config("camera")

                # Stop current capture
                if hasattr(self.camera, 'stop_capture'):
                    logger.info("Stopping camera capture...")
                    self.camera.stop_capture()

                # Release camera hardware
                if hasattr(self.camera, 'release'):
                    logger.info("Releasing camera hardware...")
                    self.camera.release()

                # Update camera configuration
                if hasattr(self.camera, 'device_id'):
                    self.camera.device_id = camera_config['device_id']
                if hasattr(self.camera, 'width'):
                    self.camera.width = camera_config['width']
                if hasattr(self.camera, 'height'):
                    self.camera.height = camera_config['height']
                if hasattr(self.camera, 'fps'):
                    self.camera.fps = camera_config['fps']

                # Reinitialize with new settings
                if hasattr(self.camera, 'initialize'):
                    logger.info(f"Reinitializing camera with device_id: {camera_config['device_id']}")
                    if self.camera.initialize():
                        logger.info("✓ Camera reinitialized")
                        hardware_reinit_status.append(("Camera", True, f"Device {camera_config['device_id']}"))

                        # Restart capture
                        if hasattr(self.camera, 'start_capture'):
                            if self.camera.start_capture():
                                logger.info("✓ Camera capture restarted")
                            else:
                                logger.warning("⚠ Camera capture restart failed")
                    else:
                        logger.warning("⚠ Camera reinitialization returned False")
                        hardware_reinit_status.append(("Camera", False, "Initialization failed"))
                else:
                    logger.warning("Camera has no initialize method")

            except Exception as e:
                logger.error(f"✗ Failed to reinitialize camera: {e}", exc_info=True)
                hardware_reinit_status.append(("Camera", False, str(e)))

        logger.info("=" * 70)

        # Step 3: Update UI status indicators
        self.update_status_indicators()

        # Step 4: Show results to user
        if fail_count == 0 and all(status[1] for status in hardware_reinit_status):
            # Perfect success
            self.statusBar().showMessage(
                f"✓ All configurations reloaded and hardware reinitialized",
                5000
            )
            logger.info(f"✓ All {success_count} tabs loaded and hardware reinitialized successfully")

            # Build success message
            success_msg = f"✓ All configurations reloaded successfully!\n\n"
            success_msg += f"Configuration tabs loaded: {success_count}\n\n"

            if hardware_reinit_status:
                success_msg += "Hardware reinitialized:\n"
                for module, success, details in hardware_reinit_status:
                    success_msg += f"  • {module}: {details}\n"

            QMessageBox.information(
                self,
                "Reload Successful",
                success_msg
            )

        else:
            # Partial failure
            self.statusBar().showMessage(
                f"⚠ {success_count} configs loaded, some issues occurred",
                5000
            )

            # Build detailed message
            warning_msg = "Configuration reload completed with some issues:\n\n"

            if fail_count > 0:
                warning_msg += f"Configuration tabs:\n"
                warning_msg += f"  • Loaded: {success_count}\n"
                warning_msg += f"  • Failed: {fail_count}\n\n"

            if hardware_reinit_status:
                warning_msg += "Hardware reinitialization:\n"
                for module, success, details in hardware_reinit_status:
                    status_icon = "✓" if success else "✗"
                    warning_msg += f"  {status_icon} {module}: {details}\n"

            warning_msg += "\nCheck logs for details."

            QMessageBox.warning(
                self,
                "Reload Issues",
                warning_msg
            )

            logger.warning(f"⚠ Reload completed with issues: {success_count} tabs succeeded, {fail_count} failed")

        # Step 5: Clear modified flags
        self.modified_tabs.clear()
        self.update_modified_indicator()

        logger.info("=" * 70)

    def save_all_configs(self):
        """Save configuration for all tabs."""
        logger.info("=" * 60)
        logger.info("Saving all configurations...")
        logger.info("=" * 60)

        success_count = 0
        fail_count = 0
        failed_tabs = []

        for name, tab in self.tabs.items():
            try:
                if tab.save_config():
                    logger.info(f"✓ {name} tab saved")
                    success_count += 1
                else:
                    logger.warning(f"⚠ {name} tab save returned False")
                    fail_count += 1
                    failed_tabs.append(name)
            except Exception as e:
                logger.error(f"✗ {name} tab save failed: {e}", exc_info=True)
                fail_count += 1
                failed_tabs.append(name)

        # Update status
        if fail_count == 0:
            self.statusBar().showMessage(f"✓ All configurations saved", 3000)
            logger.info(f"✓ All {success_count} tabs saved successfully")
            QMessageBox.information(
                self,
                "Save Successful",
                f"✓ All configurations saved successfully!\n\n"
                f"{success_count} tab(s) saved."
            )

            # Clear modified flags
            self.modified_tabs.clear()
            self.update_modified_indicator()
        else:
            self.statusBar().showMessage(
                f"⚠ {success_count} saved, {fail_count} failed", 5000
            )
            logger.warning(f"⚠ {success_count} succeeded, {fail_count} failed")
            QMessageBox.warning(
                self,
                "Save Issues",
                f"Some configurations failed to save:\n\n"
                f"Failed tabs: {', '.join(failed_tabs)}\n\n"
                f"Check logs for details."
            )

    def validate_all_configs(self):
        """Validate configuration for all tabs."""
        logger.info("=" * 60)
        logger.info("Validating all configurations...")
        logger.info("=" * 60)

        all_valid = True
        invalid_tabs = []

        for name, tab in self.tabs.items():
            try:
                if not tab.validate():
                    logger.warning(f"✗ {name} tab validation failed")
                    all_valid = False
                    invalid_tabs.append(name)
                else:
                    logger.info(f"✓ {name} tab valid")
            except Exception as e:
                logger.error(f"✗ {name} tab validation error: {e}", exc_info=True)
                all_valid = False
                invalid_tabs.append(name)

        # Show results
        if all_valid:
            self.statusBar().showMessage("✓ All configurations valid", 3000)
            logger.info("✓ All tabs validated successfully")
            QMessageBox.information(
                self,
                "Validation Passed",
                "✓ All configurations are valid!"
            )
        else:
            self.statusBar().showMessage(
                f"✗ {len(invalid_tabs)} tab(s) invalid", 5000
            )
            logger.warning(f"✗ Invalid tabs: {invalid_tabs}")
            QMessageBox.warning(
                self,
                "Validation Failed",
                f"Some configurations are invalid:\n\n"
                f"Invalid tabs: {', '.join(invalid_tabs)}\n\n"
                f"Please review and correct the issues."
            )

        return all_valid

    def reset_all_configs(self):
        """Reset all tabs to default values."""
        reply = QMessageBox.question(
            self,
            "Reset All Configurations",
            "Reset ALL configuration tabs to default values?\n\n"
            "This will discard all current settings.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        logger.info("=" * 60)
        logger.info("Resetting all configurations to defaults...")
        logger.info("=" * 60)

        for name, tab in self.tabs.items():
            try:
                tab.reset_to_defaults()
                logger.info(f"✓ {name} tab reset")
            except Exception as e:
                logger.error(f"✗ {name} tab reset failed: {e}", exc_info=True)

        self.statusBar().showMessage("All tabs reset to defaults", 3000)
        logger.info("✓ All tabs reset to defaults")

    def apply_and_close(self):
        """Apply all configurations and close the GUI."""
        logger.info("=" * 60)
        logger.info("Apply and Close requested")
        logger.info("=" * 60)

        # Check if any modifications exist
        if not self.modified_tabs:
            logger.info("No changes detected, but proceeding to close")

            # Get current configuration even if no changes
            final_config = {}
            for name, tab in self.tabs.items():
                try:
                    final_config[tab.get_module_name()] = tab.get_config()
                except Exception as e:
                    logger.error(f"Failed to get config from {name}: {e}")

            # Emit signal even with no changes
            logger.info("Emitting configuration_complete signal")
            self.configuration_complete.emit(final_config)

            logger.info("✓ Configuration applied (no changes)")
            logger.info("=" * 60)

            # Close the window
            self.close()
            return

        # Validate all configurations
        if not self.validate_all_configs():
            reply = QMessageBox.question(
                self,
                "Validation Failed",
                "Some configurations are invalid.\n\n"
                "Do you want to fix them before applying?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                return

        # Save all configurations
        logger.info("Saving all configurations before closing...")

        success_count = 0
        fail_count = 0

        for name, tab in self.tabs.items():
            try:
                if tab.save_config():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"Save failed for {name}: {e}")
                fail_count += 1

        if fail_count > 0:
            QMessageBox.warning(
                self,
                "Save Issues",
                f"Some configurations failed to save.\n\n"
                f"Saved: {success_count}\n"
                f"Failed: {fail_count}\n\n"
                f"Check logs for details."
            )
            return

        # Get final configuration
        final_config = {}
        for name, tab in self.tabs.items():
            try:
                final_config[tab.get_module_name()] = tab.get_config()
            except Exception as e:
                logger.error(f"Failed to get config from {name}: {e}")

        # Emit completion signal
        logger.info("Emitting configuration_complete signal")
        self.configuration_complete.emit(final_config)

        logger.info("✓ Configuration applied successfully")
        logger.info("=" * 60)

        # Close the window
        self.close()

    def cancel_configuration(self):
        """Cancel configuration and close without saving."""
        if self.modified_tabs:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"You have unsaved changes in {len(self.modified_tabs)} tab(s).\n\n"
                "Close without saving?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

        logger.info("Configuration cancelled by user")
        self.configuration_cancelled.emit()
        self.close()

    # ========================================================================
    # SIGNAL HANDLERS
    # ========================================================================

    def on_tab_modified(self, tab_name: str, is_modified: bool):
        """Handle modification status changes from tabs."""
        if is_modified:
            self.modified_tabs.add(tab_name)
        else:
            self.modified_tabs.discard(tab_name)

        self.update_modified_indicator()

    def on_validation_failed(self, error_message: str):
        """Handle validation failures from tabs."""
        logger.warning(f"Validation failed: {error_message}")
        self.statusBar().showMessage(f"Validation failed: {error_message}", 5000)

    def on_config_changed(self, module_name: str, config: Dict[str, Any]):
        """Handle configuration changes from tabs."""
        logger.debug(f"Configuration changed: {module_name}")
        self.statusBar().showMessage(f"{module_name} configuration updated", 2000)

    def on_tab_changed(self, index: int):
        """Handle tab switch."""
        if index >= 0:
            tab_name = self.tab_widget.tabText(index)
            self.statusBar().showMessage(f"Viewing {tab_name} configuration", 2000)

    # ========================================================================
    # STATUS UPDATES
    # ========================================================================

    def update_modified_indicator(self):
        """Update the modified indicator label."""
        if self.modified_tabs:
            count = len(self.modified_tabs)
            self.modified_indicator.setText(f"Modified: {count} tab(s)")
            self.modified_indicator.setStyleSheet("color: #E67E22; font-weight: bold;")
        else:
            self.modified_indicator.setText("Modified: No")
            self.modified_indicator.setStyleSheet("color: #27AE60; font-weight: bold;")

    def update_status_indicators(self):
        """Update module status indicators."""
        # Camera status
        if self.camera:
            try:
                if hasattr(self.camera, 'is_initialized') and self.camera.is_initialized():
                    self.camera_status.setText("Camera: Active")
                    self.camera_status.setStyleSheet("color: #27AE60; font-weight: bold;")
                else:
                    self.camera_status.setText("Camera: Initialized")
                    self.camera_status.setStyleSheet("color: #F39C12; font-weight: bold;")
            except:
                self.camera_status.setText("Camera: Available")
                self.camera_status.setStyleSheet("color: #3498DB; font-weight: bold;")
        else:
            self.camera_status.setText("Camera: Not Available")
            self.camera_status.setStyleSheet("color: gray; font-weight: bold;")

        # Hardware status
        if self.arduino:
            try:
                if self.arduino.is_ready():
                    self.hardware_status.setText("Hardware: Ready")
                    self.hardware_status.setStyleSheet("color: #27AE60; font-weight: bold;")
                else:
                    self.hardware_status.setText("Hardware: Not Ready")
                    self.hardware_status.setStyleSheet("color: #E67E22; font-weight: bold;")
            except:
                self.hardware_status.setText("Hardware: Available")
                self.hardware_status.setStyleSheet("color: #3498DB; font-weight: bold;")
        else:
            self.hardware_status.setText("Hardware: Not Available")
            self.hardware_status.setStyleSheet("color: gray; font-weight: bold;")

        # Category service status
        if self.category_service:
            try:
                primaries = self.category_service.get_primary_categories()
                self.category_status.setText(f"Categories: {len(primaries)} primary")
                self.category_status.setStyleSheet("color: #27AE60; font-weight: bold;")
            except:
                self.category_status.setText("Categories: Available")
                self.category_status.setStyleSheet("color: #3498DB; font-weight: bold;")
        else:
            self.category_status.setText("Categories: Not Available")
            self.category_status.setStyleSheet("color: gray; font-weight: bold;")

    # ========================================================================
    # WINDOW MANAGEMENT
    # ========================================================================

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event."""
        if self.modified_tabs:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"You have unsaved changes in {len(self.modified_tabs)} tab(s).\n\n"
                "Close without saving?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                event.ignore()
                return

        logger.info("Configuration GUI closing")

        # Clean up camera previews
        self.cleanup_camera_previews()

        event.accept()

    def cleanup_camera_previews(self):
        """Clean up camera preview registrations."""
        logger.info("Cleaning up camera previews...")

        if not self.camera:
            return

        # Unregister detector preview
        if "Detector" in self.tabs:
            try:
                detector_tab = self.tabs["Detector"]
                if hasattr(detector_tab, 'unregister_preview'):
                    detector_tab.unregister_preview()
                    logger.info("✓ Detector preview unregistered")
                elif hasattr(detector_tab, 'cleanup'):
                    detector_tab.cleanup()
                    logger.info("✓ Detector tab cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up detector preview: {e}")

        logger.info("✓ Camera preview cleanup complete")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

def main():
    """
    Standalone test function for the configuration GUI.

    This allows testing the GUI without the full application.
    """
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 70)
    logger.info("CONFIGURATION GUI - Standalone Test")
    logger.info("=" * 70)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Import required modules
    from enhanced_config_manager import create_config_manager

    try:
        from camera_module import create_camera
    except ImportError:
        logger.warning("Camera module not available")
        create_camera = None

    try:
        from hardware.arduino_servo_module import create_arduino_servo_controller
    except ImportError:
        logger.warning("Arduino module not available")
        create_arduino_servo_controller = None

    try:
        from processing.category_hierarchy_service import create_category_hierarchy_service
    except ImportError:
        logger.warning("Category service not available")
        create_category_hierarchy_service = None

    # Create config manager
    logger.info("Creating configuration manager...")
    config_manager = create_config_manager()

    # Create optional modules
    camera = None
    if create_camera:
        try:
            logger.info("Creating camera module...")
            camera = create_camera("webcam", config_manager)

            # Initialize camera hardware
            logger.info("Initializing camera hardware...")
            if camera.initialize():
                logger.info("✓ Camera hardware initialized")

                # Start capture
                logger.info("Starting camera capture...")
                if camera.start_capture():
                    logger.info("✓ Camera capture started")
                else:
                    logger.warning("Could not start camera capture")
            else:
                logger.warning("Could not initialize camera hardware")

        except Exception as e:
            logger.warning(f"Could not create camera: {e}")
            camera = None

    arduino = None
    if create_arduino_servo_controller:
        try:
            logger.info("Creating Arduino module...")
            arduino = create_arduino_servo_controller(config_manager)
            logger.info("✓ Arduino module created")
        except Exception as e:
            logger.warning(f"Could not create Arduino: {e}")

    category_service = None
    if create_category_hierarchy_service:
        try:
            logger.info("Creating category service...")
            category_service = create_category_hierarchy_service(config_manager)

            # Verify it works by testing a query
            primaries = category_service.get_primary_categories()
            logger.info(f"✓ Category service created - {len(primaries)} primary categories loaded")
            logger.info(f"  Primary categories: {primaries}")
            logger.info(f"  Category service object: {category_service}")
            logger.info(f"  Category service type: {type(category_service)}")

        except Exception as e:
            logger.error(f"✗ Could not create category service: {e}", exc_info=True)
            category_service = None
    else:
        logger.warning("⚠️  create_category_hierarchy_service function not available (import failed)")

    # Create configuration GUI
    logger.info("Creating configuration GUI...")
    logger.info(f"Passing to ConfigurationGUI:")
    logger.info(f"  config_manager: {config_manager is not None}")
    logger.info(f"  camera: {camera is not None}")
    logger.info(f"  arduino: {arduino is not None}")
    logger.info(f"  category_service: {category_service is not None}")

    gui = ConfigurationGUI(
        config_manager,
        camera=camera,
        arduino=arduino,
        category_service=category_service
    )

    # Connect signals
    gui.configuration_complete.connect(
        lambda config: logger.info(f"Configuration complete: {list(config.keys())}")
    )
    gui.configuration_cancelled.connect(
        lambda: logger.info("Configuration cancelled")
    )

    # Show GUI
    gui.show()
    logger.info("✓ Configuration GUI displayed")
    logger.info("=" * 70)

    # Run application
    exit_code = app.exec_()

    # Cleanup on exit
    logger.info("Application closing, cleaning up...")

    if camera:
        try:
            logger.info("Stopping camera...")
            camera.stop_capture()
            camera.release()
            logger.info("✓ Camera released")
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")

    if arduino:
        try:
            logger.info("Releasing Arduino...")
            arduino.release()
            logger.info("✓ Arduino released")
        except Exception as e:
            logger.error(f"Error releasing Arduino: {e}")

    logger.info("✓ Cleanup complete")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
