"""
detector_tab.py - Detector configuration tab with live ROI preview

FILE LOCATION: GUI/config_tabs/detector_tab.py

This tab provides detection and ROI configuration with a live camera preview.
Users can configure the Region of Interest (ROI), entry/exit zones, and detection
parameters while seeing real-time feedback on the camera feed.

Features:
- Live camera preview with ROI overlay
- Real-time ROI configuration with automatic frame boundary enforcement
- Entry and exit zone percentage controls
- Detection parameter tuning (min/max area)
- Dynamic spinbox limits based on actual camera resolution
- Visual feedback and validation

Configuration Mapping:
    {
        "detector": {
            "min_area": 1000,
            "max_area": 50000,
            "zones": {
                "entry_percentage": 15,
                "exit_percentage": 15
            }
        },
        "detector_roi": {
            "x": 100,
            "y": 50,
            "width": 1720,
            "height": 980
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QPushButton, QSpinBox, QLabel,
                             QSlider, QMessageBox, QSplitter)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from typing import Dict, Any, Optional, Tuple
import logging

# Import base class and widgets
from GUI.config_tabs.base_tab import BaseConfigTab
from GUI.widgets.camera_view import CameraViewROI
from enhanced_config_manager import ModuleConfig

# Initialize logger
logger = logging.getLogger(__name__)


class DetectorConfigTab(BaseConfigTab):
    """
    Detector configuration tab with live ROI preview.

    This tab allows users to:
    - Configure ROI boundaries with automatic frame limit enforcement
    - Set entry and exit zone percentages
    - Tune detection parameters (min/max area)
    - View live camera feed with ROI overlay
    - Get real-time visual feedback of configuration changes

    The ROI is automatically constrained to stay within the camera frame
    dimensions, preventing invalid configurations.

    Signals:
        roi_changed(tuple): Emitted when ROI changes (x, y, width, height)
        zones_changed(tuple): Emitted when zones change (entry_pct, exit_pct)
        camera_device_changed(int): Emitted when camera device changes
    """

    # Custom signals
    roi_changed = pyqtSignal(tuple)  # (x, y, width, height)
    zones_changed = pyqtSignal(tuple)  # (entry_pct, exit_pct)
    camera_device_changed = pyqtSignal(int)  # device_id

    def __init__(self, config_manager, camera=None, parent=None):
        """
        Initialize detector configuration tab.

        Args:
            config_manager: Configuration manager instance
            camera: Optional camera module reference for live preview
            parent: Parent widget
        """
        # Store camera reference
        self.camera = camera
        self.preview_active = False

        # Frame dimensions (updated when camera provides frames)
        self.frame_width = 1920  # Default, will be updated
        self.frame_height = 1080  # Default, will be updated
        self.frame_dimensions_known = False

        # Preview widget (created in init_ui)
        self.preview_widget = None

        # Call parent init
        super().__init__(config_manager, parent)

        # Initialize UI and load config
        self.init_ui()
        self.load_config()

        # Start checking for frame dimensions if camera is available
        if self.camera:
            self.start_dimension_monitoring()

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (Required by BaseConfigTab)
    # ========================================================================

    def get_module_name(self) -> str:
        """Return the configuration module name this tab manages."""
        return ModuleConfig.DETECTOR.value

    def init_ui(self):
        """Create the detector configuration user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Title
        title = QLabel("Detection Configuration")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #2E86AB;
                color: white;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(title)

        # Create splitter for side-by-side layout
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left panel: Configuration controls
        config_panel = self.create_config_panel()
        splitter.addWidget(config_panel)

        # Right panel: Live preview
        preview_panel = self.create_preview_panel()
        splitter.addWidget(preview_panel)

        # Set initial splitter sizes (40% config, 60% preview)
        splitter.setSizes([400, 600])

        # Bottom control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        self.logger.info("Detector tab UI initialized")

    def create_config_panel(self) -> QWidget:
        """Create the configuration controls panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ROI Configuration Group
        roi_group = self.create_roi_group()
        layout.addWidget(roi_group)

        # Zone Configuration Group
        zone_group = self.create_zone_group()
        layout.addWidget(zone_group)

        # Detection Parameters Group
        detection_group = self.create_detection_group()
        layout.addWidget(detection_group)

        # Add stretch to push everything to top
        layout.addStretch()

        return panel

    def create_roi_group(self) -> QGroupBox:
        """Create ROI configuration group."""
        group = QGroupBox("Region of Interest (ROI)")
        layout = QFormLayout()

        # ROI X coordinate
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, self.frame_width)
        self.roi_x_spin.setValue(100)
        self.roi_x_spin.setSuffix(" px")
        self.roi_x_spin.setToolTip("X coordinate of ROI top-left corner")
        self.roi_x_spin.valueChanged.connect(self.on_roi_changed)
        layout.addRow("X Position:", self.roi_x_spin)

        # ROI Y coordinate
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, self.frame_height)
        self.roi_y_spin.setValue(50)
        self.roi_y_spin.setSuffix(" px")
        self.roi_y_spin.setToolTip("Y coordinate of ROI top-left corner")
        self.roi_y_spin.valueChanged.connect(self.on_roi_changed)
        layout.addRow("Y Position:", self.roi_y_spin)

        # ROI Width
        self.roi_width_spin = QSpinBox()
        self.roi_width_spin.setRange(100, self.frame_width)
        self.roi_width_spin.setValue(1720)
        self.roi_width_spin.setSuffix(" px")
        self.roi_width_spin.setToolTip("Width of the ROI")
        self.roi_width_spin.valueChanged.connect(self.on_roi_changed)
        layout.addRow("Width:", self.roi_width_spin)

        # ROI Height
        self.roi_height_spin = QSpinBox()
        self.roi_height_spin.setRange(100, self.frame_height)
        self.roi_height_spin.setValue(980)
        self.roi_height_spin.setSuffix(" px")
        self.roi_height_spin.setToolTip("Height of the ROI")
        self.roi_height_spin.valueChanged.connect(self.on_roi_changed)
        layout.addRow("Height:", self.roi_height_spin)

        # Frame dimensions info label
        self.frame_info_label = QLabel(f"Frame: {self.frame_width}×{self.frame_height}")
        self.frame_info_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        layout.addRow("", self.frame_info_label)

        # Preset buttons
        preset_layout = QHBoxLayout()

        full_frame_btn = QPushButton("Full Frame")
        full_frame_btn.setToolTip("Set ROI to entire frame")
        full_frame_btn.clicked.connect(self.set_roi_full_frame)
        preset_layout.addWidget(full_frame_btn)

        center_btn = QPushButton("Center 80%")
        center_btn.setToolTip("Set ROI to centered 80% of frame")
        center_btn.clicked.connect(self.set_roi_center_80)
        preset_layout.addWidget(center_btn)

        layout.addRow("Presets:", preset_layout)

        group.setLayout(layout)
        return group

    def create_zone_group(self) -> QGroupBox:
        """Create zone configuration group."""
        group = QGroupBox("Entry/Exit Zones")
        layout = QFormLayout()

        # Entry zone percentage
        entry_layout = QHBoxLayout()
        self.entry_zone_slider = QSlider(Qt.Horizontal)
        self.entry_zone_slider.setRange(5, 50)
        self.entry_zone_slider.setValue(15)
        self.entry_zone_slider.setToolTip("Entry zone as percentage of ROI height")
        self.entry_zone_slider.valueChanged.connect(self.on_zones_changed)

        self.entry_zone_label = QLabel("15%")
        self.entry_zone_label.setMinimumWidth(40)
        self.entry_zone_slider.valueChanged.connect(
            lambda v: self.entry_zone_label.setText(f"{v}%")
        )

        entry_layout.addWidget(self.entry_zone_slider)
        entry_layout.addWidget(self.entry_zone_label)
        layout.addRow("Entry Zone:", entry_layout)

        # Exit zone percentage
        exit_layout = QHBoxLayout()
        self.exit_zone_slider = QSlider(Qt.Horizontal)
        self.exit_zone_slider.setRange(5, 50)
        self.exit_zone_slider.setValue(15)
        self.exit_zone_slider.setToolTip("Exit zone as percentage of ROI height")
        self.exit_zone_slider.valueChanged.connect(self.on_zones_changed)

        self.exit_zone_label = QLabel("15%")
        self.exit_zone_label.setMinimumWidth(40)
        self.exit_zone_slider.valueChanged.connect(
            lambda v: self.exit_zone_label.setText(f"{v}%")
        )

        exit_layout.addWidget(self.exit_zone_slider)
        exit_layout.addWidget(self.exit_zone_label)
        layout.addRow("Exit Zone:", exit_layout)

        # Info label
        info = QLabel("Zones are displayed as colored overlays on the preview")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        info.setWordWrap(True)
        layout.addRow("", info)

        group.setLayout(layout)
        return group

    def create_detection_group(self) -> QGroupBox:
        """Create detection parameters group."""
        group = QGroupBox("Detection Parameters")
        layout = QFormLayout()

        # Minimum area
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 100000)
        self.min_area_spin.setValue(1000)
        self.min_area_spin.setSuffix(" px²")
        self.min_area_spin.setSingleStep(100)
        self.min_area_spin.setToolTip("Minimum contour area to detect pieces")
        self.min_area_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Min Area:", self.min_area_spin)

        # Maximum area
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1000, 500000)
        self.max_area_spin.setValue(50000)
        self.max_area_spin.setSuffix(" px²")
        self.max_area_spin.setSingleStep(1000)
        self.max_area_spin.setToolTip("Maximum contour area to detect pieces")
        self.max_area_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Max Area:", self.max_area_spin)

        # Info label
        info = QLabel("Pieces outside these area limits will be ignored")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        info.setWordWrap(True)
        layout.addRow("", info)

        group.setLayout(layout)
        return group

    def create_preview_panel(self) -> QWidget:
        """Create the live preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Live Camera Preview")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)

        # Camera view widget
        self.preview_widget = CameraViewROI()
        self.preview_widget.set_fps_visible(True)
        layout.addWidget(self.preview_widget, stretch=1)

        # Preview controls
        controls_layout = QHBoxLayout()

        self.preview_status_label = QLabel("Preview: Not started")
        self.preview_status_label.setStyleSheet("color: orange; font-weight: bold;")
        controls_layout.addWidget(self.preview_status_label)

        controls_layout.addStretch()

        self.fps_toggle_btn = QPushButton("Toggle FPS")
        self.fps_toggle_btn.clicked.connect(self.toggle_fps)
        controls_layout.addWidget(self.fps_toggle_btn)

        layout.addLayout(controls_layout)

        return panel

    def create_control_panel(self) -> QWidget:
        """Create bottom control panel."""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # Info label
        info = QLabel("ROI and zones update in real-time • Changes are saved when you click 'Save Configuration'")
        info.setStyleSheet("color: #7F8C8D; font-style: italic;")
        layout.addWidget(info)

        layout.addStretch()

        # Apply button (for immediate preview without saving)
        apply_btn = QPushButton("Apply to Preview")
        apply_btn.setToolTip("Apply current settings to preview (without saving to config)")
        apply_btn.clicked.connect(self.apply_to_preview)
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        layout.addWidget(apply_btn)

        return panel

    # ========================================================================
    # ROI MANAGEMENT AND VALIDATION
    # ========================================================================

    def update_frame_dimensions(self, width: int, height: int):
        """
        Update frame dimensions and adjust ROI spinbox limits.

        This method is called when the actual camera frame dimensions are known.
        It updates the spinbox ranges to prevent ROI from exceeding frame bounds.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        if width == self.frame_width and height == self.frame_height:
            return  # No change

        self.frame_width = width
        self.frame_height = height
        self.frame_dimensions_known = True

        self.logger.info(f"Frame dimensions updated: {width}×{height}")

        # Update info label
        self.frame_info_label.setText(f"Frame: {width}×{height}")

        # Update spinbox ranges
        # Block signals to prevent triggering updates during range changes
        self.roi_x_spin.blockSignals(True)
        self.roi_y_spin.blockSignals(True)
        self.roi_width_spin.blockSignals(True)
        self.roi_height_spin.blockSignals(True)

        self.roi_x_spin.setMaximum(width - 100)  # Leave room for minimum width
        self.roi_y_spin.setMaximum(height - 100)  # Leave room for minimum height
        self.roi_width_spin.setMaximum(width)
        self.roi_height_spin.setMaximum(height)

        # Restore signals
        self.roi_x_spin.blockSignals(False)
        self.roi_y_spin.blockSignals(False)
        self.roi_width_spin.blockSignals(False)
        self.roi_height_spin.blockSignals(False)

        # Validate and constrain current ROI to new bounds
        self.constrain_roi_to_frame()

    def constrain_roi_to_frame(self):
        """
        Constrain ROI to stay within frame boundaries.

        This method ensures that:
        - ROI position + size doesn't exceed frame dimensions
        - ROI has minimum viable dimensions
        - All values are adjusted smoothly without jarring changes
        """
        # Get current values
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        width = self.roi_width_spin.value()
        height = self.roi_height_spin.value()

        # Constrain width and height to frame
        if width > self.frame_width:
            width = self.frame_width
        if height > self.frame_height:
            height = self.frame_height

        # Constrain position to keep ROI within frame
        if x + width > self.frame_width:
            x = self.frame_width - width
        if y + height > self.frame_height:
            y = self.frame_height - height

        # Ensure non-negative positions
        x = max(0, x)
        y = max(0, y)

        # Update spinboxes if values changed
        self.roi_x_spin.blockSignals(True)
        self.roi_y_spin.blockSignals(True)
        self.roi_width_spin.blockSignals(True)
        self.roi_height_spin.blockSignals(True)

        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_width_spin.setValue(width)
        self.roi_height_spin.setValue(height)

        self.roi_x_spin.blockSignals(False)
        self.roi_y_spin.blockSignals(False)
        self.roi_width_spin.blockSignals(False)
        self.roi_height_spin.blockSignals(False)

        # Apply to preview
        self.apply_to_preview()

    def validate_roi(self) -> bool:
        """
        Validate that ROI is within frame bounds.

        Returns:
            bool: True if ROI is valid, False otherwise
        """
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        width = self.roi_width_spin.value()
        height = self.roi_height_spin.value()

        # Check if ROI exceeds frame bounds
        if x + width > self.frame_width:
            self.show_validation_error(
                f"ROI exceeds frame width: {x + width} > {self.frame_width}"
            )
            return False

        if y + height > self.frame_height:
            self.show_validation_error(
                f"ROI exceeds frame height: {y + height} > {self.frame_height}"
            )
            return False

        # Check minimum dimensions
        if width < 100:
            self.show_validation_error("ROI width must be at least 100 pixels")
            return False

        if height < 100:
            self.show_validation_error("ROI height must be at least 100 pixels")
            return False

        return True

    # ========================================================================
    # PRESET METHODS
    # ========================================================================

    def set_roi_full_frame(self):
        """Set ROI to cover the entire frame."""
        self.roi_x_spin.setValue(0)
        self.roi_y_spin.setValue(0)
        self.roi_width_spin.setValue(self.frame_width)
        self.roi_height_spin.setValue(self.frame_height)
        self.logger.info("ROI set to full frame")

    def set_roi_center_80(self):
        """Set ROI to centered 80% of frame."""
        width = int(self.frame_width * 0.8)
        height = int(self.frame_height * 0.8)
        x = (self.frame_width - width) // 2
        y = (self.frame_height - height) // 2

        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_width_spin.setValue(width)
        self.roi_height_spin.setValue(height)
        self.logger.info("ROI set to center 80%")

    # ========================================================================
    # SIGNAL HANDLERS
    # ========================================================================

    def on_roi_changed(self):
        """Handle ROI spinbox value changes."""
        # Ensure ROI stays within bounds
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        width = self.roi_width_spin.value()
        height = self.roi_height_spin.value()

        # Adjust if exceeding frame bounds
        if x + width > self.frame_width:
            self.roi_width_spin.blockSignals(True)
            self.roi_width_spin.setValue(self.frame_width - x)
            self.roi_width_spin.blockSignals(False)
            width = self.frame_width - x

        if y + height > self.frame_height:
            self.roi_height_spin.blockSignals(True)
            self.roi_height_spin.setValue(self.frame_height - y)
            self.roi_height_spin.blockSignals(False)
            height = self.frame_height - y

        # Apply to preview and mark as modified
        self.apply_to_preview()
        self.mark_modified()

        # Emit signal
        roi = (x, y, width, height)
        self.roi_changed.emit(roi)

    def on_zones_changed(self):
        """Handle zone slider value changes."""
        self.apply_to_preview()
        self.mark_modified()

        # Emit signal
        entry_pct = self.entry_zone_slider.value() / 100.0
        exit_pct = self.exit_zone_slider.value() / 100.0
        self.zones_changed.emit((entry_pct, exit_pct))

    def apply_to_preview(self):
        """Apply current configuration to the preview widget."""
        if not self.preview_widget:
            return

        # Get current values
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        width = self.roi_width_spin.value()
        height = self.roi_height_spin.value()

        entry_pct = self.entry_zone_slider.value() / 100.0
        exit_pct = self.exit_zone_slider.value() / 100.0

        # Apply to preview widget
        self.preview_widget.set_roi(x, y, width, height)
        self.preview_widget.set_zones(entry_pct, exit_pct)

        self.logger.debug(f"Applied to preview - ROI: ({x}, {y}, {width}, {height}), "
                          f"Zones: entry={entry_pct:.2f}, exit={exit_pct:.2f}")

    def toggle_fps(self):
        """Toggle FPS display on preview."""
        if self.preview_widget:
            current = self.preview_widget.fps_visible
            self.preview_widget.set_fps_visible(not current)

    # ========================================================================
    # CAMERA INTEGRATION
    # ========================================================================

    def set_camera(self, camera):
        """
        Set or update the camera reference.

        Args:
            camera: Camera module instance
        """
        self.camera = camera
        self.start_dimension_monitoring()

    def start_dimension_monitoring(self):
        """Start monitoring camera for frame dimensions."""
        if not self.camera:
            self.logger.warning("No camera available for dimension monitoring")
            return

        # Set up a timer to check for frames and extract dimensions
        self.dimension_timer = QTimer()
        self.dimension_timer.timeout.connect(self.check_frame_dimensions)
        self.dimension_timer.start(1000)  # Check every second

    def check_frame_dimensions(self):
        """Check if frame dimensions can be determined from camera."""
        if not self.camera or self.frame_dimensions_known:
            if hasattr(self, 'dimension_timer'):
                self.dimension_timer.stop()
            return

        try:
            # Try to get camera statistics
            stats = self.camera.get_statistics()
            camera_info = stats.get('camera', {})

            # Extract resolution
            resolution = camera_info.get('resolution', None)
            if resolution and 'x' in resolution:
                # Parse resolution string like "1920x1080"
                parts = resolution.split('x')
                if len(parts) == 2:
                    width = int(parts[0])
                    height = int(parts[1])
                    self.update_frame_dimensions(width, height)
                    self.dimension_timer.stop()
                    self.logger.info(f"✓ Frame dimensions detected: {width}×{height}")
        except Exception as e:
            self.logger.debug(f"Could not get frame dimensions yet: {e}")

    def register_preview(self):
        """Register the preview widget as a camera consumer."""
        if not self.camera or not self.preview_widget:
            self.logger.warning("Cannot register preview - camera or widget not available")
            return False

        try:
            success = self.camera.register_consumer(
                name="detector_preview",
                callback=self.preview_widget.receive_frame,
                processing_type="async",
                priority=20
            )

            if success:
                self.preview_active = True
                self.preview_status_label.setText("Preview: Active")
                self.preview_status_label.setStyleSheet("color: green; font-weight: bold;")
                self.logger.info("✓ Preview registered successfully")
                return True
            else:
                self.preview_status_label.setText("Preview: Registration Failed")
                self.preview_status_label.setStyleSheet("color: red; font-weight: bold;")
                self.logger.error("✗ Failed to register preview")
                return False

        except Exception as e:
            self.logger.error(f"Error registering preview: {e}")
            return False

    def unregister_preview(self):
        """Unregister the preview widget from camera."""
        if not self.camera or not self.preview_active:
            return

        try:
            self.camera.unregister_consumer("detector_preview")
            self.preview_active = False
            self.preview_status_label.setText("Preview: Stopped")
            self.preview_status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.logger.info("Preview unregistered")
        except Exception as e:
            self.logger.error(f"Error unregistering preview: {e}")

    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================

    def load_config(self) -> bool:
        """Load configuration from config_manager and populate UI."""
        try:
            # Get detector config
            detector_config = self.get_module_config(ModuleConfig.DETECTOR.value)
            detector_roi_config = self.get_module_config(ModuleConfig.DETECTOR_ROI.value)

            # Block signals during loading
            self.roi_x_spin.blockSignals(True)
            self.roi_y_spin.blockSignals(True)
            self.roi_width_spin.blockSignals(True)
            self.roi_height_spin.blockSignals(True)
            self.entry_zone_slider.blockSignals(True)
            self.exit_zone_slider.blockSignals(True)
            self.min_area_spin.blockSignals(True)
            self.max_area_spin.blockSignals(True)

            # Load detector parameters
            if detector_config:
                self.min_area_spin.setValue(detector_config.get('min_area', 1000))
                self.max_area_spin.setValue(detector_config.get('max_area', 50000))

                zones = detector_config.get('zones', {})
                entry_pct = int(zones.get('entry_percentage', 15))
                exit_pct = int(zones.get('exit_percentage', 15))
                self.entry_zone_slider.setValue(entry_pct)
                self.exit_zone_slider.setValue(exit_pct)

            # Load ROI configuration
            if detector_roi_config:
                self.roi_x_spin.setValue(detector_roi_config.get('x', 100))
                self.roi_y_spin.setValue(detector_roi_config.get('y', 50))
                self.roi_width_spin.setValue(detector_roi_config.get('width', 1720))
                self.roi_height_spin.setValue(detector_roi_config.get('height', 980))

            # Restore signals
            self.roi_x_spin.blockSignals(False)
            self.roi_y_spin.blockSignals(False)
            self.roi_width_spin.blockSignals(False)
            self.roi_height_spin.blockSignals(False)
            self.entry_zone_slider.blockSignals(False)
            self.exit_zone_slider.blockSignals(False)
            self.min_area_spin.blockSignals(False)
            self.max_area_spin.blockSignals(False)

            # Apply to preview
            self.apply_to_preview()

            # Clear modified flag
            self.clear_modified()

            self.logger.info("Configuration loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            return False

    def save_config(self) -> bool:
        """Save current UI values to config_manager."""
        if not self.validate():
            return False

        try:
            # Get current configuration
            config = self.get_config()

            # Save detector configuration
            self.config_manager.update_module_config(
                ModuleConfig.DETECTOR.value,
                config['detector']
            )

            # Save ROI configuration
            self.config_manager.update_module_config(
                ModuleConfig.DETECTOR_ROI.value,
                config['detector_roi']
            )

            # Clear modified flag
            self.clear_modified()

            self.logger.info("Configuration saved successfully")

            # Emit config_changed signal
            self.config_changed.emit(ModuleConfig.DETECTOR.value, config)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}", exc_info=True)
            return False

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration as a dictionary."""
        return {
            'detector': {
                'min_area': self.min_area_spin.value(),
                'max_area': self.max_area_spin.value(),
                'zones': {
                    'entry_percentage': self.entry_zone_slider.value(),
                    'exit_percentage': self.exit_zone_slider.value()
                }
            },
            'detector_roi': {
                'x': self.roi_x_spin.value(),
                'y': self.roi_y_spin.value(),
                'width': self.roi_width_spin.value(),
                'height': self.roi_height_spin.value()
            }
        }

    def validate(self) -> bool:
        """Validate current configuration."""
        # Validate ROI bounds
        if not self.validate_roi():
            return False

        # Validate detection parameters
        min_area = self.min_area_spin.value()
        max_area = self.max_area_spin.value()

        if min_area >= max_area:
            self.show_validation_error(
                f"Minimum area ({min_area}) must be less than maximum area ({max_area})"
            )
            return False

        # Validate zone percentages
        entry_pct = self.entry_zone_slider.value()
        exit_pct = self.exit_zone_slider.value()

        if entry_pct + exit_pct > 80:
            self.show_validation_error(
                f"Combined zone percentages ({entry_pct + exit_pct}%) exceed 80%. "
                "Zones may overlap."
            )
            return False

        return True

    def reset_to_defaults(self):
        """Reset all values to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Reset all detector settings to default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Reset ROI (center 80%)
            self.set_roi_center_80()

            # Reset zones
            self.entry_zone_slider.setValue(15)
            self.exit_zone_slider.setValue(15)

            # Reset detection parameters
            self.min_area_spin.setValue(1000)
            self.max_area_spin.setValue(50000)

            self.mark_modified()
            self.logger.info("Settings reset to defaults")

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def cleanup(self):
        """Clean up resources when tab is closed or destroyed."""
        self.logger.info("Cleaning up detector tab...")

        # Stop dimension monitoring
        if hasattr(self, 'dimension_timer'):
            self.dimension_timer.stop()

        # Unregister preview
        self.unregister_preview()

        self.logger.info("Detector tab cleanup complete")