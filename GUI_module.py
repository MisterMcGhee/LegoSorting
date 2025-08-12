"""
GUI_module_refactored.py - Refactored GUI with EnhancedConfigManager integration

This refactored version:
1. Uses EnhancedConfigManager instead of direct JSON operations
2. Combines Camera/ROI with live preview
3. Combines Sorting Strategy and Bin Assignment
4. Uses "within" language for hierarchical sorting
5. Dynamically updates available options
6. Shows ROI overlay on camera feed
"""

import sys
import json
import numpy as np
import cv2
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QComboBox, QSlider, QCheckBox,
    QMessageBox, QGroupBox, QGridLayout, QScrollArea, QTabWidget,
    QListWidget, QListWidgetItem, QProgressBar, QFrame, QSplitter,
    QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QRect, QSize
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush
)

# Import the enhanced config manager
from enhanced_config_manager import create_config_manager, ModuleConfig
from sorting_module import create_sorting_manager


class CameraThread(QThread):
    """Thread for handling camera operations"""
    frame_ready = pyqtSignal(np.ndarray)
    resolution_changed = pyqtSignal(int, int)  # width, height

    def __init__(self, device_id=0, width=1920, height=1080):
        super().__init__()
        self.device_id = device_id
        self.target_width = width
        self.target_height = height
        self.cap = None
        self.is_running = False

    def set_resolution(self, width, height):
        """Set target resolution"""
        self.target_width = width
        self.target_height = height

        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Get actual resolution (camera might not support requested resolution)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution_changed.emit(actual_width, actual_height)

    def run(self):
        """Main thread loop"""
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.device_id}")
            return

        # Set desired resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        # Get actual resolution (camera might not support requested resolution)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera opened at {actual_width}x{actual_height} (requested {self.target_width}x{self.target_height})")
        self.resolution_changed.emit(actual_width, actual_height)

        self.is_running = True

        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                print("Warning: Failed to read frame from camera")

            # Small delay to prevent CPU overload
            self.msleep(30)  # ~30 FPS

    def stop(self):
        """Stop the camera thread"""
        self.is_running = False
        self.wait()  # Wait for thread to finish

        if self.cap:
            self.cap.release()
            self.cap = None


class CameraPreviewWidget(QLabel):
    """Widget for displaying camera feed with ROI overlay"""

    roi_changed = pyqtSignal(int, int, int, int)  # x, y, w, h

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setMaximumSize(960, 720)
        self.setScaledContents(False)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #222;")

        # ROI parameters (in camera coordinates)
        self.roi_x = 125
        self.roi_y = 200
        self.roi_w = 1550
        self.roi_h = 500

        # Camera frame dimensions
        self.frame_width = 1920
        self.frame_height = 1080

        # Display scaling
        self.scale_factor = 1.0
        self.display_width = 640
        self.display_height = 480

        # Interaction state
        self.dragging = False
        self.drag_start = None
        self.resize_handle = None

        self.setMouseTracking(True)

        # Show placeholder when no camera
        self.show_placeholder()

    def show_placeholder(self):
        """Show placeholder text when no camera feed"""
        placeholder = QPixmap(self.size())
        placeholder.fill(QColor(40, 40, 40))

        painter = QPainter(placeholder)
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        font = QFont()
        font.setPixelSize(16)
        painter.setFont(font)
        painter.drawText(
            placeholder.rect(),
            Qt.AlignCenter,
            "Camera Preview\nClick 'Start Camera' to begin"
        )
        painter.end()

        self.setPixmap(placeholder)

    def set_frame_size(self, width, height):
        """Update the actual camera frame dimensions"""
        self.frame_width = width
        self.frame_height = height
        self.constrain_roi()

    def set_roi(self, x, y, w, h):
        """Set ROI parameters and constrain to frame"""
        self.roi_x = x
        self.roi_y = y
        self.roi_w = w
        self.roi_h = h
        self.constrain_roi()
        self.update()

    def constrain_roi(self):
        """Ensure ROI stays within camera frame bounds"""
        # Constrain position
        self.roi_x = max(0, min(self.roi_x, self.frame_width - self.roi_w))
        self.roi_y = max(0, min(self.roi_y, self.frame_height - self.roi_h))

        # Constrain size
        self.roi_w = min(self.roi_w, self.frame_width - self.roi_x)
        self.roi_h = min(self.roi_h, self.frame_height - self.roi_y)

        # Ensure minimum size
        self.roi_w = max(self.roi_w, 100)
        self.roi_h = max(self.roi_h, 100)

    def update_frame(self, frame):
        """Update displayed frame"""
        if frame is None:
            return

        # Update frame dimensions
        h, w = frame.shape[:2]
        if w != self.frame_width or h != self.frame_height:
            self.set_frame_size(w, h)

        # Convert to RGB and create QImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w

        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Calculate scale to fit the entire frame in the widget
        widget_size = self.size()
        self.scale_factor = min(
            widget_size.width() / w,
            widget_size.height() / h
        )

        # Scale the image to fit
        self.display_width = int(w * self.scale_factor)
        self.display_height = int(h * self.scale_factor)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.display_width,
            self.display_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Create final pixmap centered in widget
        final_pixmap = QPixmap(widget_size)
        final_pixmap.fill(QColor(30, 30, 30))

        # Calculate position to center the scaled frame
        x_offset = (widget_size.width() - self.display_width) // 2
        y_offset = (widget_size.height() - self.display_height) // 2

        # Draw the scaled frame
        painter = QPainter(final_pixmap)
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)

        # Draw ROI overlay with offset
        self.draw_roi_overlay(painter, x_offset, y_offset)

        painter.end()

        self.setPixmap(final_pixmap)

    def draw_roi_overlay(self, painter, x_offset, y_offset):
        """Draw ROI rectangle overlay"""
        # Scale ROI to display coordinates
        x = int(self.roi_x * self.scale_factor) + x_offset
        y = int(self.roi_y * self.scale_factor) + y_offset
        w = int(self.roi_w * self.scale_factor)
        h = int(self.roi_h * self.scale_factor)

        # Draw semi-transparent overlay outside ROI (within frame bounds)
        frame_x = x_offset
        frame_y = y_offset
        frame_w = self.display_width
        frame_h = self.display_height

        # Darken areas outside ROI but inside frame
        overlay_color = QColor(0, 0, 0, 120)

        # Top
        if y > frame_y:
            painter.fillRect(frame_x, frame_y, frame_w, y - frame_y, overlay_color)

        # Left
        if x > frame_x:
            painter.fillRect(frame_x, y, x - frame_x, h, overlay_color)

        # Right
        if x + w < frame_x + frame_w:
            painter.fillRect(x + w, y, frame_x + frame_w - (x + w), h, overlay_color)

        # Bottom
        if y + h < frame_y + frame_h:
            painter.fillRect(frame_x, y + h, frame_w, frame_y + frame_h - (y + h), overlay_color)

        # Draw ROI border
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        painter.drawRect(x, y, w, h)

        # Draw corner handles
        handle_size = 8
        handle_color = QColor(0, 255, 0)
        painter.fillRect(x - handle_size // 2, y - handle_size // 2, handle_size, handle_size, handle_color)
        painter.fillRect(x + w - handle_size // 2, y - handle_size // 2, handle_size, handle_size, handle_color)
        painter.fillRect(x - handle_size // 2, y + h - handle_size // 2, handle_size, handle_size, handle_color)
        painter.fillRect(x + w - handle_size // 2, y + h - handle_size // 2, handle_size, handle_size, handle_color)

        # Draw info labels
        font = QFont()
        font.setPixelSize(12)
        painter.setFont(font)

        # ROI size and position
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        info_text = f"ROI: {self.roi_w}×{self.roi_h} @ ({self.roi_x}, {self.roi_y})"
        painter.drawText(x + 5, y - 5, info_text)

        # Frame size in corner
        frame_info = f"Frame: {self.frame_width}×{self.frame_height}"
        painter.drawText(frame_x + 5, frame_y + 15, frame_info)


class ConfigurationScreen(QWidget):
    """Main configuration screen with all settings"""

    config_confirmed = pyqtSignal()

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager

        # Initialize category database from sorting manager
        self.category_database = None
        self.init_category_database()

        # Camera thread
        self.camera_thread = None

        # Initialize actual camera frame dimensions (will be updated when camera starts)
        self.actual_frame_width = 1920
        self.actual_frame_height = 1080

        self.init_ui()
        self.load_settings()

    def init_category_database(self):
        """Initialize category database from sorting manager"""
        try:
            # Create temporary sorting manager to get category data
            temp_manager = create_sorting_manager(self.config_manager)

            # Build category database structure
            self.category_database = {
                "primary": set(),
                "primary_to_secondary": {},  # primary -> set of secondaries
                "secondary_to_tertiary": {}  # (primary, secondary) -> set of tertiaries
            }

            # Get all categories from the sorting manager's data
            categories_data = temp_manager.strategy.categories_data

            for element_id, info in categories_data.items():
                primary = info.get("primary_category", "")
                secondary = info.get("secondary_category", "")
                tertiary = info.get("tertiary_category", "")

                if primary:
                    self.category_database["primary"].add(primary)

                    if secondary:
                        if primary not in self.category_database["primary_to_secondary"]:
                            self.category_database["primary_to_secondary"][primary] = set()
                        self.category_database["primary_to_secondary"][primary].add(secondary)

                        if tertiary:
                            key = (primary, secondary)
                            if key not in self.category_database["secondary_to_tertiary"]:
                                self.category_database["secondary_to_tertiary"][key] = set()
                            self.category_database["secondary_to_tertiary"][key].add(tertiary)

        except Exception as e:
            print(f"Error initializing category database: {e}")
            self.category_database = None

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()

        # Create tab widget for different settings groups
        self.tab_widget = QTabWidget()

        # Add tabs - Camera/ROI combined, Sorting/Bins combined, Servo separate
        self.tab_widget.addTab(self.create_camera_roi_tab(), "Camera & ROI")
        self.tab_widget.addTab(self.create_sorting_bins_tab(), "Sorting Strategy & Bins")
        self.tab_widget.addTab(self.create_servo_tab(), "Servo Control")

        main_layout.addWidget(self.tab_widget)

        # Bottom button bar
        button_layout = QHBoxLayout()

        self.validate_btn = QPushButton("Validate Configuration")
        self.validate_btn.clicked.connect(self.validate_configuration)
        button_layout.addWidget(self.validate_btn)

        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_btn)

        self.start_btn = QPushButton("Start Sorting System")
        self.start_btn.clicked.connect(self.start_system)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.start_btn)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def create_camera_roi_tab(self):
        """Create combined camera and ROI settings tab with live preview"""
        widget = QWidget()
        main_layout = QHBoxLayout()

        # Left side - Camera preview
        preview_layout = QVBoxLayout()

        preview_label = QLabel("Camera Preview")
        preview_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        preview_layout.addWidget(preview_label)

        self.camera_preview = CameraPreviewWidget()
        self.camera_preview.roi_changed.connect(self.on_roi_dragged)
        preview_layout.addWidget(self.camera_preview)

        # Camera control buttons
        cam_controls = QHBoxLayout()

        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        cam_controls.addWidget(self.start_camera_btn)

        self.stop_camera_btn = QPushButton("Stop Camera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        cam_controls.addWidget(self.stop_camera_btn)

        preview_layout.addLayout(cam_controls)

        main_layout.addLayout(preview_layout, 2)  # Give more space to preview

        # Right side - Settings
        settings_layout = QVBoxLayout()

        # Camera settings
        cam_group = QGroupBox("Camera Settings")
        cam_layout = QGridLayout()

        cam_layout.addWidget(QLabel("Camera Device:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0 - Default", "1 - USB Camera", "2 - External"])
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        cam_layout.addWidget(self.camera_combo, 0, 1)

        cam_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1920x1080", "1280x720", "640x480"])
        self.resolution_combo.currentTextChanged.connect(self.on_resolution_changed)
        cam_layout.addWidget(self.resolution_combo, 1, 1)

        cam_layout.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        cam_layout.addWidget(self.fps_spin, 2, 1)

        cam_layout.addWidget(QLabel("Auto Exposure:"), 3, 0)
        self.auto_exposure_check = QCheckBox()
        self.auto_exposure_check.setChecked(True)
        cam_layout.addWidget(self.auto_exposure_check, 3, 1)

        # Camera status
        cam_layout.addWidget(QLabel("Status:"), 4, 0)
        self.camera_status_label = QLabel("Not connected")
        self.camera_status_label.setStyleSheet("color: #666;")
        cam_layout.addWidget(self.camera_status_label, 4, 1)

        cam_group.setLayout(cam_layout)
        settings_layout.addWidget(cam_group)

        # ROI settings
        roi_group = QGroupBox("Region of Interest (ROI)")
        roi_layout = QGridLayout()

        roi_layout.addWidget(QLabel("X Position:"), 0, 0)
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 3840)  # Support up to 4K width
        self.roi_x_spin.valueChanged.connect(self.on_roi_changed)
        roi_layout.addWidget(self.roi_x_spin, 0, 1)

        roi_layout.addWidget(QLabel("Y Position:"), 1, 0)
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 2160)  # Support up to 4K height
        self.roi_y_spin.valueChanged.connect(self.on_roi_changed)
        roi_layout.addWidget(self.roi_y_spin, 1, 1)

        roi_layout.addWidget(QLabel("Width:"), 2, 0)
        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(100, 3840)  # Support up to 4K width
        self.roi_w_spin.valueChanged.connect(self.on_roi_changed)
        roi_layout.addWidget(self.roi_w_spin, 2, 1)

        roi_layout.addWidget(QLabel("Height:"), 3, 0)
        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(100, 2160)  # Support up to 4K height
        self.roi_h_spin.valueChanged.connect(self.on_roi_changed)
        roi_layout.addWidget(self.roi_h_spin, 3, 1)

        # ROI quick actions
        roi_layout.addWidget(QLabel("Quick Actions:"), 4, 0)
        roi_actions = QHBoxLayout()

        self.center_roi_btn = QPushButton("Center")
        self.center_roi_btn.clicked.connect(self.center_roi)
        roi_actions.addWidget(self.center_roi_btn)

        self.maximize_roi_btn = QPushButton("Maximize")
        self.maximize_roi_btn.clicked.connect(self.maximize_roi)
        roi_actions.addWidget(self.maximize_roi_btn)

        roi_layout.addLayout(roi_actions, 4, 1)

        roi_group.setLayout(roi_layout)
        settings_layout.addWidget(roi_group)

        # Detection settings
        detect_group = QGroupBox("Detection Parameters")
        detect_layout = QGridLayout()

        detect_layout.addWidget(QLabel("Min Area:"), 0, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 10000)
        detect_layout.addWidget(self.min_area_spin, 0, 1)

        detect_layout.addWidget(QLabel("Max Area:"), 1, 0)
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1000, 100000)
        detect_layout.addWidget(self.max_area_spin, 1, 1)

        detect_group.setLayout(detect_layout)
        settings_layout.addWidget(detect_group)

        settings_layout.addStretch()
        main_layout.addLayout(settings_layout, 1)

        widget.setLayout(main_layout)
        return widget

    def create_sorting_bins_tab(self):
        """Create combined sorting strategy and bin assignment tab"""
        widget = QWidget()
        main_layout = QHBoxLayout()

        # Left side - Sorting Strategy
        strategy_layout = QVBoxLayout()

        strategy_group = QGroupBox("Sorting Strategy")
        strat_layout = QGridLayout()

        strat_layout.addWidget(QLabel("Strategy:"), 0, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["primary", "secondary", "tertiary"])
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strat_layout.addWidget(self.strategy_combo, 0, 1)

        # "Within" selectors for hierarchical sorting
        strat_layout.addWidget(QLabel("Within Primary:"), 1, 0)
        self.primary_combo = QComboBox()
        self.primary_combo.setEnabled(False)  # Disabled for primary strategy
        self.primary_combo.currentTextChanged.connect(self.on_primary_changed)
        strat_layout.addWidget(self.primary_combo, 1, 1)

        strat_layout.addWidget(QLabel("Within Secondary:"), 2, 0)
        self.secondary_combo = QComboBox()
        self.secondary_combo.setEnabled(False)  # Disabled for primary/secondary strategies
        self.secondary_combo.currentTextChanged.connect(self.on_secondary_changed)
        strat_layout.addWidget(self.secondary_combo, 2, 1)

        # Max bins and overflow
        strat_layout.addWidget(QLabel("Max Bins:"), 3, 0)
        self.max_bins_spin = QSpinBox()
        self.max_bins_spin.setRange(1, 9)
        self.max_bins_spin.setValue(7)
        self.max_bins_spin.valueChanged.connect(self.on_max_bins_changed)
        strat_layout.addWidget(self.max_bins_spin, 3, 1)

        strat_layout.addWidget(QLabel("Overflow Bin:"), 4, 0)
        overflow_label = QLabel("Bin 0 (Fixed)")
        overflow_label.setStyleSheet("color: #666;")
        strat_layout.addWidget(overflow_label, 4, 1)

        # Strategy description
        self.strategy_desc = QLabel("")
        self.strategy_desc.setWordWrap(True)
        self.strategy_desc.setStyleSheet("color: #0066cc; font-style: italic;")
        strat_layout.addWidget(self.strategy_desc, 5, 0, 1, 2)

        strategy_group.setLayout(strat_layout)
        strategy_layout.addWidget(strategy_group)

        strategy_layout.addStretch()
        main_layout.addLayout(strategy_layout, 1)

        # Right side - Bin Assignments
        assignment_layout = QVBoxLayout()

        assignment_group = QGroupBox("Bin Pre-Assignments")
        assign_layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "Pre-assign categories to specific bins. Unassigned categories "
            "will be dynamically assigned during sorting."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 11px; color: #666;")
        assign_layout.addWidget(instructions)

        # Bin assignment grid
        grid_widget = QWidget()
        grid_layout = QGridLayout()

        # Headers
        header_bin = QLabel("Bin")
        header_bin.setStyleSheet("font-weight: bold;")
        grid_layout.addWidget(header_bin, 0, 0)

        header_cat = QLabel("Category")
        header_cat.setStyleSheet("font-weight: bold;")
        grid_layout.addWidget(header_cat, 0, 1)

        # Create dropdowns for bins
        self.bin_combos = {}
        for i in range(1, 10):  # Max 9 bins
            label = QLabel(f"Bin {i}:")
            grid_layout.addWidget(label, i, 0)

            combo = QComboBox()
            combo.addItem("(Unassigned)")
            combo.currentTextChanged.connect(self.on_assignment_changed)
            grid_layout.addWidget(combo, i, 1)

            self.bin_combos[i] = combo

            # Hide bins beyond max_bins initially
            if i > 7:  # Default max_bins is 7
                label.setVisible(False)
                combo.setVisible(False)

        grid_widget.setLayout(grid_layout)

        # Scroll area for many bins
        scroll = QScrollArea()
        scroll.setWidget(grid_widget)
        scroll.setWidgetResizable(True)
        assign_layout.addWidget(scroll)

        # Clear button
        clear_btn = QPushButton("Clear All Assignments")
        clear_btn.clicked.connect(self.clear_all_assignments)
        assign_layout.addWidget(clear_btn)

        assignment_group.setLayout(assign_layout)
        assignment_layout.addWidget(assignment_group)

        main_layout.addLayout(assignment_layout, 1)

        widget.setLayout(main_layout)
        return widget

    def create_servo_tab(self):
        """Create servo control settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Servo settings
        servo_group = QGroupBox("Servo Configuration")
        servo_layout = QGridLayout()

        servo_layout.addWidget(QLabel("Min Pulse (μs):"), 0, 0)
        self.min_pulse_spin = QSpinBox()
        self.min_pulse_spin.setRange(100, 2000)
        self.min_pulse_spin.setValue(500)
        servo_layout.addWidget(self.min_pulse_spin, 0, 1)

        servo_layout.addWidget(QLabel("Max Pulse (μs):"), 1, 0)
        self.max_pulse_spin = QSpinBox()
        self.max_pulse_spin.setRange(1000, 3000)
        self.max_pulse_spin.setValue(2500)
        servo_layout.addWidget(self.max_pulse_spin, 1, 1)

        servo_layout.addWidget(QLabel("Default Angle:"), 2, 0)
        self.default_angle_spin = QSpinBox()
        self.default_angle_spin.setRange(0, 180)
        self.default_angle_spin.setValue(90)
        servo_layout.addWidget(self.default_angle_spin, 2, 1)

        servo_layout.addWidget(QLabel("Move Duration (ms):"), 3, 0)
        self.move_duration_spin = QSpinBox()
        self.move_duration_spin.setRange(100, 2000)
        self.move_duration_spin.setValue(500)
        servo_layout.addWidget(self.move_duration_spin, 3, 1)

        servo_group.setLayout(servo_layout)
        layout.addWidget(servo_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def start_camera(self):
        """Start camera preview"""
        if self.camera_thread is None or not self.camera_thread.isRunning():
            device_id = self.camera_combo.currentIndex()

            # Update status
            self.camera_status_label.setText("Connecting...")
            self.camera_status_label.setStyleSheet("color: blue;")

            # Get resolution from settings
            resolution = self.resolution_combo.currentText()
            width, height = 1920, 1080  # Default
            if "x" in resolution:
                w_str, h_str = resolution.split("x")
                width = int(w_str)
                height = int(h_str)

            self.camera_thread = CameraThread(device_id, width, height)
            self.camera_thread.frame_ready.connect(self.on_frame_received)
            self.camera_thread.resolution_changed.connect(self.on_camera_resolution_changed)
            self.camera_thread.start()

            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)

    def stop_camera(self):
        """Stop camera preview"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)

        # Update status
        self.camera_status_label.setText("Not connected")
        self.camera_status_label.setStyleSheet("color: #666;")

        # Show placeholder
        self.camera_preview.show_placeholder()

    def on_camera_changed(self):
        """Handle camera device change"""
        # Restart camera if it's running
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
            self.start_camera()

    def on_resolution_changed(self):
        """Handle resolution change"""
        # Update camera if running
        if self.camera_thread and self.camera_thread.isRunning():
            resolution = self.resolution_combo.currentText()
            if "x" in resolution:
                width, height = resolution.split("x")
                self.camera_thread.set_resolution(int(width), int(height))

    def on_camera_resolution_changed(self, width, height):
        """Handle actual camera resolution from camera thread"""
        print(f"Camera resolution: {width}x{height}")

        # Store actual frame dimensions
        self.actual_frame_width = width
        self.actual_frame_height = height

        # Update status label
        requested = self.resolution_combo.currentText()
        actual = f"{width}x{height}"
        if requested == actual:
            self.camera_status_label.setText(f"Connected @ {actual}")
            self.camera_status_label.setStyleSheet("color: green;")
        else:
            self.camera_status_label.setText(f"Connected @ {actual} (requested {requested})")
            self.camera_status_label.setStyleSheet("color: orange;")

        # Update preview widget frame size
        self.camera_preview.set_frame_size(width, height)

        # Update ROI spinbox limits
        self.roi_x_spin.setMaximum(max(0, width - 100))
        self.roi_y_spin.setMaximum(max(0, height - 100))
        self.roi_w_spin.setMaximum(width)
        self.roi_h_spin.setMaximum(height)

        # Constrain current ROI values
        self.constrain_roi_values(width, height)

    def constrain_roi_values(self, frame_width, frame_height):
        """Constrain ROI values to frame bounds"""
        # Get current values
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        w = self.roi_w_spin.value()
        h = self.roi_h_spin.value()

        # Constrain to frame
        x = min(x, max(0, frame_width - w))
        y = min(y, max(0, frame_height - h))
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)

        # Update spinboxes
        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_w_spin.setValue(w)
        self.roi_h_spin.setValue(h)

        # Update preview
        self.camera_preview.set_roi(x, y, w, h)

    def on_frame_received(self, frame):
        """Handle new camera frame"""
        self.camera_preview.update_frame(frame)

    def on_roi_changed(self):
        """Handle ROI spinbox changes"""
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        w = self.roi_w_spin.value()
        h = self.roi_h_spin.value()

        # Ensure ROI stays within frame bounds
        frame_w = self.camera_preview.frame_width
        frame_h = self.camera_preview.frame_height

        # Constrain position
        if x + w > frame_w:
            x = max(0, frame_w - w)
            self.roi_x_spin.setValue(x)

        if y + h > frame_h:
            y = max(0, frame_h - h)
            self.roi_y_spin.setValue(y)

        self.camera_preview.set_roi(x, y, w, h)

    def on_roi_dragged(self, x, y, w, h):
        """Handle ROI drag in preview"""
        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_w_spin.setValue(w)
        self.roi_h_spin.setValue(h)

    def center_roi(self):
        """Center ROI in frame"""
        # Get current ROI size
        w = self.roi_w_spin.value()
        h = self.roi_h_spin.value()

        # Get actual frame size from preview
        frame_w = self.camera_preview.frame_width
        frame_h = self.camera_preview.frame_height

        # Calculate center position
        x = max(0, (frame_w - w) // 2)
        y = max(0, (frame_h - h) // 2)

        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.on_roi_changed()

    def maximize_roi(self):
        """Maximize ROI with small margin"""
        margin = 50

        # Get actual frame size from preview
        frame_w = self.camera_preview.frame_width
        frame_h = self.camera_preview.frame_height

        # Set to nearly full frame
        x = min(margin, frame_w // 2)
        y = min(margin, frame_h // 2)
        w = max(100, frame_w - 2 * x)
        h = max(100, frame_h - 2 * y)

        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_w_spin.setValue(w)
        self.roi_h_spin.setValue(h)
        self.on_roi_changed()

    def on_strategy_changed(self, strategy):
        """Handle strategy change - update UI and available options"""
        # Update UI based on strategy
        if strategy == "primary":
            # No hierarchy needed for primary
            self.primary_combo.setEnabled(False)
            self.secondary_combo.setEnabled(False)
            self.primary_combo.clear()
            self.secondary_combo.clear()
            self.strategy_desc.setText("Sorting by primary categories (Basic, Technic, etc.)")

        elif strategy == "secondary":
            # Need to select primary category
            self.primary_combo.setEnabled(True)
            self.secondary_combo.setEnabled(False)
            self.secondary_combo.clear()
            self.update_primary_options()
            self.strategy_desc.setText("Sorting by secondary categories within a primary category")

        elif strategy == "tertiary":
            # Need to select both primary and secondary
            self.primary_combo.setEnabled(True)
            self.secondary_combo.setEnabled(True)
            self.update_primary_options()
            self.strategy_desc.setText("Sorting by tertiary categories within primary/secondary")

        # Update bin assignment options
        self.update_bin_assignment_options()

    def on_primary_changed(self, primary):
        """Handle primary category change"""
        strategy = self.strategy_combo.currentText()

        if strategy in ["secondary", "tertiary"]:
            if strategy == "tertiary":
                # Update secondary options for tertiary strategy
                self.update_secondary_options()

            # Update description with selected category
            if primary:
                if strategy == "secondary":
                    self.strategy_desc.setText(f"Sorting secondary categories within '{primary}'")
                else:
                    self.strategy_desc.setText(f"Sorting tertiary categories within '{primary}' → ...")

        # Update bin assignments
        self.update_bin_assignment_options()

    def on_secondary_changed(self, secondary):
        """Handle secondary category change"""
        if self.strategy_combo.currentText() == "tertiary":
            primary = self.primary_combo.currentText()
            if primary and secondary:
                self.strategy_desc.setText(
                    f"Sorting tertiary categories within '{primary}' → '{secondary}'"
                )

        # Update bin assignments
        self.update_bin_assignment_options()

    def on_max_bins_changed(self, value):
        """Handle max bins change - show/hide bin dropdowns"""
        for i in range(1, 10):
            if i in self.bin_combos:
                # Show bins up to max_bins, hide others
                visible = i <= value

                # Find the label for this bin
                grid_layout = self.bin_combos[i].parent().layout()
                for row in range(grid_layout.rowCount()):
                    item = grid_layout.itemAtPosition(row, 0)
                    if item and item.widget():
                        label_widget = item.widget()
                        if isinstance(label_widget, QLabel) and f"Bin {i}:" in label_widget.text():
                            label_widget.setVisible(visible)
                            break

                self.bin_combos[i].setVisible(visible)

    def update_primary_options(self):
        """Update primary category dropdown options"""
        if not self.category_database:
            return

        self.primary_combo.blockSignals(True)
        self.primary_combo.clear()
        self.primary_combo.addItem("")  # Empty option

        primaries = sorted(self.category_database["primary"])
        self.primary_combo.addItems(primaries)

        self.primary_combo.blockSignals(False)

    def update_secondary_options(self):
        """Update secondary category dropdown based on selected primary"""
        if not self.category_database:
            return

        primary = self.primary_combo.currentText()

        self.secondary_combo.blockSignals(True)
        self.secondary_combo.clear()

        if primary and primary in self.category_database["primary_to_secondary"]:
            self.secondary_combo.addItem("")  # Empty option
            secondaries = sorted(self.category_database["primary_to_secondary"][primary])
            self.secondary_combo.addItems(secondaries)

        self.secondary_combo.blockSignals(False)

    def update_bin_assignment_options(self):
        """Update available categories in bin assignment dropdowns"""
        if not self.category_database:
            return

        strategy = self.strategy_combo.currentText()
        primary = self.primary_combo.currentText()
        secondary = self.secondary_combo.currentText()

        # Get available categories based on current strategy
        available_categories = set()

        if strategy == "primary":
            available_categories = self.category_database["primary"]

        elif strategy == "secondary" and primary:
            available_categories = self.category_database["primary_to_secondary"].get(primary, set())

        elif strategy == "tertiary" and primary and secondary:
            key = (primary, secondary)
            available_categories = self.category_database["secondary_to_tertiary"].get(key, set())

        # Update each visible bin combo
        max_bins = self.max_bins_spin.value()
        for bin_num in range(1, max_bins + 1):
            if bin_num in self.bin_combos:
                combo = self.bin_combos[bin_num]
                current_selection = combo.currentText()

                combo.blockSignals(True)
                combo.clear()
                combo.addItem("(Unassigned)")

                for category in sorted(available_categories):
                    combo.addItem(category)

                # Restore selection if still valid
                if current_selection in available_categories or current_selection == "(Unassigned)":
                    combo.setCurrentText(current_selection)
                else:
                    combo.setCurrentText("(Unassigned)")

                combo.blockSignals(False)

    def clear_all_assignments(self):
        """Clear all bin assignments"""
        for combo in self.bin_combos.values():
            combo.setCurrentText("(Unassigned)")

    def on_assignment_changed(self):
        """Handle when a bin assignment changes - check for duplicates"""
        assigned_categories = {}

        for bin_num, combo in self.bin_combos.items():
            if not combo.isVisible():
                continue

            category = combo.currentText()
            if category != "(Unassigned)":
                if category in assigned_categories:
                    # Duplicate found - clear this assignment
                    combo.blockSignals(True)
                    combo.setCurrentText("(Unassigned)")
                    combo.blockSignals(False)

                    QMessageBox.warning(
                        self,
                        "Duplicate Assignment",
                        f"Category '{category}' is already assigned to Bin {assigned_categories[category]}."
                    )
                else:
                    assigned_categories[category] = bin_num

    def load_settings(self):
        """Load settings from config manager"""
        # Camera settings
        camera_config = self.config_manager.get_module_config(ModuleConfig.CAMERA.value)
        self.camera_combo.setCurrentIndex(camera_config.get("device_id", 0))

        width = camera_config.get("width", 1920)
        height = camera_config.get("height", 1080)
        resolution_text = f"{width}x{height}"
        index = self.resolution_combo.findText(resolution_text)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)

        self.fps_spin.setValue(camera_config.get("fps", 30))
        self.auto_exposure_check.setChecked(camera_config.get("auto_exposure", True))

        # Set camera preview frame size even before camera starts
        self.camera_preview.set_frame_size(width, height)

        # Update ROI spinbox limits based on configured resolution
        self.roi_x_spin.setMaximum(max(0, width - 100))
        self.roi_y_spin.setMaximum(max(0, height - 100))
        self.roi_w_spin.setMaximum(width)
        self.roi_h_spin.setMaximum(height)

        # ROI settings
        roi_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR_ROI.value)
        roi_x = roi_config.get("x", 125)
        roi_y = roi_config.get("y", 200)
        roi_w = roi_config.get("w", 1550)
        roi_h = roi_config.get("h", 500)

        # Constrain ROI to frame bounds
        roi_x = min(roi_x, max(0, width - roi_w))
        roi_y = min(roi_y, max(0, height - roi_h))
        roi_w = min(roi_w, width - roi_x)
        roi_h = min(roi_h, height - roi_y)

        self.roi_x_spin.setValue(roi_x)
        self.roi_y_spin.setValue(roi_y)
        self.roi_w_spin.setValue(roi_w)
        self.roi_h_spin.setValue(roi_h)

        # Update camera preview ROI
        self.camera_preview.set_roi(roi_x, roi_y, roi_w, roi_h)

        # Detection settings
        detector_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR.value)
        self.min_area_spin.setValue(detector_config.get("min_area", 1000))
        self.max_area_spin.setValue(detector_config.get("max_area", 50000))

        # Sorting settings
        sorting_config = self.config_manager.get_module_config(ModuleConfig.SORTING.value)
        strategy = sorting_config.get("strategy", "primary")

        # Set max bins first
        self.max_bins_spin.setValue(sorting_config.get("max_bins", 7))

        # Set strategy (this will trigger updates)
        index = self.strategy_combo.findText(strategy)
        if index >= 0:
            self.strategy_combo.setCurrentIndex(index)

        # Set target categories after strategy is set
        if strategy in ["secondary", "tertiary"]:
            primary = sorting_config.get("target_primary_category", "")
            if primary:
                index = self.primary_combo.findText(primary)
                if index >= 0:
                    self.primary_combo.setCurrentIndex(index)

        if strategy == "tertiary":
            secondary = sorting_config.get("target_secondary_category", "")
            if secondary:
                index = self.secondary_combo.findText(secondary)
                if index >= 0:
                    self.secondary_combo.setCurrentIndex(index)

        # Load pre-assignments
        pre_assignments = sorting_config.get("pre_assignments", {})
        for category, bin_num in pre_assignments.items():
            if bin_num in self.bin_combos and self.bin_combos[bin_num].isVisible():
                combo = self.bin_combos[bin_num]
                index = combo.findText(category)
                if index >= 0:
                    combo.setCurrentIndex(index)

        # Servo settings
        servo_config = self.config_manager.get_module_config(ModuleConfig.SERVO.value)
        self.min_pulse_spin.setValue(servo_config.get("min_pulse", 500))
        self.max_pulse_spin.setValue(servo_config.get("max_pulse", 2500))
        self.default_angle_spin.setValue(servo_config.get("default_angle", 90))
        self.move_duration_spin.setValue(servo_config.get("move_duration", 500))

    def save_settings(self):
        """Save settings to config manager"""
        # Camera settings
        camera_updates = {
            "device_id": self.camera_combo.currentIndex(),
            "fps": self.fps_spin.value(),
            "auto_exposure": self.auto_exposure_check.isChecked()
        }

        # Parse resolution
        resolution = self.resolution_combo.currentText()
        if "x" in resolution:
            width, height = resolution.split("x")
            camera_updates["width"] = int(width)
            camera_updates["height"] = int(height)

        self.config_manager.update_module_config(ModuleConfig.CAMERA.value, camera_updates)

        # ROI settings
        roi_updates = {
            "x": self.roi_x_spin.value(),
            "y": self.roi_y_spin.value(),
            "w": self.roi_w_spin.value(),
            "h": self.roi_h_spin.value()
        }
        self.config_manager.update_module_config(ModuleConfig.DETECTOR_ROI.value, roi_updates)

        # Detection settings
        detector_updates = {
            "min_area": self.min_area_spin.value(),
            "max_area": self.max_area_spin.value()
        }
        self.config_manager.update_module_config(ModuleConfig.DETECTOR.value, detector_updates)

        # Sorting settings
        sorting_updates = {
            "strategy": self.strategy_combo.currentText(),
            "max_bins": self.max_bins_spin.value(),
            "overflow_bin": 0
        }

        # Add target categories if needed
        strategy = self.strategy_combo.currentText()
        if strategy in ["secondary", "tertiary"]:
            sorting_updates["target_primary_category"] = self.primary_combo.currentText()
        else:
            sorting_updates["target_primary_category"] = ""

        if strategy == "tertiary":
            sorting_updates["target_secondary_category"] = self.secondary_combo.currentText()
        else:
            sorting_updates["target_secondary_category"] = ""

        # Collect pre-assignments (only from visible bins)
        pre_assignments = {}
        max_bins = self.max_bins_spin.value()
        for bin_num in range(1, max_bins + 1):
            if bin_num in self.bin_combos:
                combo = self.bin_combos[bin_num]
                category = combo.currentText()
                if category != "(Unassigned)":
                    pre_assignments[category] = bin_num

        sorting_updates["pre_assignments"] = pre_assignments

        self.config_manager.update_module_config(ModuleConfig.SORTING.value, sorting_updates)

        # Servo settings
        servo_updates = {
            "min_pulse": self.min_pulse_spin.value(),
            "max_pulse": self.max_pulse_spin.value(),
            "default_angle": self.default_angle_spin.value(),
            "move_duration": self.move_duration_spin.value()
        }
        self.config_manager.update_module_config(ModuleConfig.SERVO.value, servo_updates)

        # Save to file
        self.config_manager.save_config()

        QMessageBox.information(self, "Success", "Configuration saved successfully!")

    def validate_configuration(self):
        """Validate the current configuration"""
        # Get validation report
        report = self.config_manager.get_validation_report()

        if report["valid"]:
            QMessageBox.information(
                self,
                "Configuration Valid",
                "All configuration settings are valid."
            )
        else:
            # Build error message
            error_msg = "Configuration validation failed:\n\n"

            for module, result in report["modules"].items():
                if not result["valid"]:
                    error_msg += f"{module}:\n"
                    for error in result["errors"]:
                        error_msg += f"  • {error}\n"
                    error_msg += "\n"

            QMessageBox.warning(
                self,
                "Configuration Invalid",
                error_msg
            )

    def start_system(self):
        """Start the sorting system if configuration is valid"""
        # First save current settings
        self.save_settings()

        # Validate configuration
        report = self.config_manager.get_validation_report()

        if not report["valid"]:
            # Show validation errors
            error_msg = "Cannot start system - configuration is invalid:\n\n"

            for module, result in report["modules"].items():
                if not result["valid"]:
                    error_msg += f"{module}:\n"
                    for error in result["errors"]:
                        error_msg += f"  • {error}\n"
                    error_msg += "\n"

            QMessageBox.critical(
                self,
                "Cannot Start System",
                error_msg
            )
            return

        # Additional validation for sorting strategy
        strategy = self.strategy_combo.currentText()
        if strategy == "secondary" and not self.primary_combo.currentText():
            QMessageBox.critical(
                self,
                "Invalid Configuration",
                "Secondary sorting requires selecting a primary category to sort within."
            )
            return

        if strategy == "tertiary":
            if not self.primary_combo.currentText() or not self.secondary_combo.currentText():
                QMessageBox.critical(
                    self,
                    "Invalid Configuration",
                    "Tertiary sorting requires selecting both primary and secondary categories to sort within."
                )
                return

        # Stop camera if running
        if self.camera_thread:
            self.stop_camera()

        # Configuration is valid - emit signal to start system
        self.config_confirmed.emit()


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Create config manager
        self.config_manager = create_config_manager()

        self.setWindowTitle("LEGO Sorting System - Configuration")
        self.setGeometry(100, 100, 1200, 700)

        # Create and set configuration screen
        self.config_screen = ConfigurationScreen(self.config_manager)
        self.config_screen.config_confirmed.connect(self.on_config_confirmed)
        self.setCentralWidget(self.config_screen)

    def on_config_confirmed(self):
        """Handle configuration confirmation"""
        reply = QMessageBox.question(
            self,
            "Start Sorting System",
            "Configuration validated successfully.\n\n"
            "Ready to start the sorting system?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            QMessageBox.information(
                self,
                "System Starting",
                "The sorting system is now starting...\n\n"
                "This window will transition to the sorting view."
            )
            # Here you would emit a signal or call the main application
            # to start the actual sorting system
            # For now, we just close the config window
            # self.start_sorting_system()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()