"""
GUI_module.py - Complete GUI modules module for LEGO Sorting System with all fixes applied
This module provides both configuration and active sorting operation interfaces.
Fixed issues:
1. ROI constrained within camera frame
2. Camera selection dropdown for multiple devices
3. Bin 0 properly designated as overflow
4. Module initialization error resolved
"""

import sys
import os
import time
import cv2
import numpy as np
import threading
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QRect, QPoint,
    QMutex, QMutexLocker, QSize
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QGridLayout, QFrame,
    QSplitter, QStatusBar, QCheckBox, QSpinBox, QComboBox,
    QTextEdit, QProgressBar, QTabWidget, QMessageBox,
    QLineEdit, QListWidget, QListWidgetItem, QDialog, QScrollArea
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush,
    QFont, QPalette, QMouseEvent
)

# Import configuration manager
from enhanced_config_manager import create_config_manager, ModuleConfig

# Initialize logger
logger = logging.getLogger(__name__)


# ============= Camera Selection Dialog =============

class CameraSelectionDialog(QDialog):
    """Dialog for selecting camera from available devices"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera")
        self.setModal(True)
        self.selected_index = 0

        layout = QVBoxLayout()

        # Test available cameras
        self.available_cameras = self.detect_cameras()

        # Camera selection
        layout.addWidget(QLabel("Available Cameras:"))
        self.camera_combo = QComboBox()

        for idx, name in self.available_cameras:
            self.camera_combo.addItem(f"Camera {idx}: {name}")

        layout.addWidget(self.camera_combo)

        # Preview button
        self.preview_btn = QPushButton("Test Camera")
        self.preview_btn.clicked.connect(self.test_camera)
        layout.addWidget(self.preview_btn)

        # Buttons
        buttons = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.ok_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)

        self.setLayout(layout)

    def detect_cameras(self):
        """Detect available cameras by testing first 3 indices"""
        cameras = []
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    # Try to get camera name
                    backend = cap.getBackendName()
                    cameras.append((i, f"{backend} Device"))
                cap.release()

        if not cameras:
            cameras.append((0, "Default Camera"))

        return cameras

    def test_camera(self):
        """Test selected camera"""
        idx = self.camera_combo.currentIndex()
        if idx >= 0 and idx < len(self.available_cameras):
            camera_idx = self.available_cameras[idx][0]

            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    QMessageBox.information(self, "Camera Test",
                                            f"Camera {camera_idx} is working!\n"
                                            f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    QMessageBox.warning(self, "Camera Test",
                                        f"Camera {camera_idx} failed to capture frame")
                cap.release()

    def get_selected_camera(self):
        """Get selected camera index"""
        idx = self.camera_combo.currentIndex()
        if idx >= 0 and idx < len(self.available_cameras):
            return self.available_cameras[idx][0]
        return 0


# ============= Camera Thread =============

class CameraThread(QThread):
    """Thread for camera capture during configuration"""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.is_running = False
        self.cap = None

    def run(self):
        """Main thread loop"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return

        self.is_running = True

        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(33)  # ~30 FPS

    def stop(self):
        """Stop the thread"""
        self.is_running = False
        self.wait()
        if self.cap:
            self.cap.release()


# ============= Camera Preview Widget with ROI Constraints =============

class CameraPreviewWidget(QLabel):
    """Widget for displaying camera feed with ROI overlay"""

    roi_changed = pyqtSignal(int, int, int, int)  # x, y, w, h

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setMaximumSize(960, 720)
        self.setScaledContents(False)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #222;")

        # Initialize current_pixmap to None
        self.current_pixmap = None

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

    def update_frame(self, pixmap, frame_width, frame_height):
        """Update the displayed frame"""
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Scale pixmap to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Trigger a repaint which will draw the ROI overlay
        self.current_pixmap = scaled_pixmap
        self.update()  # This triggers paintEvent

    def paintEvent(self, event):
        """Paint event to draw frame and ROI overlay"""
        painter = QPainter(self)

        # Draw the frame if we have one
        if self.current_pixmap is not None:
            # Calculate position to center the frame
            x = (self.width() - self.current_pixmap.width()) // 2
            y = (self.height() - self.current_pixmap.height()) // 2
            painter.drawPixmap(x, y, self.current_pixmap)

            # Draw ROI overlay
            self._draw_roi_overlay(painter, x, y)
        else:
            # Draw placeholder
            self.show_placeholder()

    def _draw_roi_overlay(self, painter, offset_x, offset_y):
        """Draw the ROI rectangle overlay"""
        if not hasattr(self, 'frame_width') or not self.frame_width:
            return

        # Calculate scale factor
        scale_x = self.current_pixmap.width() / self.frame_width
        scale_y = self.current_pixmap.height() / self.frame_height

        # Scale ROI coordinates to display size
        roi_x = int(self.roi_x * scale_x) + offset_x
        roi_y = int(self.roi_y * scale_y) + offset_y
        roi_w = int(self.roi_w * scale_x)
        roi_h = int(self.roi_h * scale_y)

        # Draw ROI rectangle
        pen = QPen(QColor(0, 255, 0), 2)  # Green, 2px width
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(roi_x, roi_y, roi_w, roi_h)

        # Draw ROI info text
        painter.setPen(QColor(0, 255, 0))
        info_text = f"ROI: {self.roi_w}x{self.roi_h}"
        painter.drawText(roi_x + 5, roi_y - 5, info_text)

    def mousePressEvent(self, event):
        """Handle mouse press events for ROI interaction"""
        if event.button() == Qt.LeftButton and self.current_pixmap is not None:
            # Convert mouse position to camera coordinates
            offset_x = (self.width() - self.current_pixmap.width()) // 2
            offset_y = (self.height() - self.current_pixmap.height()) // 2

            mouse_x = event.x()
            mouse_y = event.y()

            # Check if within image area
            if (mouse_x >= offset_x and mouse_x <= offset_x + self.current_pixmap.width() and
                    mouse_y >= offset_y and mouse_y <= offset_y + self.current_pixmap.height()):

                # Convert to camera coordinates
                scale_x = self.frame_width / self.current_pixmap.width()
                scale_y = self.frame_height / self.current_pixmap.height()

                cam_x = int((mouse_x - offset_x) * scale_x)
                cam_y = int((mouse_y - offset_y) * scale_y)

                # Check if clicking on ROI
                if (cam_x >= self.roi_x and cam_x <= self.roi_x + self.roi_w and
                        cam_y >= self.roi_y and cam_y <= self.roi_y + self.roi_h):
                    self.dragging = True
                    self.drag_start = (cam_x, cam_y)
                    self.resize_handle = self._get_resize_handle(cam_x, cam_y)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.drag_start = None
            self.resize_handle = None
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for ROI dragging and resizing"""
        if not hasattr(self, 'frame_width') or not self.frame_width:
            return

        # Convert mouse position to camera coordinates
        mouse_x = event.x()
        mouse_y = event.y()

        # Calculate position in scaled coordinates
        # Check that current_pixmap exists AND is not None
        if self.current_pixmap is not None:
            offset_x = (self.width() - self.current_pixmap.width()) // 2
            offset_y = (self.height() - self.current_pixmap.height()) // 2

            # Check if mouse is within the image area
            if (mouse_x < offset_x or mouse_x > offset_x + self.current_pixmap.width() or
                    mouse_y < offset_y or mouse_y > offset_y + self.current_pixmap.height()):
                self.setCursor(Qt.ArrowCursor)
                return

            # Convert to camera coordinates
            scale_x = self.frame_width / self.current_pixmap.width()
            scale_y = self.frame_height / self.current_pixmap.height()

            cam_x = int((mouse_x - offset_x) * scale_x)
            cam_y = int((mouse_y - offset_y) * scale_y)

            if self.dragging and self.drag_start:
                # Calculate the delta from drag start
                delta_x = cam_x - self.drag_start[0]
                delta_y = cam_y - self.drag_start[1]

                if self.resize_handle:
                    # Handle resizing
                    new_x = self.roi_x
                    new_y = self.roi_y
                    new_w = self.roi_w
                    new_h = self.roi_h

                    if 'left' in self.resize_handle:
                        new_x = min(self.roi_x + delta_x, self.roi_x + self.roi_w - 100)
                        new_w = self.roi_w - (new_x - self.roi_x)
                    if 'right' in self.resize_handle:
                        new_w = max(100, self.roi_w + delta_x)
                    if 'top' in self.resize_handle:
                        new_y = min(self.roi_y + delta_y, self.roi_y + self.roi_h - 100)
                        new_h = self.roi_h - (new_y - self.roi_y)
                    if 'bottom' in self.resize_handle:
                        new_h = max(100, self.roi_h + delta_y)

                    # Constrain to frame bounds
                    new_x = max(0, min(new_x, self.frame_width - new_w))
                    new_y = max(0, min(new_y, self.frame_height - new_h))
                    new_w = min(new_w, self.frame_width - new_x)
                    new_h = min(new_h, self.frame_height - new_y)

                    self.roi_x = new_x
                    self.roi_y = new_y
                    self.roi_w = new_w
                    self.roi_h = new_h
                else:
                    # Handle dragging - maintain size, just move position
                    new_x = self.roi_x + delta_x
                    new_y = self.roi_y + delta_y

                    # Constrain position to keep ROI within frame bounds
                    # Do NOT change the size - just bump against edges
                    new_x = max(0, min(new_x, self.frame_width - self.roi_w))
                    new_y = max(0, min(new_y, self.frame_height - self.roi_h))

                    self.roi_x = new_x
                    self.roi_y = new_y

                self.roi_changed.emit(self.roi_x, self.roi_y, self.roi_w, self.roi_h)
                self.update()
            else:
                # Check if cursor is near ROI edges for resize handles
                self._update_cursor_for_resize(cam_x, cam_y)
        else:
            # No pixmap available yet, just set default cursor
            self.setCursor(Qt.ArrowCursor)

    def _get_resize_handle(self, cam_x, cam_y):
        """Determine which resize handle is being grabbed"""
        edge_threshold = 10
        handle = []

        # Check edges
        if abs(cam_x - self.roi_x) < edge_threshold:
            handle.append('left')
        elif abs(cam_x - (self.roi_x + self.roi_w)) < edge_threshold:
            handle.append('right')

        if abs(cam_y - self.roi_y) < edge_threshold:
            handle.append('top')
        elif abs(cam_y - (self.roi_y + self.roi_h)) < edge_threshold:
            handle.append('bottom')

        return handle if handle else None

    def _update_cursor_for_resize(self, cam_x, cam_y):
        """Update cursor based on proximity to ROI edges"""
        edge_threshold = 10

        # Check if near edges
        near_left = abs(cam_x - self.roi_x) < edge_threshold
        near_right = abs(cam_x - (self.roi_x + self.roi_w)) < edge_threshold
        near_top = abs(cam_y - self.roi_y) < edge_threshold
        near_bottom = abs(cam_y - (self.roi_y + self.roi_h)) < edge_threshold

        # Set appropriate cursor
        if (near_left and near_top) or (near_right and near_bottom):
            self.setCursor(Qt.SizeFDiagCursor)
        elif (near_left and near_bottom) or (near_right and near_top):
            self.setCursor(Qt.SizeBDiagCursor)
        elif near_left or near_right:
            self.setCursor(Qt.SizeHorCursor)
        elif near_top or near_bottom:
            self.setCursor(Qt.SizeVerCursor)
        elif (cam_x >= self.roi_x and cam_x <= self.roi_x + self.roi_w and
              cam_y >= self.roi_y and cam_y <= self.roi_y + self.roi_h):
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)


# ============= Configuration Screen =============

class ConfigurationScreen(QWidget):
    """Main configuration screen with all settings"""

    config_confirmed = pyqtSignal()

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager

        # Selected camera index
        self.selected_camera_index = 0

        # Initialize category database from sorting manager
        self.category_database = None
        self.init_category_database()

        # Camera thread for preview
        self.camera_thread = None

        # Frame dimensions
        self.actual_frame_width = 1920
        self.actual_frame_height = 1080

        self.init_ui()
        self.load_settings()

    def init_category_database(self):
        """Initialize category database from config manager's CSV parser"""
        try:
            hierarchy = self.config_manager.parse_categories_from_csv()

            self.category_database = {
                "primary": set(hierarchy.get("primary", [])),
                "primary_to_secondary": {},
                "secondary_to_tertiary": {}
            }

            for primary, secondaries in hierarchy.get("primary_to_secondary", {}).items():
                self.category_database["primary_to_secondary"][primary] = set(secondaries)

            for key, tertiaries in hierarchy.get("secondary_to_tertiary", {}).items():
                self.category_database["secondary_to_tertiary"][key] = set(tertiaries)

            logger.info(f"Loaded {len(self.category_database['primary'])} primary categories")

        except Exception as e:
            logger.error(f"Error initializing category database: {e}")
            self.category_database = None

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Add tabs
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
        """Create camera and ROI settings tab with camera selection"""
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

        # Camera selection button
        self.select_camera_btn = QPushButton("Select Camera")
        self.select_camera_btn.clicked.connect(self.select_camera)
        cam_controls.addWidget(self.select_camera_btn)

        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        cam_controls.addWidget(self.start_camera_btn)

        self.stop_camera_btn = QPushButton("Stop Camera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        cam_controls.addWidget(self.stop_camera_btn)

        preview_layout.addLayout(cam_controls)
        main_layout.addLayout(preview_layout, 2)

        # Right side - Settings
        settings_layout = QVBoxLayout()

        # Camera info
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QGridLayout()

        camera_layout.addWidget(QLabel("Device:"), 0, 0)
        self.camera_device_label = QLabel("Camera 0")
        camera_layout.addWidget(self.camera_device_label, 0, 1)

        camera_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.camera_resolution_label = QLabel("Unknown")
        camera_layout.addWidget(self.camera_resolution_label, 1, 1)

        camera_group.setLayout(camera_layout)
        settings_layout.addWidget(camera_group)

        # ROI settings with constraints
        roi_group = QGroupBox("ROI Settings")
        roi_layout = QGridLayout()

        roi_layout.addWidget(QLabel("X Position:"), 0, 0)
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 3840)
        self.roi_x_spin.valueChanged.connect(self.on_roi_changed)
        roi_layout.addWidget(self.roi_x_spin, 0, 1)

        roi_layout.addWidget(QLabel("Y Position:"), 1, 0)
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 2160)
        self.roi_y_spin.valueChanged.connect(self.on_roi_changed)
        roi_layout.addWidget(self.roi_y_spin, 1, 1)

        roi_layout.addWidget(QLabel("Width:"), 2, 0)
        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(100, 3840)
        self.roi_w_spin.valueChanged.connect(self.on_roi_changed)
        roi_layout.addWidget(self.roi_w_spin, 2, 1)

        roi_layout.addWidget(QLabel("Height:"), 3, 0)
        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(100, 2160)
        self.roi_h_spin.valueChanged.connect(self.on_roi_changed)
        roi_layout.addWidget(self.roi_h_spin, 3, 1)

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
        """Create sorting strategy and bin assignment tab with proper bin 0 handling"""
        widget = QWidget()
        main_layout = QHBoxLayout()

        # Left side - Sorting Strategy
        strategy_layout = QVBoxLayout()

        strategy_group = QGroupBox("Sorting Strategy")
        strategy_inner = QVBoxLayout()

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["primary", "secondary", "tertiary"])
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_inner.addWidget(QLabel("Strategy:"))
        strategy_inner.addWidget(self.strategy_combo)

        # Category selection (hidden by default)
        self.primary_combo = QComboBox()
        self.primary_label = QLabel("Primary Category:")
        self.primary_label.hide()
        self.primary_combo.hide()
        self.primary_combo.currentTextChanged.connect(self.on_primary_changed)
        strategy_inner.addWidget(self.primary_label)
        strategy_inner.addWidget(self.primary_combo)

        self.secondary_combo = QComboBox()
        self.secondary_label = QLabel("Secondary Category:")
        self.secondary_label.hide()
        self.secondary_combo.hide()
        strategy_inner.addWidget(self.secondary_label)
        strategy_inner.addWidget(self.secondary_combo)

        # Max bins (not including overflow bin 0)
        strategy_inner.addWidget(QLabel("Max Category Bins:"))
        self.max_bins_spin = QSpinBox()
        self.max_bins_spin.setRange(1, 7)  # Bins 1-7 for categories
        self.max_bins_spin.setValue(7)
        self.max_bins_spin.setToolTip("Number of bins for categories (Bin 0 is reserved for overflow)")
        # CRITICAL FIX: Connect the valueChanged signal
        self.max_bins_spin.valueChanged.connect(self.on_max_bins_changed)
        strategy_inner.addWidget(self.max_bins_spin)

        strategy_group.setLayout(strategy_inner)
        strategy_layout.addWidget(strategy_group)

        # Assignment UI - Scrollable area for categories
        assignment_group = QGroupBox("Category Assignments")
        assignment_layout = QVBoxLayout()

        # Create scroll area for assignments
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.assignment_scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(self.assignment_scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)

        assignment_layout.addWidget(scroll_area)
        assignment_group.setLayout(assignment_layout)
        strategy_layout.addWidget(assignment_group)

        strategy_layout.addStretch()
        main_layout.addLayout(strategy_layout, 1)

        # Right side - Bin Visual Status
        bins_layout = QVBoxLayout()

        bins_group = QGroupBox("Bin Pre-Assignments")
        self.bins_inner_layout = QVBoxLayout()  # Store reference for dynamic updates

        # Create initial bin display
        self.update_bin_display()

        bins_group.setLayout(self.bins_inner_layout)
        bins_layout.addWidget(bins_group)

        main_layout.addLayout(bins_layout, 1)

        widget.setLayout(main_layout)

        # Initialize assignment UI
        self.update_assignment_ui()

        return widget

    def on_max_bins_changed(self):
        """Handle max bins change - update all related UI elements"""
        max_bins = self.max_bins_spin.value()

        # Update the bin display (visual bin assignments) if we have the necessary components
        if hasattr(self, 'bins_inner_layout'):
            self.update_bin_display()

        # Update assignment UI (scrollable category list) if it exists
        if hasattr(self, 'assignment_scroll_layout'):
            self.update_assignment_ui()

        # Update all bin spinboxes with new range in assignment widgets
        if hasattr(self, 'assignment_widgets'):
            for category, widgets in self.assignment_widgets.items():
                bin_spin = widgets["spin"]
                # Update range for all spinboxes (1 to max_bins, since 0 is overflow)
                current_value = bin_spin.value()
                bin_spin.setRange(1, max_bins)
                # Keep current value if still valid, otherwise set to max
                if current_value > max_bins:
                    bin_spin.setValue(max_bins)

        # Update the summary if the method exists
        if hasattr(self, 'assignment_summary_label'):
            self.update_assignment_summary()

        # Update bin combos to reflect new bin count
        if hasattr(self, 'bin_combos'):
            # Store current selections
            current_selections = {}
            for bin_num, combo in self.bin_combos.items():
                if bin_num <= max_bins:
                    current_selections[bin_num] = combo.currentText()

            # Rebuild bin combos with new range
            self.update_bin_display()

            # Restore selections where possible
            for bin_num, selection in current_selections.items():
                if bin_num in self.bin_combos:
                    index = self.bin_combos[bin_num].findText(selection)
                    if index >= 0:
                        self.bin_combos[bin_num].setCurrentIndex(index)

    def update_bin_display(self):
        """Dynamically update the bin display based on max_bins"""
        if not hasattr(self, 'bins_inner_layout'):
            # If the layout doesn't exist yet, we can't update it
            return

        # Clear existing widgets
        while self.bins_inner_layout.count():
            child = self.bins_inner_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Get current max_bins value
        max_bins = self.max_bins_spin.value()

        # Add overflow bin 0 label (not assignable)
        overflow_container = QWidget()
        overflow_layout = QHBoxLayout()
        overflow_layout.setContentsMargins(0, 0, 0, 0)

        overflow_label = QLabel("Bin 0:")
        overflow_label.setStyleSheet("font-weight: bold; color: #ff5555;")
        overflow_layout.addWidget(overflow_label)

        overflow_info = QLabel("OVERFLOW (Auto)")
        overflow_info.setStyleSheet("color: #ff5555;")
        overflow_info.setToolTip("Bin 0 is reserved for overflow/errors/uncertainty")
        overflow_layout.addWidget(overflow_info)
        overflow_layout.addStretch()

        overflow_container.setLayout(overflow_layout)
        self.bins_inner_layout.addWidget(overflow_container)

        # Add separator
        separator = QFrame()
        separator.setFrameStyle(QFrame.HLine)
        self.bins_inner_layout.addWidget(separator)

        # Category bins (1 to max_bins)
        self.bin_combos = {}
        for i in range(1, max_bins + 1):
            bin_container = QWidget()
            bin_layout = QHBoxLayout()
            bin_layout.setContentsMargins(0, 0, 0, 0)

            bin_label = QLabel(f"Bin {i}:")
            bin_label.setMinimumWidth(50)
            bin_layout.addWidget(bin_label)

            combo = QComboBox()
            combo.addItem("Auto-assign")
            combo.setMinimumWidth(150)
            self.bin_combos[i] = combo
            bin_layout.addWidget(combo)
            bin_layout.addStretch()

            bin_container.setLayout(bin_layout)
            self.bins_inner_layout.addWidget(bin_container)

        # Add stretch at the end
        self.bins_inner_layout.addStretch()

        # Update categories in combos
        if hasattr(self, 'category_database'):
            self.update_bin_categories()

    def update_assignment_ui(self):
        """Update the bin assignment UI based on current strategy"""
        # This is a placeholder if the full implementation doesn't exist yet
        if not hasattr(self, 'assignment_scroll_layout'):
            return

        # Clear existing assignment widgets
        while self.assignment_scroll_layout.count():
            child = self.assignment_scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add a simple label for now
        info_label = QLabel(f"Max bins set to: {self.max_bins_spin.value()}")
        info_label.setStyleSheet("color: #999;")
        self.assignment_scroll_layout.addWidget(info_label)

    def update_assignment_summary(self):
        """Update the assignment summary label"""
        # This is a placeholder implementation
        max_bins = self.max_bins_spin.value()
        summary_text = f"Bins 1-{max_bins} available for assignment\nBin 0: Reserved for overflow"

        if hasattr(self, 'assignment_summary_label'):
            self.assignment_summary_label.setText(summary_text)

    def create_servo_tab(self):
        """Create servo control settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        servo_group = QGroupBox("Servo Settings")
        servo_layout = QGridLayout()

        # Servo positions for bins 0-7
        for i in range(8):  # 0-7
            if i == 0:
                label = QLabel("Bin 0 (Overflow):")
                label.setStyleSheet("font-weight: bold;")
            else:
                label = QLabel(f"Bin {i} Position:")
            servo_layout.addWidget(label, i, 0)

            spin = QSpinBox()
            spin.setRange(0, 180)
            spin.setValue(20 * i if i > 0 else 0)  # Bin 0 at 0 degrees
            servo_layout.addWidget(spin, i, 1)

        servo_group.setLayout(servo_layout)
        layout.addWidget(servo_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def select_camera(self):
        """Show camera selection dialog"""
        dialog = CameraSelectionDialog(self)
        if dialog.exec_():
            self.selected_camera_index = dialog.get_selected_camera()
            self.camera_device_label.setText(f"Camera {self.selected_camera_index}")

            # Update config
            self.config_manager.update_module_config(
                ModuleConfig.CAMERA.value,
                {"device_id": self.selected_camera_index}
            )

            # Restart camera if running
            if self.camera_thread:
                self.stop_camera()
                self.start_camera()

    def start_camera(self):
        """Start camera preview with selected device"""
        if not self.camera_thread:
            # Get selected camera index
            camera_index = getattr(self, 'selected_camera_index', 0)

            self.camera_thread = CameraThread(camera_index)
            self.camera_thread.frame_ready.connect(self.on_frame_received)
            self.camera_thread.start()

            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)

    def on_frame_received(self, frame):
        """Callback when new frame is available from camera module"""
        # Don't try to draw ROI here - just pass frame to preview widget
        # The CameraPreviewWidget will handle ROI overlay in its paintEvent

        # Convert frame for Qt display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        # Convert BGR (OpenCV) to RGB (Qt)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create QImage
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert to QPixmap and update the preview widget
        pixmap = QPixmap.fromImage(q_image)

        # Update the camera preview widget with the new frame
        # Pass pixmap, width, and height as required by update_frame method
        self.camera_preview.update_frame(pixmap, width, height)

    def stop_camera(self):
        """Stop camera preview"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.camera_resolution_label.setText("Unknown")

    def on_roi_dragged(self, x, y, w, h):
        """Handle ROI dragging from preview"""
        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_w_spin.setValue(w)
        self.roi_h_spin.setValue(h)

    def on_roi_changed(self):
        """Handle ROI spinbox changes with constraints"""
        # Get current frame dimensions
        frame_w = self.camera_preview.frame_width
        frame_h = self.camera_preview.frame_height

        # Get spinbox values
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        w = self.roi_w_spin.value()
        h = self.roi_h_spin.value()

        # Apply constraints
        x = max(0, min(x, frame_w - 100))
        y = max(0, min(y, frame_h - 100))
        w = max(100, min(w, frame_w - x))
        h = max(100, min(h, frame_h - y))

        # Update spinboxes if values were constrained
        self.roi_x_spin.blockSignals(True)
        self.roi_y_spin.blockSignals(True)
        self.roi_w_spin.blockSignals(True)
        self.roi_h_spin.blockSignals(True)

        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_w_spin.setValue(w)
        self.roi_h_spin.setValue(h)

        self.roi_x_spin.blockSignals(False)
        self.roi_y_spin.blockSignals(False)
        self.roi_w_spin.blockSignals(False)
        self.roi_h_spin.blockSignals(False)

        # Update preview
        self.camera_preview.set_roi(x, y, w, h)

    def on_strategy_changed(self, strategy):
        """Handle sorting strategy change"""
        if strategy == "primary":
            self.primary_label.hide()
            self.primary_combo.hide()
            self.secondary_label.hide()
            self.secondary_combo.hide()
        elif strategy == "secondary":
            self.primary_label.show()
            self.primary_combo.show()
            self.secondary_label.hide()
            self.secondary_combo.hide()
            self.update_primary_categories()
        elif strategy == "tertiary":
            self.primary_label.show()
            self.primary_combo.show()
            self.secondary_label.show()
            self.secondary_combo.show()
            self.update_primary_categories()

        self.update_bin_categories()

    def on_primary_changed(self, primary):
        """Handle primary category change"""
        if self.strategy_combo.currentText() == "tertiary":
            self.update_secondary_categories()
        self.update_bin_categories()

    def update_primary_categories(self):
        """Update primary category combo"""
        if not self.category_database:
            return

        self.primary_combo.clear()
        self.primary_combo.addItems(sorted(self.category_database["primary"]))

    def update_secondary_categories(self):
        """Update secondary category combo"""
        if not self.category_database:
            return

        primary = self.primary_combo.currentText()
        if primary in self.category_database["primary_to_secondary"]:
            secondaries = self.category_database["primary_to_secondary"][primary]
            self.secondary_combo.clear()
            self.secondary_combo.addItems(sorted(secondaries))

    def update_bin_categories(self):
        """Update available categories for bin assignment"""
        strategy = self.strategy_combo.currentText()

        categories = []
        if strategy == "primary" and self.category_database:
            categories = sorted(self.category_database["primary"])
        elif strategy == "secondary":
            primary = self.primary_combo.currentText()
            if primary in self.category_database["primary_to_secondary"]:
                categories = sorted(self.category_database["primary_to_secondary"][primary])
        elif strategy == "tertiary":
            primary = self.primary_combo.currentText()
            secondary = self.secondary_combo.currentText()
            key = f"{primary}_{secondary}"
            if key in self.category_database["secondary_to_tertiary"]:
                categories = sorted(self.category_database["secondary_to_tertiary"][key])

        # Update all bin combos (excluding bin 0)
        for combo in self.bin_combos.values():
            current = combo.currentText()
            combo.clear()
            combo.addItem("Auto-assign")
            combo.addItems(categories)

            # Restore selection if possible
            index = combo.findText(current)
            if index >= 0:
                combo.setCurrentIndex(index)

    def load_settings(self):
        """Load settings from config manager"""
        # Camera settings
        camera_config = self.config_manager.get_module_config(ModuleConfig.CAMERA.value)
        self.selected_camera_index = camera_config.get("device_id", 0)
        self.camera_device_label.setText(f"Camera {self.selected_camera_index}")

        # ROI settings
        roi_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR_ROI.value)
        self.roi_x_spin.setValue(roi_config.get("x", 100))
        self.roi_y_spin.setValue(roi_config.get("y", 100))
        self.roi_w_spin.setValue(roi_config.get("w", 400))
        self.roi_h_spin.setValue(roi_config.get("h", 300))

        # Detection settings
        detector_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR.value)
        self.min_area_spin.setValue(detector_config.get("min_area", 1000))
        self.max_area_spin.setValue(detector_config.get("max_area", 50000))

        # Sorting settings
        sorting_config = self.config_manager.get_module_config(ModuleConfig.SORTING.value)
        strategy = sorting_config.get("strategy", "primary")
        index = self.strategy_combo.findText(strategy)
        if index >= 0:
            self.strategy_combo.setCurrentIndex(index)

    def save_settings(self):
        """Save settings to config manager"""
        # Update camera config
        self.config_manager.update_module_config(
            ModuleConfig.CAMERA.value,
            {"device_id": self.selected_camera_index}
        )

        # Update ROI config
        self.config_manager.update_module_config(
            ModuleConfig.DETECTOR_ROI.value,
            {
                "x": self.roi_x_spin.value(),
                "y": self.roi_y_spin.value(),
                "w": self.roi_w_spin.value(),
                "h": self.roi_h_spin.value()
            }
        )

        # Update detector config
        self.config_manager.update_module_config(
            ModuleConfig.DETECTOR.value,
            {
                "min_area": self.min_area_spin.value(),
                "max_area": self.max_area_spin.value()
            }
        )

        # Update sorting config
        sorting_config = {
            "strategy": self.strategy_combo.currentText(),
            "max_bins": self.max_bins_spin.value(),
            "overflow_bin": 0  # Always use bin 0 as overflow
        }

        if self.strategy_combo.currentText() in ["secondary", "tertiary"]:
            sorting_config["target_primary_category"] = self.primary_combo.currentText()

        if self.strategy_combo.currentText() == "tertiary":
            sorting_config["target_secondary_category"] = self.secondary_combo.currentText()

        # Save pre-assignments (excluding bin 0)
        pre_assignments = {}
        for bin_num, combo in self.bin_combos.items():
            if combo.currentText() != "Auto-assign":
                pre_assignments[combo.currentText()] = bin_num

        sorting_config["pre_assignments"] = pre_assignments

        self.config_manager.update_module_config(
            ModuleConfig.SORTING.value,
            sorting_config
        )

        # Save configuration to file
        self.config_manager.save_config()

        QMessageBox.information(self, "Configuration Saved", "Settings have been saved successfully.")

    def validate_configuration(self):
        """Validate current configuration"""
        # Check ROI
        if self.roi_w_spin.value() < 100 or self.roi_h_spin.value() < 100:
            QMessageBox.warning(self, "Invalid ROI", "ROI dimensions too small.")
            return False

        # Check strategy-specific requirements
        strategy = self.strategy_combo.currentText()
        if strategy == "secondary" and not self.primary_combo.currentText():
            QMessageBox.warning(self, "Invalid Configuration",
                                "Secondary sorting requires selecting a primary category.")
            return False

        if strategy == "tertiary":
            if not self.primary_combo.currentText() or not self.secondary_combo.currentText():
                QMessageBox.warning(self, "Invalid Configuration",
                                    "Tertiary sorting requires selecting both primary and secondary categories.")
                return False

        QMessageBox.information(self, "Configuration Valid", "All settings are valid.")
        return True

    def start_system(self):
        """Start the sorting system"""
        if not self.validate_configuration():
            return

        self.save_settings()

        # Stop camera if running
        if self.camera_thread:
            self.stop_camera()

        # Emit signal to start system
        self.config_confirmed.emit()

    def cleanup(self):
        """Clean up resources"""
        if self.camera_thread:
            self.stop_camera()


# ============= Sorting Operation GUI modules Components =============

@dataclass
class TrackedPieceDisplay:
    """Display data for a tracked piece"""
    id: int
    bbox: tuple
    status: str
    update_count: int
    center: tuple


class MetricsPanel(QGroupBox):
    """Real-time metrics display panel"""

    def __init__(self):
        super().__init__("System Metrics")
        self.init_ui()
        self.metrics = {}

    def init_ui(self):
        layout = QGridLayout()

        self.labels = {
            'fps': QLabel("FPS: --"),
            'queue_size': QLabel("Queue: --"),
            'processed': QLabel("Processed: 0"),
            'detected': QLabel("Detected: 0"),
            'errors': QLabel("Errors: 0"),
            'uptime': QLabel("Uptime: 00:00:00"),
            'throughput': QLabel("Throughput: --/min"),
            'efficiency': QLabel("Efficiency: --%")
        }

        row = 0
        for key, label in self.labels.items():
            label.setStyleSheet("QLabel { color: #00ff00; font-family: monospace; }")
            layout.addWidget(label, row % 4, row // 4)
            row += 1

        self.setLayout(layout)
        self.setMaximumHeight(150)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update displayed metrics"""
        self.metrics.update(metrics)

        if 'fps' in metrics:
            self.labels['fps'].setText(f"FPS: {metrics['fps']:.1f}")

        if 'queue_size' in metrics:
            self.labels['queue_size'].setText(f"Queue: {metrics['queue_size']}")

        if 'processed_count' in metrics:
            self.labels['processed'].setText(f"Processed: {metrics['processed_count']}")

        if 'detected_count' in metrics:
            self.labels['detected'].setText(f"Detected: {metrics['detected_count']}")

        if 'error_count' in metrics:
            self.labels['errors'].setText(f"Errors: {metrics['error_count']}")

        if 'uptime' in metrics:
            uptime = int(metrics['uptime'])
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            seconds = uptime % 60
            self.labels['uptime'].setText(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")

        # Calculate throughput
        if 'processed_count' in metrics and 'uptime' in metrics:
            if metrics['uptime'] > 0:
                throughput = (metrics['processed_count'] / metrics['uptime']) * 60
                self.labels['throughput'].setText(f"Throughput: {throughput:.1f}/min")

        # Calculate efficiency
        if 'detected_count' in metrics and 'processed_count' in metrics:
            if metrics['detected_count'] > 0:
                efficiency = (metrics['processed_count'] / metrics['detected_count']) * 100
                self.labels['efficiency'].setText(f"Efficiency: {efficiency:.1f}%")


class ProcessedPiecePanel(QGroupBox):
    """Display recently processed piece information"""

    def __init__(self):
        super().__init__("Recently Processed")
        self.init_ui()
        self.current_piece = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(250, 150)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #1e1e1e; border: 1px solid #555; }")
        self.image_label.setText("No piece processed")
        layout.addWidget(self.image_label)

        # Piece information
        self.info_layout = QGridLayout()

        self.info_labels = {
            'id': QLabel("ID: --"),
            'element': QLabel("Element: --"),
            'name': QLabel("Name: --"),
            'category': QLabel("Category: --"),
            'bin': QLabel("Bin: --"),
            'time': QLabel("Time: --"),
            'confidence': QLabel("Confidence: --%")
        }

        row = 0
        for key, label in self.info_labels.items():
            label.setStyleSheet("QLabel { color: #ffffff; font-size: 10pt; }")
            self.info_layout.addWidget(label, row, 0)
            row += 1

        layout.addLayout(self.info_layout)
        self.setLayout(layout)
        self.setMaximumWidth(300)

    def update_piece(self, piece_data: Dict[str, Any]):
        """Update display with new piece data"""
        self.current_piece = piece_data

        # Update image if available
        if 'image' in piece_data and piece_data['image'] is not None:
            self.display_piece_image(piece_data['image'])

        # Update information labels
        if 'piece_id' in piece_data:
            self.info_labels['id'].setText(f"ID: {piece_data['piece_id']}")

        if 'element_id' in piece_data:
            self.info_labels['element'].setText(f"Element: {piece_data['element_id']}")

        if 'name' in piece_data:
            name = piece_data['name']
            if len(name) > 25:
                name = name[:22] + "..."
            self.info_labels['name'].setText(f"Name: {name}")

        if 'primary_category' in piece_data:
            self.info_labels['category'].setText(f"Category: {piece_data['primary_category']}")

        if 'bin_number' in piece_data:
            bin_num = piece_data['bin_number']
            if bin_num == 0:
                self.info_labels['bin'].setText("Bin: 0 (OVERFLOW)")
            else:
                self.info_labels['bin'].setText(f"Bin: {bin_num}")

        if 'processing_time' in piece_data:
            self.info_labels['time'].setText(f"Time: {piece_data['processing_time']:.2f}s")

        if 'confidence' in piece_data:
            self.info_labels['confidence'].setText(f"Confidence: {piece_data['confidence']:.1f}%")

    def display_piece_image(self, image: np.ndarray):
        """Display piece image in the panel"""
        height, width = image.shape[:2]

        if len(image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)


class BinStatusPanel(QGroupBox):
    """Display bin assignments and status with bin 0 as overflow"""

    def __init__(self, max_bins=7):
        super().__init__("Bin Assignments")
        self.max_bins = max_bins
        self.bin_assignments = {}
        self.bin_counts = {}
        self.init_ui()

    def init_ui(self):
        self.main_layout = QGridLayout()
        self.create_bin_widgets()
        self.setLayout(self.main_layout)

    def create_bin_widgets(self):
        """Create bin widgets dynamically based on max_bins"""
        # Clear existing widgets
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create bin widgets (0 plus 1 to max_bins)
        self.bin_widgets = {}
        total_bins = self.max_bins + 1  # Include bin 0

        for i in range(total_bins):
            if i == 0:
                # Bin 0 is overflow
                bin_num = 0
            else:
                bin_num = i

            bin_frame = QFrame()
            bin_frame.setFrameStyle(QFrame.Box)

            # Special styling for overflow bin
            if bin_num == 0:
                bin_frame.setStyleSheet("QFrame { background-color: #3b2b2b; border: 2px solid #aa5555; }")
            else:
                bin_frame.setStyleSheet("QFrame { background-color: #2b2b2b; border: 2px solid #555; }")

            bin_layout = QVBoxLayout()

            # Bin number with special label for overflow
            if bin_num == 0:
                bin_label = QLabel("Bin 0 (OVERFLOW)")
                bin_label.setStyleSheet("QLabel { color: #ff5555; font-weight: bold; }")
            else:
                bin_label = QLabel(f"Bin {bin_num}")
                bin_label.setStyleSheet("QLabel { color: #ffffff; font-weight: bold; }")

            bin_label.setAlignment(Qt.AlignCenter)
            bin_layout.addWidget(bin_label)

            # Category assignment
            if bin_num == 0:
                category_label = QLabel("Errors/Unknown")
                category_label.setStyleSheet("QLabel { color: #ff8888; }")
            else:
                category_label = QLabel("Unassigned")
                category_label.setStyleSheet("QLabel { color: #888888; }")

            category_label.setAlignment(Qt.AlignCenter)
            bin_layout.addWidget(category_label)

            # Count
            count_label = QLabel("Count: 0")
            count_label.setAlignment(Qt.AlignCenter)
            count_label.setStyleSheet("QLabel { color: #00ff00; }")
            bin_layout.addWidget(count_label)

            # Progress bar
            progress = QProgressBar()
            progress.setMaximum(100)
            progress.setValue(0)
            progress.setTextVisible(False)
            progress.setMaximumHeight(10)
            bin_layout.addWidget(progress)

            bin_frame.setLayout(bin_layout)

            self.bin_widgets[bin_num] = {
                'frame': bin_frame,
                'category': category_label,
                'count': count_label,
                'progress': progress
            }

            # Calculate grid position dynamically
            cols = 4  # Display in 4 columns
            row = i // cols
            col = i % cols
            self.main_layout.addWidget(bin_frame, row, col)

    def update_bin_count_display(self, new_max_bins):
        """Update display when max_bins changes"""
        self.max_bins = new_max_bins
        self.create_bin_widgets()
        # Restore counts for existing bins
        for bin_num, count in self.bin_counts.items():
            if bin_num in self.bin_widgets:
                self.update_bin_count(bin_num, count)


class VideoDisplayWidget(QLabel):
    """Widget for displaying video feed with detection overlays"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setStyleSheet("QLabel { background-color: #000000; border: 2px solid #555; }")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Waiting for video feed...")

        # Detection data
        self.detector_data = None
        self.show_overlays = True
        self.overlay_options = {
            'roi': True,
            'zones': True,
            'pieces': True,
            'ids': True,
            'fps': True
        }

        # Thread safety
        self.mutex = QMutex()

    def update_frame(self, frame: np.ndarray, detector_data: Optional[Dict] = None):
        """Update displayed frame with optional detection data"""
        with QMutexLocker(self.mutex):
            if detector_data:
                self.detector_data = detector_data

            # Apply overlays if enabled
            if self.show_overlays and self.detector_data:
                frame = self.apply_detection_overlays(frame)

            # Convert to QPixmap and display
            height, width = frame.shape[:2]

            if len(frame.shape) == 3:
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            else:
                bytes_per_line = width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

    def apply_detection_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Apply detection overlays to frame"""
        display = frame.copy()

        # Draw ROI rectangle
        if self.overlay_options['roi'] and 'roi' in self.detector_data:
            x, y, w, h = self.detector_data['roi']
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "ROI", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw entry/exit zones
        if self.overlay_options['zones'] and 'entry_zone' in self.detector_data:
            roi_y = self.detector_data['roi'][1]
            roi_h = self.detector_data['roi'][3]

            entry_x = self.detector_data['entry_zone'][1]
            cv2.line(display, (entry_x, roi_y), (entry_x, roi_y + roi_h), (255, 0, 0), 2)

            if 'exit_zone' in self.detector_data:
                exit_x = self.detector_data['exit_zone'][0]
                cv2.line(display, (exit_x, roi_y), (exit_x, roi_y + roi_h), (0, 0, 255), 2)

                exit_end = self.detector_data['exit_zone'][1]
                overlay = display.copy()
                cv2.rectangle(overlay, (exit_x, roi_y), (exit_end, roi_y + roi_h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.1, display, 0.9, 0, display)

        # Draw tracked pieces
        if self.overlay_options['pieces'] and 'tracked_pieces' in self.detector_data:
            roi_x = self.detector_data.get('roi', [0, 0, 0, 0])[0]
            roi_y = self.detector_data.get('roi', [0, 0, 0, 0])[1]

            for piece in self.detector_data['tracked_pieces']:
                px, py, pw, ph = piece['bbox']
                px += roi_x
                py += roi_y

                # Color based on status
                color = self.get_piece_color(piece)

                cv2.rectangle(display, (px, py), (px + pw, py + ph), color, 2)

                cx = px + pw // 2
                cy = py + ph // 2
                cv2.circle(display, (cx, cy), 3, color, -1)

                if self.overlay_options['ids']:
                    label = self.get_piece_label(piece)
                    cv2.putText(display, label, (px, py - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Add FPS overlay
        if self.overlay_options['fps'] and 'fps' in self.detector_data:
            fps_text = f"FPS: {self.detector_data['fps']:.1f}"
            cv2.putText(display, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return display

    def get_piece_color(self, piece: Dict) -> tuple:
        """Get color for piece based on status"""
        if piece.get('being_processed'):
            return (255, 0, 255)  # Purple
        elif piece.get('in_exit_zone'):
            return (0, 165, 255)  # Orange
        elif piece.get('captured'):
            return (0, 255, 0)  # Green
        else:
            return (0, 0, 255)  # Red

    def get_piece_label(self, piece: Dict) -> str:
        """Get label for piece"""
        piece_id = piece.get('id', 0)

        if piece.get('being_processed'):
            return f"ID:{piece_id} PROCESSING"
        elif piece.get('in_exit_zone'):
            return f"ID:{piece_id} EXIT"
        else:
            update_count = piece.get('update_count', 0)
            return f"ID:{piece_id} U:{update_count}"


# ============= Main Sorting GUI modules =============

class LegoSortingGUI(QMainWindow):
    """Main GUI modules window for sorting operation"""

    # Signals
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, config_manager=None):
        super().__init__()
        self.config_manager = config_manager
        self.is_paused = False
        self.start_time = time.time()

        # Camera registration
        self.camera = None
        self.is_registered = False

        # System modules
        self.detector = None
        self.sorting_manager = None
        self.piece_history = None
        self.thread_manager = None

        self.setWindowTitle("LEGO Sorting System - Active Sorting")
        self.setGeometry(100, 100, 1400, 900)

        self.apply_dark_theme()
        self.init_ui()
        self.setup_timers()

    def apply_dark_theme(self):
        """Apply dark theme to the window"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #444;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #888;
            }
        """)

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()

        # Control bar
        control_layout = QHBoxLayout()

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop Sorting")
        self.stop_btn.clicked.connect(self.stop_sorting)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #aa4444; }")
        control_layout.addWidget(self.stop_btn)

        control_layout.addStretch()

        # Overlay toggles
        self.overlay_checks = {
            'roi': QCheckBox("ROI"),
            'zones': QCheckBox("Zones"),
            'pieces': QCheckBox("Pieces"),
            'ids': QCheckBox("IDs")
        }

        for check in self.overlay_checks.values():
            check.setChecked(True)
            check.stateChanged.connect(self.update_overlay_settings)
            control_layout.addWidget(check)

        main_layout.addLayout(control_layout)

        # Content area
        content_splitter = QSplitter(Qt.Horizontal)

        # Left side - Video and metrics
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        self.video_display = VideoDisplayWidget()
        left_layout.addWidget(self.video_display, 1)

        self.metrics_panel = MetricsPanel()
        left_layout.addWidget(self.metrics_panel)

        left_widget.setLayout(left_layout)
        content_splitter.addWidget(left_widget)

        # Right side - Piece info and bin status
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        self.piece_panel = ProcessedPiecePanel()
        right_layout.addWidget(self.piece_panel)

        self.bin_panel = BinStatusPanel()
        right_layout.addWidget(self.bin_panel, 1)

        right_widget.setLayout(right_layout)
        content_splitter.addWidget(right_widget)

        content_splitter.setSizes([980, 420])

        main_layout.addWidget(content_splitter, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System Ready")

        central_widget.setLayout(main_layout)

    def setup_timers(self):
        """Setup periodic update timers"""
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(1000)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500)

    def register_camera_consumer(self, camera):
        """Register as a frame consumer with the camera module"""
        self.camera = camera

        if not self.is_registered:
            try:
                success = camera.register_consumer(
                    name="sorting_gui",
                    callback=self.process_frame,
                    processing_type="async",
                    priority=80
                )

                if success:
                    self.is_registered = True
                    self.status_bar.showMessage("Camera feed connected")
                else:
                    self.status_bar.showMessage("Failed to connect camera feed")
            except Exception as e:
                logger.error(f"Failed to register camera consumer: {e}")
                self.status_bar.showMessage(f"Camera error: {str(e)}")

    def unregister_camera_consumer(self):
        """Unregister from camera module"""
        if self.camera and self.is_registered:
            try:
                self.camera.unregister_consumer("sorting_gui")
                self.is_registered = False
            except Exception as e:
                logger.warning(f"Error unregistering camera consumer: {e}")

    def process_frame(self, frame: np.ndarray):
        """Frame consumer callback"""
        if self.is_paused:
            return

        # Get detector data if available
        detector_data = None
        if self.detector:
            try:
                detector_data = self.detector.get_visualization_data()
            except Exception as e:
                logger.warning(f"Failed to get detector data: {e}")

        # Update video display
        self.video_display.update_frame(frame, detector_data)

    def set_modules(self, modules: Dict[str, Any]):
        """Set references to system modules"""
        self.detector = modules.get('detector')
        self.sorting_manager = modules.get('sorting_manager')
        self.piece_history = modules.get('piece_history')
        self.thread_manager = modules.get('thread_manager')

        # Update bin assignments
        if self.sorting_manager:
            try:
                assignments = self.sorting_manager.get_bin_mapping()
                self.bin_panel.update_assignments(assignments)
            except Exception as e:
                logger.error(f"Failed to get bin assignments: {e}")

    def update_metrics(self):
        """Update metrics display"""
        uptime = time.time() - self.start_time

        metrics = {
            'uptime': uptime
        }

        # Get metrics from thread manager
        if self.thread_manager:
            try:
                metrics['queue_size'] = self.thread_manager.get_queue_size()
            except:
                pass

        # Get metrics from detector
        if self.detector:
            try:
                detector_data = self.detector.get_visualization_data()
                if 'fps' in detector_data:
                    metrics['fps'] = detector_data['fps']
                if 'tracked_pieces' in detector_data:
                    metrics['detected_count'] = len(detector_data['tracked_pieces'])
            except:
                pass

        # Get processed count from piece history
        if self.piece_history:
            try:
                metrics['processed_count'] = self.piece_history.get_total_count()
                metrics['error_count'] = self.piece_history.get_error_count()

                # Get latest piece
                latest_piece = self.piece_history.get_latest_piece()
                if latest_piece:
                    self.piece_panel.update_piece(latest_piece)

                    # Update bin count
                    bin_num = latest_piece.get('bin_number', 0)
                    self.bin_panel.increment_bin(bin_num)
            except Exception as e:
                logger.warning(f"Failed to get piece history data: {e}")

        self.metrics_panel.update_metrics(metrics)

    def update_status(self):
        """Update status bar"""
        if self.is_paused:
            self.status_bar.showMessage("System Paused")
        elif self.thread_manager:
            try:
                queue_size = self.thread_manager.get_queue_size()
                self.status_bar.showMessage(f"Sorting Active | Queue: {queue_size}")
            except:
                self.status_bar.showMessage("Sorting Active")

    def update_overlay_settings(self):
        """Update video overlay settings"""
        for key, check in self.overlay_checks.items():
            self.video_display.overlay_options[key] = check.isChecked()

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_btn.setText("Resume")
            self.pause_requested.emit()

            if self.camera:
                self.camera.pause_consumer("sorting_gui")
        else:
            self.pause_btn.setText("Pause")
            self.resume_requested.emit()

            if self.camera:
                self.camera.resume_consumer("sorting_gui")

    def stop_sorting(self):
        """Stop sorting operation"""
        self.stop_requested.emit()
        self.unregister_camera_consumer()
        self.status_bar.showMessage("Stopping sorting system...")

    def cleanup(self):
        """Clean up resources"""
        self.unregister_camera_consumer()

        # Stop timers
        if hasattr(self, 'metrics_timer'):
            self.metrics_timer.stop()
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()

    def closeEvent(self, event):
        """Handle window close event"""
        self.cleanup()
        event.accept()


# ============= Main Application Window =============

class MainWindow(QMainWindow):
    """Main application window with mode switching"""

    def __init__(self):
        super().__init__()

        # Create config manager
        self.config_manager = create_config_manager()

        # System modules
        self.system_modules = {}

        # Current mode
        self.current_mode = "configuration"

        # GUI modules screens
        self.config_screen = None
        self.sorting_gui = None

        self.setWindowTitle("LEGO Sorting System")
        self.setGeometry(100, 100, 1200, 700)

        # Start with configuration screen
        self.show_configuration_screen()

    def show_configuration_screen(self):
        """Show the configuration screen"""
        self.current_mode = "configuration"
        self.setWindowTitle("LEGO Sorting System - Configuration")

        # Clean up sorting if it exists
        if self.sorting_gui:
            self.sorting_gui.cleanup()

        # Create configuration screen
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
            # Clean up configuration screen
            self.config_screen.cleanup()

            # Start sorting mode
            self.start_sorting_mode()

    def start_sorting_mode(self):
        """Initialize and start sorting operation mode"""
        try:
            QMessageBox.information(
                self,
                "Initializing",
                "Initializing sorting system modules..."
            )

            # Initialize system modules
            self.initialize_system_modules()

            # Create sorting GUI modules
            self.sorting_gui = LegoSortingGUI(self.config_manager)

            # Connect signals
            self.sorting_gui.pause_requested.connect(self.on_pause_requested)
            self.sorting_gui.resume_requested.connect(self.on_resume_requested)
            self.sorting_gui.stop_requested.connect(self.on_stop_requested)

            # Set modules
            self.sorting_gui.set_modules(self.system_modules)

            # Register with camera
            if 'camera' in self.system_modules:
                self.sorting_gui.register_camera_consumer(self.system_modules['camera'])

            # Update window
            self.current_mode = "sorting"
            self.setWindowTitle("LEGO Sorting System - Active Sorting")
            self.setCentralWidget(self.sorting_gui)

            # Start camera capture
            if 'camera' in self.system_modules:
                self.system_modules['camera'].start_capture()

        except Exception as e:
            logger.error(f"Failed to start sorting mode: {e}")
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize sorting system:\n{str(e)}"
            )
            self.show_configuration_screen()

    def initialize_system_modules(self):
        """Initialize all system modules for sorting operation"""
        try:
            # Import necessary modules
            from camera_module import create_camera
            from Old_versions.detector_module import create_detector
            from sorting_module import create_sorting_manager
            from thread_management_module import create_thread_manager
            from piece_history_module import create_piece_history

            # Initialize modules with correct arguments
            self.system_modules = {
                'camera': create_camera(
                    camera_type="webcam",
                    config_manager=self.config_manager
                ),
                'detector': create_detector(
                    detector_type="conveyor",
                    config_manager=self.config_manager,
                    thread_manager=None
                ),
                'sorting_manager': create_sorting_manager(self.config_manager),
                'thread_manager': create_thread_manager(self.config_manager),
                'piece_history': create_piece_history()
            }

            # Update detector with thread manager if needed
            if hasattr(self.system_modules['detector'], 'set_thread_manager'):
                self.system_modules['detector'].set_thread_manager(
                    self.system_modules['thread_manager']
                )

            # Create processing worker
            from processing_module import create_processing_worker
            self.system_modules['processing_worker'] = create_processing_worker(
                self.system_modules['thread_manager'],
                self.config_manager,
                piece_history=self.system_modules['piece_history']
            )

            # Try to initialize Arduino
            try:
                from arduino_servo_module import create_arduino_servo_module
                self.system_modules['arduino'] = create_arduino_servo_module(self.config_manager)
            except Exception as e:
                logger.warning(f"Arduino module not available: {e}")

        except Exception as e:
            logger.error(f"Module initialization error details: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize modules: {str(e)}")

    def on_pause_requested(self):
        """Handle pause request from sorting GUI modules"""
        logger.info("Sorting paused")

    def on_resume_requested(self):
        """Handle resume request from sorting GUI modules"""
        logger.info("Sorting resumed")

    def on_stop_requested(self):
        """Handle stop request and return to configuration"""
        reply = QMessageBox.question(
            self,
            "Stop Sorting",
            "Are you sure you want to stop sorting and return to configuration?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.cleanup_sorting_mode()
            self.show_configuration_screen()

    def cleanup_sorting_mode(self):
        """Clean up sorting mode resources"""
        # Stop camera capture
        if 'camera' in self.system_modules:
            try:
                self.system_modules['camera'].stop_capture()
            except:
                pass

        # Clean up sorting GUI modules
        if self.sorting_gui:
            self.sorting_gui.cleanup()

        # Release resources
        for module in self.system_modules.values():
            if hasattr(module, 'release'):
                try:
                    module.release()
                except:
                    pass

        self.system_modules.clear()

    def closeEvent(self, event):
        """Handle application close event"""
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit the sorting system?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Clean up based on current mode
            if self.current_mode == "sorting":
                self.cleanup_sorting_mode()
            elif self.config_screen:
                self.config_screen.cleanup()

            event.accept()
        else:
            event.ignore()


# ============= Main Entry Point =============

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
