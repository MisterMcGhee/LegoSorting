#!/usr/bin/env python3
"""
LEGO Sorting GUI - Phase 1 Standalone
Mock GUI for testing layout and functionality without hardware dependencies
"""

import sys
import time
import cv2
import json
import csv
import os
from typing import Tuple

# PyQt imports with fallback
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *

    PYQT_VERSION = "PyQt5"
    print("Using PyQt5")
except ImportError:
    try:
        from PyQt6.QtWidgets import *
        from PyQt6.QtCore import *
        from PyQt6.QtGui import *

        PYQT_VERSION = "PyQt6"
        print("Using PyQt6")
    except ImportError:
        print("ERROR: Neither PyQt5 nor PyQt6 is installed!")
        sys.exit(1)


class InteractiveROIWidget(QWidget):
    """Camera widget with interactive ROI selection capability"""

    roi_updated = pyqtSignal(QRect)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.current_frame = None
        self.camera_error = False
        self.error_message = ""
        self.cap = None
        self._first_frame_received = False

        # ROI related
        self.roi_rect = QRect(125, 200, 1550, 500)  # Default from config
        self.edit_mode = False
        self.selecting_roi = False
        self.roi_start_point = None
        self.temp_roi = None
        self.dragging_handle = None
        self.hover_handle = None

        # Handle positions
        self.handles = {}

        # Constraints
        self.min_roi_size = QSize(100, 100)

        self.setMouseTracking(True)
        self.setMinimumHeight(400)

        # Initialize camera
        self.init_camera()

        # Timer for frame updates
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(33)  # ~30 FPS

    def init_camera(self):
        """Initialize OpenCV camera"""
        try:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Camera opened but cannot read frames")

            self.camera_error = False
            self.error_message = ""

        except Exception as e:
            self.camera_error = True
            self.error_message = str(e)
            if self.cap:
                self.cap.release()
                self.cap = None

    def update_frame(self):
        """Capture and update frame from camera"""
        if self.camera_error or not self.cap:
            return

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame = frame.copy()
                # Validate ROI when we get the first frame
                if hasattr(self, '_first_frame_received'):
                    self.validate_and_clamp_roi()
                    self._first_frame_received = True
                self.update()
        except Exception as e:
            if not self.camera_error:
                self.camera_error = True
                self.error_message = f"Camera error: {str(e)}"

    def get_scaled_roi(self) -> QRect:
        """Get ROI scaled to current widget size"""
        if self.current_frame is None or self.camera_error:  # Changed from 'if not self.current_frame'
            return self.roi_rect

        # Get frame dimensions
        h, w = self.current_frame.shape[:2]

        # Calculate scale based on widget size
        widget_aspect = self.width() / self.height()
        frame_aspect = w / h

        if widget_aspect > frame_aspect:
            scale = self.height() / h
            offset_x = (self.width() - w * scale) / 2
            offset_y = 0
        else:
            scale = self.width() / w
            offset_x = 0
            offset_y = (self.height() - h * scale) / 2

        # Scale ROI to display coordinates
        scaled_roi = QRect(
            int(self.roi_rect.x() * scale + offset_x),
            int(self.roi_rect.y() * scale + offset_y),
            int(self.roi_rect.width() * scale),
            int(self.roi_rect.height() * scale)
        )

        return scaled_roi
    def validate_and_clamp_roi(self):
        """Ensure ROI is within frame bounds"""
        if self.current_frame is None:
            return

        h, w = self.current_frame.shape[:2]

        # Clamp ROI to frame bounds
        x = max(0, min(self.roi_rect.x(), w - self.min_roi_size.width()))
        y = max(0, min(self.roi_rect.y(), h - self.min_roi_size.height()))
        width = min(self.roi_rect.width(), w - x)
        height = min(self.roi_rect.height(), h - y)

        # Ensure minimum size
        width = max(width, self.min_roi_size.width())
        height = max(height, self.min_roi_size.height())

        # Update ROI
        self.roi_rect = QRect(x, y, width, height)

    def display_to_frame_coords(self, display_point: QPoint) -> QPoint:
        """Convert display coordinates to frame coordinates"""
        if self.current_frame is None:  # Changed from 'if not self.current_frame'
            return display_point

        h, w = self.current_frame.shape[:2]

        # Calculate scale and offset
        widget_aspect = self.width() / self.height()
        frame_aspect = w / h

        if widget_aspect > frame_aspect:
            scale = self.height() / h
            offset_x = (self.width() - w * scale) / 2
            offset_y = 0
        else:
            scale = self.width() / w
            offset_x = 0
            offset_y = (self.height() - h * scale) / 2

        # Convert to frame coordinates
        frame_x = int((display_point.x() - offset_x) / scale)
        frame_y = int((display_point.y() - offset_y) / scale)

        # Clamp to frame bounds
        frame_x = max(0, min(w - 1, frame_x))
        frame_y = max(0, min(h - 1, frame_y))

        return QPoint(frame_x, frame_y)
    def update_handles(self):
        """Update handle positions based on current ROI"""
        scaled_roi = self.get_scaled_roi()

        # Define 8 handles: corners and edge midpoints
        self.handles = {
            'nw': QRect(scaled_roi.left() - 5, scaled_roi.top() - 5, 10, 10),
            'n': QRect(scaled_roi.center().x() - 5, scaled_roi.top() - 5, 10, 10),
            'ne': QRect(scaled_roi.right() - 5, scaled_roi.top() - 5, 10, 10),
            'e': QRect(scaled_roi.right() - 5, scaled_roi.center().y() - 5, 10, 10),
            'se': QRect(scaled_roi.right() - 5, scaled_roi.bottom() - 5, 10, 10),
            's': QRect(scaled_roi.center().x() - 5, scaled_roi.bottom() - 5, 10, 10),
            'sw': QRect(scaled_roi.left() - 5, scaled_roi.bottom() - 5, 10, 10),
            'w': QRect(scaled_roi.left() - 5, scaled_roi.center().y() - 5, 10, 10)
        }

    def get_handle_at_pos(self, pos: QPoint) -> str:
        """Get handle at given position"""
        for handle_name, handle_rect in self.handles.items():
            if handle_rect.contains(pos):
                return handle_name
        return None

    def mousePressEvent(self, event):
        if not self.edit_mode or event.button() != Qt.LeftButton:
            return
        # Add safety check
        if self.current_frame is None:
            return

        self.update_handles()
        handle = self.get_handle_at_pos(event.pos())

        if handle:
            self.dragging_handle = handle
        else:
            # Start new ROI selection
            self.selecting_roi = True
            self.roi_start_point = self.display_to_frame_coords(event.pos())
            self.temp_roi = QRect(self.roi_start_point, QSize(0, 0))

    def mouseMoveEvent(self, event):
        if not self.edit_mode:
            return

        if self.current_frame is None:
            return

        if self.selecting_roi and self.roi_start_point:
            # Update temporary ROI
            end_point = self.display_to_frame_coords(event.pos())
            self.temp_roi = QRect(self.roi_start_point, end_point).normalized()
            self.update()

        elif self.dragging_handle:
            # Handle dragging logic
            self.resize_roi_with_handle(event.pos())
            self.update()

        else:
            # Update cursor based on hover
            self.update_handles()
            handle = self.get_handle_at_pos(event.pos())
            if handle != self.hover_handle:
                self.hover_handle = handle
                self.update_cursor()

    def mouseReleaseEvent(self, event):
        if not self.edit_mode:
            return

        if self.current_frame is None:
            return

        if self.selecting_roi and self.temp_roi:
            # Finalize ROI selection
            if self.temp_roi.width() >= self.min_roi_size.width() and \
                    self.temp_roi.height() >= self.min_roi_size.height():
                self.roi_rect = self.temp_roi
                self.roi_updated.emit(self.roi_rect)
            self.temp_roi = None

        self.selecting_roi = False
        self.dragging_handle = None
        self.roi_start_point = None
        self.update()

    def resize_roi_with_handle(self, pos: QPoint):
        """Resize ROI based on handle drag"""
        frame_pos = self.display_to_frame_coords(pos)

        # Get current ROI bounds
        left = self.roi_rect.left()
        top = self.roi_rect.top()
        right = self.roi_rect.right()
        bottom = self.roi_rect.bottom()

        # Update bounds based on which handle is being dragged
        if 'n' in self.dragging_handle:
            top = frame_pos.y()
        if 's' in self.dragging_handle:
            bottom = frame_pos.y()
        if 'w' in self.dragging_handle:
            left = frame_pos.x()
        if 'e' in self.dragging_handle:
            right = frame_pos.x()

        # Create new rectangle and ensure minimum size
        new_roi = QRect(QPoint(left, top), QPoint(right, bottom)).normalized()

        if new_roi.width() >= self.min_roi_size.width() and \
                new_roi.height() >= self.min_roi_size.height():
            self.roi_rect = new_roi
            self.roi_updated.emit(self.roi_rect)

    def update_cursor(self):
        """Update cursor based on hover state"""
        cursor_map = {
            'nw': Qt.CursorShape.SizeFDiagCursor if PYQT_VERSION == "PyQt6" else Qt.SizeFDiagCursor,
            'ne': Qt.CursorShape.SizeBDiagCursor if PYQT_VERSION == "PyQt6" else Qt.SizeBDiagCursor,
            'sw': Qt.CursorShape.SizeBDiagCursor if PYQT_VERSION == "PyQt6" else Qt.SizeBDiagCursor,
            'se': Qt.CursorShape.SizeFDiagCursor if PYQT_VERSION == "PyQt6" else Qt.SizeFDiagCursor,
            'n': Qt.CursorShape.SizeVerCursor if PYQT_VERSION == "PyQt6" else Qt.SizeVerCursor,
            's': Qt.CursorShape.SizeVerCursor if PYQT_VERSION == "PyQt6" else Qt.SizeVerCursor,
            'e': Qt.CursorShape.SizeHorCursor if PYQT_VERSION == "PyQt6" else Qt.SizeHorCursor,
            'w': Qt.CursorShape.SizeHorCursor if PYQT_VERSION == "PyQt6" else Qt.SizeHorCursor,
        }

        if self.hover_handle in cursor_map:
            self.setCursor(cursor_map[self.hover_handle])
        else:
            self.setCursor(Qt.CursorShape.CrossCursor if PYQT_VERSION == "PyQt6" else Qt.CrossCursor)

    def opencv_to_qpixmap(self, cv_img):
        """Convert OpenCV image to QPixmap"""
        if cv_img is None:
            return None

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line,
                          QImage.Format.Format_RGB888 if PYQT_VERSION == "PyQt6" else QImage.Format_RGB888)

        return QPixmap.fromImage(qt_image)

    def paintEvent(self, event):
        painter = QPainter(self)

        if self.camera_error:
            # Draw error message
            painter.fillRect(self.rect(), QColor(40, 40, 40))
            painter.setPen(QPen(QColor(255, 100, 100), 2))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter,
                             f"Camera Error: {self.error_message}")

        elif self.current_frame is not None:
            # Draw camera feed
            pixmap = self.opencv_to_qpixmap(self.current_frame)
            if pixmap:
                # Scale to fit widget
                scaled_pixmap = pixmap.scaled(self.size(),
                                              Qt.AspectRatioMode.KeepAspectRatio if PYQT_VERSION == "PyQt6" else Qt.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation if PYQT_VERSION == "PyQt6" else Qt.SmoothTransformation)

                # Center the image
                x = (self.width() - scaled_pixmap.width()) // 2
                y = (self.height() - scaled_pixmap.height()) // 2
                painter.drawPixmap(x, y, scaled_pixmap)

                if self.edit_mode:
                    # Draw overlay
                    overlay = QPixmap(self.size())
                    overlay.fill(Qt.GlobalColor.transparent if PYQT_VERSION == "PyQt6" else Qt.transparent)
                    overlay_painter = QPainter(overlay)

                    # Semi-transparent dark overlay
                    overlay_painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

                    # Clear the ROI area
                    roi_to_draw = self.temp_roi if self.selecting_roi and self.temp_roi else self.roi_rect
                    if roi_to_draw == self.roi_rect:
                        scaled_roi = self.get_scaled_roi()
                    else:
                        # Scale temporary ROI
                        if self.current_frame is not None:
                            h, w = self.current_frame.shape[:2]
                            scale_x = scaled_pixmap.width() / w
                            scale_y = scaled_pixmap.height() / h
                            scaled_roi = QRect(
                                int(roi_to_draw.x() * scale_x + x),
                                int(roi_to_draw.y() * scale_y + y),
                                int(roi_to_draw.width() * scale_x),
                                int(roi_to_draw.height() * scale_y)
                            )
                        else:
                            scaled_roi = QRect()

                    overlay_painter.setCompositionMode(
                        QPainter.CompositionMode.CompositionMode_Clear if PYQT_VERSION == "PyQt6" else QPainter.CompositionMode_Clear)
                    overlay_painter.fillRect(scaled_roi,
                                             Qt.GlobalColor.transparent if PYQT_VERSION == "PyQt6" else Qt.transparent)
                    overlay_painter.end()

                    painter.drawPixmap(0, 0, overlay)

                    # Draw ROI border
                    painter.setPen(
                        QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine if PYQT_VERSION == "PyQt6" else Qt.DashLine))
                    painter.drawRect(scaled_roi)

                    # Draw dimensions
                    dim_text = f"{roi_to_draw.width()} × {roi_to_draw.height()}"
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.setFont(QFont("Arial", 12))
                    text_rect = painter.fontMetrics().boundingRect(dim_text)
                    text_pos = scaled_roi.center() - QPoint(text_rect.width() // 2, text_rect.height() // 2)
                    painter.fillRect(text_rect.translated(text_pos), QColor(0, 0, 0, 150))
                    painter.drawText(text_pos, dim_text)

                    # Draw handles if not selecting
                    if not self.selecting_roi:
                        self.update_handles()
                        for handle_name, handle_rect in self.handles.items():
                            if handle_name == self.hover_handle or handle_name == self.dragging_handle:
                                painter.fillRect(handle_rect, QColor(0, 255, 0))
                            else:
                                painter.fillRect(handle_rect, QColor(255, 255, 255))
                            painter.setPen(QPen(QColor(0, 0, 0), 1))
                            painter.drawRect(handle_rect)

    def cleanup(self):
        """Clean up camera resources"""
        if hasattr(self, 'frame_timer'):
            self.frame_timer.stop()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


class ROIAdjustmentDialog(QDialog):
    """Dialog for adjusting ROI with visual and numerical controls"""

    def __init__(self, camera_index, current_roi, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.original_roi = QRect(current_roi)
        self.current_roi = QRect(current_roi)

        self.setModal(True)
        self.setWindowTitle("Adjust ROI - Region of Interest")
        self.resize(1000, 700)

        self.init_ui()
        self.connect_signals()
        self.update_spinboxes()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "Click and drag to create a new ROI, or drag the handles to adjust the existing ROI. "
            "You can also use the number inputs below for precise control."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(instructions)

        # Camera widget
        self.camera_widget = InteractiveROIWidget(self.camera_index)
        self.camera_widget.edit_mode = True
        self.camera_widget.roi_rect = self.current_roi
        layout.addWidget(self.camera_widget, stretch=1)

        # Controls section
        controls_layout = QHBoxLayout()

        # Position group
        pos_group = QGroupBox("Position")
        pos_layout = QFormLayout()

        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 9999)
        self.x_spin.setSuffix(" px")

        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 9999)
        self.y_spin.setSuffix(" px")

        pos_layout.addRow("X:", self.x_spin)
        pos_layout.addRow("Y:", self.y_spin)
        pos_group.setLayout(pos_layout)
        controls_layout.addWidget(pos_group)

        # Size group
        size_group = QGroupBox("Size")
        size_layout = QFormLayout()

        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 9999)
        self.width_spin.setSuffix(" px")

        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 9999)
        self.height_spin.setSuffix(" px")

        size_layout.addRow("Width:", self.width_spin)
        size_layout.addRow("Height:", self.height_spin)
        size_group.setLayout(size_layout)
        controls_layout.addWidget(size_group)

        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout()

        center_btn = QPushButton("Center ROI")
        center_btn.clicked.connect(self.center_roi)

        maximize_btn = QPushButton("Maximize ROI")
        maximize_btn.clicked.connect(self.maximize_roi)

        reset_btn = QPushButton("Reset to Original")
        reset_btn.clicked.connect(self.reset_roi)

        actions_layout.addWidget(center_btn)
        actions_layout.addWidget(maximize_btn)
        actions_layout.addWidget(reset_btn)
        actions_group.setLayout(actions_layout)
        controls_layout.addWidget(actions_group)

        layout.addLayout(controls_layout)

        # Status bar
        self.status_label = QLabel("ROI: Ready")
        self.status_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        layout.addWidget(self.status_label)

        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        self.save_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px 20px; }")

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet("QPushButton { padding: 8px 20px; }")

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def connect_signals(self):
        """Connect widget signals"""
        # Camera widget ROI updates
        self.camera_widget.roi_updated.connect(self.on_roi_updated)

        # Spinbox changes
        self.x_spin.valueChanged.connect(self.on_spinbox_changed)
        self.y_spin.valueChanged.connect(self.on_spinbox_changed)
        self.width_spin.valueChanged.connect(self.on_spinbox_changed)
        self.height_spin.valueChanged.connect(self.on_spinbox_changed)

    def on_roi_updated(self, roi: QRect):
        """Handle ROI updates from camera widget"""
        self.current_roi = roi
        self.update_spinboxes()
        self.update_status()

    def update_spinboxes(self):
        """Update spinboxes with current ROI values"""
        # Temporarily disconnect to avoid recursion
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)

        self.x_spin.setValue(self.current_roi.x())
        self.y_spin.setValue(self.current_roi.y())
        self.width_spin.setValue(self.current_roi.width())
        self.height_spin.setValue(self.current_roi.height())

        # Reconnect signals
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)

    def on_spinbox_changed(self):
        """Handle spinbox value changes"""
        new_roi = QRect(
            self.x_spin.value(),
            self.y_spin.value(),
            self.width_spin.value(),
            self.height_spin.value()
        )

        self.current_roi = new_roi
        self.camera_widget.roi_rect = new_roi
        self.camera_widget.update()
        self.update_status()

    def center_roi(self):
        """Center ROI in frame"""
        if self.camera_widget.current_frame is not None:
            h, w = self.camera_widget.current_frame.shape[:2]
            roi_w = min(self.current_roi.width(), w - 20)
            roi_h = min(self.current_roi.height(), h - 20)

            new_x = (w - roi_w) // 2
            new_y = (h - roi_h) // 2

            self.current_roi = QRect(new_x, new_y, roi_w, roi_h)
            self.camera_widget.roi_rect = self.current_roi
            self.camera_widget.update()
            self.update_spinboxes()
            self.update_status()

    def maximize_roi(self):
        """Maximize ROI to frame size with margin"""
        if self.camera_widget.current_frame is not None:
            h, w = self.camera_widget.current_frame.shape[:2]
            margin = 10

            self.current_roi = QRect(margin, margin, w - 2 * margin, h - 2 * margin)
            self.camera_widget.roi_rect = self.current_roi
            self.camera_widget.update()
            self.update_spinboxes()
            self.update_status()

    def reset_roi(self):
        """Reset ROI to original values"""
        self.current_roi = QRect(self.original_roi)
        self.camera_widget.roi_rect = self.current_roi
        self.camera_widget.update()
        self.update_spinboxes()
        self.update_status()

    def update_status(self):
        """Update status label"""
        status = f"ROI: {self.current_roi.x()}, {self.current_roi.y()}, {self.current_roi.width()}×{self.current_roi.height()}"
        if self.current_roi != self.original_roi:
            status += " (modified)"
        self.status_label.setText(status)

    def get_roi(self) -> Tuple[int, int, int, int]:
        """Get current ROI as tuple"""
        return (self.current_roi.x(), self.current_roi.y(),
                self.current_roi.width(), self.current_roi.height())

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'camera_widget'):
            self.camera_widget.cleanup()


def load_roi_from_config(config_path="config.json", max_width=640, max_height=480) -> Tuple[int, int, int, int]:
    """Load ROI from config file and validate bounds"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            roi_config = config.get("detector_roi", {})

            # Get values from config
            x = roi_config.get("x", 125)
            y = roi_config.get("y", 200)
            w = roi_config.get("w", 1550)
            h = roi_config.get("h", 500)

            # Validate and clamp to reasonable defaults
            # This assumes a typical webcam resolution, actual validation
            # will happen when camera frame is available
            x = max(0, min(x, max_width - 100))
            y = max(0, min(y, max_height - 100))
            w = min(w, max_width - x)
            h = min(h, max_height - y)

            # Ensure minimum size
            w = max(w, 100)
            h = max(h, 100)

            return (x, y, w, h)

    except Exception as e:
        print(f"Error loading ROI from config: {e}")
        # Return sensible defaults for typical webcam
        return (50, 50, 400, 300)

def save_roi_to_config(x: int, y: int, w: int, h: int, config_path="config.json") -> bool:
    """Save ROI to config file"""
    try:
        # Load existing config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Update ROI section
        config["detector_roi"] = {
            "x": x,
            "y": y,
            "w": w,
            "h": h
        }

        # Save back to file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        print(f"ROI saved to config: {x}, {y}, {w}×{h}")
        return True

    except Exception as e:
        print(f"Error saving ROI to config: {e}")
        return False

class StartupScreen(QWidget):
    """System startup and initialization screen"""

    startup_complete = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.start_initialization()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Title
        title = QLabel("LEGO Sorting System v1.0")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)

        # LEGO icon placeholder
        icon_label = QLabel("🧱 Initializing System...")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 18px; margin: 20px;")
        layout.addWidget(icon_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setStyleSheet("QProgressBar { height: 20px; }")
        layout.addWidget(self.progress)

        # Status checks
        self.status_label = QLabel("Starting up...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.checks_widget = QWidget()
        checks_layout = QVBoxLayout(self.checks_widget)

        self.check_labels = {
            "config": QLabel("⏳ Loading configuration..."),
            "camera": QLabel("⏳ Initializing camera module..."),
            "arduino": QLabel("⏳ Connecting to Arduino servo..."),
            "database": QLabel("⏳ Loading piece identification database...")
        }

        for label in self.check_labels.values():
            label.setStyleSheet("margin: 5px 20px; font-family: monospace;")
            checks_layout.addWidget(label)

        layout.addWidget(self.checks_widget)

        # Please wait message
        wait_label = QLabel("Please wait...")
        wait_label.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        wait_label.setStyleSheet("font-style: italic; margin: 20px;")
        layout.addWidget(wait_label)

        self.setLayout(layout)

    def start_initialization(self):
        """Simulate system initialization"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.step = 0
        self.timer.start(300)  # Update every 300ms

    def update_progress(self):
        """Update initialization progress"""
        self.step += 1
        progress = min(self.step * 4, 100)
        self.progress.setValue(progress)

        # Update status checks
        if self.step >= 5:
            self.check_labels["config"].setText("✅ Configuration loaded")
        if self.step >= 10:
            self.check_labels["camera"].setText("⏳ Checking camera connection...")
        if self.step >= 15:
            # Actually test camera availability using our detection function
            available_cameras = detect_available_cameras(max_cameras=3)
            if available_cameras:
                self.check_labels["camera"].setText(f"✅ Camera(s) detected: {available_cameras}")
            else:
                self.check_labels["camera"].setText("❌ No cameras detected (will show error messages)")
        if self.step >= 17:
            self.check_labels["arduino"].setText("❌ Arduino servo not detected (will run in sim mode)")
        if self.step >= 20:
            self.check_labels["database"].setText("⏳ Loading piece identification database...")
        if self.step >= 22:
            # Load category database
            try:
                # Import at runtime to avoid circular imports
                import sys
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from config_management_module import create_config_manager

                config_mgr = create_config_manager()
                categories = config_mgr.parse_categories_from_csv()

                # Get the main window (traverse up from stacked widget)
                main_window = self.window()  # This gets the top-level window
                if main_window and hasattr(main_window, 'category_database'):
                    main_window.category_database = categories
                    self.check_labels["database"].setText("✅ Piece identification database loaded")
                else:
                    # If we can't find the main window, at least store it somewhere
                    if hasattr(self.parent(), 'window'):
                        self.parent().window().category_database = categories
                    self.check_labels["database"].setText("✅ Piece identification database loaded")

            except FileNotFoundError:
                self.check_labels["database"].setText("❌ Database CSV not found - cannot continue")
                QMessageBox.critical(self, "Critical Error",
                                     "Lego_Categories.csv not found!\n\nThe piece identification database is required for operation.")
                QTimer.singleShot(1000, lambda: sys.exit(1))
            except Exception as e:
                self.check_labels["database"].setText(f"❌ Database load error: {str(e)}")
                import traceback
                traceback.print_exc()  # Print the full error for debugging

        if self.step >= 25:
            parent_widget = self.parent()
            if parent_widget and hasattr(parent_widget, 'category_database') and parent_widget.category_database:
                count = len([p for p in parent_widget.category_database.get("primary", [])])
                self.check_labels["database"].setText(f"✅ Database loaded ({count} primary categories)")

        if progress >= 100:
            self.timer.stop()
            QTimer.singleShot(500, self.startup_complete.emit)


class ModeSelectionScreen(QWidget):
    """Mode selection screen"""

    mode_selected = pyqtSignal(str)  # "sorting" or "inventory"
    exit_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(30)

        # Title
        title = QLabel("Select Operating Mode")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)

        # Mode selection buttons
        modes_layout = QHBoxLayout()
        modes_layout.setSpacing(50)

        # Sorting mode
        sorting_widget = QWidget()
        sorting_layout = QVBoxLayout(sorting_widget)
        sorting_layout.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)

        sorting_icon = QLabel("🏗️ SORTING")
        sorting_icon.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        sorting_icon.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        sorting_layout.addWidget(sorting_icon)

        sorting_desc = QLabel("Sort bulk LEGO\npieces by type\nand category")
        sorting_desc.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        sorting_desc.setStyleSheet("margin: 10px; line-height: 1.5;")
        sorting_layout.addWidget(sorting_desc)

        sorting_btn = QPushButton("SELECT")
        sorting_btn.clicked.connect(lambda: self.mode_selected.emit("sorting"))
        sorting_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px 20px; }")
        sorting_layout.addWidget(sorting_btn)

        sorting_widget.setStyleSheet(
            "QWidget { border: 2px solid #ccc; border-radius: 10px; padding: 20px; background-color: #f9f9f9; }")
        modes_layout.addWidget(sorting_widget)

        # Inventory mode
        inventory_widget = QWidget()
        inventory_layout = QVBoxLayout(inventory_widget)
        inventory_layout.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)

        inventory_icon = QLabel("📦 INVENTORY")
        inventory_icon.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        inventory_icon.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #888;")
        inventory_layout.addWidget(inventory_icon)

        inventory_desc = QLabel("Check set\ncompleteness\n[Coming Soon]")
        inventory_desc.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        inventory_desc.setStyleSheet("margin: 10px; line-height: 1.5; color: #888;")
        inventory_layout.addWidget(inventory_desc)

        inventory_btn = QPushButton("DISABLED")
        inventory_btn.setEnabled(False)
        inventory_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px 20px; }")
        inventory_layout.addWidget(inventory_btn)

        inventory_widget.setStyleSheet(
            "QWidget { border: 2px solid #ddd; border-radius: 10px; padding: 20px; background-color: #f5f5f5; }")
        modes_layout.addWidget(inventory_widget)

        layout.addLayout(modes_layout)

        # Exit button
        exit_btn = QPushButton("EXIT")
        exit_btn.clicked.connect(self.exit_requested.emit)
        exit_btn.setStyleSheet("QPushButton { padding: 10px 30px; margin: 20px; }")
        layout.addWidget(exit_btn,
                         alignment=Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)

        self.setLayout(layout)


class ConfigurationScreen(QWidget):

    start_sorting = pyqtSignal(dict)  # Configuration dict
    back_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.available_cameras = []
        self.current_camera_index = 0
        self.category_database = None  # Will be set by main window

        # Added this line to ensure we have a default empty structure
        self.category_database = {
            "primary": set(),
            "primary_to_secondary": {},
            "secondary_to_tertiary": {}
        }

        # Load ROI from config
        roi_tuple = load_roi_from_config()
        self.current_roi = QRect(roi_tuple[0], roi_tuple[1], roi_tuple[2], roi_tuple[3])

        self.init_ui()
        self.detect_cameras()

    def set_category_database(self, category_db):
        """Set the category database for dropdowns"""
        self.category_database = category_db
        if hasattr(self, 'strategy_combo'):
            self.on_strategy_changed(self.strategy_combo.currentText())
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Title
        title = QLabel("Sorting Configuration")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)

        # Configuration options
        config_layout = QVBoxLayout()

        # Top row - strategy and camera selection
        top_settings_layout = QHBoxLayout()

        # Left side - strategy settings
        strategy_group = QGroupBox("Sorting Settings")
        strategy_layout = QVBoxLayout(strategy_group)

        # Sorting strategy
        strategy_row = QHBoxLayout()
        strategy_row.addWidget(QLabel("Sort by:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Primary Categories", "Secondary Categories", "Tertiary Categories"])
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_row.addWidget(self.strategy_combo)
        strategy_layout.addLayout(strategy_row)

        # Dynamic dropdowns container
        self.dynamic_dropdowns_layout = QVBoxLayout()
        strategy_layout.addLayout(self.dynamic_dropdowns_layout)

        # Initialize with primary (no additional dropdowns needed)
        self.primary_combo = None
        self.secondary_combo = None

        top_settings_layout.addWidget(strategy_group)

        # Right side - camera selection
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout(camera_group)

        # Camera selection
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        camera_row.addWidget(self.camera_combo)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.detect_cameras)
        refresh_btn.setMaximumWidth(80)
        camera_row.addWidget(refresh_btn)
        camera_layout.addLayout(camera_row)

        top_settings_layout.addWidget(camera_group)
        config_layout.addLayout(top_settings_layout)

        # Camera preview
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.camera_preview = RealCameraWidget(camera_index=0, show_roi=True)
        self.camera_preview.roi_rect = self.current_roi
        self.camera_preview.setMinimumHeight(400)
        preview_layout.addWidget(self.camera_preview)

        # ROI status
        roi_status_layout = QHBoxLayout()
        self.roi_status = QLabel("ROI Status: ✅ Properly configured")
        self.roi_status.setStyleSheet("color: green; font-weight: bold;")
        roi_status_layout.addWidget(self.roi_status)
        roi_status_layout.addStretch()

        adjust_roi_btn = QPushButton("Adjust ROI")
        adjust_roi_btn.clicked.connect(self.adjust_roi)
        roi_status_layout.addWidget(adjust_roi_btn)

        preview_layout.addLayout(roi_status_layout)
        config_layout.addWidget(preview_group)
        layout.addLayout(config_layout)

        # Advanced settings
        self.advanced_group = QGroupBox("Advanced Settings")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced_group)

        # Detection sensitivity
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Detection Sensitivity:"))
        self.sens_slider = QSlider(Qt.Orientation.Horizontal if PYQT_VERSION == "PyQt6" else Qt.Horizontal)
        self.sens_slider.setRange(0, 100)
        self.sens_slider.setValue(80)
        self.sens_value_label = QLabel("80%")
        self.sens_slider.valueChanged.connect(lambda v: self.sens_value_label.setText(f"{v}%"))
        sens_layout.addWidget(self.sens_slider)
        sens_layout.addWidget(self.sens_value_label)
        advanced_layout.addLayout(sens_layout)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal if PYQT_VERSION == "PyQt6" else Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(70)
        self.conf_value_label = QLabel("70%")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value_label.setText(f"{v}%"))
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        advanced_layout.addLayout(conf_layout)

        # Max bins
        bins_layout_adv = QHBoxLayout()
        bins_layout_adv.addWidget(QLabel("Max Processing Bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(1, 9)
        self.bins_spin.setValue(7)
        bins_layout_adv.addWidget(self.bins_spin)
        bins_layout_adv.addStretch()
        advanced_layout.addLayout(bins_layout_adv)

        layout.addWidget(self.advanced_group)

        # Buttons
        buttons_layout = QHBoxLayout()

        back_btn = QPushButton("BACK")
        back_btn.clicked.connect(self.back_requested.emit)
        buttons_layout.addWidget(back_btn)

        save_default_btn = QPushButton("SAVE AS DEFAULT")
        save_default_btn.clicked.connect(self.save_as_default)
        buttons_layout.addWidget(save_default_btn)

        start_btn = QPushButton("START SORTING")
        start_btn.clicked.connect(self.start_sorting_clicked)
        start_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #4CAF50; color: white; padding: 10px; }")
        buttons_layout.addWidget(start_btn)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def on_strategy_changed(self, strategy):
        """Handle strategy change and update dynamic dropdowns"""
        try:
            # Clear existing dynamic dropdowns
            while self.dynamic_dropdowns_layout.count():
                child = self.dynamic_dropdowns_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            self.primary_combo = None
            self.secondary_combo = None

            if not self.category_database:
                return

            if "Secondary" in strategy:
                # Add primary category dropdown
                primary_row = QHBoxLayout()
                primary_row.addWidget(QLabel("Within:"))
                self.primary_combo = QComboBox()

                primary_categories = sorted(self.category_database.get("primary", []))
                self.primary_combo.addItems(primary_categories)
                self.primary_combo.currentTextChanged.connect(self.on_primary_changed)

                primary_row.addWidget(self.primary_combo)
                self.dynamic_dropdowns_layout.addLayout(primary_row)

            elif "Tertiary" in strategy:
                # Add primary category dropdown
                primary_row = QHBoxLayout()
                primary_row.addWidget(QLabel("Within Primary:"))
                self.primary_combo = QComboBox()

                # Block signals during setup
                self.primary_combo.blockSignals(True)

                primary_categories = sorted(self.category_database.get("primary", []))
                self.primary_combo.addItems(primary_categories)
                self.primary_combo.currentTextChanged.connect(self.on_primary_changed)

                primary_row.addWidget(self.primary_combo)
                self.dynamic_dropdowns_layout.addLayout(primary_row)

                # Add secondary category dropdown
                secondary_row = QHBoxLayout()
                secondary_row.addWidget(QLabel("Within Secondary:"))
                self.secondary_combo = QComboBox()
                secondary_row.addWidget(self.secondary_combo)
                self.dynamic_dropdowns_layout.addLayout(secondary_row)

                # Unblock signals and trigger initial population
                self.primary_combo.blockSignals(False)
                if self.primary_combo.count() > 0:
                    self.on_primary_changed(self.primary_combo.currentText())

        except Exception as e:
            print(f"Error in on_strategy_changed: {e}")

    def on_primary_changed(self, primary_category):
        """Handle primary category selection for tertiary sorting"""
        if self.secondary_combo is None or not primary_category:
            return

        self.secondary_combo.clear()

        primary_to_secondary = self.category_database.get("primary_to_secondary", {})

        if primary_category in primary_to_secondary:
            secondaries_set = primary_to_secondary[primary_category]
            secondaries = sorted(list(secondaries_set))

            if secondaries:
                self.secondary_combo.addItems(secondaries)
                self.secondary_combo.setEnabled(True)
            else:
                self.secondary_combo.addItem("(No secondary categories)")
                self.secondary_combo.setEnabled(False)
        else:
            self.secondary_combo.addItem("(No secondary categories)")
            self.secondary_combo.setEnabled(False)
    def detect_cameras(self):
        """Detect available cameras and populate dropdown"""
        print("🔍 Detecting available cameras...")
        self.available_cameras = detect_available_cameras()

        # Update camera selection dropdown
        self.camera_combo.clear()
        if self.available_cameras:
            for cam_idx in self.available_cameras:
                self.camera_combo.addItem(f"Camera {cam_idx}")
            # Set current camera to first available
            self.current_camera_index = self.available_cameras[0]
            self.camera_preview.change_camera(self.current_camera_index)
        else:
            self.camera_combo.addItem("No cameras detected")
            self.camera_combo.setEnabled(False)

    def on_camera_changed(self, camera_text):
        """Handle camera selection change"""
        if "Camera" in camera_text:
            try:
                camera_index = int(camera_text.split()[-1])
                if camera_index != self.current_camera_index:
                    print(f"User selected camera {camera_index}")
                    self.current_camera_index = camera_index
                    self.camera_preview.change_camera(camera_index)
                    self.update_roi_status()
            except (ValueError, IndexError):
                print(f"Could not parse camera index from: {camera_text}")

    def update_roi_status(self):
        """Update ROI status based on camera state"""
        if self.camera_preview.camera_error:
            self.roi_status.setText("ROI Status: ❌ Camera not available")
            self.roi_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.roi_status.setText("ROI Status: ✅ Properly configured")
            self.roi_status.setStyleSheet("color: green; font-weight: bold;")

    def adjust_roi(self):
        """Launch ROI adjustment dialog"""
        # Make sure we have a valid camera index
        if not self.available_cameras:
            QMessageBox.warning(self, "No Camera",
                                "No camera available. Please connect a camera and refresh.")
            return

        # Launch ROI adjustment dialog
        dialog = ROIAdjustmentDialog(self.current_camera_index, self.current_roi, self)

        if dialog.exec() == QDialog.DialogCode.Accepted if PYQT_VERSION == "PyQt6" else dialog.exec_() == QDialog.Accepted:
            # Get new ROI
            roi_tuple = dialog.get_roi()
            self.current_roi = QRect(roi_tuple[0], roi_tuple[1], roi_tuple[2], roi_tuple[3])

            # Update preview
            self.camera_preview.roi_rect = self.current_roi
            self.camera_preview.update()

            # Save to config
            if save_roi_to_config(roi_tuple[0], roi_tuple[1], roi_tuple[2], roi_tuple[3]):
                self.roi_status.setText("ROI Status: ✅ Updated and saved")
                self.roi_status.setStyleSheet("color: green; font-weight: bold;")
                QMessageBox.information(self, "ROI Saved",
                                        f"ROI updated to: {roi_tuple[0]}, {roi_tuple[1]}, {roi_tuple[2]}×{roi_tuple[3]}")
            else:
                self.roi_status.setText("ROI Status: ⚠️ Updated but not saved")
                self.roi_status.setStyleSheet("color: orange; font-weight: bold;")

        # Clean up dialog
        dialog.cleanup()

    def save_as_default(self):
        """Save current settings as default"""
        reply = QMessageBox.question(self, "Save as Default",
                                     "Save current settings as default configuration?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No if PYQT_VERSION == "PyQt6" else QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.StandardButton.Yes if PYQT_VERSION == "PyQt6" else QMessageBox.Yes:
            QMessageBox.information(self, "Saved", "Settings saved as default!")

    def cleanup(self):
        """Clean up camera resources"""
        if hasattr(self, 'camera_preview'):
            self.camera_preview.cleanup()
    def start_sorting_clicked(self):
        """Start sorting with current configuration"""
        strategy_text = self.strategy_combo.currentText()

        # Determine actual strategy and targets
        if "Primary" in strategy_text:
            strategy = "primary"
            target_primary = ""
            target_secondary = ""
        elif "Secondary" in strategy_text:
            strategy = "secondary"
            target_primary = self.primary_combo.currentText() if self.primary_combo else ""
            target_secondary = ""
        else:  # Tertiary
            strategy = "tertiary"
            target_primary = self.primary_combo.currentText() if self.primary_combo else ""
            target_secondary = self.secondary_combo.currentText() if self.secondary_combo else ""

        config = {
            "strategy": strategy,
            "target_primary": target_primary,
            "target_secondary": target_secondary,
            "sensitivity": self.sens_slider.value(),
            "confidence": self.conf_slider.value(),
            "max_bins": self.bins_spin.value(),
            "camera_index": self.current_camera_index
        }

        # Save to actual config file
        if hasattr(self, 'config_manager'):
            config_manager = self.config_manager
        else:
            from config_management_module import create_config_manager
            config_manager = create_config_manager()

        config_manager.set("sorting", "strategy", strategy)
        config_manager.set("sorting", "target_primary_category", target_primary)
        config_manager.set("sorting", "target_secondary_category", target_secondary)
        config_manager.set("sorting", "max_bins", config["max_bins"])
        config_manager.save_config()

        self.start_sorting.emit(config)


class RealCameraWidget(QWidget):
    """Real camera feed widget using OpenCV"""

    def __init__(self, camera_index=0, show_roi=True):
        super().__init__()
        self.setMinimumHeight(300)
        self.camera_index = camera_index
        self.show_roi = show_roi
        # Load ROI from config
        roi_tuple = load_roi_from_config()
        self.roi_rect = QRect(roi_tuple[0], roi_tuple[1], roi_tuple[2], roi_tuple[3])
        self.current_frame = None
        self.camera_error = False
        self.error_message = ""
        self.cap = None

        # Try to initialize camera
        self.init_camera()

        # Timer for frame updates
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(33)  # ~30 FPS

    def init_camera(self):
        """Initialize OpenCV camera"""
        try:
            print(f"Attempting to open camera {self.camera_index}...")

            # Release previous camera if exists
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

            # Test frame capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Camera opened but cannot read frames")

            self.camera_error = False
            self.error_message = ""
            print(f"✅ Camera {self.camera_index} initialized successfully")
            print(
                f"   Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        except Exception as e:
            self.camera_error = True
            self.error_message = str(e)
            if self.cap:
                self.cap.release()
                self.cap = None
            print(f"❌ Camera initialization failed: {e}")

    def change_camera(self, camera_index):
        """Change to a different camera index"""
        if camera_index != self.camera_index:
            print(f"Switching from camera {self.camera_index} to camera {camera_index}")
            self.camera_index = camera_index
            self.init_camera()

    def update_frame(self):
        """Capture and update frame from camera"""
        if self.camera_error or not self.cap:
            return

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Store current frame
                self.current_frame = frame.copy()
                self.update()  # Trigger repaint
            else:
                # Frame read failed
                if not self.camera_error:
                    print("⚠️ Frame read failed, camera may be disconnected")
                    self.camera_error = True
                    self.error_message = "Lost connection to camera"

        except Exception as e:
            if not self.camera_error:
                print(f"⚠️ Error reading frame: {e}")
                self.camera_error = True
                self.error_message = f"Camera error: {str(e)}"

    def opencv_to_qpixmap(self, cv_img):
        """Convert OpenCV image to QPixmap"""
        if cv_img is None:
            return None

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Create QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line,
                          QImage.Format.Format_RGB888 if PYQT_VERSION == "PyQt6" else QImage.Format_RGB888)

        # Convert to QPixmap and scale to widget size
        pixmap = QPixmap.fromImage(qt_image)
        return pixmap.scaled(self.size(),
                             Qt.AspectRatioMode.KeepAspectRatio if PYQT_VERSION == "PyQt6" else Qt.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation if PYQT_VERSION == "PyQt6" else Qt.SmoothTransformation)

    def paintEvent(self, event):
        """Paint the camera feed"""
        painter = QPainter(self)

        if self.camera_error:
            # Draw error message
            painter.fillRect(self.rect(), QColor(40, 40, 40))
            painter.setPen(QPen(QColor(255, 100, 100), 2))
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold if PYQT_VERSION == "PyQt6" else QFont.Bold))

            error_text = f"📷 Camera {self.camera_index} Error"
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter,
                             error_text)

            # Draw detailed error message
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.setFont(QFont("Arial", 10))
            error_rect = QRect(10, self.height() // 2 + 30, self.width() - 20, 100)
            painter.drawText(error_rect, Qt.TextFlag.TextWordWrap if PYQT_VERSION == "PyQt6" else Qt.TextWordWrap | (
                Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter),
                             f"Error: {self.error_message}\n\nTry selecting a different camera or check connection.")

        elif self.current_frame is not None:
            # Draw camera feed
            pixmap = self.opencv_to_qpixmap(self.current_frame)
            if pixmap:
                # Calculate position to center the image
                x = (self.width() - pixmap.width()) // 2
                y = (self.height() - pixmap.height()) // 2
                painter.drawPixmap(x, y, pixmap)

                # Draw ROI rectangle overlay if enabled
                if self.show_roi:
                    painter.setPen(QPen(QColor(0, 255, 0), 3))  # Green ROI, thicker line

                    # Calculate ROI position relative to displayed image
                    if pixmap.width() > 0 and pixmap.height() > 0:
                        scale_x = pixmap.width() / 640  # Assuming 640 as base width
                        scale_y = pixmap.height() / 480  # Assuming 480 as base height

                        roi_display = QRect(
                            x + int(self.roi_rect.x() * scale_x),
                            y + int(self.roi_rect.y() * scale_y),
                            int(self.roi_rect.width() * scale_x),
                            int(self.roi_rect.height() * scale_y)
                        )
                        painter.drawRect(roi_display)

                        # Draw ROI label
                        painter.setPen(QPen(QColor(0, 255, 0), 1))
                        painter.setFont(
                            QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == "PyQt6" else QFont.Bold))
                        painter.drawText(roi_display.x(), roi_display.y() - 8, "ROI - Detection Zone")
        else:
            # Draw loading message
            painter.fillRect(self.rect(), QColor(40, 40, 40))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == "PyQt6" else Qt.AlignCenter,
                             f"📷 Initializing camera {self.camera_index}...")

    def set_roi(self, roi_rect):
        """Set ROI rectangle"""
        self.roi_rect = roi_rect
        self.update()

    def get_roi(self):
        """Get current ROI rectangle"""
        return self.roi_rect

    def cleanup(self):
        """Clean up camera resources"""
        if hasattr(self, 'frame_timer'):
            self.frame_timer.stop()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            print("🔌 Camera released")


def detect_available_cameras(max_cameras=3):
    """Detect available camera indices"""
    available_cameras = []

    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    print(f"✅ Found working camera at index {i}")
                else:
                    print(f"⚠️ Camera {i} opens but cannot read frames")
            cap.release()
        except:
            pass

    if not available_cameras:
        print("❌ No working cameras detected")
    else:
        print(f"📷 Available cameras: {available_cameras}")

    return available_cameras


class SortingInterface(QWidget):
    """Main sorting interface"""

    pause_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.is_paused = False
        self.processed_count = 0
        self.bin_counts = [0] * 8  # 8 bins
        self.current_piece = None
        self.uptime_start = time.time()

        # Add placeholder for pieces queue (will be populated by real detector)
        self.pieces_queue = []

        self.init_ui()
        self.setup_timers()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Top bar with title and controls
        top_layout = QHBoxLayout()

        title = QLabel(f"LEGO Sorting - {self.config['strategy']}")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        top_layout.addWidget(title)

        top_layout.addStretch()

        self.pause_btn = QPushButton("⏸️ PAUSE")
        self.pause_btn.clicked.connect(self.toggle_pause)
        top_layout.addWidget(self.pause_btn)

        stop_btn = QPushButton("⏹️ STOP")
        stop_btn.clicked.connect(self.stop_requested.emit)
        stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        top_layout.addWidget(stop_btn)

        layout.addLayout(top_layout)

        # Camera feed (full width)
        camera_index = self.config.get("camera_index", 0)
        self.camera_widget = RealCameraWidget(camera_index=camera_index, show_roi=True)
        layout.addWidget(self.camera_widget)

        # Bottom section with bin assignments and recent piece
        bottom_layout = QHBoxLayout()

        # Bin assignments
        bins_widget = QGroupBox("Bin Assignments")
        bins_layout = QVBoxLayout(bins_widget)

        self.bin_labels = []
        bin_assignments = [
            "Overflow", "Basic Bricks", "Plates", "Curves",
            "Technic", "Unassigned", "Unassigned", "Unassigned"
        ]

        for i, assignment in enumerate(bin_assignments):
            label = QLabel(f"Bin {i}: {assignment} [0]")
            label.setStyleSheet(
                "margin: 2px; padding: 5px; background-color: #f0f0f0; border-radius: 3px; font-family: monospace;")
            bins_layout.addWidget(label)
            self.bin_labels.append(label)

        bottom_layout.addWidget(bins_widget)

        # Recent piece info
        recent_widget = QGroupBox("Most Recent Piece")
        recent_layout = QVBoxLayout(recent_widget)

        self.recent_info = QLabel("No pieces processed yet")
        self.recent_info.setStyleSheet("padding: 10px; background-color: #f9f9f9; border-radius: 5px;")
        self.recent_info.setWordWrap(True)
        recent_layout.addWidget(self.recent_info)

        bottom_layout.addWidget(recent_widget)
        layout.addLayout(bottom_layout)

        # Status bar
        self.status_bar = QLabel("Status: Running | Processed: 0 pieces | Queue: 0")
        self.status_bar.setStyleSheet(
            "padding: 5px; background-color: #e0e0e0; border-radius: 3px; font-family: monospace;")
        layout.addWidget(self.status_bar)

        self.setLayout(layout)

    def setup_timers(self):
        """Setup timers for updates"""
        # Timer for updating status
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second

        # Note: Real piece processing will be triggered by the detector module
        # For now, we'll just have the status update timer

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.setText("▶️ RESUME")
            # Pause camera updates
            if hasattr(self.camera_widget, 'frame_timer'):
                self.camera_widget.frame_timer.stop()
        else:
            self.pause_btn.setText("⏸️ PAUSE")
            # Resume camera updates
            if hasattr(self.camera_widget, 'frame_timer'):
                self.camera_widget.frame_timer.start(33)

    def process_piece_detected(self, piece_data):
        """Called when detector detects a piece (placeholder for integration)"""
        # This method will be called by the real detector
        # For now, just update the UI
        self.current_piece = piece_data
        self.update_recent_piece()
        self.processed_count += 1

        # Simulate bin assignment (will come from real sorting module)
        bin_number = piece_data.get('bin_number', 0)
        self.bin_counts[bin_number] += 1
        self.update_bin_counts()

    def update_recent_piece(self):
        """Update recent piece display"""
        if self.current_piece:
            # Display placeholder info until real data is available
            info_text = f"""Waiting for piece detection...

Camera feed is active.
Real-time detection will appear here."""
            self.recent_info.setText(info_text)

    def update_bin_counts(self):
        """Update bin count displays"""
        bin_assignments = [
            "Overflow", "Dynamic", "Dynamic", "Dynamic",
            "Dynamic", "Dynamic", "Dynamic", "Dynamic"
        ]

        for i, (assignment, count) in enumerate(zip(bin_assignments, self.bin_counts)):
            self.bin_labels[i].setText(f"Bin {i}: {assignment} [{count}]")

    def update_status(self):
        """Update status bar"""
        if not self.is_paused:
            uptime = time.time() - self.uptime_start
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)

            # Use the placeholder queue for now
            queue_size = len(self.pieces_queue)

            # Calculate actual FPS from camera widget if available
            fps = 30.0  # Default
            if hasattr(self.camera_widget, 'cap') and self.camera_widget.cap:
                try:
                    fps = self.camera_widget.cap.get(cv2.CAP_PROP_FPS)
                except:
                    fps = 30.0

            status_text = (f"Status: {'Paused' if self.is_paused else 'Running'} | "
                           f"Processed: {self.processed_count} pieces | "
                           f"Queue: {queue_size}")

            # Check camera status
            camera_status = "Connected" if not self.camera_widget.camera_error else "Error"

            detail_text = (f"FPS: {fps:.1f} | "
                           f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d} | "
                           f"Camera: {camera_status}")

            self.status_bar.setText(f"{status_text}\n{detail_text}")

    def cleanup(self):
        """Clean up resources when stopping"""
        if hasattr(self, 'camera_widget'):
            self.camera_widget.cleanup()

class LegoSortingGUI(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEGO Sorting System")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)

        # Initialize category database
        self.category_database = None

        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Initialize screens
        self.setup_screens()

        # Start with startup screen
        self.show_startup()

    def setup_screens(self):
        """Setup all application screens"""
        # Startup screen
        self.startup_screen = StartupScreen()
        self.startup_screen.startup_complete.connect(self.show_mode_selection)
        self.stacked_widget.addWidget(self.startup_screen)

        # Mode selection screen
        self.mode_screen = ModeSelectionScreen()
        self.mode_screen.mode_selected.connect(self.show_configuration)
        self.mode_screen.exit_requested.connect(self.close)
        self.stacked_widget.addWidget(self.mode_screen)

        # Configuration screen
        self.config_screen = ConfigurationScreen()
        self.config_screen.start_sorting.connect(self.start_sorting)
        self.config_screen.back_requested.connect(self.show_mode_selection)
        self.stacked_widget.addWidget(self.config_screen)

    def show_startup(self):
        """Show startup screen"""
        self.stacked_widget.setCurrentWidget(self.startup_screen)

    def show_mode_selection(self):
        """Show mode selection screen"""
        self.stacked_widget.setCurrentWidget(self.mode_screen)

    def show_configuration(self, mode):
        """Show configuration screen"""
        if mode == "sorting":
            # Pass category database to config screen
            if hasattr(self, 'category_database') and self.category_database:
                self.config_screen.set_category_database(self.category_database)
            self.stacked_widget.setCurrentWidget(self.config_screen)
        else:
            QMessageBox.information(self, "Coming Soon", "Inventory mode will be available in a future update!")
    def start_sorting(self, config):
        """Start sorting with given configuration"""
        # Clean up configuration screen camera
        self.config_screen.cleanup()

        self.sorting_screen = SortingInterface(config)
        self.sorting_screen.stop_requested.connect(lambda: self.stop_sorting())
        self.stacked_widget.addWidget(self.sorting_screen)
        self.stacked_widget.setCurrentWidget(self.sorting_screen)

    def stop_sorting(self):
        """Stop sorting and return to mode selection"""
        if hasattr(self, 'sorting_screen'):
            self.sorting_screen.cleanup()
        self.show_mode_selection()

    def closeEvent(self, event):
        """Handle application close"""
        reply = QMessageBox.question(self, "Exit", "Are you sure you want to exit?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No if PYQT_VERSION == "PyQt6" else QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.StandardButton.Yes if PYQT_VERSION == "PyQt6" else QMessageBox.Yes:
            # Clean up camera resources from all screens
            if hasattr(self, 'config_screen'):
                self.config_screen.cleanup()
            if hasattr(self, 'sorting_screen') and hasattr(self.sorting_screen, 'camera_widget'):
                self.sorting_screen.camera_widget.cleanup()
            event.accept()
        else:
            event.ignore()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("LEGO Sorting System")
    app.setApplicationVersion("1.0")

    # Create and show main window
    window = LegoSortingGUI()
    window.show()

    print("LEGO Sorting GUI (Phase 2A Enhanced) started successfully!")
    print("✅ Real OpenCV camera feed with ROI preview")
    print("✅ Camera selection and detection")
    print("✅ Clean interface ready for detection integration")
    print("📷 Mock simulation removed - ready for real piece detection")

    sys.exit(app.exec() if PYQT_VERSION == "PyQt6" else app.exec_())


if __name__ == "__main__":
    main()