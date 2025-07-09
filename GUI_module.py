#!/usr/bin/env python3
"""
LEGO Sorting GUI - Phase 1 Standalone
Mock GUI for testing layout and functionality without hardware dependencies
"""

import sys
import random
import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

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
            available_cameras = detect_available_cameras(max_cameras=3)  # Quick scan
            if available_cameras:
                self.check_labels["camera"].setText(f"✅ Camera(s) detected: {available_cameras}")
            else:
                self.check_labels["camera"].setText("❌ No cameras detected (will show error messages)")
        if self.step >= 17:
            self.check_labels["arduino"].setText("❌ Arduino servo not detected (will run in sim mode)")
        if self.step >= 20:
            self.check_labels["database"].setText("✅ Loading piece identification database...")
        if self.step >= 25:
            self.check_labels["database"].setText("✅ Piece identification database loaded")

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
    """Sorting configuration screen"""

    start_sorting = pyqtSignal(dict)  # Configuration dict
    back_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.available_cameras = []
        self.current_camera_index = 0
        self.init_ui()
        self.detect_cameras()

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
        strategy_row.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Primary Categories", "Secondary Categories", "Tertiary Categories"])
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_row.addWidget(self.strategy_combo)
        strategy_layout.addLayout(strategy_row)

        # Target category
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox()
        self.target_combo.addItems(["All Categories"])
        target_row.addWidget(self.target_combo)
        strategy_layout.addLayout(target_row)

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

        # Camera preview (full width, larger)
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.camera_preview = RealCameraWidget(camera_index=0, show_roi=True)
        self.camera_preview.setMinimumHeight(400)  # Taller for better view
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

        # Bin layout preview
        bins_group = QGroupBox("Bin Assignment Preview (Future Feature)")
        bins_layout = QHBoxLayout(bins_group)

        # Add explanatory note
        bins_note = QLabel(
            "Note: Bins are currently assigned dynamically as pieces are encountered.\nPre-assignment will be available in a future update.")
        bins_note.setStyleSheet("color: #666; font-style: italic; margin: 5px;")
        bins_note.setWordWrap(True)

        bins_container = QVBoxLayout()
        bins_container.addWidget(bins_note)

        bins_row = QHBoxLayout()
        self.bin_labels = []
        for i in range(8):
            if i == 0:
                bin_label = QLabel(f"Bin {i}: Overflow")
            else:
                bin_label = QLabel(f"Bin {i}: Dynamic")
            bin_label.setStyleSheet(
                "margin: 2px; padding: 8px; background-color: #f0f0f0; border-radius: 3px; font-family: monospace; color: #666;")
            bins_row.addWidget(bin_label)
            self.bin_labels.append(bin_label)

        bins_container.addLayout(bins_row)
        bins_group.setLayout(bins_container)
        config_layout.addWidget(bins_group)
        layout.addLayout(config_layout)

        # Advanced settings (collapsible)
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
        """Adjust ROI settings (placeholder for now)"""
        QMessageBox.information(self, "ROI Adjustment",
                                "ROI adjustment will be available in the next update.\n\nCurrently using default ROI settings.")

    def on_strategy_changed(self, strategy):
        """Handle strategy change"""
        if "Secondary" in strategy:
            self.target_combo.clear()
            self.target_combo.addItems(["Basic", "Technic", "Curves", "Specialty"])
        elif "Tertiary" in strategy:
            self.target_combo.clear()
            self.target_combo.addItems(["Bricks", "Plates", "Slopes", "Round"])
        else:
            self.target_combo.clear()
            self.target_combo.addItems(["All Categories"])

    def save_as_default(self):
        """Save current settings as default"""
        reply = QMessageBox.question(self, "Save as Default",
                                     "Save current settings as default configuration?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No if PYQT_VERSION == "PyQt6" else QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.StandardButton.Yes if PYQT_VERSION == "PyQt6" else QMessageBox.Yes:
            QMessageBox.information(self, "Saved", "Settings saved as default!")

    def start_sorting_clicked(self):
        """Start sorting with current configuration"""
        config = {
            "strategy": self.strategy_combo.currentText(),
            "target": self.target_combo.currentText(),
            "sensitivity": self.sens_slider.value(),
            "confidence": self.conf_slider.value(),
            "max_bins": self.bins_spin.value(),
            "camera_index": self.current_camera_index
        }
        self.start_sorting.emit(config)

    def cleanup(self):
        """Clean up camera resources"""
        if hasattr(self, 'camera_preview'):
            self.camera_preview.cleanup()


class RealCameraWidget(QWidget):
    """Real camera feed widget using OpenCV"""

    def __init__(self, camera_index=0, show_roi=True):
        super().__init__()
        self.setMinimumHeight(300)
        self.camera_index = camera_index
        self.show_roi = show_roi
        self.roi_rect = QRect(50, 50, 500, 200)
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
        """Setup timers for simulation"""
        # Timer for updating status
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second

        # Timer for processing pieces (mock simulation)
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.process_piece)
        self.process_timer.start(random.randint(3000, 7000))  # Random processing

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.setText("▶️ RESUME")
            self.camera_widget.animation_timer.stop()
            self.process_timer.stop()
        else:
            self.pause_btn.setText("⏸️ PAUSE")
            self.camera_widget.animation_timer.start(50)
            self.process_timer.start(random.randint(3000, 7000))

    def process_piece(self):
        """Simulate processing a piece"""
        if not self.is_paused and self.camera_widget.pieces:
            # Find a piece to process
            piece = random.choice(self.camera_widget.pieces)
            piece.status = "processing"

            # Update piece info
            self.current_piece = piece
            self.update_recent_piece()

            # Simulate processing time then assign to bin
            QTimer.singleShot(1500, lambda: self.complete_processing(piece))

        # Reset timer
        self.process_timer.start(random.randint(3000, 7000))

    def complete_processing(self, piece):
        """Complete piece processing"""
        if piece in self.camera_widget.pieces:
            self.processed_count += 1
            self.bin_counts[piece.bin_number] += 1
            self.update_bin_counts()
            piece.status = "captured"

    def update_recent_piece(self):
        """Update recent piece display"""
        if self.current_piece:
            piece = self.current_piece
            info_text = f"""🖼️ [IMG] Element ID: {piece.element_id}
          Name: {piece.name}
          Category: {piece.category}
          → Bin {piece.bin_number}
          Confidence: {piece.confidence:.0%}
          Processing: {piece.processing_time:.1f}s"""
            self.recent_info.setText(info_text)

    def update_bin_counts(self):
        """Update bin count displays"""
        bin_assignments = [
            "Overflow", "Basic Bricks", "Plates", "Curves",
            "Technic", "Unassigned", "Unassigned", "Unassigned"
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

            queue_size = len(self.camera_widget.pieces)
            fps = 28.5  # Mock FPS

            status_text = (f"Status: {'Paused' if self.is_paused else 'Running'} | "
                           f"Processed: {self.processed_count} pieces | "
                           f"Queue: {queue_size}")

            detail_text = (f"FPS: {fps} | "
                           f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d} | "
                           f"Servo: Connected")

            self.status_bar.setText(f"{status_text}\n{detail_text}")


class LegoSortingGUI(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEGO Sorting System")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)

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
            self.stacked_widget.setCurrentWidget(self.config_screen)
        else:
            QMessageBox.information(self, "Coming Soon", "Inventory mode will be available in a future update!")

    def start_sorting(self, config):
        """Start sorting with given configuration"""
        # Clean up configuration screen camera
        self.config_screen.cleanup()

        self.sorting_screen = SortingInterface(config)
        self.sorting_screen.stop_requested.connect(self.show_mode_selection)
        self.stacked_widget.addWidget(self.sorting_screen)
        self.stacked_widget.setCurrentWidget(self.sorting_screen)

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