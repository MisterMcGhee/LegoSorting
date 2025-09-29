"""
gui_common.py - Common utilities and base classes for GUI modules modules
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
from typing import Dict, Any, Optional


# ============= Style Constants =============
class GUIStyles:
    """Centralized style definitions"""

    DARK_THEME = """
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
        QSlider::groove:horizontal {
            height: 8px;
            background: #555;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #888;
            width: 20px;
            margin: -6px 0;
            border-radius: 10px;
        }
        QComboBox, QSpinBox, QLineEdit {
            background-color: #555;
            color: white;
            border: 1px solid #666;
            padding: 5px;
            border-radius: 3px;
        }
        QTabWidget::pane {
            border: 1px solid #555;
            background-color: #2b2b2b;
        }
        QTabBar::tab {
            background-color: #444;
            color: white;
            padding: 8px 20px;
        }
        QTabBar::tab:selected {
            background-color: #666;
        }
        QStatusBar {
            background-color: #333;
            color: white;
        }
    """


# ============= Base GUI modules Class =============
class BaseGUIWindow(QMainWindow):
    """Base class for all GUI modules windows with common functionality"""

    def __init__(self, config_manager=None, window_title="Lego Sorting System"):
        super().__init__()
        self.config_manager = config_manager
        self.setWindowTitle(window_title)
        self.apply_theme()

    def apply_theme(self):
        """Apply the dark theme to the window"""
        self.setStyleSheet(GUIStyles.DARK_THEME)

    def center_window(self):
        """Center the window on screen"""
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)


# ============= Common Widgets =============
class StatusDisplay(QWidget):
    """Common status display widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Ready")
        self.fps_label = QLabel("FPS: 0.0")
        self.processed_label = QLabel("Processed: 0")

        layout.addWidget(self.status_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.processed_label)

        self.setLayout(layout)

    def update_status(self, status: str):
        self.status_label.setText(f"Status: {status}")

    def update_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_processed_count(self, count: int):
        self.processed_label.setText(f"Processed: {count}")


class VideoWidget(QLabel):
    """Widget for displaying video frames"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self.setText("No Video Signal")
        self.setStyleSheet("border: 2px solid #555; background-color: #1a1a1a;")

    def update_frame(self, frame: np.ndarray):
        """Update the displayed frame"""
        if frame is None:
            self.setText("No Video Signal")
            return

        # Convert frame to Qt format
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        # Ensure frame is in RGB format (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create QImage and QPixmap
        q_image = QImage(rgb_frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Scale to widget size while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


# ============= Common Dialogs =============
class ConfirmationDialog(QMessageBox):
    """Standard confirmation dialog"""

    @staticmethod
    def confirm(parent, title: str, message: str) -> bool:
        reply = QMessageBox.question(parent, title, message,
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        return reply == QMessageBox.Yes


# ============= Utility Functions =============
def validate_config(config_manager) -> tuple[bool, str]:
    """
    Validate that essential configuration is present
    Returns: (is_valid, error_message)
    """
    required_modules = ['camera', 'detector', 'sorting', 'api']

    for module in required_modules:
        config = config_manager.get_module_config(module)
        if not config:
            return False, f"Missing configuration for {module}"

    # Check specific critical settings
    camera_config = config_manager.get_module_config('camera')
    if camera_config.get('device_id') is None:
        return False, "Camera device ID not set"

    return True, ""


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"