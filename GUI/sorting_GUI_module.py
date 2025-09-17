"""
sorting_gui_module.py - Active sorting GUI for Lego Sorting System
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List
from GUI.gui_common import BaseGUIWindow, VideoWidget, StatusDisplay, format_time

import logging
logger = logging.getLogger(__name__)
logger.info("sorting_GUI_module imported")



class SortingGUI(BaseGUIWindow):
    """Main GUI window for active sorting operation"""

    # Signals
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    settings_requested = pyqtSignal()  # Return to configuration

    def __init__(self, config_manager=None):
        logger.info("SortingGUI.__init__ starting")

        super().__init__(config_manager, "Lego Sorting System - Active Sorting")
        logger.info("Parent class initialized")

        # State
        self.is_paused = False
        self.start_time = time.time()
        self.frame_count = 0
        self.piece_count = 0
        logger.info("State variables initialized")

        # ROI and zone configuration (set once during initialization)
        self.roi_config = None
        self.roi_bounds = None
        self.zones = None
        logger.info("ROI variables initialized")

        # Window setup
        self.setGeometry(100, 100, 1400, 900)
        logger.info("Geometry set")

        self.center_window()
        logger.info("Window centered")

        # Initialize UI
        logger.info("Calling init_ui...")
        self.init_ui()
        logger.info("init_ui complete")

        logger.info("Calling setup_timers...")
        self.setup_timers()
        logger.info("setup_timers complete")

        logger.info("SortingGUI.__init__ complete")

    def set_roi_configuration(self, roi_data: Dict[str, Any]):
        """
        Set the ROI configuration - PHASE 1: STORE BUT DON'T USE
        """
        if "error" in roi_data:
            self.add_log_message(f"ROI configuration error: {roi_data['error']}", "ERROR")
            return

        # PHASE 1: Store the configuration but don't use it for drawing yet
        self.roi_bounds = roi_data.get("roi")
        self.zones = {
            "entry": roi_data.get("entry_zone"),
            "valid": roi_data.get("valid_zone"),
            "exit": roi_data.get("exit_zone")
        }

        if self.roi_bounds:
            x, y, w, h = self.roi_bounds
            # PHASE 1: Just log that we received ROI, don't use it for drawing
            self.add_log_message(f"ROI configured but not displayed: {w}x{h} at ({x},{y})", "INFO")

    def init_ui(self):
        """Initialize the user interface"""
        logger.debug("Starting UI initialization...")

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        logger.debug("Central widget created")

        # Main layout - horizontal split
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        logger.debug("Main layout created")

        # Left panel - Video and controls
        logger.debug("Creating left panel...")
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=3)
        logger.debug("Left panel added")

        # Right panel - Information displays
        logger.debug("Creating right panel...")
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        logger.debug("Right panel added")

        # Status bar
        logger.debug("Creating status bar...")
        self.create_status_bar()
        logger.debug("UI initialization complete")

    def create_left_panel(self) -> QWidget:
        """Create left panel with video and controls"""
        logger.info("create_left_panel starting")

        panel = QWidget()
        layout = QVBoxLayout()
        logger.info("Left panel base created")

        # Debug: Check before creating VideoWidget
        logger.info(f"Creating VideoWidget - current parent: {panel}")

        # Video display
        self.video_widget = VideoWidget(panel)
        logger.info(f"VideoWidget created successfully")

        self.video_widget.setMinimumSize(800, 600)
        logger.info("VideoWidget size set")

        layout.addWidget(self.video_widget)
        logger.info("VideoWidget added to layout")

        # Control buttons
        controls = self.create_control_buttons()
        logger.info("Control buttons created")

        layout.addWidget(controls)
        logger.info("Controls added to layout")

        panel.setLayout(layout)
        logger.info("create_left_panel complete")
        return panel

    def create_right_panel(self) -> QWidget:
        """Create right panel with information displays"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Metrics display
        metrics_group = QGroupBox("System Metrics")
        metrics_layout = QFormLayout()

        self.fps_label = QLabel("0.0")
        metrics_layout.addRow("FPS:", self.fps_label)

        self.processed_label = QLabel("0")
        metrics_layout.addRow("Pieces Processed:", self.processed_label)

        self.runtime_label = QLabel("00:00")
        metrics_layout.addRow("Runtime:", self.runtime_label)

        self.queue_label = QLabel("0")
        metrics_layout.addRow("Queue Size:", self.queue_label)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Current piece display
        piece_group = QGroupBox("Current Piece")
        piece_layout = QVBoxLayout()

        self.piece_image_label = QLabel()
        self.piece_image_label.setMinimumHeight(150)
        self.piece_image_label.setStyleSheet("border: 1px solid #555;")
        self.piece_image_label.setScaledContents(True)
        self.piece_image_label.setText("No piece")
        self.piece_image_label.setAlignment(Qt.AlignCenter)
        piece_layout.addWidget(self.piece_image_label)

        self.piece_info_label = QLabel("Waiting for piece...")
        self.piece_info_label.setWordWrap(True)
        piece_layout.addWidget(self.piece_info_label)

        piece_group.setLayout(piece_layout)
        layout.addWidget(piece_group)

        # Bin status display
        bin_group = QGroupBox("Bin Status")
        bin_layout = QGridLayout()

        self.bin_indicators = []
        for i in range(10):
            indicator = QLabel(f"Bin {i}")
            indicator.setAlignment(Qt.AlignCenter)
            indicator.setStyleSheet("""
                QLabel {
                    background-color: #444;
                    border: 1px solid #666;
                    padding: 5px;
                    border-radius: 3px;
                }
            """)
            bin_layout.addWidget(indicator, i // 5, i % 5)
            self.bin_indicators.append(indicator)

        bin_group.setLayout(bin_layout)
        layout.addWidget(bin_group)

        # System log
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_control_buttons(self) -> QWidget:
        """Create control button panel"""
        widget = QWidget()
        layout = QHBoxLayout()

        # Pause/Resume button
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setMinimumWidth(100)

        # Stop button
        self.stop_btn = QPushButton("Stop Sorting")
        self.stop_btn.clicked.connect(self.stop_sorting)
        self.stop_btn.setMinimumWidth(100)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        # Settings button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        self.settings_btn.setMinimumWidth(100)

        layout.addStretch()
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.settings_btn)
        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add permanent widgets
        self.status_label = QLabel("Sorting Active")
        self.camera_status = QLabel("Camera: OK")
        self.arduino_status = QLabel("Arduino: OK")

        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.camera_status)
        self.status_bar.addPermanentWidget(self.arduino_status)

    def setup_timers(self):
        """Setup update timers"""
        logger.info("setup_timers starting")

        # UI update timer (10 Hz)
        self.ui_timer = QTimer()
        logger.info("UI timer created")

        self.ui_timer.timeout.connect(self.update_ui)
        logger.info("UI timer connected")

        self.ui_timer.start(100)
        logger.info("UI timer started")

        # Metrics update timer (1 Hz)
        self.metrics_timer = QTimer()
        logger.info("Metrics timer created")

        self.metrics_timer.timeout.connect(self.update_metrics)
        logger.info("Metrics timer connected")

        self.metrics_timer.start(1000)
        logger.info("Metrics timer started")

        logger.info("setup_timers complete")

    # ========== Update Methods ==========

    def update_frame(self, frame: np.ndarray, pieces_data: Dict[str, Any] = None):
        """PHASE 1: Display raw camera feed only"""

        if frame is None:
            logger.warning("Received None frame in update_frame")
            return

        self.frame_count += 1

        if self.frame_count % 30 == 0:
            logger.info(f"GUI processing frame {self.frame_count}")

        # PHASE 1: COMMENT OUT THESE LINES
        # if self.roi_bounds:
        #     frame = self.draw_roi_overlay(frame)

        # if pieces_data and 'tracked_pieces' in pieces_data:
        #     frame = self.draw_tracked_pieces(frame, pieces_data['tracked_pieces'])

        # PHASE 1: ONLY THIS LINE SHOULD RUN
        self.video_widget.update_frame(frame)

    def draw_roi_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw static ROI and zone boundaries on frame - DISABLED IN PHASE 1"""
        # PHASE 1: This method exists but won't be called
        display_frame = frame.copy()

        # Draw main ROI rectangle
        if self.roi_bounds:
            x, y, w, h = self.roi_bounds
            cv2.rectangle(display_frame,
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0), 2)  # Green for ROI

            # Draw zone dividers if available
            if self.zones:
                # Entry zone boundary
                if self.zones.get("entry"):
                    entry_end = self.zones["entry"][1]
                    cv2.line(display_frame,
                             (entry_end, y),
                             (entry_end, y + h),
                             (255, 255, 0), 1)  # Yellow line

                # Exit zone boundary
                if self.zones.get("exit"):
                    exit_start = self.zones["exit"][0]
                    cv2.line(display_frame,
                             (exit_start, y),
                             (exit_start, y + h),
                             (0, 255, 255), 1)  # Cyan line

        return display_frame

    def draw_tracked_pieces(self, frame: np.ndarray, pieces: List[Dict]) -> np.ndarray:
        """Draw tracked pieces with color coding based on status"""
        display_frame = frame.copy()

        for piece in pieces:
            # Get piece position (now already in frame coordinates!)
            x, y, w, h = piece['bbox']

            # Get color based on status
            status = piece.get('status', 'detected')
            color = self.get_piece_color(status)

            # Draw rectangle around piece
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

            # Add label with piece ID
            label = f"ID: {piece.get('id', '?')}"
            if piece.get('in_exit_zone'):
                label += " [EXIT]"

            cv2.putText(display_frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return display_frame

    def get_piece_color(self, status: str) -> tuple:
        """
        Get color for piece based on status string.

        Args:
            status: Status string ('detected', 'processing', 'processed', 'error')

        Returns:
            Tuple of BGR color values for OpenCV
        """
        colors = {
            'detected': (0, 0, 255),  # Red - newly detected piece
            'processing': (0, 165, 255),  # Orange - being processed by API
            'processed': (0, 255, 0),  # Green - processing complete
            'error': (0, 0, 128)  # Dark red - processing error
        }

        # Return the color for the status, or gray if status unknown
        return colors.get(status, (128, 128, 128))

    def update_piece_display(self, piece_data: Dict[str, Any]):
        """Update current piece display"""
        if 'image' in piece_data:
            # Convert numpy array to QPixmap
            image = piece_data['image']
            height, width = image.shape[:2]

            if len(image.shape) == 3:
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height,
                                 bytes_per_line, QImage.Format_RGB888)
            else:
                q_image = QImage(image.data, width, height,
                                 width, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(150, 150, Qt.KeepAspectRatio)
            self.piece_image_label.setPixmap(scaled)

        # Update piece info
        info_text = f"ID: {piece_data.get('id', 'Unknown')}\n"
        info_text += f"Category: {piece_data.get('category', 'Unknown')}\n"
        info_text += f"Bin: {piece_data.get('bin', 'TBD')}"
        self.piece_info_label.setText(info_text)

        self.piece_count += 1

    def update_metrics(self):
        """Update metrics display"""
        # Calculate runtime
        runtime = time.time() - self.start_time
        self.runtime_label.setText(format_time(runtime))

        # Calculate FPS
        if runtime > 0:
            fps = self.frame_count / runtime
            self.fps_label.setText(f"{fps:.1f}")

    def update_ui(self):
        """Regular UI updates"""
        # This is called by timer for any regular updates needed
        pass

    def update_bin_status(self, bin_number: int, status: str):
        """Update bin indicator status"""
        if 0 <= bin_number < len(self.bin_indicators):
            colors = {
                'active': '#4CAF50',  # Green
                'idle': '#444',  # Gray
                'error': '#f44336'  # Red
            }
            color = colors.get(status, '#444')
            self.bin_indicators[bin_number].setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    border: 1px solid #666;
                    padding: 5px;
                    border-radius: 3px;
                }}
            """)

    def add_log_message(self, message: str, level: str = "INFO"):
        """Add message to system log"""
        timestamp = time.strftime("%H:%M:%S")

        colors = {
            "DEBUG": "#888",
            "INFO": "#FFF",
            "WARNING": "#FFA500",
            "ERROR": "#FF4444"
        }
        color = colors.get(level, "#FFF")

        html_message = f'<span style="color: {color};">[{timestamp}] {message}</span>'
        self.log_text.append(html_message)

        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ========== Control Methods ==========

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_btn.setText("Resume")
            self.status_label.setText("Sorting Paused")
            self.pause_requested.emit()
            self.add_log_message("Sorting paused", "INFO")
        else:
            self.pause_btn.setText("Pause")
            self.status_label.setText("Sorting Active")
            self.resume_requested.emit()
            self.add_log_message("Sorting resumed", "INFO")

    def stop_sorting(self):
        """Stop sorting operation"""
        reply = QMessageBox.question(self, "Stop Sorting",
                                     "Are you sure you want to stop sorting?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.add_log_message("Stopping sorting system...", "WARNING")
            self.stop_requested.emit()

    def open_settings(self):
        """Request to open settings"""
        if self.is_paused or QMessageBox.question(self, "Open Settings",
                                                  "This will pause sorting. Continue?",
                                                  QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:

            if not self.is_paused:
                self.toggle_pause()
            self.settings_requested.emit()

    # ========== Public Interface ==========

    def set_camera_status(self, is_connected: bool):
        """Update camera status indicator"""
        if is_connected:
            self.camera_status.setText("Camera: OK")
            self.camera_status.setStyleSheet("color: #4CAF50;")
        else:
            self.camera_status.setText("Camera: Error")
            self.camera_status.setStyleSheet("color: #f44336;")

    def set_arduino_status(self, is_connected: bool):
        """Update Arduino status indicator"""
        if is_connected:
            self.arduino_status.setText("Arduino: OK")
            self.arduino_status.setStyleSheet("color: #4CAF50;")
        else:
            self.arduino_status.setText("Arduino: N/A")
            self.arduino_status.setStyleSheet("color: #FFA500;")

    def set_queue_size(self, size: int):
        """Update queue size display"""
        self.queue_label.setText(str(size))

        # Change color based on queue size
        if size > 50:
            self.queue_label.setStyleSheet("color: #FFA500;")  # Orange
        elif size > 75:
            self.queue_label.setStyleSheet("color: #f44336;")  # Red
        else:
            self.queue_label.setStyleSheet("color: #4CAF50;")  # Green

    def set_processed_count(self, count: int):
        """Update processed piece count"""
        self.processed_label.setText(str(count))

    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(self, "Exit",
                                     "Stop sorting and exit?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.stop_requested.emit()
            event.accept()
        else:
            event.ignore()