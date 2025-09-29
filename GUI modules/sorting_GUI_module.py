"""
sorting_GUI_module.py - Phase 3.1: Live Camera Feed

This module implements the LEGO Sorting System GUI modules with live camera feed.
Phase 3.1 adds actual camera frame consumption and display.

Phase 3.1 Goals:
- Replace camera placeholder with live video feed
- Register as camera consumer using existing pattern
- Convert OpenCV frames to Qt format
- Handle frame display and updates
- Maintain proper cleanup when closing
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional
from GUI.gui_common import BaseGUIWindow, VideoWidget

import logging
logger = logging.getLogger(__name__)


class SortingGUI(BaseGUIWindow):
    """
    Main GUI modules window for active sorting operation

    Phase 3.1: Live camera feed implementation
    """

    # Signals for control
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    settings_requested = pyqtSignal()

    # Signal for frame updates (emitted in main thread)
    frame_update_signal = pyqtSignal(np.ndarray)

    def __init__(self, config_manager=None):
        super().__init__(config_manager, "LEGO Sorting System - Active Sorting (Phase 3.1)")
        logger.info("SortingGUI Phase 3.1 - Live Camera Feed")

        # State variables
        self.is_paused = False
        self.roi_config = None

        # Camera integration - Phase 3.1
        self.camera = None
        self.camera_active = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0

        # Window setup
        self.setGeometry(100, 100, 1400, 900)
        self.center_window()

        # Connect frame update signal to slot (for thread safety)
        self.frame_update_signal.connect(self.update_camera_display)

        # Initialize UI
        self.init_ui()

        logger.info("SortingGUI Phase 3.1 initialization complete")

    def init_ui(self):
        """Initialize the user interface with placeholder panels"""
        logger.info("Initializing Phase 2 UI layout...")

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left side: Camera feed (largest section)
        self.create_camera_section(main_layout)

        # Right side: Information and controls
        self.create_right_panel(main_layout)

        # Set layout proportions: 70% camera, 30% right panel
        main_layout.setStretch(0, 7)  # Camera section
        main_layout.setStretch(1, 3)  # Right panel

        logger.info("Phase 2 UI layout created successfully")

    def create_camera_section(self, parent_layout):
        """Create the camera feed section with live video display"""
        logger.info("Creating camera section with live feed...")

        # Camera section frame
        camera_frame = QFrame()
        camera_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        camera_frame.setLineWidth(2)

        # Camera layout
        camera_layout = QVBoxLayout(camera_frame)

        # Title
        camera_title = QLabel("Live Camera Feed")
        camera_title.setAlignment(Qt.AlignCenter)
        camera_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2E86AB;
                padding: 5px;
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
                border-radius: 5px;
            }
        """)
        camera_layout.addWidget(camera_title)

        # Phase 3.1: Replace placeholder with actual VideoWidget
        self.video_widget = VideoWidget()
        self.video_widget.setText("Initializing Camera Feed...")
        self.video_widget.setStyleSheet("""
            VideoWidget {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 2px solid #34495E;
                border-radius: 10px;
                min-height: 400px;
            }
        """)
        camera_layout.addWidget(self.video_widget)

        # Camera status bar
        camera_status_layout = QHBoxLayout()

        self.camera_status_label = QLabel("Camera: Connecting...")
        self.camera_status_label.setStyleSheet("color: #F39C12; font-weight: bold;")
        camera_status_layout.addWidget(self.camera_status_label)

        camera_status_layout.addStretch()

        self.frame_counter_label = QLabel("Frames: 0")
        self.frame_counter_label.setStyleSheet("color: #7F8C8D;")
        camera_status_layout.addWidget(self.frame_counter_label)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #7F8C8D;")
        camera_status_layout.addWidget(self.fps_label)

        camera_layout.addLayout(camera_status_layout)

        parent_layout.addWidget(camera_frame)
        logger.info("Camera section with live feed created")

    def create_right_panel(self, parent_layout):
        """Create the right panel with info and controls"""
        logger.info("Creating right panel sections...")

        # Right panel container
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        # Recently processed piece section
        self.create_recent_piece_section(right_layout)

        # Bin status section
        self.create_bin_status_section(right_layout)

        # Control panel section
        self.create_control_section(right_layout)

        # Add stretch to push everything up
        right_layout.addStretch()

        parent_layout.addWidget(right_panel)
        logger.info("Right panel sections created")

    def create_recent_piece_section(self, parent_layout):
        """Create the recently processed piece panel"""
        logger.info("Creating recent piece section placeholder...")

        # Section frame
        piece_frame = QFrame()
        piece_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        piece_frame.setFixedHeight(200)

        piece_layout = QVBoxLayout(piece_frame)

        # Title
        piece_title = QLabel("🧱 Recently Processed Piece")
        piece_title.setAlignment(Qt.AlignCenter)
        piece_title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #27AE60;
                padding: 5px;
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
                border-radius: 3px;
            }
        """)
        piece_layout.addWidget(piece_title)

        # Placeholder content
        self.piece_placeholder = QLabel()
        self.piece_placeholder.setAlignment(Qt.AlignCenter)
        self.piece_placeholder.setText(
            "📦 RECENT PIECE PLACEHOLDER\n\n"
            "Phase 4 Implementation:\n"
            "• Piece image thumbnail\n"
            "• Element ID and name\n"
            "• Confidence score\n"
            "• Category information\n"
            "• Bin assignment"
        )
        self.piece_placeholder.setStyleSheet("""
            QLabel {
                background-color: #F7F9FC;
                color: #2C3E50;
                border: 1px dashed #BDC3C7;
                border-radius: 5px;
                padding: 10px;
                font-size: 11px;
            }
        """)
        piece_layout.addWidget(self.piece_placeholder)

        parent_layout.addWidget(piece_frame)
        logger.info("Recent piece section placeholder created")

    def create_bin_status_section(self, parent_layout):
        """Create the bin status display section"""
        logger.info("Creating bin status section placeholder...")

        # Section frame
        bin_frame = QFrame()
        bin_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        bin_frame.setFixedHeight(250)

        bin_layout = QVBoxLayout(bin_frame)

        # Title
        bin_title = QLabel("📊 Bin Status")
        bin_title.setAlignment(Qt.AlignCenter)
        bin_title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #8E44AD;
                padding: 5px;
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
                border-radius: 3px;
            }
        """)
        bin_layout.addWidget(bin_title)

        # Placeholder content
        self.bin_placeholder = QLabel()
        self.bin_placeholder.setAlignment(Qt.AlignCenter)
        self.bin_placeholder.setText(
            "🗂️ BIN STATUS PLACEHOLDER\n\n"
            "Phase 5 Implementation:\n"
            "• Bin 1-9 + Overflow\n"
            "• Assigned categories\n"
            "• Current piece counts\n"
            "• Capacity progress bars\n"
            "• Reset bin buttons\n"
            "• Warning indicators"
        )
        self.bin_placeholder.setStyleSheet("""
            QLabel {
                background-color: #FDF6E3;
                color: #2C3E50;
                border: 1px dashed #BDC3C7;
                border-radius: 5px;
                padding: 10px;
                font-size: 11px;
            }
        """)
        bin_layout.addWidget(self.bin_placeholder)

        parent_layout.addWidget(bin_frame)
        logger.info("Bin status section placeholder created")

    def create_control_section(self, parent_layout):
        """Create the control panel section"""
        logger.info("Creating control section...")

        # Section frame
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        control_frame.setFixedHeight(180)

        control_layout = QVBoxLayout(control_frame)

        # Title
        control_title = QLabel("🎮 System Controls")
        control_title.setAlignment(Qt.AlignCenter)
        control_title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #E67E22;
                padding: 5px;
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
                border-radius: 3px;
            }
        """)
        control_layout.addWidget(control_title)

        # Control buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        # Pause/Resume button
        self.pause_button = QPushButton("⏸️ Pause Sorting")
        self.pause_button.setFixedHeight(35)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
            QPushButton:pressed {
                background-color: #D35400;
            }
        """)
        button_layout.addWidget(self.pause_button)

        # Stop button
        self.stop_button = QPushButton("⏹️ Stop Sorting")
        self.stop_button.setFixedHeight(35)
        self.stop_button.clicked.connect(self.stop_sorting)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:pressed {
                background-color: #A93226;
            }
        """)
        button_layout.addWidget(self.stop_button)

        # Settings button
        self.settings_button = QPushButton("⚙️ Settings")
        self.settings_button.setFixedHeight(35)
        self.settings_button.clicked.connect(self.open_settings)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
            QPushButton:pressed {
                background-color: #6C7B7D;
            }
        """)
        button_layout.addWidget(self.settings_button)

        control_layout.addLayout(button_layout)

        # Status display
        self.status_label = QLabel("Status: Ready for Phase 3")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #27AE60;
                font-weight: bold;
                font-size: 11px;
                padding: 5px;
                background-color: #D5EDDA;
                border: 1px solid #C3E6CB;
                border-radius: 3px;
            }
        """)
        control_layout.addWidget(self.status_label)

        parent_layout.addWidget(control_frame)
        logger.info("Control section created")

    # ========== PHASE 3.1: CAMERA INTEGRATION METHODS ==========

    def set_camera(self, camera):
        """Set camera instance for live feed"""
        self.camera = camera
        logger.info("Camera instance set for sorting GUI modules")

    def register_camera_consumer(self, camera):
        """Register as camera consumer during initialization (before capture starts)"""
        if not camera:
            logger.error("No camera provided for consumer registration")
            return False

        try:
            logger.info("Registering sorting GUI modules as camera consumer...")

            # Debug: Check for any OpenCV windows before registration
            import cv2
            logger.info(f"OpenCV windows before consumer registration: {cv2.getWindowImageRect('any_window') if hasattr(cv2, 'getWindowImageRect') else 'Cannot check'}")

            success = camera.register_consumer(
                name="sorting_gui",
                callback=self._camera_frame_callback,
                processing_type="async",
                priority=20  # High priority for GUI modules display
            )

            if success:
                self.camera = camera
                self.camera_active = True
                logger.info("Sorting GUI modules registered as camera consumer successfully")

                # Debug: Check for OpenCV windows after registration
                logger.info(f"OpenCV windows after consumer registration: {cv2.getWindowImageRect('any_window') if hasattr(cv2, 'getWindowImageRect') else 'Cannot check'}")

                return True
            else:
                logger.error("Failed to register sorting GUI modules as camera consumer")
                return False

        except Exception as e:
            logger.error(f"Error registering sorting GUI modules as camera consumer: {e}", exc_info=True)
            return False

    def activate_camera_display(self):
        """Activate camera display after capture has started"""
        logger.info("Activating camera display...")

        # Debug: Check for OpenCV windows before activation
        import cv2
        try:
            # Try to detect any OpenCV windows
            logger.info("Checking for OpenCV windows...")
            # OpenCV doesn't have a direct way to list windows, so we'll check other things
        except Exception as e:
            logger.warning(f"Could not check OpenCV windows: {e}")

        if self.camera_active:
            self.camera_status_label.setText("Camera: Active")
            self.camera_status_label.setStyleSheet("color: #27AE60; font-weight: bold;")
            logger.info("Camera display activated")
        else:
            self.camera_status_label.setText("Camera: Not Registered")
            self.camera_status_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
            logger.warning("Camera display activation failed - not registered as consumer")

    def stop_camera_feed(self):
        """Stop camera feed and unregister consumer"""
        if self.camera and self.camera_active:
            try:
                self.camera.unregister_consumer("sorting_gui")
                self.camera_active = False
                self.camera_status_label.setText("Camera: Stopped")
                self.camera_status_label.setStyleSheet("color: #95A5A6; font-weight: bold;")
                self.video_widget.setText("Camera Feed Stopped")
                logger.info("Camera feed stopped for sorting GUI modules")
            except Exception as e:
                logger.error(f"Error stopping camera feed: {e}")

    def _camera_frame_callback(self, frame: np.ndarray):
        """
        Callback function for camera frames.

        This runs in a camera thread, so we emit a signal to update GUI modules in main thread.
        """
        try:
            # Debug: Log first frame callback
            if self.frame_count == 0:
                logger.info(f"FIRST camera callback - frame shape: {frame.shape}, dtype: {frame.dtype}")
                # Check if any windows exist when first frame arrives
                import cv2
                logger.info("First frame callback - checking for any OpenCV windows or displays")

            # Update frame counter
            self.frame_count += 1

            # Calculate FPS periodically
            current_time = time.time()
            self.fps_counter += 1

            if current_time - self.last_fps_time >= 1.0:  # Update every second
                fps = self.fps_counter / (current_time - self.last_fps_time)
                self.fps_counter = 0
                self.last_fps_time = current_time

                # Log FPS periodically
                if self.frame_count % 30 == 0:
                    logger.info(f"GUI modules camera feed FPS: {fps:.1f}")

                # Emit signal to update FPS in main thread
                QMetaObject.invokeMethod(
                    self.fps_label,
                    "setText",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"FPS: {fps:.1f}")
                )

            # Emit signal for frame update (thread-safe)
            self.frame_update_signal.emit(frame.copy())

        except Exception as e:
            logger.error(f"Error in camera frame callback: {e}", exc_info=True)

    @pyqtSlot(np.ndarray)
    def update_camera_display(self, frame):
        """
        Update camera display in main GUI modules thread.

        This slot receives frames via signal and updates the VideoWidget.
        """
        try:
            # CHECKPOINT 5: Before first VideoWidget update
            if self.frame_count == 1:
                reply = QMessageBox.question(
                    self,
                    "Black Window Debug - Checkpoint 5",
                    "First camera frame received and about to update VideoWidget.\n\n"
                    "Is there a black window visible now?\n"
                    "(About to call VideoWidget.update_frame())",
                    QMessageBox.Yes | QMessageBox.No
                )
                logger.info(f"CHECKPOINT 5 - Black window present: {reply == QMessageBox.Yes}")

            # Update frame counter display
            self.frame_counter_label.setText(f"Frames: {self.frame_count}")

            # Update video widget - this might be where extra window appears
            self.video_widget.update_frame(frame)

            # CHECKPOINT 6: After first VideoWidget update
            if self.frame_count == 1:
                reply = QMessageBox.question(
                    self,
                    "Black Window Debug - Checkpoint 6",
                    "VideoWidget.update_frame() completed for first frame.\n\n"
                    "Is there a black window visible now?\n"
                    "(First frame has been displayed in VideoWidget)",
                    QMessageBox.Yes | QMessageBox.No
                )
                logger.info(f"CHECKPOINT 6 - Black window present: {reply == QMessageBox.Yes}")
                logger.info("Black window debugging checkpoints complete - continuing normal operation")

        except Exception as e:
            logger.error(f"Error updating camera display: {e}", exc_info=True)

    # ========== PLACEHOLDER METHODS FOR FUTURE PHASES ==========

    def set_roi_configuration(self, roi_data: Dict[str, Any]):
        """Store ROI configuration for future phases"""
        self.roi_config = roi_data
        logger.info(f"ROI configuration stored for Phase 3.2: {roi_data}")

    def set_arduino_status(self, is_connected: bool):
        """Update Arduino status (placeholder for now)"""
        status = "Connected" if is_connected else "Simulation Mode"
        logger.info(f"Arduino status: {status}")

    # ========== CONTROL METHODS ==========

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_button.setText("▶️ Resume Sorting")
            self.status_label.setText("Status: Paused")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #E67E22;
                    font-weight: bold;
                    font-size: 11px;
                    padding: 5px;
                    background-color: #FDF2E9;
                    border: 1px solid #FADBD8;
                    border-radius: 3px;
                }
            """)
            self.pause_requested.emit()
            logger.info("Pause requested from GUI modules")
        else:
            self.pause_button.setText("⏸️ Pause Sorting")
            self.status_label.setText("Status: Sorting Active")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #27AE60;
                    font-weight: bold;
                    font-size: 11px;
                    padding: 5px;
                    background-color: #D5EDDA;
                    border: 1px solid #C3E6CB;
                    border-radius: 3px;
                }
            """)
            self.resume_requested.emit()
            logger.info("Resume requested from GUI modules")

    def stop_sorting(self):
        """Stop sorting operation"""
        reply = QMessageBox.question(
            self,
            "Stop Sorting",
            "Are you sure you want to stop sorting and return to configuration?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            logger.info("Stop sorting requested from GUI modules")
            self.stop_requested.emit()

    def open_settings(self):
        """Open settings/configuration"""
        reply = QMessageBox.question(
            self,
            "Open Settings",
            "This will pause sorting and return to configuration. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            logger.info("Settings requested from GUI modules")
            self.settings_requested.emit()

    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(
            self,
            "Exit Sorting",
            "Stop sorting and exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            logger.info("Window close requested")

            # Phase 3.1: Clean up camera feed
            self.stop_camera_feed()

            self.stop_requested.emit()
            event.accept()
        else:
            event.ignore()

    # ========== PHASE 3.1 INFO METHOD ==========

    def show_phase_info(self):
        """Show information about current phase"""
        QMessageBox.information(
            self,
            "Phase 3.1: Live Camera Feed Complete",
            "Phase 3.1 Successfully Implemented!\n\n"
            "Current Features:\n"
            "• Live camera feed display\n"
            "• Frame rate monitoring\n"
            "• Camera consumer registration\n"
            "• Thread-safe frame updates\n\n"
            "Next: Phase 3.2 - ROI Overlay\n"
            "Ready to add detection region visualization!"
        )