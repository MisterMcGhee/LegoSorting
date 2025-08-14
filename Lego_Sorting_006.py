#!/usr/bin/env python3
"""
Lego_Sorting_006_Fixed.py - Corrected GUI-First Lego Sorting Application

This version fixes the import errors and implements missing functionality
for proper GUI integration with the sorting system.

Author: Lego Sorting System Team
Version: 6.1.0
"""

import os
import sys
import argparse
import signal
import time
import logging
import json
import threading
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime

# PyQt5 imports
from PyQt5.QtCore import (
    QObject, QThread, pyqtSignal, pyqtSlot, QTimer,
    QMutex, QMutexLocker, Qt
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QPushButton, QLabel,
    QMessageBox, QStatusBar, QToolBar, QAction, QMenuBar,
    QProgressBar, QGroupBox, QGridLayout, QTextEdit,
    QSplitter, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont

# Import modules
from enhanced_config_manager import create_config_manager, ModuleConfig
from camera_module import CameraModule
from detector_module import create_detector
from sorting_module import create_sorting_manager
from thread_management_module import create_thread_manager, PieceMessage
from arduino_servo_module import create_arduino_servo_module
from piece_history_module import create_piece_history
from error_module import setup_logging, get_logger
from GUI_module import ConfigurationScreen, MainWindow, LegoSortingGUI  # Fixed import
from ui_module import create_ui_manager

# Import cv2 and numpy
import cv2
import numpy as np

# Initialize logger
logger = get_logger(__name__)


# ============= Application States =============

class ApplicationState(Enum):
    """Application state machine states"""
    STARTUP = "startup"
    CONFIGURING = "configuring"
    VALIDATING = "validating"
    INITIALIZING = "initializing"
    SORTING = "sorting"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    SHUTDOWN = "shutdown"


# ============= Camera Resource Manager =============

class CameraResourceManager:
    """Manages camera resource sharing between GUI and sorting system"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.camera = None
                    cls._instance.users = set()
                    cls._instance.config_manager = None
        return cls._instance

    def acquire_camera(self, user_id: str, config_manager=None):
        """Acquire camera for a user"""
        with self._lock:
            if self.camera is None:
                if config_manager:
                    self.config_manager = config_manager
                if self.config_manager:
                    self.camera = CameraModule.get_instance(self.config_manager)
            self.users.add(user_id)
            logger.debug(f"Camera acquired by {user_id}. Active users: {self.users}")
            return self.camera

    def release_camera(self, user_id: str):
        """Release camera from a user"""
        with self._lock:
            self.users.discard(user_id)
            logger.debug(f"Camera released by {user_id}. Active users: {self.users}")
            if not self.users and self.camera:
                self.camera.release()
                self.camera = None
                logger.info("Camera fully released")


# ============= State Manager =============

class StateManager(QObject):
    """Manages application state transitions"""

    state_changed = pyqtSignal(ApplicationState)

    def __init__(self):
        super().__init__()
        self.current_state = ApplicationState.STARTUP
        self.state_history = []
        self.mutex = QMutex()

        # Define valid transitions
        self.transitions = {
            ApplicationState.STARTUP: [ApplicationState.CONFIGURING],
            ApplicationState.CONFIGURING: [ApplicationState.VALIDATING, ApplicationState.SHUTDOWN],
            ApplicationState.VALIDATING: [ApplicationState.CONFIGURING, ApplicationState.INITIALIZING],
            ApplicationState.INITIALIZING: [ApplicationState.SORTING, ApplicationState.ERROR],
            ApplicationState.SORTING: [ApplicationState.PAUSED, ApplicationState.STOPPING, ApplicationState.ERROR],
            ApplicationState.PAUSED: [ApplicationState.SORTING, ApplicationState.STOPPING],
            ApplicationState.STOPPING: [ApplicationState.CONFIGURING, ApplicationState.SHUTDOWN],
            ApplicationState.ERROR: [ApplicationState.CONFIGURING, ApplicationState.SHUTDOWN],
            ApplicationState.SHUTDOWN: []
        }

    def can_transition_to(self, new_state: ApplicationState) -> bool:
        """Check if transition to new state is valid"""
        with QMutexLocker(self.mutex):
            return new_state in self.transitions.get(self.current_state, [])

    def transition_to(self, new_state: ApplicationState) -> bool:
        """Attempt to transition to new state"""
        with QMutexLocker(self.mutex):
            if new_state not in self.transitions.get(self.current_state, []):
                logger.warning(f"Invalid transition: {self.current_state} -> {new_state}")
                return False

            # Record history
            self.state_history.append({
                'from': self.current_state,
                'to': new_state,
                'timestamp': datetime.now()
            })

            # Update state
            old_state = self.current_state
            self.current_state = new_state
            logger.info(f"State transition: {old_state.value} -> {new_state.value}")

            # Emit signal
            self.state_changed.emit(new_state)
            return True


# ============= Metrics Tracker =============

class MetricsTracker(QObject):
    """Tracks and reports system metrics"""

    metrics_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.metrics = {
            'pieces_detected': 0,
            'pieces_processed': 0,
            'pieces_sorted': 0,
            'errors': 0,
            'uptime': 0,
            'average_processing_time': 0.0,
            'efficiency': 0.0,
            'servo_movements': 0
        }
        self.start_time = time.time()
        self.processing_times = []
        self.mutex = QMutex()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(1000)  # Update every second

    def update_metrics(self):
        """Update calculated metrics"""
        with QMutexLocker(self.mutex):
            # Update uptime
            self.metrics['uptime'] = int(time.time() - self.start_time)

            # Calculate efficiency
            if self.metrics['pieces_detected'] > 0:
                self.metrics['efficiency'] = (
                                                     self.metrics['pieces_sorted'] / self.metrics['pieces_detected']
                                             ) * 100

            # Calculate average processing time
            if self.processing_times:
                self.metrics['average_processing_time'] = sum(self.processing_times) / len(self.processing_times)

            self.metrics_updated.emit(self.metrics.copy())

    def record_detection(self):
        """Record a piece detection"""
        with QMutexLocker(self.mutex):
            self.metrics['pieces_detected'] += 1

    def record_processing(self, processing_time: float):
        """Record piece processing completion"""
        with QMutexLocker(self.mutex):
            self.metrics['pieces_processed'] += 1
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:  # Keep last 100 times
                self.processing_times.pop(0)

    def record_sorting(self):
        """Record successful sorting"""
        with QMutexLocker(self.mutex):
            self.metrics['pieces_sorted'] += 1
            self.metrics['servo_movements'] += 1

    def record_error(self):
        """Record an error"""
        with QMutexLocker(self.mutex):
            self.metrics['errors'] += 1




# ============= Processing Worker =============

class ProcessingWorker(QThread):
    """Worker thread for processing detected pieces"""

    piece_processed = pyqtSignal(dict)
    processing_error = pyqtSignal(str)

    def __init__(self, config_manager, modules):
        super().__init__()
        self.config_manager = config_manager
        self.modules = modules
        self.is_running = False
        self.piece_queue = []
        self.queue_mutex = QMutex()

    def add_piece(self, piece_msg: PieceMessage):
        """Add piece to processing queue"""
        with QMutexLocker(self.queue_mutex):
            self.piece_queue.append(piece_msg)

    def run(self):
        """Process pieces from queue"""
        self.is_running = True

        while self.is_running:
            piece_msg = None

            # Get piece from queue
            with QMutexLocker(self.queue_mutex):
                if self.piece_queue:
                    piece_msg = self.piece_queue.pop(0)

            if piece_msg:
                try:
                    result = self.process_piece(piece_msg)
                    if result:
                        self.piece_processed.emit(result)
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    self.processing_error.emit(str(e))
            else:
                self.msleep(10)  # Short sleep when queue is empty

    def process_piece(self, piece_msg: PieceMessage) -> Optional[Dict[str, Any]]:
        """Process a single piece"""
        try:
            # Save image
            save_dir = self.config_manager.get('processing.save_directory', 'LegoPictures')
            os.makedirs(save_dir, exist_ok=True)

            filename = f"Lego_{piece_msg.piece_id:06d}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, piece_msg.cropped_image)

            # Simulate API identification (replace with actual API call)
            api_result = {
                'element_id': f'ELEM{piece_msg.piece_id:04d}',
                'name': f'Test Brick {piece_msg.piece_id}',
                'category': 'Brick',
                'subcategory': 'Basic'
            }

            # Determine bin
            if 'sorting' in self.modules:
                sorting_decision = self.modules['sorting'].determine_bin(
                    element_id=api_result.get('element_id'),
                    primary_category=api_result.get('category'),
                    secondary_category=api_result.get('subcategory')
                )
            else:
                sorting_decision = {'bin_number': 1}

            # Queue Arduino movement if available
            if 'arduino' in self.modules and self.modules['arduino']:
                self.modules['arduino'].queue_movement(sorting_decision['bin_number'])

            # Create result
            result = {
                'piece_id': piece_msg.piece_id,
                'timestamp': piece_msg.timestamp,
                'image_path': filepath,
                'element_id': api_result.get('element_id'),
                'name': api_result.get('name'),
                'category': api_result.get('category'),
                'bin_number': sorting_decision['bin_number']
            }

            # Add to history
            if 'history' in self.modules:
                self.modules['history'].add_piece(result)

            return result

        except Exception as e:
            logger.error(f"Failed to process piece {piece_msg.piece_id}: {e}")
            return None

    def stop(self):
        """Stop processing"""
        self.is_running = False


# ============= Main Application =============

class LegoSorting006(QObject):
    """Main application controller"""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()

        # Initialize configuration
        self.config_manager = create_config_manager(config_path)

        # Initialize camera resource manager
        self.camera_manager = CameraResourceManager()

        # Initialize state manager
        self.state_manager = StateManager()
        self.state_manager.state_changed.connect(self.on_state_changed)

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()

        # Windows
        self.config_window = None
        self.sorting_gui = None

        # Modules
        self.modules = {}
        self.camera = None

        # Processing
        self.processing_worker = None

        # Start with configuration
        self.state_manager.transition_to(ApplicationState.CONFIGURING)

    @pyqtSlot(ApplicationState)
    def on_state_changed(self, new_state: ApplicationState):
        """Handle state changes"""
        logger.info(f"Entering state: {new_state.value}")

        if new_state == ApplicationState.CONFIGURING:
            self.enter_configuration_state()
        elif new_state == ApplicationState.INITIALIZING:
            self.enter_initialization_state()
        elif new_state == ApplicationState.SORTING:
            self.enter_sorting_state()
        elif new_state == ApplicationState.PAUSED:
            self.enter_paused_state()
        elif new_state == ApplicationState.STOPPING:
            self.enter_stopping_state()
        elif new_state == ApplicationState.ERROR:
            self.enter_error_state()
        elif new_state == ApplicationState.SHUTDOWN:
            self.shutdown()

    def enter_configuration_state(self):
        """Enter configuration state"""
        # Create configuration window
        self.config_window = MainWindow()
        self.config_window.config_screen.config_confirmed.connect(self.on_config_confirmed)
        self.config_window.show()

    def on_config_confirmed(self):
        """Handle configuration confirmation"""
        logger.info("Configuration confirmed, initializing system")

        # Validate configuration
        validation_report = self.config_manager.get_validation_report()
        if not validation_report['overall_valid']:
            QMessageBox.critical(
                self.config_window,
                "Configuration Error",
                "Invalid configuration detected. Please review settings."
            )
            return

        # Save configuration
        self.config_manager.save_config()

        # Transition to initialization
        self.state_manager.transition_to(ApplicationState.INITIALIZING)

    def enter_initialization_state(self):
        """Initialize system for sorting"""
        try:
            # Hide configuration window
            if self.config_window:
                self.config_window.hide()

            # Create sorting GUI
            self.sorting_gui = LegoSortingGUI(self.config_manager)
            self.sorting_gui.pause_requested.connect(self.pause_sorting)
            self.sorting_gui.resume_requested.connect(self.resume_sorting)
            self.sorting_gui.stop_requested.connect(self.stop_sorting)

            # Connect metrics
            self.metrics_tracker.metrics_updated.connect(self.sorting_gui.update_metrics)

            # Show sorting GUI
            self.sorting_gui.show()
            self.sorting_gui.status_bar.showMessage("Initializing modules...")

            # Initialize camera
            self.camera = self.camera_manager.acquire_camera("sorting", self.config_manager)

            # Create modules
            self.modules = {
                'detector': create_detector(self.config_manager),
                'sorting': create_sorting_manager(self.config_manager),
                'history': create_piece_history(),
                'thread_mgr': create_thread_manager(self.config_manager)
            }

            # Try to create Arduino module
            try:
                self.modules['arduino'] = create_arduino_servo_module(self.config_manager)
            except Exception as e:
                logger.warning(f"Arduino module not available: {e}")
                self.modules['arduino'] = None

            # Create processing worker
            self.processing_worker = ProcessingWorker(self.config_manager, self.modules)
            self.processing_worker.piece_processed.connect(self.on_piece_processed)

            # Transition to sorting
            self.state_manager.transition_to(ApplicationState.SORTING)

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            QMessageBox.critical(None, "Initialization Error", f"Failed to initialize: {str(e)}")
            self.state_manager.transition_to(ApplicationState.ERROR)

    def enter_sorting_state(self):
        """Start sorting operation"""
        logger.info("Starting sorting operation")
        self.sorting_gui.status_bar.showMessage("Sorting Active")

        # Start processing worker
        if self.processing_worker:
            self.processing_worker.start()

        # Start detection loop (simplified for this example)
        self.start_detection_loop()

    def enter_paused_state(self):
        """Enter paused state"""
        logger.info("System paused")
        # Pause detection and processing

    def enter_stopping_state(self):
        """Stop sorting operation"""
        logger.info("Stopping sorting operation")

        # Stop processing
        if self.processing_worker:
            self.processing_worker.stop()
            self.processing_worker.wait()

        # Release camera
        self.camera_manager.release_camera("sorting")

        # Close sorting GUI
        if self.sorting_gui:
            self.sorting_gui.close()

        # Return to configuration
        self.state_manager.transition_to(ApplicationState.CONFIGURING)

    def enter_error_state(self):
        """Handle error state"""
        logger.error("System in error state")
        QMessageBox.critical(
            None,
            "System Error",
            "A critical error has occurred. The system will return to configuration."
        )
        self.state_manager.transition_to(ApplicationState.CONFIGURING)

    def start_detection_loop(self):
        """Start the detection loop (simplified)"""
        # This would normally be more complex with proper threading
        # For now, just demonstrate the connection
        pass

    @pyqtSlot(dict)
    def on_piece_processed(self, piece_record):
        """Handle processed piece"""
        self.metrics_tracker.record_sorting()
        self.sorting_gui.add_piece_to_display(piece_record)

    def pause_sorting(self):
        """Pause sorting operation"""
        if self.state_manager.can_transition_to(ApplicationState.PAUSED):
            self.state_manager.transition_to(ApplicationState.PAUSED)

    def resume_sorting(self):
        """Resume sorting operation"""
        if self.state_manager.can_transition_to(ApplicationState.SORTING):
            self.state_manager.transition_to(ApplicationState.SORTING)

    def stop_sorting(self):
        """Stop sorting operation"""
        if self.state_manager.can_transition_to(ApplicationState.STOPPING):
            self.state_manager.transition_to(ApplicationState.STOPPING)

    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down application")

        # Stop all workers
        if self.processing_worker:
            self.processing_worker.stop()
            self.processing_worker.wait()

        # Release resources
        self.camera_manager.release_camera("sorting")

        # Close windows
        if self.config_window:
            self.config_window.close()
        if self.sorting_gui:
            self.sorting_gui.close()

        QApplication.quit()


# ============= Main Entry Point =============

def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Lego Sorting System v6.1")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--calibrate-servo', action='store_true', help='Enable servo calibration mode')
    parser.add_argument('--calibrate-sorting', action='store_true', help='Enable sorting calibration mode')
    parser.add_argument('--profile', type=str, help='Load configuration profile')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(console_level=log_level)

    logger.info("=" * 60)
    logger.info("  Lego Sorting System v6.1 - Starting")
    logger.info("=" * 60)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Lego Sorting System")
    app.setApplicationDisplayName("Lego Sorting System v6.1")

    try:
        # Create main application
        lego_app = LegoSorting006(args.config)

        # Handle command line options
        if args.calibrate_servo:
            lego_app.config_manager.set("servo", "calibration_mode", True)
            logger.info("Servo calibration mode enabled")

        if args.calibrate_sorting:
            lego_app.config_manager.set("sorting", "calibrate_sorting_strategy", True)
            logger.info("Sorting calibration mode enabled")

        if args.profile:
            profile_path = f"profiles/{args.profile}.json"
            if os.path.exists(profile_path):
                lego_app.config_manager.import_config(profile_path)
                logger.info(f"Loaded profile: {args.profile}")
            else:
                logger.warning(f"Profile not found: {profile_path}")

        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, lambda s, f: lego_app.shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: lego_app.shutdown())

        # Run application
        sys.exit(app.exec_())

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        QMessageBox.critical(None, "Fatal Error",
                             f"Application failed to start:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()