#!/usr/bin/env python3
"""
Lego_Sorting_006.py - Integrated Modular GUI modules Version

This version integrates the refactored modular GUI modules components while maintaining
all existing functionality. The GUI modules is now split into three modules:
- GUI modules/gui_common.py: Shared utilities and base classes
- GUI modules/config_gui_module.py: Configuration interface
- GUI modules/sorting_gui_module.py: Active sorting visualization

Main improvements:
- Better separation of concerns
- Easier maintenance and testing
- Cleaner state management
- Improved error handling
"""

import sys
import os
import signal
import argparse
import logging
import time
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# NEW MODULAR GUI modules IMPORTS
from GUI.config_GUI_module import ConfigurationGUI
from GUI.sorting_GUI_module import SortingGUI
from GUI.gui_common import validate_config

# Existing module imports
from enhanced_config_manager import create_config_manager
from camera_module import create_camera
from processing_module import processing_worker_thread
from Old_versions.detector_module import create_detector
from sorting_module import create_sorting_manager
from piece_history_module import create_piece_history
from thread_management_module import create_thread_manager
from api_module import create_api_client
from arduino_servo_module import create_arduino_servo_module
from error_module import setup_logging, get_logger

# Initialize logger
logger = get_logger(__name__)


# ============= Application States =============

class ApplicationState(Enum):
    """Application state enumeration"""
    INITIALIZING = "initializing"
    CONFIGURING = "configuring"
    SORTING = "sorting"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class StateTransition:
    """State transition definition"""
    from_state: ApplicationState
    to_state: ApplicationState
    condition: Optional[callable] = None
    action: Optional[callable] = None


# ============= State Manager =============

class ApplicationStateManager(QObject):
    """Manages application state transitions"""

    state_changed = pyqtSignal(ApplicationState, ApplicationState)

    def __init__(self):
        super().__init__()
        self.current_state = ApplicationState.INITIALIZING
        self.previous_state = None
        self.transitions = {}
        self._setup_transitions()

    def _setup_transitions(self):
        """Define valid state transitions"""
        self.transitions = {
            (ApplicationState.INITIALIZING, ApplicationState.CONFIGURING): True,
            (ApplicationState.CONFIGURING, ApplicationState.INITIALIZING): True,
            (ApplicationState.INITIALIZING, ApplicationState.SORTING): True,
            (ApplicationState.SORTING, ApplicationState.PAUSED): True,
            (ApplicationState.PAUSED, ApplicationState.SORTING): True,
            (ApplicationState.SORTING, ApplicationState.CONFIGURING): True,
            (ApplicationState.PAUSED, ApplicationState.CONFIGURING): True,
            (ApplicationState.SORTING, ApplicationState.SHUTTING_DOWN): True,
            (ApplicationState.CONFIGURING, ApplicationState.SHUTTING_DOWN): True,
            (ApplicationState.ERROR, ApplicationState.SHUTTING_DOWN): True,
        }

    def can_transition(self, to_state: ApplicationState) -> bool:
        """Check if transition is valid"""
        return (self.current_state, to_state) in self.transitions

    def transition_to(self, new_state: ApplicationState) -> bool:
        """Transition to new state"""
        if not self.can_transition(new_state):
            logger.warning(f"Invalid transition: {self.current_state} -> {new_state}")
            return False

        self.previous_state = self.current_state
        self.current_state = new_state
        logger.info(f"State transition: {self.previous_state} -> {self.current_state}")
        self.state_changed.emit(self.previous_state, self.current_state)
        return True

# ============= Metrics Tracker =============

class MetricsTracker(QObject):
    """Tracks and reports system metrics"""

    metrics_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        self.frame_count = 0
        self.piece_count = 0
        self.last_update = time.time()
        self.last_frame_count = 0

    def record_frame(self):
        """Record processed frame"""
        self.frame_count += 1

    def record_piece_processed(self):
        """Record processed piece"""
        self.piece_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        update_interval = current_time - self.last_update

        # Calculate FPS
        frames_in_interval = self.frame_count - self.last_frame_count
        fps = frames_in_interval / update_interval if update_interval > 0 else 0

        self.last_update = current_time
        self.last_frame_count = self.frame_count

        return {
            'fps': fps,
            'processed_count': self.piece_count,
            'frame_count': self.frame_count,
            'runtime': elapsed,
            'camera_status': True,  # Will be updated based on actual status
            'queue_size': 0  # Will be updated from thread manager
        }

    def emit_metrics(self):
        """Emit current metrics"""
        metrics = self.get_metrics()
        self.metrics_updated.emit(metrics)

    def _update_metrics(self):
        """Update system metrics (called by timer)"""
        try:
            # Get queue size from thread manager
            queue_size = self.modules['thread_mgr'].get_queue_size()

            # Get camera statistics
            camera_stats = self.camera.get_statistics()

            # Update GUI modules
            if self.sorting_gui:
                # Update FPS
                fps = camera_stats['camera']['fps_actual']
                self.sorting_gui.fps_label.setText(f"{fps:.1f}")

                # Update queue size
                self.sorting_gui.set_queue_size(queue_size)

                # Update processed count
                self.sorting_gui.set_processed_count(self.processed_pieces)

                # Update camera status
                self.sorting_gui.set_camera_status(self.camera.is_capturing)

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

# ============= Main Application =============

class LegoSorting006(QObject):
    """Main application controller with modular GUI modules"""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()

        # Initialize configuration
        self.config_manager = create_config_manager(config_path)

        # Initialize state manager
        self.state_manager = ApplicationStateManager()
        self.state_manager.state_changed.connect(self.on_state_changed)

        # Initialize camera manager
        self.camera_manager = CameraManagerSingleton()

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()

        # GUI modules components
        self.config_window = None
        self.sorting_gui = None

        # System modules
        self.camera = None
        self.modules = {}
        self.processing_worker = None

        # Processing state
        self.is_running = False
        self.is_paused = False

        # Timers
        self.processing_timer = None
        self.preview_timer = None
        self.metrics_timer = None

        # Statistics
        self.start_time = None
        self.frame_count = 0
        self.processed_pieces = 0

        # Start in configuration state
        self.state_manager.transition_to(ApplicationState.CONFIGURING)

    # ============= State Change Handler =============

    def on_state_changed(self, old_state: ApplicationState, new_state: ApplicationState):
        """Handle state changes"""
        logger.info(f"State changed: {old_state} -> {new_state}")

        # Exit old state
        exit_method = f"exit_{old_state.value}_state"
        if hasattr(self, exit_method):
            getattr(self, exit_method)()

        # Enter new state
        enter_method = f"enter_{new_state.value}_state"
        if hasattr(self, enter_method):
            getattr(self, enter_method)()

    # ============= Configuration State =============

    def enter_configuring_state(self):
        """Enter configuration state"""
        logger.info("Entering configuration state")

        try:
            # Clean up sorting GUI modules if it exists
            if hasattr(self, 'sorting_gui') and self.sorting_gui:
                self.sorting_gui.hide()
                self.sorting_gui.deleteLater()
                self.sorting_gui = None

            # Create configuration GUI modules
            self.config_window = ConfigurationGUI(self.config_manager)

            # Connect configuration signals
            self.config_window.configuration_complete.connect(self.on_configuration_complete)
            self.config_window.preview_requested.connect(self.start_camera_preview)

            # Connect window close event to application shutdown
            self.config_window.destroyed.connect(self.handle_window_closed)

            # Center and show window
            self.config_window.center_window()
            self.config_window.show()

            # Update status
            self.config_window.status_bar.showMessage("Ready to configure")
            logger.info("Configuration window displayed")

        except Exception as e:
            logger.error(f"Failed to enter configuration state: {e}")
            QMessageBox.critical(None, "State Error",
                                 f"Failed to enter configuration: {str(e)}")
            self.state_manager.transition_to(ApplicationState.ERROR)

    def exit_configuring_state(self):
        """Exit configuration state"""
        logger.info("Exiting configuration state")

        # Stop camera preview if running
        if hasattr(self, 'preview_timer') and self.preview_timer:
            self.preview_timer.stop()
            self.preview_timer = None

    def on_configuration_complete(self, config: Dict[str, Any]):
        """Handle configuration completion

        Args:
            config: Configuration dictionary from GUI modules
        """
        logger.info("Configuration completed")

        # Validate configuration
        is_valid, error_msg = validate_config(self.config_manager)

        if not is_valid:
            QMessageBox.warning(self.config_window, "Configuration Invalid",
                                f"Cannot start sorting: {error_msg}")
            return

        # Stop camera preview if running
        if hasattr(self, 'preview_timer') and self.preview_timer:
            self.preview_timer.stop()
            self.preview_timer = None

        # Update configuration manager with GUI modules settings
        for module, settings in config.items():
            self.config_manager.update_module_config(module, settings)

        # Save configuration
        self.config_manager.save_config()
        logger.info("Configuration saved")

        # Transition to initialization
        self.state_manager.transition_to(ApplicationState.INITIALIZING)

    # ============= Initialization State =============

    def enter_initialization_state(self):
        """Initialize system for sorting"""
        try:
            logger.info("Initializing sorting system")

            # Hide configuration window but don't destroy it yet
            if self.config_window:
                self.config_window.hide()

            # Show loading dialog
            loading = QProgressDialog("Initializing sorting system...", None, 0, 0)
            loading.setWindowTitle("Initializing")
            loading.setWindowModality(Qt.WindowModal)
            loading.show()
            QApplication.processEvents()

            try:
                # Initialize camera (singleton handles instance management)
                self.camera = create_camera(config_manager=self.config_manager)

                # Ensure camera is initialized
                if not self.camera.is_initialized:
                    self.camera.initialize()

                logger.info("Camera initialized for sorting")

                # Create system modules
                self.modules = {
                    'detector': create_detector(self.config_manager),
                    'sorting': create_sorting_manager(self.config_manager),
                    'history': create_piece_history(),
                    'thread_mgr': create_thread_manager(self.config_manager),
                    'api': create_api_client(self.config_manager)
                }

                # Try to create Arduino module
                try:
                    self.modules['arduino'] = create_arduino_servo_module(self.config_manager)
                    logger.info("Arduino module initialized")
                except Exception as e:
                    logger.warning(f"Arduino module not available: {e}")
                    self.modules['arduino'] = None

                # Initialize detector ROI
                frame = self.camera.get_frame()  # Use get_frame for initial setup
                if frame is not None:
                    self.modules['detector'].load_roi_from_config(frame, self.config_manager)

                # Register frame consumers AFTER modules are created
                self._register_frame_consumers()

                # Start processing worker thread
                self._start_processing_worker()

                logger.info("All modules initialized successfully")

            finally:
                loading.close()

            # Create sorting GUI modules
            self.sorting_gui = SortingGUI(self.config_manager)

            # Connect sorting GUI modules signals
            self.sorting_gui.pause_requested.connect(self.pause_sorting)
            self.sorting_gui.resume_requested.connect(self.resume_sorting)
            self.sorting_gui.stop_requested.connect(self.stop_sorting)
            self.sorting_gui.settings_requested.connect(self.return_to_configuration)

            # Connect metrics signals
            self.metrics_tracker.metrics_updated.connect(self.update_sorting_metrics)

            # Update Arduino status in GUI modules
            if self.modules.get('arduino'):
                self.sorting_gui.set_arduino_status(True)
            else:
                self.sorting_gui.set_arduino_status(False)

            # Show sorting GUI modules
            self.sorting_gui.center_window()
            self.sorting_gui.show()

            # Destroy configuration window now that sorting is ready
            if self.config_window:
                self.config_window.deleteLater()
                self.config_window = None

            # Transition to sorting state
            self.state_manager.transition_to(ApplicationState.SORTING)

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            QMessageBox.critical(None, "Initialization Error",
                                 f"Failed to initialize: {str(e)}")
            self.state_manager.transition_to(ApplicationState.ERROR)

    def _register_frame_consumers(self):
        """Register consumers for the camera frame distribution system"""
        logger.info("Registering frame consumers")

        # Define callback for detector processing
        def detector_callback(frame):
            """Process frame for piece detection"""
            if not self.is_running or self.is_paused:
                return

            try:
                self.frame_count += 1
                self.metrics_tracker.record_frame()

                # Detect pieces
                detections = self.modules['detector'].detect_pieces(frame)

                # Process each detected piece
                for piece in detections:
                    # Check if this is a new piece that needs processing
                    if hasattr(piece, 'is_new') and piece.is_new and not piece.being_processed:
                        # Mark as being processed
                        piece.being_processed = True

                        # Extract piece image
                        x, y, w, h = piece.bbox
                        roi_x, roi_y = self.modules['detector'].roi[:2]
                        abs_x = roi_x + x
                        abs_y = roi_y + y
                        piece_image = frame[abs_y:abs_y + h, abs_x:abs_x + w].copy()

                        # Queue for processing
                        self.modules['thread_mgr'].add_message(
                            piece_id=piece.id,
                            image=piece_image,
                            frame_number=self.frame_count,
                            position=(abs_x, abs_y, w, h),
                            priority=0 if piece.in_exit_zone else 1,
                            in_exit_zone=piece.in_exit_zone
                        )

                        logger.debug(f"Queued piece {piece.id} for processing")

                # Update GUI modules with detections
                if hasattr(self, 'sorting_gui') and self.sorting_gui:
                    detection_data = {
                        'roi': self.modules['detector'].roi,
                        'pieces': [
                            {
                                'id': p.id,
                                'bbox': p.bbox,
                                'status': 'processing' if p.being_processed else 'detected',
                                'in_exit_zone': p.in_exit_zone
                            }
                            for p in detections
                        ]
                    }

                    # Use Qt's thread-safe signal
                    QMetaObject.invokeMethod(
                        self,
                        "_update_gui_frame",
                        Qt.QueuedConnection,
                        Q_ARG(object, frame),
                        Q_ARG(object, detection_data)
                    )

            except Exception as e:
                logger.error(f"Error in detector callback: {e}")

        # Register detector as high priority synchronous consumer
        self.camera.register_consumer(
            name="detector",
            callback=detector_callback,
            processing_type="sync",
            priority=90
        )

        logger.info("Frame consumers registered successfully")

    @pyqtSlot(object, object)
    def _update_gui_frame(self, frame, detection_data):
        """Thread-safe GUI modules update method"""
        if hasattr(self, 'sorting_gui') and self.sorting_gui:
            self.sorting_gui.update_frame(frame, detection_data)

    def _start_processing_worker(self):
        """Start the processing worker thread"""
        logger.info("Starting processing worker")

        # Register callbacks for processing results
        self.modules['thread_mgr'].register_callback(
            "piece_processed",
            self._on_piece_processed_callback
        )

        self.modules['thread_mgr'].register_callback(
            "piece_error",
            self._on_piece_error_callback
        )

        # Start the processing worker thread
        success = self.modules['thread_mgr'].start_worker(
            name="processing_worker",
            target=processing_worker_thread,
            args=(
                self.modules['thread_mgr'],
                self.config_manager,
                self.modules['history']
            ),
            daemon=True
        )

        if success:
            logger.info("Processing worker started successfully")
        else:
            logger.error("Failed to start processing worker")
            raise RuntimeError("Failed to start processing worker")

    # ============= Sorting State =============

    def enter_sorting_state(self):
        """Start sorting operation"""
        logger.info("Starting sorting operation")

        try:
            # Set running flag
            self.is_running = True
            self.is_paused = False

            # Reset metrics
            self.start_time = time.time()
            self.frame_count = 0
            self.processed_pieces = 0

            # Start camera capture if not already running
            if not self.camera.is_capturing:
                if not self.camera.start_capture():
                    raise RuntimeError("Failed to start camera capture")

            # Start metrics timer
            self.metrics_timer = QTimer()
            self.metrics_timer.timeout.connect(self._update_metrics)
            self.metrics_timer.start(1000)  # Update every second

            # Update GUI modules status
            if self.sorting_gui:
                self.sorting_gui.status_label.setText("Sorting Active")
                self.sorting_gui.add_log_message("Sorting started", "INFO")

            logger.info("Sorting operation started successfully")

        except Exception as e:
            logger.error(f"Failed to start sorting: {e}")
            if self.sorting_gui:
                self.sorting_gui.add_log_message(f"Error starting sorting: {e}", "ERROR")
            self.state_manager.transition_to(ApplicationState.ERROR)

    def exit_sorting_state(self):
        """Exit sorting state"""
        logger.info("Exiting sorting state")

        # Stop processing
        self.is_running = False

        # Stop timers
        if self.processing_timer:
            self.processing_timer.stop()

        if self.metrics_timer:
            self.metrics_timer.stop()

        # Stop worker thread
        if self.processing_worker and self.processing_worker.isRunning():
            self.processing_worker.stop()
            self.processing_worker.wait(5000)

    # ============= Processing Methods =============

    def on_piece_processed(self, result: Dict[str, Any]):
        """Handle processed piece result

        Args:
            result: Processing result dictionary
        """
        if not result:
            return

        try:
            # Update processed count
            self.processed_pieces += 1
            self.metrics_tracker.record_piece_processed()

            # Update sorting GUI modules
            if hasattr(self, 'sorting_gui') and self.sorting_gui:
                # Update piece display
                piece_data = {
                    'id': result.get('piece_id'),
                    'image': result.get('image'),
                    'category': result.get('category'),
                    'bin': result.get('bin_number'),
                    'element_id': result.get('element_id'),
                    'name': result.get('name')
                }
                self.sorting_gui.update_piece_display(piece_data)

                # Update bin indicator
                bin_number = result.get('bin_number', 0)
                self.sorting_gui.update_bin_status(bin_number, 'active')

                # Add to log
                message = f"Piece {result.get('piece_id')} → Bin {bin_number}"
                self.sorting_gui.add_log_message(message, "INFO")

                # Reset bin indicator after delay
                QTimer.singleShot(500,
                                  lambda: self.sorting_gui.update_bin_status(bin_number, 'idle'))

        except Exception as e:
            logger.error(f"Error handling processed piece: {e}")

    def update_sorting_metrics(self, metrics: Dict[str, Any]):
        """Update sorting GUI modules with latest metrics

        Args:
            metrics: Dictionary containing system metrics
        """
        if not hasattr(self, 'sorting_gui') or not self.sorting_gui:
            return

        try:
            # Update FPS
            if 'fps' in metrics:
                self.sorting_gui.fps_label.setText(f"{metrics['fps']:.1f}")

            # Update processed count
            if 'processed_count' in metrics:
                self.sorting_gui.set_processed_count(metrics['processed_count'])

            # Update camera status
            if 'camera_status' in metrics:
                self.sorting_gui.set_camera_status(metrics['camera_status'])

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _on_piece_processed_callback(self, result: Dict[str, Any]):
        """Callback when piece processing completes

        Args:
            result: Processing result dictionary
        """
        try:
            # Update metrics
            self.processed_pieces += 1
            self.metrics_tracker.record_piece_processed()

            # Emit signal for GUI modules update (thread-safe)
            if hasattr(self, 'sorting_gui'):
                QMetaObject.invokeMethod(
                    self,
                    "_handle_processed_piece",
                    Qt.QueuedConnection,
                    Q_ARG(object, result)
                )

            logger.info(f"Piece {result.get('piece_id')} processed successfully")

        except Exception as e:
            logger.error(f"Error handling processed piece: {e}")

    def _on_piece_error_callback(self, piece_id: int, error: str, result: Optional[Dict] = None):
        """Callback when piece processing fails

        Args:
            piece_id: ID of the piece that failed
            error: Error message
            result: Optional error details
        """
        logger.error(f"Processing failed for piece {piece_id}: {error}")

        # Update error metrics
        if hasattr(self, 'metrics_tracker'):
            self.metrics_tracker.record_error()

        # Update GUI modules with error
        if hasattr(self, 'sorting_gui'):
            self.sorting_gui.add_log_message(
                f"Error processing piece {piece_id}: {error}",
                "ERROR"
            )

    @pyqtSlot(object)
    def _handle_processed_piece(self, result: Dict[str, Any]):
        """Thread-safe handler for processed pieces"""
        self.on_piece_processed(result)

    # ============= Control Methods =============

    def pause_sorting(self):
        """Pause sorting operation"""
        if self.state_manager.current_state != ApplicationState.SORTING:
            return

        logger.info("Pausing sorting")
        self.is_paused = True
        self.state_manager.transition_to(ApplicationState.PAUSED)

        if self.sorting_gui:
            self.sorting_gui.add_log_message("Sorting paused", "INFO")

    def resume_sorting(self):
        """Resume sorting operation"""
        if self.state_manager.current_state != ApplicationState.PAUSED:
            return

        logger.info("Resuming sorting")
        self.is_paused = False
        self.state_manager.transition_to(ApplicationState.SORTING)

        if self.sorting_gui:
            self.sorting_gui.add_log_message("Sorting resumed", "INFO")

    def stop_sorting(self):
        """Stop sorting and return to configuration"""
        logger.info("Stopping sorting")

        # Stop processing
        self.is_running = False

        # Log session statistics
        self.log_session_statistics()

        # Transition to configuration
        self.state_manager.transition_to(ApplicationState.CONFIGURING)

    def return_to_configuration(self):
        """Return to configuration from sorting state"""
        logger.info("Returning to configuration")

        # Pause sorting first
        if not self.is_paused:
            self.pause_sorting()

        # Confirm with user
        reply = QMessageBox.question(
            self.sorting_gui,
            "Return to Configuration",
            "Return to configuration? Current sorting will be stopped.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Stop sorting
            self.is_running = False

            # Stop timers
            if hasattr(self, 'processing_timer'):
                self.processing_timer.stop()

            # Stop worker thread
            if hasattr(self, 'processing_worker'):
                self.processing_worker.stop()

            # Release camera from sorting
            if self.camera:
                self.camera_manager.release_camera("sorting")
                self.camera = None

            # Log final statistics
            self.log_session_statistics()

            # Transition back to configuration
            self.state_manager.transition_to(ApplicationState.CONFIGURING)

    # ============= Camera Preview Methods =============

    def start_camera_preview(self):
        """Start camera preview for configuration"""
        logger.info("Starting camera preview")

        try:
            # Get or create camera instance
            if not hasattr(self, 'camera') or not self.camera:
                self.camera = create_camera(config_manager=self.config_manager)
                if not self.camera.is_initialized:
                    self.camera.initialize()

            # Define preview callback
            def preview_callback(frame):
                if self.config_window:
                    # Use thread-safe GUI modules update
                    QMetaObject.invokeMethod(
                        self.config_window,
                        "update_preview_frame",
                        Qt.QueuedConnection,
                        Q_ARG(object, frame)
                    )

            # Register preview consumer
            self.camera.register_consumer(
                name="config_preview",
                callback=preview_callback,
                processing_type="async",
                priority=50
            )

            # Start capture if not running
            if not self.camera.is_capturing:
                self.camera.start_capture()

            logger.info("Camera preview started")

        except Exception as e:
            logger.error(f"Failed to start preview: {e}")
            if self.config_window:
                QMessageBox.warning(self.config_window, "Preview Error",
                                    f"Failed to start camera preview: {str(e)}")

    def stop_camera_preview(self):
        """Stop camera preview"""
        logger.info("Stopping camera preview")

        if hasattr(self, 'camera') and self.camera:
            # Unregister preview consumer
            self.camera.unregister_consumer("config_preview")

            # Stop capture if no other consumers
            if not self.camera.consumers:
                self.camera.stop_capture()

        if self.config_window:
            self.config_window.stop_camera_preview()

    def update_camera_preview(self):
        """Update camera preview frame"""
        if not self.camera or not self.config_window:
            return

        try:
            frame = self.camera.get_frame()
            if frame is not None:
                self.config_window.update_preview_frame(frame)
        except Exception as e:
            logger.error(f"Error updating preview: {e}")
    # ============= Utility Methods =============

    def log_session_statistics(self):
        """Log statistics for the current session"""
        if not hasattr(self, 'start_time'):
            return

        elapsed = time.time() - self.start_time

        stats = {
            'duration': elapsed,
            'frames_processed': self.frame_count,
            'pieces_processed': self.processed_pieces,
            'average_fps': self.frame_count / elapsed if elapsed > 0 else 0
        }

        logger.info("=" * 60)
        logger.info("Session Statistics:")
        logger.info(f"  Duration: {stats['duration']:.1f} seconds")
        logger.info(f"  Frames: {stats['frames_processed']}")
        logger.info(f"  Pieces: {stats['pieces_processed']}")
        logger.info(f"  Avg FPS: {stats['average_fps']:.1f}")
        logger.info("=" * 60)

        # Also show in GUI modules if available
        if hasattr(self, 'sorting_gui') and self.sorting_gui:
            self.sorting_gui.add_log_message(
                f"Session ended: {stats['pieces_processed']} pieces in {stats['duration']:.1f}s",
                "INFO"
            )

    def handle_window_closed(self):
        """Handle window close event"""
        logger.info("Window closed, shutting down")
        self.shutdown()

# ============= Shutdown Methods =============

    def shutdown(self):
        """Shutdown application cleanly"""
        logger.info("Shutting down application")

        try:
            # Transition to shutting down state
            self.state_manager.transition_to(ApplicationState.SHUTTING_DOWN)

            # Stop processing
            self.is_running = False

            # Stop all timers
            if hasattr(self, 'metrics_timer') and self.metrics_timer:
                self.metrics_timer.stop()

            # Stop camera capture
            if hasattr(self, 'camera') and self.camera:
                self.camera.stop_capture()

            # Stop thread manager workers
            if hasattr(self, 'modules') and 'thread_mgr' in self.modules:
                self.modules['thread_mgr'].stop_all_workers(timeout=5.0)

            # Release camera
            if hasattr(self, 'camera'):
                self.camera.release()

            # Close Arduino connection
            if hasattr(self, 'modules') and self.modules.get('arduino'):
                self.modules['arduino'].release()

            # Log final statistics
            self.log_session_statistics()

            # Close all windows
            if hasattr(self, 'config_window') and self.config_window:
                self.config_window.close()

            if hasattr(self, 'sorting_gui') and self.sorting_gui:
                self.sorting_gui.close()

            logger.info("Application shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        finally:
            QApplication.quit()

# ============= Main Entry Point =============

def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Lego Sorting System v6 - Modular GUI modules")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--calibrate-servo', action='store_true',
                        help='Enable servo calibration mode')
    parser.add_argument('--calibrate-sorting', action='store_true',
                        help='Enable sorting calibration mode')
    parser.add_argument('--profile', type=str, help='Load configuration profile')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(console_level=log_level)

    logger.info("=" * 60)
    logger.info("  Lego Sorting System v6 - Modular GUI modules Edition")
    logger.info("  Starting application...")
    logger.info("=" * 60)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Lego Sorting System")
    app.setApplicationDisplayName("Lego Sorting System v6 - Modular GUI modules")

    # Register custom types for thread-safe signals
    from PyQt5.QtCore import qRegisterMetaType
    qRegisterMetaType('object')

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

        # Add timer to handle Ctrl+C in console
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(500)

        logger.info("Application initialized successfully")
        logger.info("Starting GUI modules...")

        # Run application
        sys.exit(app.exec_())

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        QMessageBox.critical(None, "Fatal Error",
                             f"Application failed to start:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()