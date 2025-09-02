#!/usr/bin/env python3
"""
Lego_Sorting_007.py - Main orchestration for the Lego Sorting System

This version correctly wires together all the modular components with proper
data flow and error handling. Key improvements:
- Correct module initialization order
- Proper queue management between detector and processing
- Thread-safe GUI updates
- Robust state management
- Clean shutdown procedures
"""

import sys
import os
import signal
import argparse
import logging
import time
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import cv2
import numpy as np

# PyQt5 imports for GUI
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QMetaObject, Q_ARG, QTimer

# GUI modules
from GUI.config_GUI_module import ConfigurationGUI
from GUI.sorting_GUI_module import SortingGUI
from GUI.gui_common import validate_config

# Core system modules - CORRECT IMPORTS
from enhanced_config_manager import create_config_manager, ModuleConfig
from camera_module import create_camera
from detector_module import create_detector
from sorting_module import create_sorting_manager
from api_module import create_api_client
from arduino_servo_module import create_arduino_servo_module
from piece_history_module import create_piece_history

# Threading and queue management - CRITICAL IMPORTS
from thread_manager import create_thread_manager, ThreadManager
from piece_queue_manager import create_piece_queue_manager, PieceQueueManager, PieceMessage
from processing_module import ProcessingWorker, create_processing_worker

# Error handling
from error_module import setup_logging, get_logger

logger = get_logger(__name__)


# ============= Application States =============

class ApplicationState(Enum):
    """Application lifecycle states"""
    STARTUP = "startup"
    CONFIGURING = "configuring"
    INITIALIZING = "initializing"
    SORTING = "sorting"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


# ============= Metrics Tracker =============

class MetricsTracker(QObject):
    """Track and emit application metrics"""

    metrics_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.frames_processed = 0
        self.pieces_detected = 0
        self.pieces_processed = 0
        self.pieces_sorted = {}
        self.errors = 0

    def start_session(self):
        """Start a new sorting session"""
        self.start_time = time.time()
        self.frames_processed = 0
        self.pieces_detected = 0
        self.pieces_processed = 0
        self.pieces_sorted = {}
        self.errors = 0

    def record_frame(self):
        """Record a processed frame"""
        self.frames_processed += 1

    def record_detection(self):
        """Record a piece detection"""
        self.pieces_detected += 1

    def record_processed(self, bin_number: int):
        """Record a processed piece"""
        self.pieces_processed += 1
        if bin_number not in self.pieces_sorted:
            self.pieces_sorted[bin_number] = 0
        self.pieces_sorted[bin_number] += 1
        self.emit_update()

    def record_error(self):
        """Record an error"""
        self.errors += 1

    def emit_update(self):
        """Emit current metrics"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frames_processed / elapsed if elapsed > 0 else 0
            pps = self.pieces_processed / elapsed if elapsed > 0 else 0
        else:
            elapsed = 0
            fps = 0
            pps = 0

        metrics = {
            'elapsed_time': elapsed,
            'fps': fps,
            'frames_processed': self.frames_processed,
            'pieces_detected': self.pieces_detected,
            'pieces_processed': self.pieces_processed,
            'pieces_per_second': pps,
            'pieces_sorted': self.pieces_sorted,
            'errors': self.errors
        }

        self.metrics_updated.emit(metrics)


# ============= Main Application Class =============

class LegoSortingApplication(QObject):
    """Main application orchestrator"""

    # Signals for GUI updates
    gui_update_signal = pyqtSignal(object, object)  # frame, detection_data
    error_signal = pyqtSignal(str)
    state_changed = pyqtSignal(ApplicationState)

    def __init__(self):
        super().__init__()

        # Core components (initialized in order)
        self.config_manager = None
        self.camera = None
        self.detector = None
        self.thread_manager = None
        self.queue_manager = None
        self.processing_worker = None
        self.sorting_manager = None
        self.servo_module = None
        self.api_client = None
        self.piece_history = None

        # GUI components
        self.config_gui = None
        self.sorting_gui = None

        # State management
        self.current_state = ApplicationState.STARTUP
        self.is_running = False
        self.is_paused = False

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.frame_count = 0

        # Processing thread reference
        self.processing_thread = None

        # Update timer for periodic GUI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system_status)

    # ========================================================================
    # INITIALIZATION SECTION
    # ========================================================================

    def initialize_system(self) -> bool:
        """Initialize all system components in the correct order"""

        logger.info("Starting system initialization")

        # Show progress dialog
        progress = QProgressDialog("Initializing system components...", None, 0, 10)
        progress.setWindowTitle("System Initialization")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            # 1. Configuration Manager (first - everything depends on this)
            progress.setLabelText("Loading configuration...")
            progress.setValue(1)
            QApplication.processEvents()

            self.config_manager = create_config_manager()
            logger.info("Configuration manager initialized")

            # 2. Thread Manager (infrastructure for threading)
            progress.setLabelText("Setting up thread management...")
            progress.setValue(2)
            QApplication.processEvents()

            self.thread_manager = create_thread_manager(self.config_manager)
            logger.info("Thread manager initialized")

            # 3. Queue Manager (needs thread manager)
            progress.setLabelText("Creating piece queue...")
            progress.setValue(3)
            QApplication.processEvents()

            self.queue_manager = create_piece_queue_manager(
                thread_manager=self.thread_manager,
                config_manager=self.config_manager
            )
            logger.info("Queue manager initialized")

            # 4. Camera (singleton, can be created early)
            progress.setLabelText("Initializing camera...")
            progress.setValue(4)
            QApplication.processEvents()

            self.camera = create_camera(config_manager=self.config_manager)
            if not self.camera.is_initialized:
                if not self.camera.initialize():
                    raise RuntimeError("Failed to initialize camera")
            logger.info("Camera initialized")

            # 5. Detector (needs configuration)
            progress.setLabelText("Setting up detector...")
            progress.setValue(5)
            QApplication.processEvents()

            self.detector = create_detector(
                detector_type="conveyor",
                config_manager=self.config_manager
            )

            # Set ROI from config or use defaults
            frame = self.camera.get_frame()
            if frame is not None:
                self.detector.load_roi_from_config(frame, self.config_manager)
            logger.info("Detector initialized with ROI")

            # 6. API Client
            progress.setLabelText("Connecting to API...")
            progress.setValue(6)
            QApplication.processEvents()

            self.api_client = create_api_client(
                api_type="brickognize",
                config_manager=self.config_manager
            )
            logger.info("API client initialized")

            # 7. Sorting Manager
            progress.setLabelText("Loading sorting strategy...")
            progress.setValue(7)
            QApplication.processEvents()

            self.sorting_manager = create_sorting_manager(self.config_manager)
            logger.info("Sorting manager initialized")

            # 8. Arduino Servo (may fail if not connected)
            progress.setLabelText("Connecting to Arduino...")
            progress.setValue(8)
            QApplication.processEvents()

            try:
                self.servo_module = create_arduino_servo_module(self.config_manager)
                logger.info("Arduino servo module initialized")
            except Exception as e:
                logger.warning(f"Arduino servo not available: {e}")
                self.servo_module = None

            # 9. Piece History
            progress.setLabelText("Loading piece history...")
            progress.setValue(9)
            QApplication.processEvents()

            self.piece_history = create_piece_history(self.config_manager)
            logger.info("Piece history initialized")

            # 10. Processing Worker (needs all the above)
            progress.setLabelText("Creating processing worker...")
            progress.setValue(10)
            QApplication.processEvents()

            self.processing_worker = create_processing_worker(
                queue_manager=self.queue_manager,
                thread_manager=self.thread_manager,
                config_manager=self.config_manager,
                piece_history=self.piece_history
            )
            logger.info("Processing worker initialized")

            progress.close()
            logger.info("System initialization complete")
            return True

        except Exception as e:
            progress.close()
            logger.error(f"System initialization failed: {e}", exc_info=True)
            QMessageBox.critical(None, "Initialization Failed",
                                 f"Failed to initialize system:\n{str(e)}")
            return False

    # ========================================================================
    # DATA FLOW CONNECTIONS
    # ========================================================================

    def setup_camera_to_detector_flow(self):
        """Connect camera frame distribution to detector"""

        def detector_callback(frame: np.ndarray):
            """Process each camera frame for piece detection"""

            if not self.is_running or self.is_paused:
                return

            try:
                # Update frame counter
                self.frame_count += 1
                self.metrics_tracker.record_frame()

                # Run detection
                detections = self.detector.detect_pieces(frame)

                # Process each detected piece
                new_pieces = 0
                for piece in detections:
                    # Only process new pieces that aren't already being processed
                    if (hasattr(piece, 'is_new') and piece.is_new and
                            not piece.being_processed):

                        # Queue the piece for processing
                        if self._queue_piece_for_processing(piece, frame):
                            new_pieces += 1
                            self.metrics_tracker.record_detection()

                # Update GUI with detection visualization
                if self.sorting_gui:
                    detection_data = self.detector.get_visualization_data()
                    # Add frame count to detection data
                    detection_data['frame_count'] = self.frame_count
                    detection_data['new_pieces'] = new_pieces

                    # Use thread-safe signal
                    self.gui_update_signal.emit(frame.copy(), detection_data)

            except Exception as e:
                logger.error(f"Error in detector callback: {e}", exc_info=True)
                self.error_signal.emit(str(e))

        # Register detector as synchronous high-priority consumer
        self.camera.register_consumer(
            name="detector",
            callback=detector_callback,
            processing_type="sync",  # Synchronous for frame-by-frame processing
            priority=90  # High priority
        )
        logger.info("Camera to detector flow established")

    def _queue_piece_for_processing(self, piece, frame) -> bool:
        """Queue a detected piece for API processing"""

        try:
            # Mark piece as being processed
            piece.being_processed = True

            # Extract piece image from frame
            x, y, w, h = piece.bbox
            roi_x, roi_y = self.detector.roi[:2]

            # Convert to absolute coordinates
            abs_x = roi_x + x
            abs_y = roi_y + y

            # Crop piece image with padding
            padding = 20
            x1 = max(0, abs_x - padding)
            y1 = max(0, abs_y - padding)
            x2 = min(frame.shape[1], abs_x + w + padding)
            y2 = min(frame.shape[0], abs_y + h + padding)

            piece_image = frame[y1:y2, x1:x2].copy()

            # Add to queue with proper priority
            # CRITICAL: Use queue_manager.add_piece, not thread_manager.add_message!
            success = self.queue_manager.add_piece(
                piece_id=piece.id,
                image=piece_image,
                frame_number=self.frame_count,
                position=(abs_x, abs_y, w, h),
                in_exit_zone=piece.in_exit_zone  # Priority handled internally
            )

            if success:
                logger.debug(f"Queued piece {piece.id} (exit_zone={piece.in_exit_zone})")
                return True
            else:
                logger.warning(f"Failed to queue piece {piece.id} - queue may be full")
                piece.being_processed = False  # Reset flag if queueing failed
                return False

        except Exception as e:
            logger.error(f"Error queueing piece {piece.id}: {e}")
            piece.being_processed = False
            return False

    def start_processing_worker(self):
        """Start the background processing worker thread"""

        def processing_thread_function():
            """Main processing thread loop"""
            logger.info("Processing thread started")

            try:
                # Start the worker
                self.processing_worker.start()

                # Main processing loop
                while self.is_running:
                    try:
                        # Get next piece from queue (with timeout to check running flag)
                        message = self.queue_manager.get_next_piece(timeout=0.1)

                        if message:
                            # Process the piece
                            result = self.processing_worker._process_message(message)

                            if result:
                                # Success - update statistics and GUI
                                self.queue_manager.mark_piece_completed(
                                    message.piece_id, result
                                )

                                # Record in metrics
                                bin_number = result.get('bin_number', 0)
                                self.metrics_tracker.record_processed(bin_number)

                                logger.info(f"Processed piece {message.piece_id} -> bin {bin_number}")

                            else:
                                # Failure - mark as failed
                                self.queue_manager.mark_piece_failed(
                                    message.piece_id,
                                    "Processing failed"
                                )
                                self.metrics_tracker.record_error()

                    except Exception as e:
                        logger.error(f"Error in processing loop: {e}")
                        self.metrics_tracker.record_error()

            except Exception as e:
                logger.error(f"Processing thread error: {e}", exc_info=True)
            finally:
                self.processing_worker.stop()
                logger.info("Processing thread stopped")

        # Start the thread using thread manager
        success = self.thread_manager.start_worker(
            name="processing_worker",
            target=processing_thread_function,
            daemon=True,
            restart_on_failure=True
        )

        if not success:
            raise RuntimeError("Failed to start processing worker")

        logger.info("Processing worker thread started successfully")

    # ========================================================================
    # GUI INTEGRATION
    # ========================================================================

    def show_configuration_gui(self):
        """Display configuration interface"""

        self.config_gui = ConfigurationGUI(self.config_manager)

        # Connect signals
        self.config_gui.configuration_complete.connect(self.on_configuration_complete)
        self.config_gui.preview_requested.connect(self.show_camera_preview)

        self.config_gui.show()
        self.config_gui.center_window()
        self.current_state = ApplicationState.CONFIGURING
        logger.info("Configuration GUI displayed")

    def show_sorting_gui(self):
        """Display sorting interface"""

        self.sorting_gui = SortingGUI(self.config_manager)

        # Connect control signals
        self.sorting_gui.pause_requested.connect(self.pause_sorting)
        self.sorting_gui.resume_requested.connect(self.resume_sorting)
        self.sorting_gui.stop_requested.connect(self.stop_sorting)
        self.sorting_gui.settings_requested.connect(self.return_to_configuration)

        # Connect update signals
        self.gui_update_signal.connect(self.sorting_gui.update_frame)
        self.metrics_tracker.metrics_updated.connect(self.sorting_gui.update_metrics)

        # Set Arduino status
        self.sorting_gui.set_arduino_status(self.servo_module is not None)

        self.sorting_gui.show()
        self.sorting_gui.center_window()
        self.current_state = ApplicationState.SORTING
        logger.info("Sorting GUI displayed")

    def show_camera_preview(self):
        """Show camera preview in configuration window"""

        if not self.camera:
            self.camera = create_camera(config_manager=self.config_manager)
            if not self.camera.is_initialized:
                self.camera.initialize()

        # Get a frame and display it
        frame = self.camera.get_frame()
        if frame is not None and self.config_gui:
            self.config_gui.update_preview(frame)

    @pyqtSlot(dict)
    def on_configuration_complete(self, config: Dict[str, Any]):
        """Handle configuration completion"""

        logger.info("Configuration completed, starting sorting")

        # Update configuration
        for module, settings in config.items():
            self.config_manager.update_module_config(module, settings)

        # Start sorting
        self.start_sorting()

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    def start_sorting(self):
        """Transition from configuration to sorting"""

        self.current_state = ApplicationState.INITIALIZING

        # Validate configuration first
        is_valid, error_msg = validate_config(self.config_manager)
        if not is_valid:
            QMessageBox.warning(None, "Invalid Configuration", error_msg)
            self.current_state = ApplicationState.CONFIGURING
            return False

        # Save configuration
        self.config_manager.save_config()
        logger.info("Configuration saved")

        # Initialize system
        if not self.initialize_system():
            self.current_state = ApplicationState.ERROR
            return False

        # Setup data flows
        self.setup_camera_to_detector_flow()
        self.start_processing_worker()

        # Start metrics tracking
        self.metrics_tracker.start_session()

        # Start camera capture
        self.camera.start_capture()

        # Update state
        self.is_running = True
        self.current_state = ApplicationState.SORTING

        # Show sorting GUI
        if self.config_gui:
            self.config_gui.hide()
        self.show_sorting_gui()

        # Start periodic updates
        self.update_timer.start(1000)  # Update every second

        logger.info("Sorting started successfully")
        self.state_changed.emit(ApplicationState.SORTING)
        return True

    @pyqtSlot()
    def pause_sorting(self):
        """Pause sorting operation"""

        if self.current_state == ApplicationState.SORTING:
            self.is_paused = True
            self.current_state = ApplicationState.PAUSED

            # Pause camera consumers
            if self.camera:
                self.camera.pause_consumer("detector")

            logger.info("Sorting paused")
            self.state_changed.emit(ApplicationState.PAUSED)

    @pyqtSlot()
    def resume_sorting(self):
        """Resume sorting operation"""

        if self.current_state == ApplicationState.PAUSED:
            self.is_paused = False
            self.current_state = ApplicationState.SORTING

            # Resume camera consumers
            if self.camera:
                self.camera.resume_consumer("detector")

            logger.info("Sorting resumed")
            self.state_changed.emit(ApplicationState.SORTING)

    @pyqtSlot()
    def stop_sorting(self):
        """Stop sorting and return to configuration"""

        logger.info("Stopping sorting")

        # Stop updates
        self.update_timer.stop()

        # Set state
        self.is_running = False
        self.current_state = ApplicationState.SHUTTING_DOWN

        # Cleanup will happen in return_to_configuration
        self.return_to_configuration()

    @pyqtSlot()
    def return_to_configuration(self):
        """Return to configuration screen"""

        logger.info("Returning to configuration")

        # Cleanup current session
        self.cleanup_session()

        # Hide sorting GUI
        if self.sorting_gui:
            self.sorting_gui.hide()
            self.sorting_gui = None

        # Show configuration GUI
        self.show_configuration_gui()

    def update_system_status(self):
        """Periodic system status update"""

        if self.queue_manager and self.sorting_gui:
            # Get queue statistics
            queue_stats = self.queue_manager.get_statistics()

            # Get processing statistics
            if self.processing_worker:
                processing_stats = self.processing_worker.get_statistics()
            else:
                processing_stats = {}

            # Update GUI with combined stats
            status = {
                'queue_size': queue_stats.get('current_queue_size', 0),
                'processing': queue_stats.get('current_processing', 0),
                'processed_total': processing_stats.get('processed_count', 0),
                'errors': processing_stats.get('error_count', 0)
            }

            # This will be handled by metrics_tracker signal instead
            self.metrics_tracker.emit_update()

    # ========================================================================
    # CLEANUP AND SHUTDOWN
    # ========================================================================

    def cleanup_session(self):
        """Clean up current sorting session"""

        logger.info("Cleaning up sorting session")

        # Stop accepting new pieces
        if self.camera:
            self.camera.stop_capture()

        # Give time for queue to process remaining pieces
        if self.queue_manager:
            remaining = self.queue_manager.get_statistics()['current_queue_size']
            if remaining > 0:
                logger.info(f"Waiting for {remaining} pieces to process...")
                # Create a small delay but don't block
                QTimer.singleShot(2000, self.finish_cleanup)
            else:
                self.finish_cleanup()
        else:
            self.finish_cleanup()

    def finish_cleanup(self):
        """Finish cleanup after queue is empty"""

        # Stop processing worker
        if self.processing_worker:
            self.processing_worker.stop()

        # Stop thread manager
        if self.thread_manager:
            self.thread_manager.shutdown(timeout=3.0)

        # Reset components
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0

        logger.info("Session cleanup complete")

    def cleanup(self):
        """Complete application cleanup - CRITICAL ORDER"""

        logger.info("Starting application cleanup")
        self.current_state = ApplicationState.SHUTTING_DOWN
        self.is_running = False

        # Stop update timer
        self.update_timer.stop()

        # 1. Stop accepting new pieces
        if self.camera:
            self.camera.stop_capture()
            logger.info("Camera capture stopped")

        # 2. Process remaining pieces in queue
        if self.queue_manager:
            remaining = self.queue_manager.get_statistics()['current_queue_size']
            if remaining > 0:
                logger.info(f"Processing {remaining} remaining pieces...")
                # Give time for queue to empty
                time.sleep(2)

        # 3. Stop processing worker
        if self.processing_worker:
            self.processing_worker.stop()
            logger.info("Processing worker stopped")

        # 4. Stop thread manager (waits for threads to finish)
        if self.thread_manager:
            self.thread_manager.shutdown(timeout=5.0)
            logger.info("Thread manager shutdown")

        # 5. Release hardware resources
        if self.servo_module:
            try:
                self.servo_module.move_to_bin(0)  # Return to home position
                self.servo_module.close()
                logger.info("Servo released")
            except:
                pass

        if self.camera:
            self.camera.release()
            logger.info("Camera released")

        # 6. Save final statistics
        if self.piece_history:
            stats = self.piece_history.get_statistics()
            logger.info(f"Final statistics: {stats}")

        # 7. Close GUIs
        if self.sorting_gui:
            self.sorting_gui.close()
        if self.config_gui:
            self.config_gui.close()

        logger.info("Cleanup complete")


# ============= Main Entry Point =============

def main():
    """Main application entry point"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lego Sorting System v007')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--no-arduino', action='store_true', help='Run without Arduino')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(console_level=log_level)

    logger.info("=" * 60)
    logger.info("Lego Sorting System v007 Starting")
    logger.info("=" * 60)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Lego Sorting System")
    app.setOrganizationName("LegoSort")
    app.setStyle('Fusion')  # Modern look

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    lego_app = None

    try:
        # Create application instance
        lego_app = LegoSortingApplication()

        # Load custom config if specified
        if args.config and lego_app.config_manager:
            try:
                lego_app.config_manager.load_config(args.config)
                logger.info(f"Loaded configuration from {args.config}")
            except Exception as e:
                logger.warning(f"Failed to load config from {args.config}: {e}")

        # Disable Arduino if requested
        if args.no_arduino and lego_app.config_manager:
            lego_app.config_manager.update_module_config(
                ModuleConfig.ARDUINO_SERVO.value,
                {"simulation_mode": True}
            )
            logger.info("Arduino disabled - running in simulation mode")

        # Start with configuration GUI
        lego_app.show_configuration_gui()

        # Run Qt event loop
        logger.info("Starting Qt event loop")
        exit_code = app.exec_()

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        QMessageBox.critical(None, "Fatal Error",
                             f"A critical error occurred:\n\n{str(e)}\n\n"
                             "Please check the log file for details.")
        exit_code = 1

    finally:
        # Ensure cleanup runs
        logger.info("Performing final cleanup")
        if lego_app:
            try:
                lego_app.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        logger.info("=" * 60)
        logger.info("Lego Sorting System v007 Shutdown Complete")
        logger.info("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()