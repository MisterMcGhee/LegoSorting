"""
Lego_Sorting_007.py - Main orchestration for the Lego Sorting System

PHASE 1: Sorting GUI modules references removed for clean implementation foundation
- Removed SortingGUI import and all references
- Replaced show_sorting_gui() with placeholder message
- Removed GUI modules signals and connections
- System can initialize all modules and run in headless mode
- Ready for Phase 2 implementation
"""

import sys
import signal
import argparse
import logging
import time
from typing import Dict, Any
from enum import Enum
import numpy as np

# PyQt5 imports for GUI modules
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QTimer

# GUI modules modules - PHASE 2: Re-add SortingGUI import
from GUI.config_GUI_module import ConfigurationGUI
from GUI.sorting_GUI_module import SortingGUI
from GUI.gui_common import validate_config

# Core system modules - CORRECT IMPORTS
from enhanced_config_manager import create_config_manager, ModuleConfig
from camera_module import create_camera
from Old_versions.detector_module import create_detector
from Old_versions.sorting_module import create_sorting_manager
from processing.identification_api_handler import create_api_client
from arduino_servo_module import create_arduino_servo_module
from piece_history_module import create_piece_history

# Threading and queue management - CRITICAL IMPORTS
from thread_manager import create_thread_manager
from piece_queue_manager import create_piece_queue_manager
from processing_module import create_processing_worker

# Error handling
from error_module import setup_logging, get_logger

logger = get_logger(__name__)


def list_all_windows():
    """Debug function to list all Qt windows"""
    from PyQt5.QtWidgets import QApplication
    windows = []
    for widget in QApplication.topLevelWidgets():
        if widget.isVisible():
            windows.append(f"  - {widget.__class__.__name__}: {widget.windowTitle()} (size: {widget.size()})")
    return windows


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

    # Signals for system events - REMOVED: gui_update_signal
    error_signal = pyqtSignal(str)
    state_changed = pyqtSignal(ApplicationState)

    def __init__(self):
        super().__init__()

        # CRITICAL: Initialize config_manager FIRST and DON'T overwrite it!
        self.config_manager = create_config_manager()
        logger.info("Configuration manager initialized during startup")

        # Core components (will be initialized later in initialize_system)
        # Note: config_manager is already set above, so DON'T set it to None here!
        self.camera = None
        self.detector = None
        self.thread_manager = None
        self.queue_manager = None
        self.processing_worker = None
        self.sorting_manager = None
        self.servo_module = None
        self.api_client = None
        self.piece_history = None

        # GUI modules components - PHASE 2: Re-add sorting_gui
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

        # Update timer for periodic system updates
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
            # 1. Configuration Manager is already initialized in __init__
            # Just verify it exists
            progress.setLabelText("Verifying configuration...")
            progress.setValue(1)
            QApplication.processEvents()

            if not self.config_manager:
                raise RuntimeError("Configuration manager not initialized")

            logger.info("Configuration manager verified")

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

    def setup_camera_consumers(self):
        """Register detector consumer only - GUI modules consumer registered separately when GUI modules exists"""
        logger.info("Registering detector camera consumer...")

        if not self.camera:
            logger.error("No camera available for consumer registration")
            return False

        # Only unregister detector consumer (sorting_gui doesn't exist yet)
        self.camera.unregister_consumer("detector")

        # Register detector consumer
        def detector_callback(frame: np.ndarray):
            """Process each camera frame for piece detection"""
            if not self.is_running or self.is_paused:
                return

            try:
                self.frame_count += 1

                # Reduced logging frequency for headless mode
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed frame {self.frame_count} (detector)")

                self.metrics_tracker.record_frame()

                # Process frame in detector
                detection_result = self.detector.process_frame_for_consumer(frame)

                # Handle piece processing
                if detection_result.get('new_pieces'):
                    for piece in detection_result['new_pieces']:
                        if self._queue_piece_for_processing(piece, frame):
                            logger.debug(f"Queued piece {piece.id} for processing")

            except Exception as e:
                logger.error(f"Error in detector callback: {e}", exc_info=True)
                self.error_signal.emit(str(e))

        # Register detector consumer
        detector_success = self.camera.register_consumer(
            name="detector",
            callback=detector_callback,
            processing_type="sync",
            priority=90  # Highest priority for detection
        )

        if detector_success:
            logger.info("Detector consumer registered successfully")
        else:
            logger.error("Failed to register detector consumer")

        return detector_success

    def register_gui_consumer(self):
        """Register GUI modules consumer after GUI modules is created"""
        logger.info("=== register_gui_consumer() method called ===")

        if not self.sorting_gui:
            logger.error("Cannot register GUI modules consumer - sorting_gui is None")
            return False

        if not self.camera:
            logger.error("Cannot register GUI modules consumer - camera is None")
            return False

        logger.info(f"Both GUI modules and camera available - GUI modules type: {type(self.sorting_gui)}")
        logger.info(f"Camera state: initialized={self.camera.is_initialized}, capturing={self.camera.is_capturing}")

        try:
            logger.info("Calling sorting_gui.register_camera_consumer()...")
            success = self.sorting_gui.register_camera_consumer(self.camera)
            logger.info(f"sorting_gui.register_camera_consumer() returned: {success}")

            if success:
                logger.info("GUI modules consumer registered successfully")
                # Activate the display status
                self.sorting_gui.activate_camera_display()
                logger.info("Camera display activated")
            else:
                logger.error("Failed to register GUI modules consumer")

            return success

        except Exception as e:
            logger.error(f"Exception in register_gui_consumer: {e}", exc_info=True)
            return False

    def setup_camera_to_detector_flow(self):
        """Legacy method - now just calls setup_camera_consumers for compatibility"""
        return self.setup_camera_consumers()

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

        logger.info("Starting processing worker - Phase 1 debugging...")

        # Debug: Check if thread manager is working
        if not self.thread_manager:
            logger.error("Thread manager is None!")
            raise RuntimeError("Thread manager not initialized")

        # Debug: Check thread manager state
        logger.info(f"Thread manager running: {self.thread_manager.running}")
        logger.info(f"Thread manager workers: {list(self.thread_manager.workers.keys())}")

        # Import the proper processing worker thread function
        try:
            from processing_module import processing_worker_thread
            logger.info("Successfully imported processing_worker_thread")
        except ImportError as e:
            logger.error(f"Failed to import processing_worker_thread: {e}")
            raise RuntimeError(f"Import error: {e}")

        # Debug: Check all dependencies
        missing_deps = []
        if not self.queue_manager: missing_deps.append("queue_manager")
        if not self.thread_manager: missing_deps.append("thread_manager")
        if not self.config_manager: missing_deps.append("config_manager")
        if not self.piece_history: missing_deps.append("piece_history")

        if missing_deps:
            error_msg = f"Missing dependencies: {missing_deps}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("All dependencies verified")

        # PHASE 1 SIMPLIFIED: Skip complex threading for now
        # Just create and start the worker directly without thread manager
        logger.info("PHASE 1: Using simplified processing worker approach")

        try:
            # Create the worker directly
            from processing_module import create_processing_worker
            self.processing_worker = create_processing_worker(
                queue_manager=self.queue_manager,
                thread_manager=self.thread_manager,
                config_manager=self.config_manager,
                piece_history=self.piece_history
            )

            # Start the worker (this just sets internal state)
            self.processing_worker.start()
            logger.info("Processing worker started successfully (simplified mode)")

            # For Phase 1, we'll process pieces manually in the detector callback
            # This avoids the complex threading issues for now

        except Exception as e:
            logger.error(f"Error creating processing worker: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create processing worker: {e}")

    # ========================================================================
    # GUI modules INTEGRATION
    # ========================================================================

    def show_configuration_gui(self):
        """Display configuration interface"""
        if not self.config_manager:
            QMessageBox.critical(None, "Configuration Error",
                                 "Configuration manager not initialized")
            return

        self.config_gui = ConfigurationGUI(self.config_manager)

        # Connect only the configuration complete signal
        self.config_gui.configuration_complete.connect(self.on_configuration_complete)

        # Connect Arduino mode change signal to reinitialization method
        self.config_gui.arduino_mode_changed.connect(self.reinitialize_arduino_module)
        logger.info("Connected arduino_mode_changed signal to reinitialize handler")

        self.config_gui.show()
        self.config_gui.center_window()
        self.current_state = ApplicationState.CONFIGURING
        logger.info(f"Configuration GUI modules displayed with config_manager: {self.config_manager}")

    def show_sorting_gui(self):
        """
        Phase 3.1: Display sorting GUI modules
        """
        logger.info("=== PHASE 3.1: CREATING SORTING GUI modules ===")

        try:
            # Create the Phase 3.1 SortingGUI
            self.sorting_gui = SortingGUI(self.config_manager)
            logger.info("SortingGUI Phase 3.1 created successfully")

            # Set camera instance (consumer registration happens separately)
            if self.camera:
                self.sorting_gui.set_camera(self.camera)
                logger.info("Camera instance set in GUI modules")
            else:
                logger.warning("No camera available for GUI modules")

            # Set ROI configuration if available
            if self.detector:
                try:
                    roi_config = self.detector.get_roi_configuration()
                    self.sorting_gui.set_roi_configuration(roi_config)
                    logger.info("ROI configuration sent to GUI modules")
                except Exception as e:
                    logger.warning(f"Could not get ROI configuration: {e}")
                    self.sorting_gui.set_roi_configuration({"roi": None})
            else:
                logger.warning("No detector available for ROI configuration")
                self.sorting_gui.set_roi_configuration({"roi": None})

            # Connect control signals
            logger.info("Connecting control signals...")
            self.sorting_gui.pause_requested.connect(self.pause_sorting)
            self.sorting_gui.resume_requested.connect(self.resume_sorting)
            self.sorting_gui.stop_requested.connect(self.stop_sorting)
            self.sorting_gui.settings_requested.connect(self.return_to_configuration)
            logger.info("Control signals connected")

            # Set Arduino status
            arduino_connected = self.servo_module is not None
            self.sorting_gui.set_arduino_status(arduino_connected)
            logger.info(f"Arduino status set: {arduino_connected}")

            # Show the GUI modules window
            self.sorting_gui.show()
            self.sorting_gui.raise_()
            self.sorting_gui.activateWindow()
            logger.info("SortingGUI window displayed")

            # Show Phase 3.1 completion message
            QTimer.singleShot(1000, self.sorting_gui.show_phase_info)

            logger.info("Phase 3.1 GUI modules created successfully")

        except Exception as e:
            logger.error(f"Error creating Phase 3.1 sorting GUI modules: {e}", exc_info=True)
            QMessageBox.critical(None, "Phase 3.1 Error", f"Failed to create sorting GUI modules: {e}")

    def start_sorting_pipeline(self):
        """Start the sorting pipeline without GUI modules (headless mode)"""
        logger.info("Starting sorting pipeline in headless mode")

        try:
            # Start camera capture
            if self.camera and not self.camera.is_capturing:
                if self.camera.start_capture():
                    logger.info("Camera capture started")
                else:
                    logger.error("Failed to start camera capture")

            # Setup detector flow
            self.setup_camera_to_detector_flow()

            # Set running flag
            self.is_running = True

            # Start processing worker
            self.start_processing_worker()

            # Start metrics tracking
            self.metrics_tracker.start_session()

            # Start periodic updates
            self.update_timer.start(1000)

            logger.info("Sorting pipeline started successfully (headless mode)")

        except Exception as e:
            logger.error(f"Error starting sorting pipeline: {e}", exc_info=True)
            raise

    def show_camera_preview(self):
        """Show camera preview in configuration window"""

        if not self.camera:
            self.camera = create_camera(config_manager=self.config_manager)
            if not self.camera.is_initialized:
                self.camera.initialize()

        # Get a frame and display it
        frame = self.camera.get_frame()
        if frame is not None and self.config_gui:
            self.config_gui.update_preview_frame(frame)

    def reinitialize_arduino_module(self, simulation_mode):
        """
        Reinitialize the Arduino servo module with updated simulation mode setting.

        This method is called when the user manually changes between simulation and
        hardware modes in the configuration GUI modules. It cleanly shuts down any existing
        Arduino connection and creates a new module with the updated settings.

        Args:
            simulation_mode: True for simulation mode, False for hardware mode
        """
        logger.info(f"Reinitializing Arduino module (simulation_mode={simulation_mode})")

        try:
            # Step 1: Clean up existing Arduino module if it exists
            if self.servo_module:
                try:
                    logger.info("Releasing existing Arduino module...")

                    # Move servo to safe position before disconnecting
                    if not self.servo_module.simulation_mode:
                        self.servo_module.move_to_bin(0)  # Move to home/overflow position
                        time.sleep(0.5)  # Wait for movement to complete

                    # Release hardware resources
                    self.servo_module.release()
                    self.servo_module = None
                    logger.info("Existing Arduino module released successfully")

                except Exception as e:
                    logger.warning(f"Error releasing Arduino module: {e}")
                    self.servo_module = None

            # Step 2: Update configuration using EnhancedConfigManager API
            if self.config_manager:
                self.config_manager.update_module_config("arduino_servo", {
                    "simulation_mode": simulation_mode
                })
                self.config_manager.save_config()
                logger.info(f"Configuration updated: simulation_mode={simulation_mode}")

            # Step 3: Create new Arduino module with updated configuration
            try:
                self.servo_module = create_arduino_servo_module(self.config_manager)

                # Verify initialization was successful
                if self.servo_module and self.servo_module.initialized:
                    mode_str = "simulation" if simulation_mode else "hardware"
                    logger.info(f"Arduino module reinitialized successfully in {mode_str} mode")

                    # Show success message in config GUI modules if it's open
                    if self.config_gui:
                        QMessageBox.information(
                            self.config_gui,
                            "Arduino Module Updated",
                            f"Arduino module successfully switched to {mode_str} mode."
                        )

                else:
                    raise Exception("Arduino module initialization returned invalid state")

            except Exception as e:
                logger.error(f"Failed to create new Arduino module: {e}")
                self.servo_module = None

                # Inform user of the failure
                error_msg = (f"Failed to initialize Arduino in "
                             f"{'simulation' if simulation_mode else 'hardware'} mode.\n\n"
                             f"Error: {str(e)}\n\n"
                             f"The system will continue without servo control.")

                if self.config_gui:
                    QMessageBox.critical(self.config_gui, "Arduino Initialization Failed", error_msg)

        except Exception as e:
            logger.error(f"Critical error during Arduino reinitialization: {e}", exc_info=True)
            self.servo_module = None

    @pyqtSlot(dict)
    def on_configuration_complete(self, config: Dict[str, Any]):
        """Handle configuration completion"""

        logger.info("Configuration completed, starting sorting")

        # Update configuration
        for module, settings in config.items():
            self.config_manager.update_module_config(module, settings)

        # Ensure detector ROI is set before starting sorting
        if self.detector and self.camera:
            try:
                # Get a frame to initialize ROI if needed
                if not hasattr(self.detector, 'roi') or self.detector.roi is None:
                    ret, frame = self.camera.get_frame() if hasattr(self.camera,
                                                                    'get_frame') else self.camera.cap.read()
                    if ret and frame is not None:
                        detector_config = self.config_manager.get_module_config("detector")
                        if detector_config and "roi" in detector_config:
                            roi = detector_config["roi"]
                            if len(roi) == 4:
                                self.detector.set_roi(tuple(roi), frame)
                                logger.info(f"Detector ROI initialized: {roi}")
            except Exception as e:
                logger.warning(f"Could not initialize ROI: {e}")

        # Start sorting
        self.start_sorting()

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    def start_sorting(self):
        """Transition from configuration to sorting"""

        self.current_state = ApplicationState.INITIALIZING

        # Checkpoint 1
        logger.info("=== CHECKPOINT 1: Before validation ===")
        windows = list_all_windows()
        logger.info(f"Windows open: {len(windows)}")
        for w in windows:
            logger.info(w)

        # Validate configuration first
        is_valid, error_msg = validate_config(self.config_manager)
        if not is_valid:
            QMessageBox.warning(None, "Invalid Configuration", error_msg)
            self.current_state = ApplicationState.CONFIGURING
            return False

        # Checkpoint 2
        logger.info("=== CHECKPOINT 2: Before initialize_system ===")
        windows = list_all_windows()
        logger.info(f"Windows open: {len(windows)}")
        for w in windows:
            logger.info(w)

        # Save configuration
        self.config_manager.save_config()
        logger.info("Configuration saved")

        # Initialize system
        if not self.initialize_system():
            self.current_state = ApplicationState.ERROR
            return False

        # Setup data flows
        logger.info("Setting up camera to detector flow...")
        self.setup_camera_to_detector_flow()

        # Set running flag BEFORE starting workers
        self.is_running = True
        self.current_state = ApplicationState.SORTING

        # Start processing worker (it checks is_running)
        self.start_processing_worker()

        # Start metrics tracking
        self.metrics_tracker.start_session()

        # Start camera capture
        logger.info("Starting camera capture...")
        if self.camera:
            self.camera.start_capture()
            logger.info("Camera capture started")

        # Hide config GUI modules and show sorting GUI modules placeholder
        if self.config_gui:
            self.config_gui.hide()

        logger.info("Showing sorting GUI modules placeholder...")
        self.show_sorting_gui()

        # Start periodic updates
        self.update_timer.start(1000)

        logger.info("Sorting started successfully")
        self.state_changed.emit(ApplicationState.SORTING)
        return True

    @pyqtSlot()
    def pause_sorting(self):
        """Pause sorting operation"""
        logger.info("Pause sorting requested")

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
        logger.info("Resume sorting requested")

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
        logger.info("Stop sorting requested")

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

        # Show configuration GUI modules
        self.show_configuration_gui()

    def update_system_status(self):
        """Periodic system status update"""

        if self.queue_manager:
            # Get queue statistics
            queue_stats = self.queue_manager.get_statistics()

            # Get processing statistics
            if self.processing_worker:
                processing_stats = self.processing_worker.get_statistics()
            else:
                processing_stats = {}

            # Log status periodically
            status = {
                'queue_size': queue_stats.get('current_queue_size', 0),
                'processing': queue_stats.get('current_processing', 0),
                'processed_total': processing_stats.get('processed_count', 0),
                'errors': processing_stats.get('error_count', 0)
            }

            # Log status every 30 seconds
            if hasattr(self, '_last_status_log'):
                if time.time() - self._last_status_log > 30:
                    logger.info(f"System status: {status}")
                    self._last_status_log = time.time()
            else:
                self._last_status_log = time.time()
                logger.info(f"System status: {status}")

            # Emit metrics update
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
            try:
                # Try to get piece count instead of full statistics
                piece_count = self.piece_history.get_piece_count()
                logger.info(f"Final piece count: {piece_count}")
            except AttributeError:
                # If method doesn't exist, just log that piece history was active
                logger.info("Piece history module was active during session")
            except Exception as e:
                logger.warning(f"Error getting piece history stats: {e}")

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
    parser = argparse.ArgumentParser(description='Lego Sorting System v007 - Phase 1')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--no-arduino', action='store_true', help='Run without Arduino')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(console_level=log_level)

    logger.info("=" * 60)
    logger.info("Lego Sorting System v007 - Phase 3.1 Starting")
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

        # Start with configuration GUI modules
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
        logger.info("Lego Sorting System v007 - Phase 3.1 Shutdown Complete")
        logger.info("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()