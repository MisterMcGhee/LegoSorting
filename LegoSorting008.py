"""
LegoSorting008.py - Main Orchestrator for the Lego Sorting Machine

CURRENT PHASE: Phase 3 - Detection Pipeline Testing

This is the entry point and main coordinator for the complete Lego sorting system.
It initializes all subsystems, manages the application lifecycle, and coordinates
data flow between modules through callbacks.

ARCHITECTURE:
    Phase 1: Configuration
        - Launch ConfigurationGUI for user setup
        - Create temporary modules for preview/testing
        - Save configuration to config.json

    Phase 2: Initialization
        - Create all modules (camera, detector, processing, hardware)
        - Register callbacks between modules
        - Connect GUI to system events

    Phase 3: Detection Testing (CURRENT)
        - Test detector coordinator processes frames
        - Verify piece tracking works
        - Test capture controller captures pieces
        - Validate images saved correctly

    Phase 4-7: Full System (TODO)
        - Processing pipeline
        - Hardware pipeline
        - Complete sorting operation

CRITICAL DATA STRUCTURES:
    - identified_pieces_dict: Maps piece_id → IdentifiedPiece
        This is the central repository of all processed pieces, used by
        hardware coordinator to look up pieces ready for sorting

    - current_state: Tracks application state for control flow

    - Module references: Maintains references to all coordinators

STATE FLOW:
    STARTING → CONFIGURING → INITIALIZING → RUNNING → STOPPING → STOPPED
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import sys
import signal
import logging
import time
from typing import Dict, Optional
from enum import Enum

# PyQt5 for GUI framework
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

# Configuration management
from enhanced_config_manager import create_config_manager, EnhancedConfigManager

# GUI modules
from GUI.configuration_gui import ConfigurationGUI
from GUI.sorting_gui import SortingGUI

# Camera module (frame producer)
from camera_module import create_camera

# Detection pipeline modules
from detector.detector_coordinator import DetectorCoordinator, create_detector_coordinator
from detector.capture_controller import CaptureController, create_capture_controller
from detector.zone_manager import ZoneManager, create_zone_manager
from detector.detector_data_models import CapturePackage, TrackedPiece

# Processing pipeline modules
from processing.processing_coordinator import ProcessingCoordinator, create_processing_coordinator
from processing.processing_worker import ProcessingWorker
from processing.processing_queue_manager import ProcessingQueueManager, create_processing_queue_manager
from processing.identification_api_handler import create_identification_api_handler
from processing.category_lookup_module import create_category_lookup
from processing.bin_assignment_module import create_bin_assignment_module
from processing.category_hierarchy_service import create_category_hierarchy_service
from processing.processing_data_models import IdentifiedPiece

# Hardware pipeline modules
from hardware.hardware_coordinator import HardwareCoordinator, create_hardware_coordinator
from hardware.arduino_servo_module import create_arduino_servo_controller
from hardware.bin_capacity_module import create_bin_capacity_manager


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legosorting008.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# APPLICATION STATE ENUM
# ============================================================================

class ApplicationState(Enum):
    """
    Defines the different states the application can be in.

    State transitions follow this pattern:
        STARTING → CONFIGURING → INITIALIZING → RUNNING → STOPPING → STOPPED
    """
    STARTING = "starting"  # Initial application startup
    CONFIGURING = "configuring"  # ConfigurationGUI is active
    INITIALIZING = "initializing"  # Creating modules after configuration
    RUNNING = "running"  # Actively sorting pieces
    STOPPING = "stopping"  # Cleanup in progress
    STOPPED = "stopped"  # Application has shut down


# ============================================================================
# MAIN ORCHESTRATOR CLASS
# ============================================================================

class LegoSorting008(QObject):
    """
    Main orchestrator for the Lego Sorting Machine.

    This class is responsible for:
        1. Application lifecycle (startup → configuration → running → shutdown)
        2. Module initialization and coordination
        3. Callback registration and data routing
        4. State management
        5. GUI management
    """

    # ========================================================================
    # SIGNALS
    # ========================================================================

    state_changed = pyqtSignal(ApplicationState)  # Emitted when state changes
    error_occurred = pyqtSignal(str)  # Emitted on critical errors

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self):
        """Initialize the orchestrator and prepare for configuration."""
        super().__init__()

        logger.info("=" * 70)
        logger.info("LEGOSORTING008 - INITIALIZING")
        logger.info("=" * 70)

        # Create configuration manager FIRST
        logger.info("Creating configuration manager...")
        self.config_manager: EnhancedConfigManager = create_config_manager()
        logger.info("✓ Configuration manager initialized")

        # State management
        self.current_state = ApplicationState.STARTING
        self.is_running = False

        # Identified pieces dictionary (central repository)
        self.identified_pieces_dict: Dict[int, IdentifiedPiece] = {}

        # Module references (created during initialization)
        self.config_gui: Optional[ConfigurationGUI] = None
        self.sorting_gui: Optional[SortingGUI] = None

        # Camera module
        self.camera = None

        # Detection pipeline
        self.detector_coordinator: Optional[DetectorCoordinator] = None
        self.capture_controller: Optional[CaptureController] = None
        self.zone_manager: Optional[ZoneManager] = None

        # Processing pipeline
        self.processing_queue_manager: Optional[ProcessingQueueManager] = None
        self.processing_coordinator: Optional[ProcessingCoordinator] = None
        self.processing_workers: list = []

        # Hardware pipeline
        self.hardware_coordinator: Optional[HardwareCoordinator] = None
        self.hardware_controller = None  # Alias for GUI compatibility
        self.servo_controller = None
        self.bin_capacity_manager = None
        self.bin_assignment_module = None

        # Signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("✓ Orchestrator initialization complete")
        logger.info("Ready to launch configuration GUI")

    # ========================================================================
    # PHASE 1: CONFIGURATION
    # ========================================================================

    def show_configuration_gui(self):
        """Launch the configuration GUI for user setup."""
        logger.info("=" * 70)
        logger.info("PHASE 1: CONFIGURATION")
        logger.info("=" * 70)

        self.current_state = ApplicationState.CONFIGURING
        self.state_changed.emit(self.current_state)

        # Create temporary modules for config GUI
        logger.info("Creating temporary modules for configuration GUI...")

        temp_camera = None
        try:
            temp_camera = create_camera("webcam", self.config_manager)
            logger.info("  ✓ Temporary camera created")
        except Exception as e:
            logger.warning(f"  ⚠ Could not create temporary camera: {e}")

        temp_arduino = None
        try:
            temp_arduino = create_arduino_servo_controller(self.config_manager)
            logger.info("  ✓ Temporary arduino created")
        except Exception as e:
            logger.warning(f"  ⚠ Could not create temporary arduino: {e}")

        temp_category_service = None
        try:
            temp_category_service = create_category_hierarchy_service(self.config_manager)
            logger.info("  ✓ Temporary category service created")
        except Exception as e:
            logger.warning(f"  ⚠ Could not create temporary category service: {e}")

        # Create configuration GUI with temporary modules
        logger.info("Creating configuration GUI with temporary modules...")
        self.config_gui = ConfigurationGUI(
            self.config_manager,
            camera=temp_camera,
            arduino=temp_arduino,
            category_service=temp_category_service
        )

        # Connect signals
        self.config_gui.configuration_complete.connect(self.on_configuration_complete)
        self.config_gui.configuration_cancelled.connect(self.on_configuration_cancelled)

        # Show GUI
        self.config_gui.show()
        logger.info("✓ Configuration GUI displayed with camera preview support")
        logger.info("Waiting for user configuration...")

    @pyqtSlot(dict)
    def on_configuration_complete(self, updated_config: Dict):
        """Handle completion of configuration."""
        logger.info("Configuration completed by user")

        # Save configuration updates
        for module_name, settings in updated_config.items():
            self.config_manager.update_module_config(module_name, settings)

        logger.info("✓ Configuration saved")

        # Clean up temporary modules
        logger.info("Cleaning up temporary configuration modules...")

        if self.config_gui:
            self.config_gui.close()
            self.config_gui = None

        logger.info("✓ Temporary modules cleaned up")

        # Proceed to module initialization
        self.initialize_modules()

    @pyqtSlot()
    def on_configuration_cancelled(self):
        """Handle cancellation of configuration."""
        logger.info("Configuration cancelled by user")
        self.shutdown_system()

    # ========================================================================
    # PHASE 2 & 3: MODULE INITIALIZATION + DETECTION TESTING
    # ========================================================================

    def initialize_modules(self):
        """Create and initialize all system modules."""
        logger.info("=" * 70)
        logger.info("PHASE 2: MODULE INITIALIZATION")
        logger.info("=" * 70)

        self.current_state = ApplicationState.INITIALIZING
        self.state_changed.emit(self.current_state)

        try:
            # ================================================================
            # STEP 1: Thread Manager
            # No longer needed. Processing_queue_manager owns threading control.
            # ================================================================

            # STEP 2: Camera Module
            # ================================================================
            logger.info("Step 2: Creating camera module...")

            # Factory handles initialization and raises CameraError if it fails
            self.camera = create_camera("webcam", self.config_manager)

            logger.info("✓ Camera module created and initialized")

            # ================================================================
            # STEP 3: Detection Pipeline
            # ================================================================
            logger.info("Step 3: Creating detection pipeline...")

            # Create zone manager
            self.zone_manager = create_zone_manager(self.config_manager)
            logger.info("  ✓ Zone manager created")

            # Create detector coordinator
            self.detector_coordinator = create_detector_coordinator(
                self.config_manager,
                self.zone_manager
            )
            logger.info("  ✓ Detector coordinator created")

            # Create capture controller
            self.capture_controller = create_capture_controller(
                self.config_manager,
                self.zone_manager
            )
            logger.info("  ✓ Capture controller created")

            logger.info("✓ Detection pipeline ready")

            logger.info("Configuring ROI for detection pipeline...")

            # Get a sample frame
            sample_frame = self.camera.get_frame()
            if sample_frame is None:
                raise RuntimeError("Could not get sample frame for ROI configuration")

            # Get ROI from config
            roi_config = self.config_manager.get_module_config("detector_roi")
            roi_coords = (
                roi_config["x"],
                roi_config["y"],
                roi_config["w"],
                roi_config["h"]
            )

            logger.info(f"Setting ROI: {roi_coords}")

            # Configure detector coordinator with ROI
            self.detector_coordinator.set_roi_from_sample_frame(sample_frame, roi_coords)

            logger.info("✓ ROI configured for detection pipeline")
            # ================================================================
            # PHASE 3: DETECTION PIPELINE TEST
            # ================================================================
            logger.info("")
            logger.info("=" * 70)
            logger.info("PHASE 3: DETECTION PIPELINE TEST")
            logger.info("=" * 70)

            # Run detection test
            self.test_detection_pipeline()

            # If we reach here, test passed and user wants to continue
            logger.info("Continuing to Phase 4...")

            # ================================================================
            # STEP 4: Processing Pipeline (TODO - Phase 4)
            # ================================================================
            logger.info("Step 4: Creating processing pipeline...")

            # Create category hierarchy service
            category_service = create_category_hierarchy_service(self.config_manager)
            logger.info("  ✓ Category hierarchy service created")

            # Create processing sub-modules
            api_handler = create_identification_api_handler(self.config_manager)
            logger.info("  ✓ API handler created")

            category_lookup = create_category_lookup(self.config_manager)
            logger.info("  ✓ Category lookup created")

            bin_assignment = create_bin_assignment_module(
                self.config_manager,
                category_service
            )
            self.bin_assignment_module = bin_assignment
            logger.info("  ✓ Bin assignment module created")

            # Create processing coordinator
            self.processing_coordinator = create_processing_coordinator(
                api_handler,
                category_lookup,
                bin_assignment,
                self.config_manager
            )
            logger.info("  ✓ Processing coordinator created")

            # Create processing queue manager
            self.processing_queue_manager = create_processing_queue_manager(
                self.config_manager
            )
            logger.info("  ✓ Processing queue manager created")

            # Create processing workers
            num_workers = self.config_manager.get_module_config("processing_workers").get("num_workers", 3)

            for worker_id in range(num_workers):
                worker = ProcessingWorker(
                    worker_id=worker_id,
                    queue_manager=self.processing_queue_manager,
                    coordinator=self.processing_coordinator,
                    config_manager=self.config_manager
                )
                self.processing_workers.append(worker)
                logger.info(f"  ✓ Processing worker {worker_id} created")

            logger.info("✓ Processing pipeline ready")

            # ================================================================
            # STEP 5: Hardware Pipeline (TODO - Phase 5)
            # ================================================================
            logger.info("Step 5: Creating hardware pipeline...")

            # Create bin capacity manager
            self.bin_capacity_manager = create_bin_capacity_manager(self.config_manager)
            logger.info("  ✓ Bin capacity manager created")

            # Create servo controller
            self.servo_controller = create_arduino_servo_controller(self.config_manager)
            logger.info("  ✓ Servo controller created")

            # Create hardware coordinator
            self.hardware_coordinator = create_hardware_coordinator(
                self.bin_capacity_manager,
                self.servo_controller,
                self.config_manager
            )
            self.hardware_controller = self.hardware_coordinator
            logger.info("  ✓ Hardware coordinator created")

            logger.info("✓ Hardware pipeline ready")

            # ================================================================
            # STEP 6: Register Callbacks (TODO - Phase 4)
            # ================================================================
            logger.info("Step 6: Registering callbacks...")
            self.register_callbacks()
            logger.info("✓ Callbacks registered")

            # ================================================================
            # STEP 7: Create Sorting GUI (TODO - Phase 6)
            # ================================================================
            logger.info("Step 7: Creating sorting GUI...")
            self.create_sorting_gui()
            logger.info("✓ Sorting GUI created")

            logger.info("=" * 70)
            logger.info("✓ ALL MODULES INITIALIZED SUCCESSFULLY")
            logger.info("=" * 70)

            # Ready to start sorting
            self.start_sorting()

        except Exception as e:
            logger.error(f"FATAL: Module initialization failed: {e}", exc_info=True)
            self.error_occurred.emit(f"Initialization failed: {e}")
            self.shutdown_system()

    # ========================================================================
    # PHASE 3: DETECTION PIPELINE TEST METHOD
    # ========================================================================

    def test_detection_pipeline(self):
        """
        Test the detection pipeline with real camera and pieces.

        This test:
        1. Starts camera capture
        2. Processes frames through detector
        3. Tests capture controller
        4. Logs results for verification
        5. Runs for configurable duration
        """

        # Test configuration
        test_duration = 30  # seconds
        test_frame_count = 0
        test_captures_count = 0
        test_pieces_seen = set()

        logger.info("Starting camera capture for detection test...")
        if not self.camera.start_capture():
            raise RuntimeError("Failed to start camera capture")
        logger.info("✓ Camera capturing at 30 FPS")

        # Frame processing callback
        def detection_test_callback(frame):
            """Process frames through detection pipeline."""
            nonlocal test_frame_count, test_captures_count

            test_frame_count += 1

            # Process frame through detector (this updates internal tracking state)
            detection_result = self.detector_coordinator.process_frame_for_consumer(frame)

            # Get TrackedPiece OBJECTS directly (not from the dict result)
            tracked_pieces = self.detector_coordinator.get_tracked_pieces()

            # Log every 30 frames (once per second at 30 FPS)
            if test_frame_count % 30 == 0:
                logger.info(f"Frame {test_frame_count}: {len(tracked_pieces)} pieces tracked")

                for piece in tracked_pieces:
                    test_pieces_seen.add(piece.id)  # ✓ TrackedPiece object attribute

                    # Build zone status string
                    zones = []
                    if piece.in_entry_zone:  # ✓ TrackedPiece object attribute
                        zones.append("ENTRY")
                    if piece.in_valid_zone:
                        zones.append("VALID")
                    if piece.in_exit_zone:
                        zones.append("EXIT")
                    zone_str = "+".join(zones) if zones else "NONE"

                    logger.info(f"  Piece {piece.id}: "  # ✓ Object attributes
                                f"center={piece.center}, "
                                f"zones=[{zone_str}], "
                                f"captured={piece.captured}, "
                                f"updates={piece.update_count}")

            # Test capture controller with TrackedPiece objects
            capture_packages = self.capture_controller.check_and_process_captures(
                frame,
                tracked_pieces  # ✓ Pass TrackedPiece objects
            )

            # Log captures
            for package in capture_packages:
                test_captures_count += 1
                logger.info(f"🎯 CAPTURED: Piece {package.piece_id} at position {package.capture_position}")
                logger.info(f"   Image saved to captured_pieces/ directory")

        # Register test callback
        logger.info("Registering detection test callback...")
        self.camera.register_consumer(
            name="phase3_test",
            callback=detection_test_callback,
            processing_type="sync",
            priority=100
        )
        logger.info("✓ Test callback registered")

        # Display instructions
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 3 TEST RUNNING")
        logger.info("=" * 70)
        logger.info(f"Testing detection for {test_duration} seconds...")
        logger.info("")
        logger.info("🔍 PLACE LEGO PIECES ON CONVEYOR NOW")
        logger.info("")
        logger.info("What to observe:")
        logger.info("  1. Pieces should be detected (logged every second)")
        logger.info("  2. Each piece gets a unique ID")
        logger.info("  3. Zone flags update as pieces move")
        logger.info("  4. Pieces captured in valid zone")
        logger.info("  5. Images saved to captured_pieces/")
        logger.info("")
        logger.info(f"Monitoring for {test_duration} seconds...")
        logger.info("=" * 70)
        logger.info("")

        # Run test
        start_time = time.time()
        while (time.time() - start_time) < test_duration:
            QApplication.processEvents()
            time.sleep(0.01)

        # Test complete - show summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 3 TEST COMPLETE")
        logger.info("=" * 70)
        logger.info(f"✓ Frames processed: {test_frame_count}")
        logger.info(f"✓ Unique pieces detected: {len(test_pieces_seen)}")
        logger.info(f"✓ Pieces captured: {test_captures_count}")
        logger.info("")

        # Provide guidance
        if test_frame_count < 100:
            logger.warning("⚠ Very few frames processed - camera may not be working")
        elif len(test_pieces_seen) == 0:
            logger.warning("⚠ No pieces detected - check ROI configuration")
        elif test_captures_count == 0:
            logger.warning("⚠ No captures - check capture settings or piece placement")
        else:
            logger.info("✅ Detection pipeline working correctly!")
            logger.info("")
            logger.info("Check captured_pieces/ directory for saved images")
            logger.info("")

        # Cleanup
        self.camera.unregister_consumer("phase3_test")
        logger.info("✓ Test callback unregistered")

        self.camera.stop_capture()
        logger.info("✓ Camera stopped")

        logger.info("=" * 70)
        logger.info("PHASE 3 TEST FINISHED")
        logger.info("=" * 70)
        logger.info("")

        # Ask user to review results
        logger.info("📋 REVIEW TEST RESULTS:")
        logger.info("")
        logger.info("Did the detection test work correctly?")
        logger.info("  - Were pieces detected? (check logs above)")
        logger.info("  - Were zone flags correct?")
        logger.info("  - Were pieces captured?")
        logger.info("  - Are images in captured_pieces/ directory?")
        logger.info("")
        logger.info("If YES: System will continue to Phase 4 automatically")
        logger.info("If NO: Press Ctrl+C to stop and review configuration")
        logger.info("")
        logger.info("Continuing in 10 seconds...")
        logger.info("")

        # Give user time to read and interrupt if needed
        for i in range(10, 0, -1):
            logger.info(f"  Continuing in {i}...")
            time.sleep(1)
            QApplication.processEvents()

        logger.info("")
        logger.info("✓ Proceeding to Phase 4")
        logger.info("")

    # ========================================================================
    # PHASE 4: CALLBACK REGISTRATION
    # ========================================================================

    def register_callbacks(self):
        """Register all callbacks to connect modules."""
        logger.info("Registering callbacks between modules...")

        # Processing → Orchestrator callback
        def on_piece_identified_orchestrator(identified_piece: IdentifiedPiece):
            """Store identified piece in central dictionary."""
            logger.info(f"Callback: Piece {identified_piece.piece_id} identified, storing in dict")
            self.identified_pieces_dict[identified_piece.piece_id] = identified_piece

        self.processing_coordinator.register_identification_callback(
            on_piece_identified_orchestrator
        )
        logger.info("  ✓ Processing → Orchestrator callback registered")

        logger.info("✓ Core callbacks registered")

    # ========================================================================
    # PHASE 5: GUI CREATION
    # ========================================================================

    def create_sorting_gui(self):
        """Create and configure the sorting GUI."""
        logger.info("Creating sorting GUI...")

        self.sorting_gui = SortingGUI(self)
        logger.info("  ✓ SortingGUI instance created")

        self.sorting_gui.show()
        self.sorting_gui.raise_()
        self.sorting_gui.activateWindow()
        logger.info("  ✓ GUI window displayed")

        logger.info("✓ Sorting GUI ready")

    # ========================================================================
    # PHASE 6: RUNTIME OPERATION
    # ========================================================================

    def get_identified_pieces(self) -> Dict[int, IdentifiedPiece]:
        """Get the dictionary of identified pieces."""
        return self.identified_pieces_dict

    def start_sorting(self):
        """Start the sorting operation."""
        logger.info("=" * 70)
        logger.info("PHASE 6: STARTING SORTING OPERATION")
        logger.info("=" * 70)

        try:
            # Start camera capture
            logger.info("Starting camera capture...")
            if not self.camera.start_capture():
                raise RuntimeError("Failed to start camera capture")
            logger.info("✓ Camera capturing at 30 FPS")

            # Register camera consumers
            logger.info("Registering camera consumers...")

            self.camera.register_consumer(
                name="detector",
                callback=self.on_camera_frame,
                processing_type="sync",
                priority=100
            )
            logger.info("  ✓ Detector registered as camera consumer")

            # Start processing workers
            logger.info("Starting processing workers...")
            for worker in self.processing_workers:
                worker.start()
            logger.info(f"✓ {len(self.processing_workers)} workers started")

            # Update state
            self.current_state = ApplicationState.RUNNING
            self.is_running = True
            self.state_changed.emit(self.current_state)

            logger.info("=" * 70)
            logger.info("✓ SYSTEM RUNNING - Sorting in progress")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"FATAL: Failed to start sorting: {e}", exc_info=True)
            self.error_occurred.emit(f"Start failed: {e}")
            self.shutdown_system()

    def on_camera_frame(self, frame):
        """Process each camera frame."""
        if not self.is_running:
            return

        # Run detection
        detection_result = self.detector_coordinator.process_frame_for_consumer(frame)
        tracked_pieces = detection_result.get('tracked_pieces', [])

        # Check for captures
        capture_packages = self.capture_controller.check_and_process_captures(
            frame,
            tracked_pieces
        )

        # Add captures to queue
        for capture_package in capture_packages:
            self.processing_queue_manager.submit_piece(capture_package)

        # Check for pieces ready to sort
        for piece in tracked_pieces:
            if not piece.in_exit_zone:
                continue

            identified_piece = self.identified_pieces_dict.get(piece.id)

            if identified_piece is None:
                continue

            if not identified_piece.complete:
                continue

            if not getattr(piece, 'is_rightmost_in_exit_zone', False):
                continue

            logger.info(f"Triggering sort for piece {piece.id} → bin {identified_piece.bin_number}")

            success = self.hardware_coordinator.trigger_sort_for_piece(
                piece_id=piece.id,
                bin_number=identified_piece.bin_number
            )

            if not success:
                logger.error(f"Failed to sort piece {piece.id}")

        # Update GUI
        if self.sorting_gui:
            self.sorting_gui.update_tracking_display(tracked_pieces)

    # ========================================================================
    # PHASE 7: SHUTDOWN
    # ========================================================================

    @pyqtSlot()
    def stop_sorting(self):
        """Stop sorting operation and begin shutdown."""
        logger.info("Stop sorting requested")

        if self.is_running:
            self.is_running = False
            logger.info("✓ Sorting stopped")

        self.shutdown_system()

    def shutdown_system(self):
        """Clean up all resources and shut down gracefully."""
        logger.info("=" * 70)
        logger.info("SHUTDOWN SEQUENCE INITIATED")
        logger.info("=" * 70)

        self.current_state = ApplicationState.STOPPING
        self.state_changed.emit(self.current_state)

        # Stop camera
        if self.camera:
            try:
                logger.info("Stopping camera...")
                self.camera.stop_capture()
                self.camera.release()
                logger.info("✓ Camera stopped")
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")

        # Stop processing workers
        if self.processing_workers:
            try:
                logger.info("Stopping processing workers...")
                for worker in self.processing_workers:
                    worker.stop(timeout=5.0)
                logger.info("✓ Workers stopped")
            except Exception as e:
                logger.error(f"Error stopping workers: {e}")

        # Release hardware
        if self.hardware_coordinator:
            try:
                logger.info("Releasing hardware...")
                self.hardware_coordinator.release()
                logger.info("✓ Hardware released")
            except Exception as e:
                logger.error(f"Error releasing hardware: {e}")

        # Close GUIs
        if self.sorting_gui:
            try:
                logger.info("Closing sorting GUI...")
                self.sorting_gui.close()
                logger.info("✓ Sorting GUI closed")
            except Exception as e:
                logger.error(f"Error closing GUI: {e}")

        if self.config_gui:
            try:
                logger.info("Closing config GUI...")
                self.config_gui.close()
                logger.info("✓ Config GUI closed")
            except Exception as e:
                logger.error(f"Error closing GUI: {e}")

        self.current_state = ApplicationState.STOPPED
        self.state_changed.emit(self.current_state)

        logger.info("=" * 70)
        logger.info("✓ SHUTDOWN COMPLETE")
        logger.info("=" * 70)

        QApplication.quit()

    def _signal_handler(self, signum, frame):
        """Handle system signals (Ctrl+C, etc)."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_system()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Application entry point."""
    logger.info("=" * 70)
    logger.info("LEGOSORTING MACHINE - STARTING")
    logger.info("=" * 70)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Create orchestrator
    orchestrator = LegoSorting008()

    # Launch configuration GUI
    orchestrator.show_configuration_gui()

    # Start Qt event loop
    exit_code = app.exec_()

    logger.info(f"Application exited with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()