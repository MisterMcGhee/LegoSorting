"""
detector_coordinator.py - Orchestrates vision processing and piece tracking modules

This module serves as the orchestration layer that coordinates the modular detection pipeline.
It replaces the original monolithic detector_module.py with a clean separation of concerns:

- Vision Processor: Stateless computer vision (detections in ROI coordinates)
- Piece Tracker: Stateful object tracking (pieces in ROI coordinates)
- Detector Coordinator: Interface layer and coordinate system management

KEY COORDINATE SYSTEM RESPONSIBILITY:
The coordinator handles the critical transition from ROI coordinates (used internally
by vision and tracking modules) to frame coordinates (required by GUI and consumers).
This design keeps the underlying modules simple while providing the interface that
downstream modules expect.

Architecture Benefits:
- Clean separation: Each module has a single, focused responsibility
- Maintainable: Vision and tracking algorithms can evolve independently
- Testable: Each component can be tested in isolation
- Flexible: Easy to swap different vision or tracking algorithms
"""

import time
import logging
import cv2
from typing import List, Dict, Any, Optional, Tuple
from data_models import RegionOfInterest
from vision_processor import create_vision_processor
from piece_tracker import create_piece_tracker
from enhanced_config_manager import ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN DETECTOR COORDINATOR CLASS
# ============================================================================

class DetectorCoordinator:
    """
    Orchestrates the modular detection pipeline and manages coordinate systems.

    This coordinator replaces the monolithic detector with a clean architecture
    that separates computer vision, object tracking, and interface management.

    CRITICAL COORDINATE SYSTEM MANAGEMENT:
    - Vision processor works in ROI coordinates (0,0 = top-left of ROI)
    - Piece tracker works in ROI coordinates (same as its input detections)
    - GUI and consumers need frame coordinates (0,0 = top-left of camera frame)
    - Coordinator converts ROI coordinates → frame coordinates for all output

    This separation allows each module to work in its most natural coordinate
    space while providing the interface that consumers expect.
    """

    def __init__(self, config_manager):
        """
        Initialize the detector coordinator with modular components.

        Args:
            config_manager: Enhanced config manager instance (required)

        Raises:
            ValueError: If config_manager is None or invalid
        """
        if not config_manager:
            raise ValueError("DetectorCoordinator requires a valid config_manager")

        logger.info("Initializing DetectorCoordinator")

        self.config_manager = config_manager
        self._roi = None  # Will be loaded from configuration
        self._roi_offset_x = 0  # For coordinate conversion
        self._roi_offset_y = 0  # For coordinate conversion

        # Initialize sub-modules
        self.vision_processor = create_vision_processor(config_manager)
        self.piece_tracker = create_piece_tracker(config_manager)

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.fps = 0.0

        # Load ROI configuration and set up modules
        self._load_roi_configuration()

        logger.info("DetectorCoordinator initialized successfully")

    # ========================================================================
    # CONFIGURATION AND SETUP
    # ========================================================================

    def _load_roi_configuration(self):
        """
        Load ROI configuration and initialize submodules.

        The ROI defines the detection area and is used by both vision processing
        and piece tracking. It also provides the coordinate system offset needed
        for converting internal coordinates to frame coordinates.

        Raises:
            Exception: If ROI configuration cannot be loaded
        """
        # Get ROI configuration from unified system
        roi_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR_ROI.value)

        # Extract ROI parameters
        x = roi_config.get("x", 0)
        y = roi_config.get("y", 0)
        width = roi_config.get("w", 100)
        height = roi_config.get("h", 100)

        # Create ROI object
        self._roi = RegionOfInterest(x=x, y=y, width=width, height=height)

        # Store offset for coordinate conversions
        self._roi_offset_x = x
        self._roi_offset_y = y

        # Configure submodules with ROI
        self.piece_tracker.set_roi(self._roi)

        logger.info(f"ROI configured: {self._roi.to_tuple()}")
        logger.info(f"Coordinate conversion offset: ({self._roi_offset_x}, {self._roi_offset_y})")

    def set_roi_from_sample_frame(self, frame, roi_coordinates: Tuple[int, int, int, int]):
        """
        Set ROI configuration using a sample frame for background model training.

        This method allows dynamic ROI setup (e.g., from calibration GUI) and
        ensures the vision processor gets a proper background model.

        Args:
            frame: Sample frame for training background subtraction
            roi_coordinates: ROI as (x, y, width, height) tuple

        Raises:
            ValueError: If ROI coordinates are invalid for the frame
        """
        x, y, w, h = roi_coordinates
        frame_height, frame_width = frame.shape[:2]

        # Validate ROI is within frame bounds
        if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
            raise ValueError(f"ROI {roi_coordinates} is outside frame bounds "
                             f"({frame_width}x{frame_height})")

        # Update ROI configuration
        self._roi = RegionOfInterest(x=x, y=y, width=w, height=h)
        self._roi_offset_x = x
        self._roi_offset_y = y

        # Configure submodules with new ROI
        self.vision_processor.set_roi(self._roi, frame)
        self.piece_tracker.set_roi(self._roi)

        logger.info(f"ROI updated to {roi_coordinates} with sample frame training")

    # ========================================================================
    # MAIN PROCESSING PIPELINE
    # ========================================================================

    def process_frame_for_consumer(self, frame) -> Dict[str, Any]:
        """
        Process a camera frame through the complete detection pipeline.

        This is the main entry point for frame processing. It coordinates
        vision processing and piece tracking, then formats results for
        consumer modules (GUI, capture controller, etc.).

        COORDINATE SYSTEM FLOW:
        1. Frame comes in camera/frame coordinates
        2. Vision processor extracts ROI and works in ROI coordinates
        3. Piece tracker receives detections in ROI coordinates
        4. Coordinator converts tracked pieces to frame coordinates for output

        Args:
            frame: Camera frame as numpy array

        Returns:
            Dictionary with tracking results in frame coordinates:
            {
                "tracked_pieces": [list of piece dictionaries],
                "statistics": {performance and count metrics},
                "roi_info": {ROI configuration for visualization},
                "error": None or error message
            }
        """
        try:
            if self._roi is None:
                return {"error": "ROI not configured"}

            # Update performance metrics
            self._update_performance_metrics()

            # Step 1: Computer vision processing (returns detections in ROI coordinates)
            detections = self.vision_processor.process_frame(frame)

            # Step 2: Object tracking (maintains pieces in ROI coordinates)
            tracked_pieces = self.piece_tracker.update_tracking(detections)

            # Step 3: Convert to frame coordinates and format for consumers
            formatted_results = self._format_results_for_consumers(tracked_pieces, detections)

            logger.debug(f"Processed frame: {len(detections)} detections, "
                         f"{len(tracked_pieces)} tracked pieces")

            return formatted_results

        except Exception as e:
            logger.error(f"Error in detection pipeline: {e}")
            return {
                "error": f"Detection pipeline error: {str(e)}",
                "tracked_pieces": [],
                "statistics": self._get_error_statistics(),
                "roi_info": self._get_roi_info()
            }

    # ========================================================================
    # COORDINATE SYSTEM CONVERSION
    # ========================================================================

    def _convert_roi_to_frame_coordinates(self, roi_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Convert bounding box from ROI coordinates to frame coordinates.

        CRITICAL CONVERSION FUNCTION:
        This handles the coordinate system transition that allows internal modules
        to work in ROI space while providing frame coordinates to consumers.

        ROI coordinates: (0,0) = top-left corner of ROI
        Frame coordinates: (0,0) = top-left corner of camera frame

        Args:
            roi_bbox: Bounding box in ROI coordinates (x, y, width, height)

        Returns:
            Bounding box in frame coordinates (x, y, width, height)
        """
        x, y, w, h = roi_bbox

        # Add ROI offset to position coordinates
        # Width and height remain unchanged
        frame_x = x + self._roi_offset_x
        frame_y = y + self._roi_offset_y

        return (frame_x, frame_y, w, h)

    def _convert_roi_center_to_frame(self, roi_center: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert center point from ROI coordinates to frame coordinates.

        Args:
            roi_center: Center point in ROI coordinates (x, y)

        Returns:
            Center point in frame coordinates (x, y)
        """
        x, y = roi_center

        # Add ROI offset to center coordinates
        frame_x = x + self._roi_offset_x
        frame_y = y + self._roi_offset_y

        return (frame_x, frame_y)

    # ========================================================================
    # DATA FORMATTING FOR CONSUMERS
    # ========================================================================

    def _format_results_for_consumers(self, tracked_pieces, detections) -> Dict[str, Any]:
        """
        Format tracking results for consumption by GUI and other modules.

        This method performs the critical coordinate conversion and creates
        the data structures that consumer modules expect.

        OUTPUT FORMAT FOR GUI:
        Each piece becomes a dictionary with:
        - id: Unique piece identifier
        - bbox: Bounding box in FRAME coordinates for drawing
        - status: String for color-coding visualization
        - center: Center point in FRAME coordinates

        Args:
            tracked_pieces: List of TrackedPiece objects (ROI coordinates)
            detections: List of Detection objects from vision processor

        Returns:
            Formatted dictionary with frame coordinate data
        """
        # Convert tracked pieces to consumer format
        formatted_pieces = []
        for piece in tracked_pieces:
            # Convert coordinates from ROI to frame space
            frame_bbox = self._convert_roi_to_frame_coordinates(piece.bbox)
            frame_center = self._convert_roi_center_to_frame(piece.center)

            # Determine piece status for GUI color coding
            status = self._determine_piece_status(piece)

            # Create formatted piece data
            piece_data = {
                "id": piece.id,
                "bbox": frame_bbox,  # FRAME coordinates for GUI drawing
                "center": frame_center,  # FRAME coordinates for GUI drawing
                "status": status,  # For color-coding visualization
                "fully_in_frame": piece.fully_in_frame,
                "update_count": piece.update_count,
                "velocity": piece.velocity  # For debugging/monitoring
            }

            formatted_pieces.append(piece_data)

        # Aggregate statistics from both modules
        statistics = self._aggregate_performance_statistics(tracked_pieces, detections)

        # Include ROI information for visualization overlay
        roi_info = self._get_roi_info()

        return {
            "tracked_pieces": formatted_pieces,
            "statistics": statistics,
            "roi_info": roi_info,
            "error": None
        }

    def _determine_piece_status(self, piece) -> str:
        """
        Determine the status string for GUI color coding.

        Status affects how the GUI draws bounding boxes:
        - "detected": Basic detection (neutral color)
        - "processing": Image captured, being analyzed (yellow/orange)
        - "identified": Analysis complete (green)
        - "stable": Tracked reliably but not yet captured (blue)

        Args:
            piece: TrackedPiece object

        Returns:
            Status string for GUI consumption
        """
        if piece.being_processed:
            return "processing"
        elif piece.captured:
            return "identified"
        elif piece.fully_in_frame and piece.update_count >= 3:
            return "stable"
        else:
            return "detected"

    # ========================================================================
    # PERFORMANCE MONITORING
    # ========================================================================

    def _update_performance_metrics(self):
        """
        Update FPS and performance tracking for monitoring.

        This provides unified performance metrics across the entire
        detection pipeline for debugging and optimization.
        """
        self.frame_count += 1
        current_time = time.time()

        # Update FPS every second
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.last_fps_update = current_time

    def _aggregate_performance_statistics(self, tracked_pieces, detections) -> Dict[str, Any]:
        """
        Aggregate performance statistics from vision and tracking modules.

        Combines metrics from both submodules into unified statistics
        for monitoring system performance and debugging issues.

        Args:
            tracked_pieces: Current tracked pieces
            detections: Current frame detections

        Returns:
            Dictionary with aggregated performance metrics
        """
        # Get statistics from piece tracker
        tracking_stats = self.piece_tracker.get_tracking_statistics()

        # Get vision processor background model info
        vision_stats = self.vision_processor.get_background_model_info()

        # Count pieces by status
        status_counts = {
            "detected": 0,
            "stable": 0,
            "processing": 0,
            "identified": 0
        }

        for piece in tracked_pieces:
            status = self._determine_piece_status(piece)
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            # Overall pipeline metrics
            "fps": self.fps,
            "frame_count": self.frame_count,
            "detections_this_frame": len(detections),

            # Piece tracking metrics
            "total_pieces": len(tracked_pieces),
            "status_breakdown": status_counts,
            "next_piece_id": tracking_stats.get("next_piece_id", 0),

            # Performance metrics
            "average_velocity": tracking_stats.get("average_velocity", 0.0),
            "tracking_efficiency": len(tracked_pieces) / max(1, len(detections)) if detections else 1.0,

            # System health indicators
            "vision_roi_configured": vision_stats.get("roi_set", False),
            "background_model_ready": "threshold" in vision_stats
        }

    def _get_error_statistics(self) -> Dict[str, Any]:
        """
        Get minimal statistics when pipeline errors occur.

        Returns:
            Basic statistics dictionary for error conditions
        """
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "detections_this_frame": 0,
            "total_pieces": 0,
            "error_state": True
        }

    def _get_roi_info(self) -> Dict[str, Any]:
        """
        Get ROI information for visualization overlay.

        The GUI needs ROI boundaries to draw the detection area overlay.
        ROI coordinates are already in frame coordinates.

        Returns:
            Dictionary with ROI configuration data
        """
        if not self._roi:
            return {"configured": False}

        return {
            "configured": True,
            "roi": self._roi.to_tuple(),  # Already in frame coordinates
            "offset": (self._roi_offset_x, self._roi_offset_y)
        }

    # ========================================================================
    # DATA ACCESS METHODS
    # ========================================================================

    def get_tracked_pieces(self) -> List:
        """
        Get current tracked pieces in frame coordinates.

        This provides direct access to tracked pieces for modules that
        need to work with piece data outside the main processing loop.

        Returns:
            List of TrackedPiece objects with frame coordinate bboxes
        """
        roi_pieces = self.piece_tracker.get_active_pieces()

        # Convert to frame coordinates
        frame_pieces = []
        for piece in roi_pieces:
            # Create copy with frame coordinates
            frame_piece = piece
            frame_piece.bbox = self._convert_roi_to_frame_coordinates(piece.bbox)
            frame_piece.center = self._convert_roi_center_to_frame(piece.center)
            frame_pieces.append(frame_piece)

        return frame_pieces

    def get_piece_by_id(self, piece_id: int):
        """
        Get a specific tracked piece by ID with frame coordinates.

        Args:
            piece_id: Unique ID of piece to retrieve

        Returns:
            TrackedPiece with frame coordinates or None if not found
        """
        roi_piece = self.piece_tracker.get_piece_by_id(piece_id)

        if roi_piece:
            # Convert to frame coordinates
            roi_piece.bbox = self._convert_roi_to_frame_coordinates(roi_piece.bbox)
            roi_piece.center = self._convert_roi_center_to_frame(roi_piece.center)

        return roi_piece

    def get_fully_visible_pieces(self) -> List:
        """
        Get pieces that are fully visible, with frame coordinates.

        This is used by capture controller and other modules that need
        to work with complete, stable pieces.

        Returns:
            List of fully visible TrackedPiece objects in frame coordinates
        """
        roi_pieces = self.piece_tracker.get_fully_visible_pieces()

        # Convert to frame coordinates
        frame_pieces = []
        for piece in roi_pieces:
            piece.bbox = self._convert_roi_to_frame_coordinates(piece.bbox)
            piece.center = self._convert_roi_center_to_frame(piece.center)
            frame_pieces.append(piece)

        return frame_pieces

    # ========================================================================
    # SYSTEM MANAGEMENT
    # ========================================================================

    def reset_detection_system(self):
        """
        Reset the entire detection pipeline.

        This clears all tracked pieces and resets background models.
        Useful when the conveyor is cleared or system is restarted.
        """
        logger.info("Resetting detection system")

        # Reset tracking state
        self.piece_tracker.reset_tracking()

        # Reset performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        logger.info("Detection system reset complete")

    def update_background_model(self, empty_frame):
        """
        Update the vision processor background model.

        This is useful when lighting conditions change or when you want
        to retrain the background with a known empty conveyor frame.

        Args:
            empty_frame: Frame showing empty conveyor for background training
        """
        logger.info("Updating background model with new empty frame")
        self.vision_processor.update_background_model(empty_frame)

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for monitoring and debugging.

        Returns:
            Dictionary with detailed system state information
        """
        tracking_stats = self.piece_tracker.get_tracking_statistics()
        vision_stats = self.vision_processor.get_background_model_info()

        return {
            "roi_configured": self._roi is not None,
            "vision_processor_ready": vision_stats.get("roi_set", False),
            "piece_tracker_ready": True,  # Tracker is always ready once initialized
            "current_piece_count": tracking_stats.get("total_pieces", 0),
            "system_fps": self.fps,
            "frames_processed": self.frame_count,
            "coordinate_offset": (self._roi_offset_x, self._roi_offset_y)
        }

    def release(self):
        """
        Clean up resources when shutting down.

        This ensures proper cleanup of submodules and prevents
        resource leaks during system shutdown.
        """
        logger.info("Releasing DetectorCoordinator resources")

        # Submodules may have their own cleanup needs
        if hasattr(self.vision_processor, 'release'):
            self.vision_processor.release()

        # Reset state
        self._roi = None
        self.frame_count = 0

        logger.info("DetectorCoordinator cleanup complete")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_detector_coordinator(config_manager) -> DetectorCoordinator:
    """
    Create a DetectorCoordinator instance with unified configuration.

    This is the standard way to create the detection system using the
    enhanced_config_manager for consistent configuration management.

    Args:
        config_manager: Enhanced config manager instance (required)

    Returns:
        Configured DetectorCoordinator instance

    Raises:
        ValueError: If config_manager is None or invalid
    """
    if config_manager is None:
        raise ValueError("create_detector_coordinator requires a valid config_manager")

    logger.info("Creating DetectorCoordinator with enhanced_config_manager")
    return DetectorCoordinator(config_manager)


# ============================================================================
# TESTING AND UTILITIES
# ============================================================================

if __name__ == "__main__":
    """
    Test the detector coordinator with synthetic data.

    This demonstrates the complete pipeline: vision → tracking → coordinate conversion
    """
    import sys
    import numpy as np

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing DetectorCoordinator with synthetic data")


    # Create a mock config manager for testing
    class MockConfigManager:
        """Mock config manager with test configuration"""

        def get_module_config(self, module_name):
            configs = {
                "detector": {
                    "min_area": 1000, "max_area": 50000,
                    "min_aspect_ratio": 0.3, "max_aspect_ratio": 3.0,
                    "bg_history": 500, "bg_threshold": 800.0, "learn_rate": 0.005,
                    "gaussian_blur_size": 7, "morph_kernel_size": 5,
                    "min_contour_points": 5,
                    "match_distance_threshold": 100.0, "match_x_weight": 2.0,
                    "match_y_weight": 1.0, "min_updates_for_stability": 3,
                    "piece_timeout_seconds": 2.0, "fully_in_frame_margin": 20,
                    "min_velocity_samples": 2, "max_velocity_change": 50.0
                },
                "detector_roi": {
                    "x": 100, "y": 100, "w": 600, "h": 200
                }
            }
            return configs.get(module_name, {})


    try:
        # Create coordinator with mock config
        mock_config = MockConfigManager()
        coordinator = create_detector_coordinator(mock_config)

        # Create test frame with realistic size
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Initialize with properly sized ROI for test frame
        coordinator.set_roi_from_sample_frame(test_frame, (50, 100, 400, 200))

        print(f"\nDetectorCoordinator Test Results:")
        print(f"=" * 50)

        # Test frame processing
        for frame_num in range(5):
            # Add some white rectangles as "pieces" in the frame
            test_frame.fill(0)  # Clear frame
            cv2.rectangle(test_frame, (200, 150), (240, 180), (255, 255, 255), -1)

            # Process frame
            results = coordinator.process_frame_for_consumer(test_frame)

            print(f"\nFrame {frame_num + 1}:")
            print(f"  Error: {results.get('error', 'None')}")
            print(f"  Tracked pieces: {len(results.get('tracked_pieces', []))}")
            print(f"  FPS: {results.get('statistics', {}).get('fps', 0):.1f}")

            # Show piece details
            for piece in results.get('tracked_pieces', []):
                print(f"    Piece {piece['id']}: bbox={piece['bbox']}, status={piece['status']}")

        # Test system status
        status = coordinator.get_system_status()
        print(f"\nSystem Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

        logger.info("DetectorCoordinator test completed successfully")

    except Exception as e:
        logger.error(f"DetectorCoordinator test failed: {e}")
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        try:
            coordinator.release()
        except:
            pass
