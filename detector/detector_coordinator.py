"""
detector_coordinator.py - Orchestrates the complete computer vision pipeline

This module coordinates all detection subsystems and manages the critical
coordinate system conversions between ROI space and frame space.

Key responsibilities:
- Initialize and coordinate vision_processor, piece_tracker, and zone_manager
- Manage ROI configuration
- Convert coordinates from ROI to frame reference
- Provide unified interface to consumer modules
- Aggregate performance statistics

CRITICAL ARCHITECTURAL NOTES:
- Internal modules (vision_processor, piece_tracker, zone_manager) work in ROI coordinates
- Consumer modules (GUI, capture_controller) need frame coordinates
- This coordinator performs all coordinate conversions

The pipeline flow:
1. vision_processor: frame → detections (ROI cords)
2. piece_tracker: detections → tracked_pieces (ROI cords)
3. zone_manager: tracked_pieces → tracked_pieces with zone flags (ROI cords)
4. coordinator: tracked_pieces (ROI cords) → formatted results (frame cords)
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict

from detector.vision_processor import VisionProcessor
from detector.piece_tracker import PieceTracker
from detector.zone_manager import ZoneManager
from detector.detector_data_models import RegionOfInterest, TrackedPiece

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN DETECTOR COORDINATOR CLASS
# ============================================================================

class DetectorCoordinator:
    """
    Coordinates the complete detection pipeline and manages coordinate systems.

    This class is the single entry point for frame processing. It orchestrates
    all subsystems and ensures proper data flow and coordinate conversion.
    """

    def __init__(self, config_manager, zone_manager: Optional[ZoneManager] = None):
        """
        Initialize the detector coordinator with all subsystems.

        Args:
            config_manager: Enhanced config manager instance (required)
            zone_manager: ZoneManager instance (optional, will be created if None)
        """
        logger.info("Initializing DetectorCoordinator")

        # Store configuration manager
        self.config_manager = config_manager

        # Initialize subsystems
        self.vision_processor = VisionProcessor(config_manager)
        self.piece_tracker = PieceTracker(config_manager)

        # Zone manager can be injected or created
        if zone_manager is None:
            from detector.zone_manager import create_zone_manager
            self.zone_manager = create_zone_manager(config_manager)
            logger.info("Created new ZoneManager instance")
        else:
            self.zone_manager = zone_manager
            logger.info("Using provided ZoneManager instance")

        # ROI configuration
        self._roi: Optional[RegionOfInterest] = None
        self._roi_offset_x = 0
        self._roi_offset_y = 0

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        logger.info("DetectorCoordinator initialized successfully")

    # ========================================================================
    # ROI CONFIGURATION
    # ========================================================================

    def set_roi_from_sample_frame(self, frame: np.ndarray,
                                  roi_coordinates: Tuple[int, int, int, int]):
        """
        Configure ROI and initialize all subsystems with a sample frame.

        This method allows dynamic ROI setup and ensures all subsystems
        are properly configured with the ROI boundaries.

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

        # Configure all subsystems with new ROI
        self.vision_processor.set_roi(self._roi, frame)
        self.piece_tracker.set_roi(self._roi)
        self.zone_manager.set_roi(self._roi)

        logger.info(f"ROI configured: {roi_coordinates}")
        logger.info(f"All subsystems initialized with ROI")

    # ========================================================================
    # EXIT DETECTION
    # ========================================================================

    def _update_exit_flags(self, tracked_pieces: List[TrackedPiece]) -> None:
        """
        Update exit flags for pieces that have left the ROI.

        A piece is considered "exited" when its right edge moves past
        the right edge of the ROI. We record the exit timestamp for
        fall time calculations.

        Args:
            tracked_pieces: List of TrackedPiece objects in ROI coordinates
        """
        if not self._roi:
            return  # ROI not configured yet

        roi_right_edge = self._roi.width  # Right edge in ROI coordinates

        for piece in tracked_pieces:
            # Skip if already marked as exited
            if piece.has_exited_roi:
                continue

            # Check if piece right edge is past ROI right edge
            # piece.right_edge is already in ROI coordinates
            if piece.right_edge > roi_right_edge:
                # Piece has exited!
                piece.has_exited_roi = True
                piece.exit_timestamp = time.time()

                logger.info(f"🚪 Piece {piece.id} exited ROI at x={piece.right_edge:.1f}")

    # ========================================================================
    # MAIN PROCESSING PIPELINE
    # ========================================================================

    def process_frame_for_consumer(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a camera frame through the complete detection pipeline.

        This is the main entry point for frame processing. It coordinates
        vision processing, piece tracking, zone management, and coordinate conversion.

        PIPELINE FLOW:
        1. Frame comes in camera/frame coordinates
        2. vision_processor: extracts ROI, returns detections in ROI coordinates
        3. piece_tracker: matches and tracks pieces in ROI coordinates
        4. zone_manager: updates zone flags on pieces (still ROI coordinates)
        5. coordinator: converts pieces to frame coordinates
        6. Returns formatted results ready for consumers

        Args:
            frame: Camera frame as numpy array

        Returns:
            Dictionary with tracking results in frame coordinates:
            {
                "tracked_pieces": [list of complete piece dictionaries],
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

            # Step 3: Zone status management (updates zone flags, still ROI coordinates)
            tracked_pieces = self.zone_manager.update_piece_zones(tracked_pieces)

            # NEW: Step 3.5: Update exit detection flags
            self._update_exit_flags(tracked_pieces)

            # Step 4: Convert to frame coordinates and format for consumers
            formatted_results = self._format_results_for_consumers(tracked_pieces, detections)

            logger.debug(f"Processed frame: {len(detections)} detections, "
                         f"{len(tracked_pieces)} tracked pieces")

            return formatted_results

        except Exception as e:
            logger.error(f"Error in detection pipeline: {e}", exc_info=True)
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

        Args:
            roi_bbox: Bounding box in ROI coordinates (x, y, w, h)

        Returns:
            Bounding box in frame coordinates (x, y, w, h)
        """
        x, y, w, h = roi_bbox
        return x + self._roi_offset_x, y + self._roi_offset_y, w, h

    def _convert_roi_center_to_frame(self, roi_center: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert center point from ROI coordinates to frame coordinates.

        Args:
            roi_center: Center point in ROI coordinates (x, y)

        Returns:
            Center point in frame coordinates (x, y)
        """
        x, y = roi_center
        return x + self._roi_offset_x, y + self._roi_offset_y

    def _convert_tracked_piece_to_frame_coords(self, piece: TrackedPiece) -> TrackedPiece:
        """
        Convert a TrackedPiece from ROI coordinates to frame coordinates.

        This creates a new TrackedPiece with updated coordinate fields.
        The TrackedPiece.__post_init__() will automatically calculate
        center, left_edge, right_edge, and area from the frame_bbox.

        Args:
            piece: TrackedPiece in ROI coordinates

        Returns:
            New TrackedPiece with frame coordinates
        """
        # Convert bbox to frame coordinates
        frame_bbox = self._convert_roi_to_frame_coordinates(piece.bbox)

        # Create a new TrackedPiece with frame coordinates
        # Note: contour is preserved as-is (doesn't need coordinate conversion)
        # The __post_init__() will calculate center, edges, and area automatically
        frame_piece = TrackedPiece(
            id=piece.id,
            bbox=frame_bbox,
            contour=piece.contour,  # Required field - preserve original contour
            first_detected=piece.first_detected,
            last_updated=piece.last_updated,
            captured=piece.captured,
            being_processed=piece.being_processed,
            processing_start_time=piece.processing_start_time,
            fully_in_frame=piece.fully_in_frame,
            in_entry_zone=piece.in_entry_zone,
            in_valid_zone=piece.in_valid_zone,
            in_exit_zone=piece.in_exit_zone,
            exit_zone_entry_time=piece.exit_zone_entry_time,
            update_count=piece.update_count,
            last_matched_confidence=piece.last_matched_confidence,
            velocity=piece.velocity,
            trajectory=piece.trajectory,
            predicted_exit_time=piece.predicted_exit_time
        )

        return frame_piece

    # ========================================================================
    # RESULT FORMATTING
    # ========================================================================

    def _format_results_for_consumers(self, tracked_pieces: List[TrackedPiece],
                                      detections: List) -> Dict[str, Any]:
        """
        Format tracking results for consumer modules.

        This converts TrackedPiece objects (in ROI coordinates) to complete
        dictionaries (in frame coordinates) that include ALL piece information.

        Args:
            tracked_pieces: List of TrackedPiece objects (ROI coordinates)
            detections: List of Detection objects from vision processor

        Returns:
            Formatted dictionary with frame coordinate data
        """
        # Convert tracked pieces to frame coordinates and then to dictionaries
        formatted_pieces = []

        for piece in tracked_pieces:
            # Convert to frame coordinates
            frame_piece = self._convert_tracked_piece_to_frame_coords(piece)

            # Convert to dictionary with all fields
            piece_dict = asdict(frame_piece)

            # Add computed status for visualization
            piece_dict["status"] = self._determine_piece_status(frame_piece)

            formatted_pieces.append(piece_dict)

        # Aggregate statistics from all modules
        statistics = self._aggregate_performance_statistics(tracked_pieces, detections)

        # Include ROI information for visualization overlay
        roi_info = self._get_roi_info()

        return {
            "tracked_pieces": formatted_pieces,
            "statistics": statistics,
            "roi_info": roi_info,
            "error": None
        }

    def _determine_piece_status(self, piece: TrackedPiece) -> str:
        """
        Determine the status string for GUI color coding.

        Args:
            piece: TrackedPiece to evaluate

        Returns:
            Status string: "detected", "stable", "processing", or "identified"
        """
        if piece.being_processed:
            return "processing"
        elif piece.captured:
            return "identified"
        elif piece.update_count >= 5:  # Stable threshold
            return "stable"
        else:
            return "detected"

    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================

    def _update_performance_metrics(self):
        """Update FPS and frame count metrics."""
        self.frame_count += 1

        # Calculate FPS every 30 frames
        if self.frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed

    def _aggregate_performance_statistics(self, tracked_pieces: List[TrackedPiece],
                                          detections: List) -> Dict[str, Any]:
        """
        Combine metrics from all subsystems into unified statistics.

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

        Returns:
            Dictionary with ROI configuration
        """
        if self._roi is None:
            return {"configured": False}

        return {
            "configured": True,
            "roi": self._roi.to_tuple()
        }

    # ========================================================================
    # DIRECT ACCESS METHODS
    # ========================================================================

    def get_tracked_pieces(self) -> List[TrackedPiece]:
        """
        Get currently tracked pieces in frame coordinates.

        This is used by modules that need direct access to TrackedPiece objects
        (like capture_controller) rather than formatted dictionaries.

        Note: Zone flags are already updated by process_frame_for_consumer(),
        so we don't need to update them again here.

        Returns:
            List of TrackedPiece objects in frame coordinates
        """
        # Get pieces from tracker (ROI coordinates)
        # Zone flags are already up-to-date from the main pipeline
        roi_pieces = self.piece_tracker.get_active_pieces()

        # Convert to frame coordinates
        frame_pieces = []
        for piece in roi_pieces:
            frame_piece = self._convert_tracked_piece_to_frame_coords(piece)
            frame_pieces.append(frame_piece)

        return frame_pieces

    def get_fully_visible_pieces(self) -> List[TrackedPiece]:
        """
        Get only fully visible pieces in frame coordinates.

        Note: Zone flags are already updated by process_frame_for_consumer(),
        so we don't need to update them again here.

        Returns:
            List of fully visible TrackedPiece objects in frame coordinates
        """
        roi_pieces = self.piece_tracker.get_fully_visible_pieces()

        # Convert to frame coordinates
        frame_pieces = []
        for piece in roi_pieces:
            frame_piece = self._convert_tracked_piece_to_frame_coords(piece)
            frame_pieces.append(frame_piece)

        return frame_pieces

    def mark_piece_as_captured(self, piece_id: int) -> bool:
        """
        Mark a piece as captured in the piece tracker.

        This method updates the authoritative ROI piece stored in piece_tracker,
        ensuring that subsequent calls to get_tracked_pieces() return pieces
        with the correct captured status.

        ARCHITECTURAL NOTE:
        The piece_tracker maintains pieces in ROI coordinates (source of truth).
        When get_tracked_pieces() is called, it creates NEW TrackedPiece objects
        in frame coordinates by copying data from the ROI pieces. Therefore,
        status flags like 'captured' must be updated on the ROI pieces, not on
        the frame coordinate copies.

        WORKFLOW:
        1. CaptureController captures piece (using frame coordinates)
        2. Orchestrator calls this method to update ROI piece
        3. GUI calls get_tracked_pieces() and receives updated frame pieces

        Args:
            piece_id: The ID of the piece to mark as captured

        Returns:
            True if piece was found and marked, False if piece_id not found

        """
        # Get the authoritative ROI pieces from tracker
        roi_pieces = self.piece_tracker.get_active_pieces()

        # Find the piece by ID and update its captured status
        for piece in roi_pieces:
            if piece.id == piece_id:
                # Update capture flags on the ROI piece (source of truth)
                piece.captured = True
                piece.being_processed = True
                piece.processing_start_time = time.time()

                logger.debug(f"Marked piece {piece_id} as captured in piece_tracker")
                return True

        # Piece not found - may have already been removed from tracking
        logger.warning(f"Could not mark piece {piece_id} as captured - piece not found in tracker")
        return False

    def mark_piece_processing_complete(self, piece_id: int) -> bool:
        """
        Mark a piece's processing as complete in the piece tracker.

        This method updates the processing status flags when identification
        is complete. This is typically called by the processing coordinator
        callback in the orchestrator.

        Args:
            piece_id: The ID of the piece that finished processing

        Returns:
            True if piece was found and marked, False if piece_id not found

        """
        # Get the authoritative ROI pieces from tracker
        roi_pieces = self.piece_tracker.get_active_pieces()

        # Find the piece by ID and update its processing status
        for piece in roi_pieces:
            if piece.id == piece_id:
                # Mark processing as complete
                piece.being_processed = False

                logger.debug(f"Marked piece {piece_id} processing as complete in piece_tracker")
                return True

        # Piece not found - may have already exited tracking
        logger.warning(f"Could not mark piece {piece_id} processing complete - piece not found in tracker")
        return False

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

    def update_background_model(self, empty_frame: np.ndarray):
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
            "piece_tracker_ready": True,
            "zone_manager_ready": self.zone_manager.get_zone_configuration() is not None,
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

def create_detector_coordinator(config_manager, zone_manager: Optional[ZoneManager] = None) -> DetectorCoordinator:
    """
    Create a DetectorCoordinator instance with unified configuration.

    This is the standard way to create the detection system using the
    enhanced_config_manager for consistent configuration management.

    Args:
        config_manager: Enhanced config manager instance (required)
        zone_manager: Optional ZoneManager instance (will be created if None)

    Returns:
        Configured DetectorCoordinator instance

    Raises:
        ValueError: If config_manager is None or invalid
    """
    if config_manager is None:
        raise ValueError("create_detector_coordinator requires a valid config_manager")

    logger.info("Creating DetectorCoordinator with enhanced_config_manager")
    return DetectorCoordinator(config_manager, zone_manager)
