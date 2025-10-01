"""
piece_tracker.py - Object tracking for Lego pieces moving on conveyor belt

This module bridges the gap between stateless detection and stateful tracking.
It takes Detection objects from the vision processor and maintains TrackedPiece
objects that persist across frames, providing continuity and motion tracking.

Key characteristics:
- STATEFUL: Maintains TrackedPiece objects across multiple frames
- MATCHING: Associates new detections with existing tracked pieces
- LIFECYCLE: Creates new pieces and removes stale/exited pieces
- MOTION AWARE: Tracks simple velocity for better matching and coordination

This module handles:
- Piece identity continuity as objects move through the ROI
- Position and velocity updates for each tracked piece
- Fully-in-frame detection as pieces enter the conveyor
- Piece removal when they exit or timeout

This module does NOT:
- Make capture decisions (that's the capture controller's job)
- Handle zone-specific business logic (that's the zone manager's job)
- Perform computer vision (that's the vision processor's job)
"""

import time
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from detector.data_models import Detection, TrackedPiece, RegionOfInterest, create_tracked_piece_from_detection
from enhanced_config_manager import ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN PIECE TRACKER CLASS
# ============================================================================

class PieceTracker:
    """
    Manages the lifecycle and tracking of Lego pieces moving on the conveyor.

    This class maintains a collection of TrackedPiece objects and updates them
    as new Detection data arrives from the vision processor. It handles the
    complex task of maintaining piece identity across frames despite movement.

    The tracker is optimized for the specific constraints of the conveyor system:
    - Pieces move at constant speed from left to right
    - Pieces are separated (no overlapping or touching)
    - Single direction motion simplifies matching algorithms
    """

    def __init__(self, config_manager):
        """
        Initialize the piece tracker with unified configuration.

        Args:
            config_manager: Enhanced config manager instance (required)

        Raises:
            ValueError: If config_manager is None or invalid
        """
        if not config_manager:
            raise ValueError("PieceTracker requires a valid config_manager")

        logger.info("Initializing PieceTracker")

        self.config_manager = config_manager
        self._roi = None  # Will be set when ROI is configured

        # Core tracking state
        self.tracked_pieces: List[TrackedPiece] = []
        self.piece_id_counter = 0  # For generating unique piece IDs

        # Load configuration from unified system
        self._load_configuration()

        logger.info("PieceTracker initialized successfully")

    def _load_configuration(self):
        """
        Load all tracking parameters from enhanced_config_manager.

        Configuration comes from the detector module since tracking
        parameters are closely related to detection settings.

        Raises:
            Exception: If configuration cannot be loaded
        """
        # Get detector configuration from unified system
        detector_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR.value)

        # Matching parameters - how close detections must be to existing pieces
        self.match_distance_threshold = detector_config.get("match_distance_threshold", 100.0)
        self.match_x_weight = detector_config.get("match_x_weight", 2.0)  # X movement more important
        self.match_y_weight = detector_config.get("match_y_weight", 1.0)  # Y movement less important

        # Piece lifecycle parameters
        self.min_updates_for_stability = detector_config.get("min_updates_for_stability", 3)
        self.piece_timeout_seconds = detector_config.get("piece_timeout_seconds", 2.0)
        self.fully_in_frame_margin = detector_config.get("fully_in_frame_margin", 20)

        # Motion tracking parameters
        self.min_velocity_samples = detector_config.get("min_velocity_samples", 2)
        self.max_velocity_change = detector_config.get("max_velocity_change", 50.0)  # pixels/second

        logger.info("Tracking configuration loaded from enhanced_config_manager")

    # ========================================================================
    # ROI CONFIGURATION
    # ========================================================================

    def set_roi(self, roi: RegionOfInterest):
        """
        Set the region of interest for tracking operations.

        The ROI defines the boundaries used for determining when pieces
        are fully in frame and when they have exited the tracking area.

        Args:
            roi: Region of interest configuration
        """
        self._roi = roi
        logger.info(f"PieceTracker ROI set to {roi.to_tuple()}")

    # ========================================================================
    # MAIN TRACKING PIPELINE
    # ========================================================================

    def update_tracking(self, detections: List[Detection]) -> List[TrackedPiece]:
        """
        Update tracking with new detections from the vision processor.

        This is the main entry point for tracking updates. It processes
        new detections and maintains the collection of tracked pieces.

        Args:
            detections: List of Detection objects from vision processor

        Returns:
            List of currently active TrackedPiece objects
        """
        current_time = time.time()

        # Step 1: Mark all existing pieces as not updated this frame
        for piece in self.tracked_pieces:
            piece.updated = False

        # Step 2: Match detections to existing pieces
        unmatched_detections = self._match_detections_to_pieces(detections, current_time)

        # Step 3: Create new pieces for unmatched detections
        self._create_new_pieces(unmatched_detections, current_time)

        # Step 4: Update piece states (fully_in_frame, motion, etc.)
        self._update_piece_states(current_time)

        # Step 5: Remove stale and exited pieces
        self._remove_inactive_pieces(current_time)

        logger.debug(f"Tracking update complete: {len(self.tracked_pieces)} active pieces")

        # 🔍 DIAGNOSTIC: Check internal state before returning
        if self.tracked_pieces:
            logger.warning(
                f"🔍 TRACKER INTERNAL - Piece {self.tracked_pieces[0].id} center: {self.tracked_pieces[0].center}")

        return self.tracked_pieces.copy()

    # ========================================================================
    # DETECTION MATCHING ALGORITHMS
    # ========================================================================

    def _match_detections_to_pieces(self, detections: List[Detection],
                                    current_time: float) -> List[Detection]:
        """
        Match incoming detections to existing tracked pieces.

        Uses position-based matching optimized for left-to-right conveyor movement.
        Prioritizes X-coordinate proximity since pieces primarily move horizontally.

        Args:
            detections: New detections to match
            current_time: Current timestamp

        Returns:
            List of detections that couldn't be matched to existing pieces
        """
        unmatched_detections = []

        for detection in detections:
            # Try to find the best matching piece for this detection
            best_match = self._find_best_matching_piece(detection)

            if best_match:
                # Update the matched piece with new detection data
                self._update_piece_with_detection(best_match, detection, current_time)
                best_match.updated = True
                logger.debug(f"Matched detection to piece {best_match.id}")
            else:
                # No match found - detection represents a new piece
                unmatched_detections.append(detection)
                logger.debug(f"Detection at {detection.get_center()} has no match")

        return unmatched_detections

    def _find_best_matching_piece(self, detection: Detection) -> Optional[TrackedPiece]:
        """
        Find the tracked piece that best matches a detection.

        Uses weighted distance matching that prioritizes X-axis movement
        since pieces move horizontally on the conveyor.

        Args:
            detection: Detection to match

        Returns:
            Best matching TrackedPiece or None if no good match found
        """
        if not self.tracked_pieces:
            return None

        detection_center = detection.get_center()
        best_piece = None
        best_distance = float('inf')

        for piece in self.tracked_pieces:
            # Skip pieces already matched this frame
            if piece.updated:
                continue

            # Calculate weighted distance emphasizing X-axis movement
            piece_center = piece.center

            dx = abs(detection_center[0] - piece_center[0])
            dy = abs(detection_center[1] - piece_center[1])

            # Weighted distance calculation
            weighted_distance = (dx * self.match_x_weight) + (dy * self.match_y_weight)

            # Check if this is the best match so far
            if weighted_distance < best_distance and weighted_distance < self.match_distance_threshold:
                best_distance = weighted_distance
                best_piece = piece

        if best_piece:
            logger.debug(f"Best match for detection: piece {best_piece.id} "
                         f"(distance: {best_distance:.1f})")

        return best_piece

    def _update_piece_with_detection(self, piece: TrackedPiece, detection: Detection,
                                     current_time: float):
        """
        Update a tracked piece with new detection data.

        This updates position, motion tracking, and derived properties
        while maintaining the piece's identity and history.

        Args:
            piece: TrackedPiece to update
            detection: New detection data
            current_time: Current timestamp
        """
        # Store old center for motion calculation
        old_center = piece.center

        # Update basic properties from detection
        piece.bbox = detection.bbox
        piece.contour = detection.contour
        piece.last_updated = current_time
        piece.update_count += 1

        # Recalculate derived properties (center, edges, area)
        piece.update_derived_properties()

        # Update motion tracking with new position
        piece.update_motion(piece.center, current_time)

        # Update confidence based on tracking stability
        self._update_piece_confidence(piece, detection)

        logger.debug(f"Updated piece {piece.id}: center moved from "
                     f"{old_center} to {piece.center}")

    def _update_piece_confidence(self, piece: TrackedPiece, detection: Detection):
        """
        Update the confidence score for a tracked piece.

        Confidence increases with successful matches and decreases
        if the piece behaves unexpectedly (large jumps, etc.).

        Args:
            piece: TrackedPiece to update
            detection: Latest detection data
        """
        # Start with detection confidence
        confidence = detection.confidence

        # Increase confidence for stable pieces
        if piece.update_count >= self.min_updates_for_stability:
            confidence = min(1.0, confidence + 0.1)

        # Decrease confidence for large velocity changes
        if len(piece.trajectory) >= 2:
            current_velocity = np.sqrt(piece.velocity[0] ** 2 + piece.velocity[1] ** 2)
            if current_velocity > self.max_velocity_change:
                confidence = max(0.1, confidence - 0.2)

        piece.last_matched_confidence = confidence

    # ========================================================================
    # PIECE LIFECYCLE MANAGEMENT
    # ========================================================================

    def _create_new_pieces(self, detections: List[Detection], current_time: float):
        """
        Create new TrackedPiece objects for unmatched detections.

        Each unmatched detection potentially represents a new piece
        entering the tracking area.

        Args:
            detections: Unmatched detections to convert to tracked pieces
            current_time: Current timestamp
        """
        for detection in detections:
            # Create new tracked piece
            new_piece = create_tracked_piece_from_detection(
                detection=detection,
                piece_id=self.piece_id_counter,
                timestamp=current_time
            )

            # Add to tracking list
            self.tracked_pieces.append(new_piece)
            self.piece_id_counter += 1

            logger.info(f"Created new piece {new_piece.id} at {new_piece.center}")

    def _update_piece_states(self, current_time: float):
        """
        Update state flags and derived information for all tracked pieces.

        This handles transitions like pieces becoming fully visible
        and other state changes that don't depend on new detections.

        Args:
            current_time: Current timestamp
        """
        if not self._roi:
            logger.warning("Cannot update piece states: ROI not configured")
            return

        for piece in self.tracked_pieces:
            # Update fully_in_frame status
            self._update_fully_in_frame_status(piece)

            # Update velocity if piece has enough samples
            self._update_piece_velocity(piece)

    def _update_fully_in_frame_status(self, piece: TrackedPiece):
        """
        Determine if a piece is fully visible within the ROI.

        A piece is considered fully in frame when its left edge is
        sufficiently far from the left edge of the ROI.

        Args:
            piece: TrackedPiece to evaluate
        """
        if not self._roi:
            return

        # Check if piece's left edge is past the entry margin
        left_boundary = self._roi.x + self.fully_in_frame_margin
        piece_left_edge = piece.left_edge + self._roi.x  # Convert to frame coordinates

        # Update status if piece crosses the threshold
        if not piece.fully_in_frame and piece_left_edge >= left_boundary:
            piece.fully_in_frame = True
            logger.info(f"Piece {piece.id} is now fully in frame")

    def _update_piece_velocity(self, piece: TrackedPiece):
        """
        Update velocity calculation for a piece with sufficient trajectory data.

        This provides simple velocity tracking for coordination with other
        modules and basic motion understanding.

        Args:
            piece: TrackedPiece to update velocity for
        """
        # Need at least 2 trajectory points for velocity
        if len(piece.trajectory) < self.min_velocity_samples:
            return

        # Calculate average velocity over recent trajectory
        recent_points = piece.trajectory[-self.min_velocity_samples:]

        if len(recent_points) >= 2:
            # Get first and last points
            start_point = recent_points[0]
            end_point = recent_points[-1]

            # Calculate time difference
            dt = end_point[2] - start_point[2]  # timestamp difference

            if dt > 0:
                # Calculate velocity components
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]

                piece.velocity = (dx / dt, dy / dt)

    def _remove_inactive_pieces(self, current_time: float):
        """
        Remove pieces that are no longer being tracked.

        Pieces are removed if they:
        1. Haven't been detected recently (timeout)
        2. Have exited the ROI boundary (fell off conveyor)

        Args:
            current_time: Current timestamp
        """
        pieces_to_remove = []

        for piece in self.tracked_pieces:
            # Check for timeout condition
            time_since_update = current_time - piece.last_updated
            if time_since_update > self.piece_timeout_seconds:
                pieces_to_remove.append(piece)
                logger.info(f"Removing piece {piece.id}: timeout "
                            f"({time_since_update:.1f}s since last update)")
                continue

            # Check for exit condition (if ROI is configured)
            if self._roi and self._has_piece_exited_roi(piece):
                pieces_to_remove.append(piece)
                logger.info(f"Removing piece {piece.id}: exited ROI")
                continue

        # Remove flagged pieces
        for piece in pieces_to_remove:
            self.tracked_pieces.remove(piece)

        if pieces_to_remove:
            logger.debug(f"Removed {len(pieces_to_remove)} inactive pieces")

    def _has_piece_exited_roi(self, piece: TrackedPiece) -> bool:
        """
        Check if a piece has moved beyond the ROI boundary.

        Pieces exit when their right edge moves past the right edge of the ROI.

        Args:
            piece: TrackedPiece to check

        Returns:
            True if piece has exited the ROI
        """
        if not self._roi:
            return False

        # Convert piece position to frame coordinates
        piece_right_edge = piece.right_edge + self._roi.x
        roi_right_edge = self._roi.x + self._roi.width

        return piece_right_edge > roi_right_edge

    # ========================================================================
    # DATA ACCESS METHODS
    # ========================================================================

    def get_active_pieces(self) -> List[TrackedPiece]:
        """
        Get all currently tracked pieces.

        Returns:
            Copy of the tracked pieces list
        """
        return self.tracked_pieces.copy()

    def get_piece_by_id(self, piece_id: int) -> Optional[TrackedPiece]:
        """
        Get a specific tracked piece by its ID.

        Args:
            piece_id: Unique ID of the piece to find

        Returns:
            TrackedPiece with matching ID or None if not found
        """
        for piece in self.tracked_pieces:
            if piece.id == piece_id:
                return piece
        return None

    def get_fully_visible_pieces(self) -> List[TrackedPiece]:
        """
        Get only pieces that are fully visible in the frame.

        This is useful for modules that need to work with complete pieces
        rather than partially visible ones.

        Returns:
            List of TrackedPiece objects with fully_in_frame=True
        """
        return [piece for piece in self.tracked_pieces if piece.fully_in_frame]

    def get_stable_pieces(self) -> List[TrackedPiece]:
        """
        Get pieces that have been tracked long enough to be considered stable.

        Stable pieces have been successfully matched across multiple frames
        and are less likely to be noise or false detections.

        Returns:
            List of TrackedPiece objects with sufficient update count
        """
        return [piece for piece in self.tracked_pieces
                if piece.update_count >= self.min_updates_for_stability]

    def get_tracking_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current tracking state.

        This is useful for monitoring system performance and debugging.

        Returns:
            Dictionary with tracking metrics
        """
        total_pieces = len(self.tracked_pieces)
        fully_visible = len(self.get_fully_visible_pieces())
        stable_pieces = len(self.get_stable_pieces())

        # Calculate average velocity for moving pieces
        moving_pieces = [p for p in self.tracked_pieces if p.velocity != (0.0, 0.0)]
        avg_velocity = 0.0
        if moving_pieces:
            velocities = [np.sqrt(p.velocity[0] ** 2 + p.velocity[1] ** 2) for p in moving_pieces]
            avg_velocity = np.mean(velocities)

        return {
            "total_pieces": total_pieces,
            "fully_visible_pieces": fully_visible,
            "stable_pieces": stable_pieces,
            "moving_pieces": len(moving_pieces),
            "average_velocity": avg_velocity,
            "next_piece_id": self.piece_id_counter
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def reset_tracking(self):
        """
        Clear all tracked pieces and reset the tracker state.

        This is useful when restarting the system or when the conveyor
        has been cleared of all pieces.
        """
        logger.info("Resetting piece tracker - clearing all tracked pieces")
        self.tracked_pieces.clear()
        self.piece_id_counter = 0

    def update_configuration(self):
        """
        Reload configuration from the config manager.

        This allows dynamic updates to tracking parameters without
        restarting the entire system.
        """
        logger.info("Reloading tracking configuration")
        self._load_configuration()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_piece_tracker(config_manager) -> PieceTracker:
    """
    Create a PieceTracker instance with unified configuration.

    This is the standard way to create a piece tracker using the
    enhanced_config_manager system for consistent configuration management.

    Args:
        config_manager: Enhanced config manager instance (required)

    Returns:
        Configured PieceTracker instance

    Raises:
        ValueError: If config_manager is None or invalid
    """
    if config_manager is None:
        raise ValueError("create_piece_tracker requires a valid config_manager")

    logger.info("Creating PieceTracker with enhanced_config_manager")
    return PieceTracker(config_manager)


# ============================================================================
# TESTING AND UTILITIES
# ============================================================================

def create_test_detection(center_x: float, center_y: float,
                          width: int = 40, height: int = 30) -> Detection:
    """
    Create a synthetic Detection for testing purposes.

    Args:
        center_x: X coordinate of detection center
        center_y: Y coordinate of detection center
        width: Width of detection bounding box
        height: Height of detection bounding box

    Returns:
        Detection object with specified properties
    """
    # Calculate bounding box from center and size
    x = int(center_x - width / 2)
    y = int(center_y - height / 2)
    bbox = (x, y, width, height)

    # Create simple rectangular contour
    contour = np.array([
        [[x, y]],
        [[x + width, y]],
        [[x + width, y + height]],
        [[x, y + height]]
    ], dtype=np.int32)

    return Detection(
        contour=contour,
        bbox=bbox,
        area=width * height,
        confidence=0.9
    )


if __name__ == "__main__":
    """
    Test the piece tracker with synthetic data.

    This demonstrates piece creation, matching, and lifecycle management
    using simulated detections moving across the ROI.
    """
    import sys

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing PieceTracker with synthetic data")


    # Create a mock config manager for testing
    class MockConfigManager:
        """Simple mock config manager for testing purposes"""

        def get_module_config(self, module_name):
            if module_name == "detector":
                return {
                    "match_distance_threshold": 100.0,
                    "match_x_weight": 2.0,
                    "match_y_weight": 1.0,
                    "min_updates_for_stability": 3,
                    "piece_timeout_seconds": 2.0,
                    "fully_in_frame_margin": 20,
                    "min_velocity_samples": 2,
                    "max_velocity_change": 50.0
                }
            return {}


    try:
        # Create tracker with mock config
        mock_config = MockConfigManager()
        tracker = create_piece_tracker(mock_config)

        # Set up test ROI
        roi = RegionOfInterest(x=100, y=100, width=600, height=200)
        tracker.set_roi(roi)

        print(f"\nPieceTracker Test Simulation")
        print(f"ROI: {roi.to_tuple()}")
        print(f"=" * 50)

        # Simulate a piece moving across the ROI over multiple frames
        piece_start_x = 50  # Starts outside ROI (partially visible)
        piece_y = 200  # Consistent Y position
        velocity = 25  # pixels per frame

        for frame in range(15):  # Simulate 15 frames
            print(f"\nFrame {frame + 1}:")

            # Calculate piece position
            piece_x = piece_start_x + (frame * velocity)

            # Create detection for moving piece
            detections = [create_test_detection(piece_x, piece_y)]

            # Update tracking
            tracked_pieces = tracker.update_tracking(detections)

            # Report results
            stats = tracker.get_tracking_statistics()
            print(f"  Piece position: ({piece_x}, {piece_y})")
            print(f"  Tracked pieces: {stats['total_pieces']}")
            print(f"  Fully visible: {stats['fully_visible_pieces']}")
            print(f"  Stable pieces: {stats['stable_pieces']}")

            if tracked_pieces:
                piece = tracked_pieces[0]
                print(f"  Piece {piece.id}: center={piece.center}, "
                      f"velocity={piece.velocity}, updates={piece.update_count}")
                print(f"  Fully in frame: {piece.fully_in_frame}")

            # Stop if piece has exited the system
            if not tracked_pieces:
                print(f"  Piece has exited tracking area")
                break

        print(f"\nFinal statistics:")
        final_stats = tracker.get_tracking_statistics()
        for key, value in final_stats.items():
            print(f"  {key}: {value}")

        logger.info("PieceTracker test completed successfully")

    except Exception as e:
        logger.error(f"PieceTracker test failed: {e}")
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()