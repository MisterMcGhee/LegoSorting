"""
detector_data_models.py - Core data structures for the Lego piece detection and tracking system

This module defines the fundamental data classes used throughout the detection pipeline:
- TrackedPiece: Represents a physical object moving on the conveyor belt
- Detection: Raw computer vision results from frame processing
- IdentifiedPiece: Complete identification and categorization results from API processing

These data structures are designed to be simple, focused containers that separate
the concerns of tracking (physical movement) from identification (what the piece is).
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import cv2

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================
# These define standard values used throughout the system

class PieceStatus(Enum):
    """
    Represents the current processing state of a tracked piece.
    This helps other modules know what operations are valid on a piece.
    """
    DETECTED = "detected"  # Piece found but not yet captured
    PROCESSING = "processing"  # Image captured, being analyzed by API
    IDENTIFIED = "identified"  # Analysis complete, results available
    SORTED = "sorted"  # Piece has been physically sorted into bin


# ============================================================================
# RAW DETECTION DATA
# ============================================================================
# This represents what the computer vision system found in a single frame

@dataclass
class Detection:
    """
    Raw detection result from computer vision processing.

    This represents a single moving object found in one frame, before any
    tracking or identification has occurred. The vision processor creates
    these from background subtraction and contour detection.
    """
    # Shape and position information
    contour: np.ndarray  # Exact outline of the detected object
    bbox: Tuple[int, int, int, int]  # Bounding box (x, y, width, height)
    area: float  # Area in pixels

    # Quality metrics
    confidence: float = 1.0  # How confident we are this is a real piece (0-1)

    def get_center(self) -> Tuple[float, float]:
        """
        Calculate the center point of this detection.

        Returns:
            Tuple of (x, y) coordinates for the center
        """
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2

    def get_right_edge(self) -> float:
        """Get the rightmost x-coordinate of this detection."""
        x, y, w, h = self.bbox
        return x + w

    def get_left_edge(self) -> float:
        """Get the leftmost x-coordinate of this detection."""
        x, y, w, h = self.bbox
        return x


# ============================================================================
# TRACKED PIECE DATA
# ============================================================================
# This represents a piece being followed across multiple frames

@dataclass
class TrackedPiece:
    """
    Represents a single Lego piece being tracked as it moves on the conveyor belt.

    This class tracks the physical properties and movement of a piece over time.
    It includes position, velocity, zone status, and processing state, but does NOT
    include identification results (what type of piece it is) - that's separate.

    The piece goes through this lifecycle:
    1. First detected in entry zone
    2. Moves to valid zone where it can be captured
    3. Image captured and sent for identification
    4. Moves to exit zone where sorting is triggered
    """

    # ========================================================================
    # BASIC IDENTIFICATION
    # ========================================================================
    # Core properties that identify this unique piece

    id: int  # Unique identifier for this specific piece
    bbox: Tuple[int, int, int, int]  # Current bounding box (x, y, width, height)
    contour: np.ndarray  # Current shape outline from computer vision

    # ========================================================================
    # TIMING INFORMATION
    # ========================================================================
    # When events occurred for this piece

    first_detected: float  # Timestamp when piece first appeared
    last_updated: float  # Timestamp when piece was last seen

    # ========================================================================
    # PROCESSING STATUS
    # ========================================================================
    # Flags that control whether this piece should be captured or processed

    captured: bool = False  # Has image been saved and sent for analysis?
    being_processed: bool = False  # Is API currently analyzing this piece?
    processing_start_time: Optional[float] = None  # When did API processing begin?

    # ========================================================================
    # PHYSICAL STATUS
    # ========================================================================
    # Information about the piece's position and visibility

    fully_in_frame: bool = False  # Is entire piece visible (not cut off)?
    in_entry_zone: bool = False  # The center of the piece currently in which zone?
    in_valid_zone: bool = False
    in_exit_zone: bool = False
    exit_zone_entry_time: Optional[float] = None  # When did piece enter exit zone?

    has_exited_roi: bool = False  # Piece moved past right edge of ROI
    exit_timestamp: Optional[float] = None  # When piece exited ROI

    # ========================================================================
    # TRACKING QUALITY METRICS
    # ========================================================================
    # Information about how well we're tracking this piece

    update_count: int = 1  # How many frames has this piece appeared in?
    last_matched_confidence: float = 1.0  # How confident we are in the last match

    # ========================================================================
    # MOTION TRACKING
    # ========================================================================
    # Data for predicting where the piece will be in future frames

    velocity: Tuple[float, float] = (0.0, 0.0)  # Speed in pixels/second (x, y)
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # History of (x, y, timestamp)
    predicted_exit_time: Optional[float] = None  # When will piece fall off conveyor?

    # ========================================================================
    # CACHED CALCULATIONS
    # ========================================================================
    # Pre-calculated values to avoid repeated computation

    center: Tuple[float, float] = field(init=False)  # Center point of piece
    right_edge: float = field(init=False)  # Rightmost x-coordinate
    left_edge: float = field(init=False)  # Leftmost x-coordinate
    area: float = field(init=False)  # Area of bounding box

    def __post_init__(self):
        """
        Calculate derived properties after the piece is created.

        This automatically runs when a TrackedPiece is instantiated to set up
        all the calculated fields based on the bounding box.
        """
        self.update_derived_properties()

        # Initialize trajectory with first position
        if not self.trajectory:
            self.trajectory = [(self.center[0], self.center[1], self.first_detected)]

    def update_derived_properties(self):
        """
        Recalculate all values derived from the bounding box.

        This centralizes coordinate calculations, so we compute them once
        instead of repeatedly. Call this whenever the bbox changes.
        """
        x, y, w, h = self.bbox

        # Calculate center point once and cache it
        self.center = (x + w / 2, y + h / 2)

        # Cache edge positions for quick zone checking
        self.right_edge = x + w
        self.left_edge = x

        # Cache area for quick access
        self.area = w * h

    def update_motion(self, new_center: Tuple[float, float], timestamp: float):
        """
        Update velocity and trajectory based on new position.

        This tracks how fast and in what direction the piece is moving,
        which allows us to predict where it will be in future frames.
        This is crucial for maintaining tracking when pieces move quickly.

        Args:
            new_center: Current center position (x, y)
            timestamp: Current time in seconds
        """
        # Calculate velocity if we have previous positions
        if len(self.trajectory) > 0:
            last_x, last_y, last_time = self.trajectory[-1]
            dt = timestamp - last_time

            # Only update velocity if enough time has passed (avoid divide by zero)
            if dt > 0.001:  # At least 1 millisecond
                # Velocity = distance / time for each axis
                self.velocity = (
                    (new_center[0] - last_x) / dt,  # Horizontal speed (pixels/second)
                    (new_center[1] - last_y) / dt  # Vertical speed (pixels/second)
                )

        # Add current position to trajectory history
        self.trajectory.append((new_center[0], new_center[1], timestamp))

        # Keep only recent history (last 10 positions) to save memory
        # Older positions aren't useful for current velocity calculations
        if len(self.trajectory) > 10:
            self.trajectory.pop(0)

    def predict_position(self, future_time: float) -> Tuple[float, float]:
        """
        Predict where this piece will be at a future time.

        Uses the piece's current velocity to extrapolate its position.
        This helps with matching pieces between frames and predicting
        when they'll reach the exit zone for sorting.

        Args:
            future_time: Time to predict position for (in seconds)

        Returns:
            Predicted (x, y) position
        """
        if self.trajectory and len(self.trajectory) > 0:
            # Get most recent known position
            last_x, last_y, last_time = self.trajectory[-1]

            # Calculate how much time into the future we're predicting
            dt = future_time - last_time

            # Extrapolate position using current velocity
            # New position = old position + velocity * time
            predicted_x = last_x + self.velocity[0] * dt
            predicted_y = last_y + self.velocity[1] * dt

            return predicted_x, predicted_y

        # If no trajectory data, return current center as best guess
        return self.center

    def get_status(self) -> PieceStatus:
        """
        Get the current processing status of this piece.

        Returns:
            PieceStatus enum indicating current state
        """
        if self.being_processed:
            return PieceStatus.PROCESSING
        elif self.captured:
            return PieceStatus.IDENTIFIED  # Captured means processing is complete
        else:
            return PieceStatus.DETECTED

    def is_ready_for_capture(self, min_updates: int = 5) -> bool:
        """
        Check if this piece is ready to have its image captured.

        A piece is ready for capture when:
        - It's fully visible in the frame
        - It's been tracked for enough frames to be stable
        - It's not already being processed

        Args:
            min_updates: Minimum number of frames piece must be tracked

        Returns:
            True if piece can be captured now
        """
        return (
                self.fully_in_frame and
                self.update_count >= min_updates and
                not self.captured and
                not self.being_processed
        )

    def time_in_exit_zone(self) -> Optional[float]:
        """
        Get how long this piece has been in the exit zone.

        Returns:
            Seconds in exit zone, or None if not in exit zone
        """
        if self.in_exit_zone and self.exit_zone_entry_time:
            return time.time() - self.exit_zone_entry_time
        return None


# ============================================================================
# CAPTURE PACKAGE DATA STRUCTURE
# ============================================================================

@dataclass
class CapturePackage:
    """
    Complete package of captured and processed piece image ready for identification.

    This contains everything needed to identify a piece:
    - The processed image with piece ID overlay
    - Piece ID for tracking through the identification pipeline
    - Capture metadata for logging and debugging
    """
    piece_id: int  # Unique ID of the captured piece
    processed_image: np.ndarray  # Cropped and labeled image ready for API
    capture_timestamp: float  # When the capture occurred
    capture_position: Tuple[int, int]  # Frame coordinates where piece was captured
    original_bbox: Tuple[int, int, int, int]  # Original bounding box in frame coordinates
    image_path: Optional[str] = None  # File location for passing to the processing coordinator

    def save_image(self, file_path: str) -> bool:
        """
        Save the processed image to disk as JPG.

        Args:
            file_path: Path where image should be saved

        Returns:
            True if save was successful, False otherwise
        """
        try:
            success = cv2.imwrite(file_path, self.processed_image)
            if success:
                logger.debug(f"Saved capture image for piece {self.piece_id} to {file_path}")
            else:
                logger.error(f"Failed to save image for piece {self.piece_id} to {file_path}")
            return success
        except Exception as e:
            logger.error(f"Error saving image for piece {self.piece_id}: {e}")
            return False


# ============================================================================
# COORDINATE AND REGION UTILITIES
# ============================================================================
# Helper classes for managing spatial information

@dataclass
class RegionOfInterest:
    """
    Defines a rectangular region where piece detection occurs.

    This represents the area of the camera view where we look for pieces.
    All coordinates are in pixel units relative to the full camera frame.
    """
    x: int  # Left edge of region
    y: int  # Top edge of region
    width: int  # Width of region
    height: int  # Height of region

    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is inside this region.

        Args:
            x, y: Point coordinates to check

        Returns:
            True if point is inside region
        """
        return (
                self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height
        )

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, width, height) tuple for OpenCV functions."""
        return self.x, self.y, self.width, self.height


def create_tracked_piece_from_detection(detection: Detection, piece_id: int,
                                        timestamp: float) -> TrackedPiece:
    """
    Create a TrackedPiece from a Detection result.

    This is the standard way to promote a raw detection into a tracked piece
    when we determine it represents a new object to follow.

    Args:
        detection: Raw detection from computer vision
        piece_id: Unique ID to assign to new piece
        timestamp: When this piece was first detected

    Returns:
        New TrackedPiece ready for tracking
    """
    return TrackedPiece(
        id=piece_id,
        bbox=detection.bbox,
        contour=detection.contour,
        first_detected=timestamp,
        last_updated=timestamp,
        fully_in_frame=True  # Assume detection only happens for fully visible pieces
    )
