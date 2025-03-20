"""
detector_module.py - Module for detecting and tracking Lego pieces
"""

import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np

from error_module import DetectorError

# Set up module logger
logger = logging.getLogger(__name__)


@dataclass
class TrackedPiece:
    """Represents a piece being tracked through the ROI"""
    id: int  # Unique identifier for this piece
    contour: np.ndarray  # Current contour
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    entry_time: float  # When the piece entered tracking
    captured: bool = False  # Whether this piece has been photographed
    image_number: Optional[int] = None  # The number used when saving the image
    updated: bool = True  # Whether this piece was updated in the current frame
    last_update_time: float = None  # The time when this piece was last updated

    def __post_init__(self):
        """Initialize the last_update_time to the current time"""
        self.last_update_time = time.time()


class TrackedLegoDetector:
    """Detects and tracks Lego pieces moving through a region of interest"""

    def __init__(self, config_manager=None):
        """Initialize the detector with configuration parameters

        Args:
            config_manager: Optional configuration manager
        """
        # Set default values
        self.min_piece_area = 1000
        self.max_piece_area = 100000
        self.buffer_percent = 0.01
        self.crop_padding = 20
        self.history_length = 2
        self.grid_size = (10, 10)
        self.color_margin = 40
        self.show_grid = False
        self.track_timeout = 1.0  # Time in seconds before removing a track that hasn't been updated
        self.new_piece_grace_period = 0.5  # Minimum time a piece must be tracked before capture

        # Capture cooldown settings
        self.capture_min_interval = 3.0  # Minimum seconds between captures
        self.spatial_cooldown_radius = 150  # Pixels - don't capture if too close to a previous capture
        self.spatial_cooldown_time = 10.0  # Seconds to maintain spatial cooldown zones

        # Initialize from config if provided
        if config_manager:
            logger.info("Initializing detector from configuration")
            try:
                self.min_piece_area = config_manager.get("detector", "min_piece_area", self.min_piece_area)
                self.max_piece_area = config_manager.get("detector", "max_piece_area", self.max_piece_area)
                self.buffer_percent = config_manager.get("detector", "buffer_percent", self.buffer_percent)
                self.crop_padding = config_manager.get("detector", "crop_padding", self.crop_padding)
                self.history_length = config_manager.get("detector", "history_length", self.history_length)
                grid_size = config_manager.get("detector", "grid_size", self.grid_size)
                if isinstance(grid_size, list):
                    self.grid_size = tuple(grid_size)
                self.color_margin = config_manager.get("detector", "color_margin", self.color_margin)
                self.show_grid = config_manager.get("detector", "show_grid", self.show_grid)
                self.track_timeout = config_manager.get("detector", "track_timeout", self.track_timeout)
                self.capture_min_interval = config_manager.get("detector", "capture_min_interval",
                                                               self.capture_min_interval)
                self.spatial_cooldown_radius = config_manager.get("detector", "spatial_cooldown_radius",
                                                                  self.spatial_cooldown_radius)
                self.spatial_cooldown_time = config_manager.get("detector", "spatial_cooldown_time",
                                                                self.spatial_cooldown_time)

                logger.debug(f"Detector configuration: min_area={self.min_piece_area}, "
                             f"max_area={self.max_piece_area}, buffer={self.buffer_percent}")
            except Exception as e:
                logger.error(f"Error loading detector configuration: {str(e)}")
                raise DetectorError(f"Configuration error: {str(e)}")

        # Initialize background model
        self.background_model = None

        # Initialize tracking-related variables
        self.tracked_pieces = []
        self.next_piece_id = 1
        self.roi = None
        self.entry_zone = None
        self.exit_zone = None
        self.valid_zone = None
        self.last_capture_time = 0

        # Add spatial cooldown tracking
        self.captured_locations = []  # List of (center_x, center_y, capture_time)

        logger.info("Detector initialized successfully")

    def calibrate(self, frame):
        """Get user-selected ROI and calculate buffer zones

        Args:
            frame: The video frame to use for calibration

        Returns:
            tuple: The selected ROI as (x, y, w, h)

        Raises:
            DetectorError: If calibration fails
        """
        if frame is None:
            error_msg = "Cannot calibrate with empty frame"
            logger.error(error_msg)
            raise DetectorError(error_msg)

        try:
            # Get ROI selection from user
            logger.info("Waiting for user to select ROI")
            roi = cv2.selectROI("Select Belt Region", frame, False)
            cv2.destroyWindow("Select Belt Region")

            # Validate ROI
            if roi[2] <= 0 or roi[3] <= 0:
                error_msg = "Invalid ROI selected (zero width or height)"
                logger.error(error_msg)
                raise DetectorError(error_msg)

            self.roi = roi
            roi_width = roi[2]

            # Calculate buffer zones
            buffer_width = int(roi_width * self.buffer_percent)
            self.entry_zone = (roi[0], roi[0] + buffer_width)
            self.exit_zone = (roi[0] + roi[2] - buffer_width, roi[0] + roi[2])
            self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

            logger.info(f"ROI selected: {roi}")
            logger.debug(f"Entry buffer: {self.entry_zone}")
            logger.debug(f"Valid zone: {self.valid_zone}")
            logger.debug(f"Exit buffer: {self.exit_zone}")

            # Initialize background model
            x, y, w, h = self.roi
            roi_frame = frame[y:y + h, x:x + w]
            self._initialize_background_model(roi_frame)

            return roi

        except Exception as e:
            if isinstance(e, DetectorError):
                raise
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise DetectorError(error_msg)

    def _initialize_background_model(self, roi_frame):
        """Initialize the background model by computing average colors for each grid cell

        Args:
            roi_frame: The ROI portion of the frame

        Raises:
            DetectorError: If background model initialization fails
        """
        try:
            h, w = roi_frame.shape[:2]
            grid_h, grid_w = self.grid_size

            # Create storage for grid cell averages
            self.background_model = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

            # Calculate average color for each grid cell
            for i in range(grid_h):
                for j in range(grid_w):
                    cell_y1 = i * (h // grid_h)
                    cell_y2 = (i + 1) * (h // grid_h) if i < grid_h - 1 else h
                    cell_x1 = j * (w // grid_w)
                    cell_x2 = (j + 1) * (w // grid_w) if j < grid_w - 1 else w

                    cell = roi_frame[cell_y1:cell_y2, cell_x1:cell_x2]
                    self.background_model[i, j] = np.mean(cell, axis=(0, 1))

            logger.info(f"Background model initialized with {grid_h}x{grid_w} grid")

        except Exception as e:
            error_msg = f"Failed to initialize background model: {str(e)}"
            logger.error(error_msg)
            raise DetectorError(error_msg)

    def _preprocess_frame(self, frame):
        """Extract and preprocess the ROI using color-based detection with efficient NumPy operations

        Args:
            frame: Input frame to process

        Returns:
            numpy.ndarray: Binary mask of detected objects

        Raises:
            DetectorError: If ROI is not configured or preprocessing fails
        """
        if self.roi is None:
            error_msg = "Detector not calibrated. Call calibrate() first."
            logger.error(error_msg)
            raise DetectorError(error_msg)

        try:
            # Extract ROI
            x, y, w, h = self.roi
            roi = frame[y:y + h, x:x + w]

            # Create grid
            grid_h, grid_w = self.grid_size

            # Initialize the mask
            mask = np.zeros((h, w), dtype=np.uint8)

            # Process each grid cell with vectorized operations
            for i in range(grid_h):
                for j in range(grid_w):
                    # Get cell coordinates
                    cell_y1 = i * (h // grid_h)
                    cell_y2 = (i + 1) * (h // grid_h) if i < grid_h - 1 else h
                    cell_x1 = j * (w // grid_w)
                    cell_x2 = (j + 1) * (w // grid_w) if j < grid_w - 1 else w

                    # Extract cell
                    cell = roi[cell_y1:cell_y2, cell_x1:cell_x2]

                    # Get background model for this cell
                    avg_color = self.background_model[i, j]

                    # Calculate color difference for all pixels in the cell at once
                    diff = np.abs(cell - avg_color)

                    # Find pixels where any channel differs by more than color_margin
                    color_mask = np.max(diff, axis=2) > self.color_margin

                    # Copy to the main mask
                    mask[cell_y1:cell_y2, cell_x1:cell_x2] = color_mask.astype(np.uint8) * 255

            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            error_msg = f"Frame preprocessing failed: {str(e)}"
            logger.error(error_msg)
            raise DetectorError(error_msg)

    def _find_new_contours(self, mask):
        """Find contours in the current frame

        Args:
            mask: The binary mask to find contours in

        Returns:
            list: List of (contour, bounding_box) tuples
        """
        try:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            valid_contours = []
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < self.min_piece_area or area > self.max_piece_area:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                if self.roi[0] <= x < self.roi[0] + self.roi[2]:  # Entire ROI
                    valid_contours.append((contour, (x, y, w, h)))

            logger.debug(f"Found {len(valid_contours)} valid contours")
            return valid_contours

        except Exception as e:
            error_msg = f"Contour detection failed: {str(e)}"
            logger.error(error_msg)
            return []  # Return empty list instead of raising to allow recovery

    def _match_and_update_tracks(self, new_contours):
        """Match new contours to existing tracks or create new tracks"""
        try:
            # First, mark all existing tracks as not updated
            for piece in self.tracked_pieces:
                piece.updated = False

            # Try to match new contours to existing tracks
            matched_pieces = set()  # Keep track of which pieces have been matched

            for contour, bbox in new_contours:
                x, y, w, h = bbox

                # Skip pieces that haven't entered the ROI yet
                if x < self.entry_zone[0]:
                    continue

                # Find the best match for this contour
                best_match = None
                best_distance = float('inf')

                for i, piece in enumerate(self.tracked_pieces):
                    if i in matched_pieces:  # Skip pieces that are already matched
                        continue

                    old_x, old_y, old_w, old_h = piece.bounding_box

                    # Calculate center points
                    new_center_x = x + w / 2
                    new_center_y = y + h / 2
                    old_center_x = old_x + old_w / 2
                    old_center_y = old_y + old_h / 2

                    # Calculate distance between centers
                    distance = ((new_center_x - old_center_x) ** 2 +
                                (new_center_y - old_center_y) ** 2) ** 0.5

                    # Track the best match (lowest distance)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = (i, piece)

                # Check if the best match is within our threshold
                max_allowed_distance = max(w, h) * 0.5  # Adjust threshold as needed

                if best_match is not None and best_distance < max_allowed_distance:
                    index, piece = best_match
                    piece.contour = contour
                    piece.bounding_box = bbox
                    piece.updated = True
                    piece.last_update_time = time.time()
                    matched_pieces.add(index)
                else:
                    # If no match found, create a new track
                    new_piece = TrackedPiece(
                        id=self.next_piece_id,
                        contour=contour,
                        bounding_box=bbox,
                        entry_time=time.time()
                    )
                    self.tracked_pieces.append(new_piece)
                    logger.debug(f"Created new track ID: {self.next_piece_id}")
                    self.next_piece_id += 1

            # Remove stale tracks that haven't been updated recently
            current_time = time.time()
            updated_tracks = []

            for piece in self.tracked_pieces:
                # Keep track if:
                # 1. It was just updated
                # 2. It hasn't timed out yet
                # 3. It has been captured (we want to keep it visible a bit longer)
                if (piece.updated or
                        (current_time - piece.last_update_time < self.track_timeout) or
                        (piece.captured and current_time - piece.last_update_time < self.track_timeout * 2)):
                    updated_tracks.append(piece)
                else:
                    logger.debug(
                        f"Removing stale track ID: {piece.id}, Last update: {current_time - piece.last_update_time:.2f}s ago")

            self.tracked_pieces = updated_tracks

        except Exception as e:
            error_msg = f"Error updating tracks: {str(e)}"
            logger.error(error_msg)
            # Continue execution despite errors

    def _clean_captured_locations(self):
        """Remove expired spatial cooldown zones"""
        try:
            current_time = time.time()
            expired_threshold = current_time - self.spatial_cooldown_time
            original_count = len(self.captured_locations)
            self.captured_locations = [(x, y, t) for x, y, t in self.captured_locations if t > expired_threshold]
            if original_count != len(self.captured_locations):
                logger.debug(f"Cleaned {original_count - len(self.captured_locations)} expired capture locations")
        except Exception as e:
            logger.error(f"Error cleaning captured locations: {str(e)}")
            # Continue execution despite errors

    def _should_capture(self, piece: TrackedPiece) -> bool:
        """Determine if the piece should be captured."""
        # Skip if already marked as captured
        if piece.captured:
            return False

        try:
            current_time = time.time()

            # Don't capture if it's too soon after the last capture
            if current_time - self.last_capture_time < self.capture_min_interval:
                return False

            # Don't capture pieces that just appeared (avoid post-stutter duplicates)
            if current_time - piece.entry_time < self.new_piece_grace_period:
                return False

            # Get piece position
            x, y, w, h = piece.bounding_box
            center_x = x + w / 2
            center_y = y + h / 2

            # Only capture in the valid zone
            if not (self.valid_zone[0] <= x and x + w <= self.valid_zone[1]):
                return False

            # Check if we're too close to a previously captured location
            for loc_x, loc_y, _ in self.captured_locations:
                distance = ((center_x - loc_x) ** 2 + (center_y - loc_y) ** 2) ** 0.5
                if distance < self.spatial_cooldown_radius:
                    logger.debug(
                        f"Skipping piece ID {piece.id} - too close to previous capture location ({distance:.1f}px)")
                    piece.captured = True  # Mark as captured to avoid repeated checks
                    return False

            # If all checks passed, this piece should be captured
            return True

        except Exception as e:
            logger.error(f"Error in capture decision: {str(e)}")
            return False  # In case of error, don't capture

    def _crop_piece_image(self, frame, piece: TrackedPiece) -> np.ndarray:
        """Crop the frame to just the piece with padding

        Args:
            frame: The video frame to crop
            piece: The TrackedPiece to crop around

        Returns:
            np.ndarray: The cropped image
        """
        try:
            x, y, w, h = piece.bounding_box
            roi_x, roi_y = self.roi[0], self.roi[1]

            # Add padding and ensure within frame bounds
            pad = self.crop_padding
            x1 = max(0, x + roi_x - pad)
            y1 = max(0, y + roi_y - pad)
            x2 = min(frame.shape[1], x + roi_x + w + pad)
            y2 = min(frame.shape[0], y + roi_y + h + pad)

            return frame[y1:y2, x1:x2]

        except Exception as e:
            error_msg = f"Error cropping piece image: {str(e)}"
            logger.error(error_msg)
            raise DetectorError(error_msg)

    def process_frame(self, frame, current_count=None) -> Tuple[List[TrackedPiece], Optional[np.ndarray]]:
        """Process a frame and return any pieces to capture.

        Args:
            frame: The video frame to process
            current_count: The current image count (optional)

        Returns:
            tuple: (tracked_pieces, cropped_image)
                - tracked_pieces: List of all currently tracked pieces
                - cropped_image: Cropped image of a piece to capture, or None

        Raises:
            DetectorError: If the detector is not calibrated
        """
        if self.roi is None:
            error_msg = "Detector not calibrated. Call calibrate() first."
            logger.error(error_msg)
            raise DetectorError(error_msg)

        try:
            # Clean up expired capture locations
            self._clean_captured_locations()

            mask = self._preprocess_frame(frame)
            new_contours = self._find_new_contours(mask)

            # Update tracking
            self._match_and_update_tracks(new_contours)

            # Check for pieces to capture
            for piece in self.tracked_pieces:
                if self._should_capture(piece):
                    # Get piece position for spatial cooldown
                    x, y, w, h = piece.bounding_box
                    center_x = x + w / 2
                    center_y = y + h / 2

                    # Crop the image
                    cropped_image = self._crop_piece_image(frame, piece)

                    # Mark as captured and update tracking
                    piece.captured = True
                    piece.image_number = current_count
                    self.last_capture_time = time.time()

                    # Register capture location
                    self.captured_locations.append((center_x, center_y, time.time()))

                    logger.info(f"Capturing piece ID {piece.id} at position ({center_x:.1f}, {center_y:.1f})")
                    return self.tracked_pieces, cropped_image

            return self.tracked_pieces, None

        except Exception as e:
            if isinstance(e, DetectorError):
                raise
            error_msg = f"Error processing frame: {str(e)}"
            logger.error(error_msg)
            raise DetectorError(error_msg)

    def draw_debug(self, frame):
        """Draw debug visualization with piece tracking, capture status, and grid overlay.

        Args:
            frame: The video frame to draw on

        Returns:
            np.ndarray: The debug frame with visualization
        """
        if self.roi is None:
            return frame

        try:
            debug_frame = frame.copy()

            # Draw ROI - Green boundary
            x, y, w, h = self.roi
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw entry zone - Blue
            cv2.line(debug_frame,
                     (self.entry_zone[1], y),
                     (self.entry_zone[1], y + h),
                     (255, 0, 0), 2)

            # Draw exit zone - Red
            cv2.line(debug_frame,
                     (self.exit_zone[0], y),
                     (self.exit_zone[0], y + h),
                     (0, 0, 255), 2)

            # Draw grid if enabled
            if self.show_grid:
                grid_h, grid_w = self.grid_size
                cell_h, cell_w = h // grid_h, w // grid_w

                # Draw horizontal grid lines
                for i in range(1, grid_h):
                    y_pos = y + i * cell_h
                    cv2.line(debug_frame, (x, y_pos), (x + w, y_pos), (200, 200, 200), 1)

                # Draw vertical grid lines
                for j in range(1, grid_w):
                    x_pos = x + j * cell_w
                    cv2.line(debug_frame, (x_pos, y), (x_pos, y + h), (200, 200, 200), 1)

            # Draw spatial cooldown zones
            for loc_x, loc_y, t in self.captured_locations:
                # Calculate age as percentage of total cooldown time
                current_time = time.time()
                age_pct = min(1.0, (current_time - t) / self.spatial_cooldown_time)

                # Color fades from red to transparent as it ages
                color = (0, 0, 255)

                # Draw circle showing the cooldown zone
                cv2.circle(debug_frame,
                           (int(loc_x + self.roi[0]), int(loc_y + self.roi[1])),
                           int(self.spatial_cooldown_radius),
                           color,
                           1)  # Thickness

            # Get current time for age calculation
            current_time = time.time()

            for piece in self.tracked_pieces:
                x, y, w, h = piece.bounding_box
                roi_x, roi_y = self.roi[0], self.roi[1]

                # Adjust coordinates to the full frame
                x += roi_x
                y += roi_y

                # Determine box color based on captured status and update freshness
                age = current_time - piece.last_update_time

                if piece.captured:
                    # Green for captured pieces
                    box_color = (0, 255, 0)
                elif not piece.updated and age > 0.2:
                    # Yellow for pieces that haven't been updated recently but aren't stale
                    freshness = max(0, 1 - (age / self.track_timeout))
                    box_color = (0, 255 * freshness, 255)  # Blend from cyan to blue as track ages
                else:
                    # Red for active, uncaptured pieces
                    box_color = (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), box_color, 2)

                # Add label
                if piece.captured:
                    label_text = f"#{piece.image_number}"
                else:
                    # Show ID and age for debugging
                    label_text = f"ID:{piece.id} ({age:.1f}s)"

                label_color = box_color
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(debug_frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 0, 0), -1)
                cv2.putText(debug_frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)

            return debug_frame

        except Exception as e:
            logger.error(f"Error drawing debug visualization: {str(e)}")
            return frame  # Return original frame if debug drawing fails

    def release(self):
        """Release any resources used by the detector"""
        logger.info("Releasing detector resources")
        # Currently no resources to release, but adding for API consistency
        self.tracked_pieces = []
        self.captured_locations = []


# Factory function to create a detector
def create_detector(detector_type="tracked", config_manager=None):
    """Create a detector of the specified type

    Args:
        detector_type: Type of detector ("tracked" or future types)
        config_manager: Configuration manager

    Returns:
        Detector instance

    Raises:
        DetectorError: If creation fails or unsupported type is specified
    """
    logger.info(f"Creating detector of type: {detector_type}")

    try:
        if detector_type == "tracked":
            return TrackedLegoDetector(config_manager)
        else:
            error_msg = f"Unsupported detector type: {detector_type}. Currently only 'tracked' is supported."
            logger.error(error_msg)
            raise DetectorError(error_msg)
    except Exception as e:
        if isinstance(e, DetectorError):
            raise
        error_msg = f"Failed to create detector: {str(e)}"
        logger.error(error_msg)
        raise DetectorError(error_msg)

