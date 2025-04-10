import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
import threading


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
    # New fields for threading support
    being_processed: bool = False  # Whether this piece is being processed by worker thread
    processing_start_time: Optional[float] = None  # When processing started
    # New field to store the scaling factor used for this piece
    scale_factor: Tuple[float, float] = (1.0, 1.0)  # Scale factor (fx, fy) to convert between resolutions

    def __post_init__(self):
        """Initialize the last_update_time to the current time"""
        self.last_update_time = time.time()


class TrackedLegoDetector:
    """Detects and tracks Lego pieces moving through a region of interest"""

    def __init__(self, config_manager=None, thread_manager=None):
        """Initialize the detector with configuration parameters

        Args:
            config_manager: Optional configuration manager
            thread_manager: Optional thread manager for asynchronous operation
        """
        # Store the config manager for later use
        self.config_manager = config_manager

        # Set default values
        self.min_piece_area = 1000
        self.max_piece_area = 100000
        self.buffer_percent = 0.01
        self.crop_padding = 20
        self.history_length = 2
        self.grid_size = (5, 5)  # Changed from (10, 10) to (5, 5)
        self.color_margin = 40
        self.show_grid = False
        self.track_timeout = 1.0  # Time in seconds before removing a track that hasn't been updated

        # Capture cooldown settings
        self.capture_min_interval = 3.0  # Minimum seconds between captures
        self.spatial_cooldown_radius = 150  # Pixels - don't capture if too close to a previous capture
        self.spatial_cooldown_time = 10.0  # Seconds to maintain spatial cooldown zones

        # New parameters for dual resolution processing
        self.detection_scale_factor = 0.5  # Process at half resolution for detection
        self.original_roi = None  # Will store the ROI in original resolution
        self.detection_roi = None  # Will store the ROI in detection resolution

        # Initialize from config if provided
        if config_manager:
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

            # Adjust min and max piece area for the detection resolution
            self.detection_min_piece_area = int(
                self.min_piece_area * self.detection_scale_factor * self.detection_scale_factor)
            self.detection_max_piece_area = int(
                self.max_piece_area * self.detection_scale_factor * self.detection_scale_factor)

            # Check if threading is enabled
            if config_manager.is_threading_enabled():
                self.processing_timeout = config_manager.get("threading", "processing_timeout", 60.0)
            else:
                self.processing_timeout = 60.0  # Default timeout

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

        # Thread-safety
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.thread_manager = thread_manager

        # Asynchronous mode flag
        self.async_mode = thread_manager is not None

    def load_roi_from_config(self, frame, config_manager=None):
        """Load ROI from configuration if available, otherwise calibrate manually.

        Args:
            frame: The video frame to use for calibration if needed
            config_manager: Configuration manager with ROI settings (optional)

        Returns:
            tuple: The selected/loaded ROI as (x, y, w, h)
        """
        if config_manager is None:
            config_manager = self.config_manager

        if config_manager:
            # Check if we have ROI saved in configuration
            roi_config = config_manager.get_section("detector_roi")

            if roi_config and "x" in roi_config and "y" in roi_config and "w" in roi_config and "h" in roi_config:
                # ROI exists in config, use it
                roi = (roi_config["x"], roi_config["y"], roi_config["w"], roi_config["h"])
                print(f"Loading ROI from config: {roi}")

                with self.lock:
                    # Store original resolution ROI
                    self.original_roi = roi

                    # Create detection resolution ROI
                    detection_x = int(roi[0] * self.detection_scale_factor)
                    detection_y = int(roi[1] * self.detection_scale_factor)
                    detection_w = int(roi[2] * self.detection_scale_factor)
                    detection_h = int(roi[3] * self.detection_scale_factor)
                    self.detection_roi = (detection_x, detection_y, detection_w, detection_h)

                    # Use detection ROI for internal processing
                    self.roi = self.detection_roi
                    roi_width = self.roi[2]

                    # Calculate buffer zones
                    buffer_width = int(roi_width * self.buffer_percent)
                    self.entry_zone = (self.roi[0], self.roi[0] + buffer_width)
                    self.exit_zone = (self.roi[0] + self.roi[2] - buffer_width, self.roi[0] + self.roi[2])
                    self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

                    print(f"Original ROI: {self.original_roi}")
                    print(f"Detection ROI: {self.detection_roi}")
                    print(f"Entry buffer: {self.entry_zone}")
                    print(f"Valid zone: {self.valid_zone}")
                    print(f"Exit buffer: {self.exit_zone}")

                    # Initialize background model with detection resolution
                    x, y, w, h = self.roi
                    # Create a downscaled version of the frame for detection
                    detection_frame = cv2.resize(frame, (0, 0), fx=self.detection_scale_factor,
                                                 fy=self.detection_scale_factor)
                    roi_frame = detection_frame[y:y + h, x:x + w]
                    self._initialize_background_model(roi_frame)

                return roi

        # If we get here, either no config manager or no ROI in config
        # Fall back to manual calibration
        print("No ROI configuration found. Please select ROI manually.")
        return self.calibrate(frame)

    def calibrate(self, frame):
        """Get user-selected ROI and calculate buffer zones

        Args:
            frame: The video frame to use for calibration

        Returns:
            tuple: The selected ROI as (x, y, w, h)
        """
        # Get ROI selection from user
        print("\nSelect belt region by dragging a rectangle with your mouse.")
        print("Press SPACE or ENTER to confirm selection")
        print("Press ESC to exit")
        roi = cv2.selectROI("Select Belt Region", frame, False)
        cv2.destroyWindow("Select Belt Region")

        with self.lock:
            # Store original resolution ROI
            self.original_roi = roi

            # Create detection resolution ROI
            detection_x = int(roi[0] * self.detection_scale_factor)
            detection_y = int(roi[1] * self.detection_scale_factor)
            detection_w = int(roi[2] * self.detection_scale_factor)
            detection_h = int(roi[3] * self.detection_scale_factor)
            self.detection_roi = (detection_x, detection_y, detection_w, detection_h)

            # Use detection ROI for internal processing
            self.roi = self.detection_roi
            roi_width = self.roi[2]

            # Calculate buffer zones
            buffer_width = int(roi_width * self.buffer_percent)
            self.entry_zone = (self.roi[0], self.roi[0] + buffer_width)
            self.exit_zone = (self.roi[0] + self.roi[2] - buffer_width, self.roi[0] + self.roi[2])
            self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

            print(f"Original ROI: {self.original_roi}")
            print(f"Detection ROI: {self.detection_roi}")
            print(f"Entry buffer: {self.entry_zone}")
            print(f"Valid zone: {self.valid_zone}")
            print(f"Exit buffer: {self.exit_zone}")

            # Initialize background model with detection resolution
            x, y, w, h = self.roi
            # Create a downscaled version of the frame for detection
            detection_frame = cv2.resize(frame, (0, 0), fx=self.detection_scale_factor, fy=self.detection_scale_factor)
            roi_frame = detection_frame[y:y + h, x:x + w]
            self._initialize_background_model(roi_frame)

            # Save ROI to config
            if hasattr(self, 'config_manager') and self.config_manager:
                roi_config = {
                    "x": roi[0],
                    "y": roi[1],
                    "w": roi[2],
                    "h": roi[3]
                }
                self.config_manager.update_section("detector_roi", roi_config)
                self.config_manager.save_config()
                print("ROI saved to configuration")

        return roi  # Return the original ROI for display purposes

    def _initialize_background_model(self, roi_frame):
        """Initialize the background model by computing average colors for each grid cell

        Args:
            roi_frame: The ROI portion of the frame
        """
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

        print(f"Background model initialized with {grid_h}x{grid_w} grid")

    def _preprocess_frame(self, frame):
        """Extract and preprocess the ROI using color-based detection with efficient NumPy operations"""
        if self.roi is None:
            raise ValueError("Detector not calibrated. Call calibrate() first.")

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

    def _find_new_contours(self, mask):
        """Find contours in the current frame

        Args:
            mask: The binary mask to find contours in

        Returns:
            list: List of (contour, bounding_box) tuples
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = []
        for contour in contours:
            # Filter by area using detection resolution area thresholds
            area = cv2.contourArea(contour)
            if area < self.detection_min_piece_area or area > self.detection_max_piece_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            if self.roi[0] <= x < self.roi[0] + self.roi[2]:  # Entire ROI
                valid_contours.append((contour, (x, y, w, h)))

        return valid_contours

    def _match_and_update_tracks(self, new_contours):
        """Match new contours to existing tracks or create new tracks"""
        with self.lock:
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
                    if i in matched_pieces:  # Skip pieces that are already matched to prevent duplicates
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
                        entry_time=time.time(),
                        scale_factor=(1.0 / self.detection_scale_factor, 1.0 / self.detection_scale_factor)
                        # Store scale factor
                    )
                    self.tracked_pieces.append(new_piece)
                    self.next_piece_id += 1

            # Check for timed out pieces being processed
            current_time = time.time()
            for piece in self.tracked_pieces:
                if (piece.being_processed and
                        piece.processing_start_time is not None and
                        current_time - piece.processing_start_time > self.processing_timeout):
                    # Reset processing status if timeout exceeded
                    print(f"Processing timeout for piece ID {piece.id}, resetting status")
                    piece.being_processed = False
                    piece.processing_start_time = None

            # Remove stale tracks that haven't been updated recently
            updated_tracks = []

            for piece in self.tracked_pieces:
                # Keep track if:
                # 1. It was just updated
                # 2. It hasn't timed out yet
                # 3. It has been captured (we want to keep it visible a bit longer)
                # 4. It is currently being processed
                if (piece.updated or
                        (current_time - piece.last_update_time < self.track_timeout) or
                        (piece.captured and current_time - piece.last_update_time < self.track_timeout * 2) or
                        piece.being_processed):
                    updated_tracks.append(piece)
                else:
                    print(
                        f"Removing stale track ID: {piece.id}, Last update: {current_time - piece.last_update_time:.2f}s ago")

            self.tracked_pieces = updated_tracks

    def _clean_captured_locations(self):
        """Remove expired spatial cooldown zones"""
        with self.lock:
            current_time = time.time()
            expired_threshold = current_time - self.spatial_cooldown_time
            self.captured_locations = [(x, y, t) for x, y, t in self.captured_locations if t > expired_threshold]

    def _should_capture(self, piece: TrackedPiece) -> bool:
        """Determine if the piece should be captured."""
        with self.lock:
            # Skip if already marked as captured or being processed
            if piece.captured or piece.being_processed:
                return False

            current_time = time.time()

            # Don't capture if it's too soon after the last capture
            if current_time - self.last_capture_time < self.capture_min_interval:
                return False

            # Don't capture pieces that just appeared (avoid post-stutter duplicates)
            # Either use this with a class variable:
            new_piece_grace_period = 0.5  # Local variable definition
            if current_time - piece.entry_time < new_piece_grace_period:
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
                    print(f"Skipping piece ID {piece.id} - too close to previous capture location ({distance:.1f}px)")
                    piece.captured = True  # Mark as captured to avoid repeated checks
                    return False

            # If all checks passed, this piece should be captured
            return True

    def _scale_to_original_coordinates(self, piece: TrackedPiece) -> Tuple[int, int, int, int]:
        """Scale detection coordinates to original resolution.

        Args:
            piece: The tracked piece with detection resolution coordinates

        Returns:
            Tuple[int, int, int, int]: Scaled coordinates (x, y, w, h) in original resolution
        """
        x, y, w, h = piece.bounding_box
        fx, fy = piece.scale_factor

        # Scale the coordinates
        orig_x = int(x * fx)
        orig_y = int(y * fy)
        orig_w = int(w * fx)
        orig_h = int(h * fy)

        return (orig_x, orig_y, orig_w, orig_h)

    def _crop_piece_image(self, frame, piece: TrackedPiece) -> np.ndarray:
        """Crop the frame to just the piece with padding

        Args:
            frame: The video frame to crop (original resolution)
            piece: The TrackedPiece to crop around (detection resolution coordinates)

        Returns:
            np.ndarray: The cropped image
        """
        # Scale coordinates to original resolution
        orig_x, orig_y, orig_w, orig_h = self._scale_to_original_coordinates(piece)

        # Apply original ROI offset
        roi_x, roi_y = self.original_roi[0], self.original_roi[1]

        # Add padding and ensure within frame bounds
        pad = self.crop_padding
        x1 = max(0, orig_x + roi_x - pad)
        y1 = max(0, orig_y + roi_y - pad)
        x2 = min(frame.shape[1], orig_x + roi_x + orig_w + pad)
        y2 = min(frame.shape[0], orig_y + roi_y + orig_h + pad)

        return frame[y1:y2, x1:x2]

    def process_frame(self, frame, current_count=None) -> Tuple[List[TrackedPiece], Optional[np.ndarray]]:
        """Process a frame and return any pieces to capture.

        Args:
            frame: The original resolution video frame to process
            current_count: The current image count (optional)

        Returns:
            tuple: (tracked_pieces, cropped_image)
                - tracked_pieces: List of all currently tracked pieces
                - cropped_image: Cropped image of a piece to capture, or None
        """
        if self.roi is None:
            raise ValueError("Detector not calibrated. Call calibrate() first.")

        # Clean up expired capture locations
        self._clean_captured_locations()

        # Create a downscaled version of the frame for detection
        detection_frame = cv2.resize(frame, (0, 0), fx=self.detection_scale_factor, fy=self.detection_scale_factor)

        # Process the downscaled frame
        mask = self._preprocess_frame(detection_frame)
        new_contours = self._find_new_contours(mask)

        # Update tracking
        self._match_and_update_tracks(new_contours)

        # Check for pieces to capture
        with self.lock:
            for piece in self.tracked_pieces:
                if self._should_capture(piece):
                    # Get piece position for spatial cooldown (in detection coordinates)
                    x, y, w, h = piece.bounding_box
                    center_x = x + w / 2
                    center_y = y + h / 2

                    # Crop the image from the original full-resolution frame
                    cropped_image = self._crop_piece_image(frame, piece)

                    # If in asynchronous mode, add to processing queue and don't wait
                    if self.async_mode and self.thread_manager:
                        # Mark piece as being processed
                        piece.being_processed = True
                        piece.processing_start_time = time.time()

                        # Add to processing queue without waiting
                        self.thread_manager.add_message(
                            piece_id=piece.id,
                            image=cropped_image.copy(),  # Make a copy to avoid reference issues
                            frame_number=current_count,
                            position=self._scale_to_original_coordinates(piece),  # Use original coordinates
                            priority=0  # Normal priority
                        )

                        # Mark as captured to prevent duplicate processing
                        piece.captured = True
                        self.last_capture_time = time.time()

                        # Register capture location
                        self.captured_locations.append((center_x, center_y, time.time()))

                        print(f"Queueing piece ID {piece.id} at position ({center_x:.1f}, {center_y:.1f})")

                        # Return no image to indicate async processing
                        return self.tracked_pieces, None

                    # Synchronous mode (original behavior)
                    else:
                        # Mark as captured and update tracking
                        piece.captured = True
                        if current_count is not None:
                            piece.image_number = current_count
                        self.last_capture_time = time.time()

                        # Register capture location
                        self.captured_locations.append((center_x, center_y, time.time()))

                        print(f"Capturing piece ID {piece.id} at position ({center_x:.1f}, {center_y:.1f})")
                        return self.tracked_pieces, cropped_image

        return self.tracked_pieces, None

    def process_frame_async(self, frame, current_count=None) -> List[TrackedPiece]:
        """Process a frame with asynchronous piece detection and capture.

        This method ensures detection continues without blocking even when
        pieces are captured.

        Args:
            frame: The video frame to process
            current_count: The current image count (optional)

        Returns:
            List[TrackedPiece]: All currently tracked pieces
        """
        if not self.async_mode:
            raise ValueError("Detector not in asynchronous mode. Initialize with thread_manager.")

        # Process frame normally but discard the cropped image
        tracked_pieces, _ = self.process_frame(frame, current_count)
        return tracked_pieces

    def draw_debug(self, frame):
        """Draw debug visualization with piece tracking, capture status, and grid overlay.

        Args:
            frame: The video frame to draw on

        Returns:
            np.ndarray: The debug frame with visualization
        """
        if self.roi is None:
            return frame

        debug_frame = frame.copy()

        # Draw original ROI - Green boundary
        x, y, w, h = self.original_roi
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate entry and exit zones in original coordinates
        orig_entry_x = int(self.entry_zone[1] / self.detection_scale_factor)
        orig_exit_x = int(self.exit_zone[0] / self.detection_scale_factor)

        # Draw entry zone - Blue (scaled to original coordinates)
        cv2.line(debug_frame,
                 (x + orig_entry_x, y),
                 (x + orig_entry_x, y + h),
                 (255, 0, 0), 2)

        # Draw exit zone - Red (scaled to original coordinates)
        cv2.line(debug_frame,
                 (x + orig_exit_x, y),
                 (x + orig_exit_x, y + h),
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

        # Draw spatial cooldown zones (scaled to original coordinates)
        with self.lock:
            for loc_x, loc_y, t in self.captured_locations:
                # Calculate age as percentage of total cooldown time
                current_time = time.time()
                age_pct = min(1.0, (current_time - t) / self.spatial_cooldown_time)

                # Color fades from red to transparent as it ages
                color = (0, 0, 255)

                # Scale coordinates to original resolution
                orig_x = int(loc_x / self.detection_scale_factor)
                orig_y = int(loc_y / self.detection_scale_factor)
                orig_radius = int(self.spatial_cooldown_radius / self.detection_scale_factor)

                # Draw circle showing the cooldown zone
                cv2.circle(debug_frame,
                           (int(orig_x + self.original_roi[0]), int(orig_y + self.original_roi[1])),
                           orig_radius,
                           color,
                           1)  # Thickness

            # Get current time for age calculation
            current_time = time.time()

            # Draw thread manager queue size if in async mode
            if self.async_mode and self.thread_manager:
                queue_size = self.thread_manager.get_queue_size()
                cv2.putText(debug_frame, f"Queue: {queue_size}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            for piece in self.tracked_pieces:
                # Scale detection coordinates to original resolution
                orig_x, orig_y, orig_w, orig_h = self._scale_to_original_coordinates(piece)

                # Apply original ROI offset
                roi_x, roi_y = self.original_roi[0], self.original_roi[1]
                x = orig_x + roi_x
                y = orig_y + roi_y
                w = orig_w
                h = orig_h

                # Determine box color based on captured status and update freshness
                age = current_time - piece.last_update_time

                if piece.being_processed:
                    # Purple for pieces being processed
                    box_color = (255, 0, 255)
                elif piece.captured:
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
                if piece.being_processed:
                    proc_time = time.time() - piece.processing_start_time if piece.processing_start_time else 0
                    label_text = f"ID:{piece.id} (proc:{proc_time:.1f}s)"
                elif piece.captured and piece.image_number is not None:
                    label_text = f"#{piece.image_number}"
                else:
                    # Show ID and age for debugging
                    label_text = f"ID:{piece.id} ({age:.1f}s)"

                label_color = box_color
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(debug_frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 0, 0), -1)
                cv2.putText(debug_frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)

        return debug_frame


# Factory function to create a detector
def create_detector(detector_type="tracked", config_manager=None, thread_manager=None):
    """Create a detector of the specified type

    Args:
        detector_type: Type of detector ("tracked" or future types)
        config_manager: Configuration manager
        thread_manager: Thread manager for asynchronous operation

    Returns:
        Detector instance
    """
    if detector_type == "tracked":
        return TrackedLegoDetector(config_manager, thread_manager)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}. Currently only 'tracked' is supported.")