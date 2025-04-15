"""
detector_module.py - Simplified detector for Lego sorting application

This module handles the detection and tracking of Lego pieces moving through a region of interest.
It uses OpenCV's background subtraction and position-based tracking for efficient detection.
"""

import time
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np
import threading
import logging

# Get module logger
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
    update_count: int = 1  # Number of times this piece has been updated
    # Threading support
    being_processed: bool = False  # Whether this piece is being processed by worker thread
    processing_start_time: Optional[float] = None  # When processing started

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
        self.track_timeout = 1.0  # Time in seconds before removing a track that hasn't been updated
        self.min_tracking_updates = 5  # Minimum number of tracking updates before capture

        # Background subtraction parameters
        self.history_length = 200
        self.var_threshold = 36
        self.morphology_size = 5

        # Capture cooldown settings
        self.capture_min_interval = 3.0  # Minimum seconds between captures

        # Initialize from config if provided
        if config_manager:
            self.min_piece_area = config_manager.get("detector", "min_piece_area", self.min_piece_area)
            self.max_piece_area = config_manager.get("detector", "max_piece_area", self.max_piece_area)
            self.buffer_percent = config_manager.get("detector", "buffer_percent", self.buffer_percent)
            self.crop_padding = config_manager.get("detector", "crop_padding", self.crop_padding)
            self.track_timeout = config_manager.get("detector", "track_timeout", self.track_timeout)
            self.capture_min_interval = config_manager.get("detector", "capture_min_interval",
                                                           self.capture_min_interval)
            self.min_tracking_updates = config_manager.get("detector", "min_tracking_updates",
                                                           self.min_tracking_updates)
            self.history_length = config_manager.get("detector", "history_length", self.history_length)
            self.var_threshold = config_manager.get("detector", "var_threshold", self.var_threshold)
            self.morphology_size = config_manager.get("detector", "morphology_size", self.morphology_size)

            # Threading settings
            if config_manager.is_threading_enabled():
                self.processing_timeout = config_manager.get("threading", "processing_timeout", 60.0)
            else:
                self.processing_timeout = 60.0  # Default timeout

        # Initialize background subtractor
        self.bg_subtractor = None

        # Initialize tracking-related variables
        self.tracked_pieces = []
        self.next_piece_id = 1
        self.roi = None
        self.entry_zone = None
        self.exit_zone = None
        self.valid_zone = None
        self.last_capture_time = 0

        # For visualization
        self.last_frame_time = time.time()
        self.frame_rate = 0
        # Whether to show background subtraction in debug view
        self.show_bg_subtraction = True

        # Thread-safety
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.thread_manager = thread_manager

        # Asynchronous mode flag (always true since we're focusing on multithread approach)
        self.async_mode = True

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

            # Validate ROI configuration - check if all required values exist AND are non-zero
            if (roi_config and
                    "x" in roi_config and "y" in roi_config and
                    "w" in roi_config and "h" in roi_config and
                    roi_config["w"] > 0 and roi_config["h"] > 0):  # Ensure width and height are positive

                # ROI exists in config, use it
                roi = (roi_config["x"], roi_config["y"], roi_config["w"], roi_config["h"])
                print(f"Loading ROI from config: {roi}")

                with self.lock:
                    self.roi = roi
                    roi_width = self.roi[2]

                    # Calculate buffer zones
                    buffer_width = int(roi_width * self.buffer_percent)
                    self.entry_zone = (self.roi[0], self.roi[0] + buffer_width)
                    self.exit_zone = (self.roi[0] + self.roi[2] - buffer_width, self.roi[0] + self.roi[2])
                    self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

                    print(f"ROI: {self.roi}")
                    print(f"Entry buffer: {self.entry_zone}")
                    print(f"Valid zone: {self.valid_zone}")
                    print(f"Exit buffer: {self.exit_zone}")

                    # Initialize background model
                    x, y, w, h = self.roi
                    roi_frame = frame[y:y + h, x:x + w]
                    self._initialize_background_model(roi_frame)

                return roi
            else:
                print("ROI configuration is invalid or incomplete. Running manual calibration.")

        # If we get here, either no config manager or no valid ROI in config
        # Fall back to manual calibration
        print("No valid ROI configuration found. Please select ROI manually.")
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
            self.roi = roi
            roi_width = self.roi[2]

            # Calculate buffer zones
            buffer_width = int(roi_width * self.buffer_percent)
            self.entry_zone = (self.roi[0], self.roi[0] + buffer_width)
            self.exit_zone = (self.roi[0] + self.roi[2] - buffer_width, self.roi[0] + self.roi[2])
            self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

            print(f"ROI: {self.roi}")
            print(f"Entry buffer: {self.entry_zone}")
            print(f"Valid zone: {self.valid_zone}")
            print(f"Exit buffer: {self.exit_zone}")

            # Initialize background model
            x, y, w, h = self.roi
            roi_frame = frame[y:y + h, x:x + w]
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

        return roi

    def _initialize_background_model(self, roi_frame):
        """Initialize the background subtractor with the ROI frame

        Args:
            roi_frame: The ROI portion of the frame
        """
        # Create background subtractor with tuned parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history_length,  # Frames to use for background model
            varThreshold=self.var_threshold,  # Higher threshold = less sensitive
            detectShadows=False  # Disable shadow detection for better performance
        )

        # Learn initial background from several frames
        for _ in range(30):  # Increase frames for better learning
            self.bg_subtractor.apply(roi_frame)

        logger.info("Background subtractor initialized")

    def _preprocess_frame(self, frame):
        """Process frame with enhanced background subtraction

        Args:
            frame: Full camera frame

        Returns:
            Binary mask of detected foreground objects
        """
        if self.roi is None:
            raise ValueError("Detector not calibrated. Call calibrate() first.")

        # Extract ROI
        x, y, w, h = self.roi
        roi = frame[y:y + h, x:x + w]

        # Apply background subtraction
        mask = self.bg_subtractor.apply(roi)

        # Convert ROI to grayscale for additional processing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold to get additional detail
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Combine the results (logical AND)
        combined = cv2.bitwise_and(mask, thresh)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphology_size, self.morphology_size))

        # Open operation (erosion followed by dilation) removes small objects
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Close operation (dilation followed by erosion) closes small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # Additional morphological operations to remove noise
        # Erosion to remove small isolated regions
        eroded = cv2.erode(cleaned, kernel, iterations=1)
        # Dilation to restore the size of remaining regions
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        return dilated

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
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_piece_area or area > self.max_piece_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            if self.roi[0] <= x < self.roi[0] + self.roi[2]:  # Entire ROI
                valid_contours.append((contour, (x, y, w, h)))

        return valid_contours

    def _update_tracking(self, frame):
        """Update tracking with new frame

        Args:
            frame: Current video frame
        """
        with self.lock:
            # Process new contours to find new potential objects
            mask = self._preprocess_frame(frame)
            new_contours = self._find_new_contours(mask)

            # Track existing pieces
            valid_pieces = []
            current_time = time.time()

            # Convert contours to simplified objects with just position info
            contour_objects = []
            for contour, bbox in new_contours:
                x, y, w, h = bbox
                center_x = x + w / 2
                center_y = y + h / 2
                contour_objects.append({
                    "contour": contour,
                    "bbox": bbox,
                    "center": (center_x, center_y),
                    "matched": False
                })

            # First, try to update existing tracks
            for piece in self.tracked_pieces:
                # Skip pieces outside the ROI
                if piece.bounding_box is None:
                    continue

                # Get current position
                px, py, pw, ph = piece.bounding_box
                center_x = px + pw / 2
                center_y = py + ph / 2

                # Find closest new contour
                best_match = None
                best_distance = float('inf')

                for obj in contour_objects:
                    if obj["matched"]:
                        continue

                    # Calculate distance between centers
                    dx = center_x - obj["center"][0]
                    dy = center_y - obj["center"][1]
                    distance = math.sqrt(dx * dx + dy * dy)

                    # Maximum allowed distance is half the width or height
                    max_distance = max(pw, ph) / 2

                    if distance < max_distance and distance < best_distance:
                        best_distance = distance
                        best_match = obj

                if best_match:
                    # Update with matched contour
                    best_match["matched"] = True
                    piece.contour = best_match["contour"]
                    piece.bounding_box = best_match["bbox"]
                    piece.updated = True
                    piece.last_update_time = current_time
                    piece.update_count += 1
                    valid_pieces.append(piece)
                elif (piece.being_processed or
                      piece.captured or
                      current_time - piece.last_update_time < self.track_timeout):
                    # Keep pieces that are being processed or were recently updated
                    piece.updated = False
                    valid_pieces.append(piece)

            # Check for timed out pieces being processed
            for piece in valid_pieces:
                if (piece.being_processed and
                        piece.processing_start_time is not None and
                        current_time - piece.processing_start_time > self.processing_timeout):
                    # Reset processing status if timeout exceeded
                    logger.info(f"Processing timeout for piece ID {piece.id}, resetting status")
                    piece.being_processed = False
                    piece.processing_start_time = None

            # Add new tracks for unmatched contours
            for obj in contour_objects:
                if obj["matched"]:
                    continue

                x, y, w, h = obj["bbox"]

                # Skip pieces that haven't entered the ROI yet
                if x < self.entry_zone[0]:
                    continue

                # Create new tracked piece
                new_piece = TrackedPiece(
                    id=self.next_piece_id,
                    contour=obj["contour"],
                    bounding_box=obj["bbox"],
                    entry_time=time.time()
                )

                valid_pieces.append(new_piece)
                self.next_piece_id += 1

            # Update tracked pieces list
            self.tracked_pieces = valid_pieces

    def _should_capture(self, piece):
        """Determine if the piece should be captured.

        Args:
            piece: TrackedPiece to check

        Returns:
            bool: True if the piece should be captured
        """
        # Skip if already marked as captured or being processed
        if piece.captured or piece.being_processed:
            return False

        current_time = time.time()

        # Don't capture if it's too soon after the last capture
        if current_time - self.last_capture_time < self.capture_min_interval:
            return False

        # Don't capture pieces that just appeared (avoid post-stutter duplicates)
        if current_time - piece.entry_time < 0.5:  # Half second grace period
            return False

        # Get piece position
        x, y, w, h = piece.bounding_box

        # Only capture in the valid zone
        if not (self.valid_zone[0] <= x and x + w <= self.valid_zone[1]):
            return False

        # Ensure piece has been tracked for enough frames
        if piece.update_count < self.min_tracking_updates:
            return False

        # If all checks passed, this piece should be captured
        return True

    def _crop_piece_image(self, frame, piece):
        """Crop the frame to just the piece with padding

        Args:
            frame: The video frame to crop
            piece: The TrackedPiece to crop around

        Returns:
            np.ndarray: The cropped image
        """
        # Get bounding box
        x, y, w, h = piece.bounding_box

        # Add ROI offset
        roi_x, roi_y = self.roi[0], self.roi[1]
        x += roi_x
        y += roi_y

        # Add padding and ensure within frame bounds
        pad = self.crop_padding
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        return frame[y1:y2, x1:x2]

    def process_frame(self, frame, current_count=None):
        """Process a frame and return any pieces to capture.

        Args:
            frame: The video frame to process
            current_count: The current image count (optional)

        Returns:
            tuple: (tracked_pieces, None)
                - tracked_pieces: List of all currently tracked pieces
                - Second element is always None in multithreaded mode
        """
        if self.roi is None:
            raise ValueError("Detector not calibrated. Call calibrate() first.")

        # Calculate FPS
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        if delta_time > 0:
            self.frame_rate = 1.0 / delta_time
        self.last_frame_time = current_time

        # Update tracking with new frame
        self._update_tracking(frame)

        # Check for pieces to capture and queue for processing
        with self.lock:
            for piece in self.tracked_pieces:
                if self._should_capture(piece):
                    # Crop the image
                    cropped_image = self._crop_piece_image(frame, piece)

                    # Mark piece as being processed
                    piece.being_processed = True
                    piece.processing_start_time = time.time()

                    # Add to processing queue
                    self.thread_manager.add_message(
                        piece_id=piece.id,
                        image=cropped_image.copy(),
                        frame_number=current_count,
                        position=piece.bounding_box,
                        priority=0
                    )

                    # Mark as captured to prevent duplicate processing
                    piece.captured = True
                    self.last_capture_time = time.time()

                    # Log capture event
                    logger.info(f"Queueing piece ID {piece.id} for processing")

        # Always return None as second element since processing is async
        return self.tracked_pieces, None

    def draw_debug(self, frame):
        """Draw debug visualization with piece tracking and capture status.

        Args:
            frame: The video frame to draw on

        Returns:
            np.ndarray: The debug frame with visualization
        """
        if self.roi is None:
            return frame

        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]

        # Draw ROI - Green boundary
        x, y, roi_w, roi_h = self.roi
        cv2.rectangle(debug_frame, (x, y), (x + roi_w, y + roi_h), (0, 255, 0), 2)

        # Draw entry zone - Blue
        cv2.line(debug_frame,
                 (x + self.entry_zone[1] - self.roi[0], y),
                 (x + self.entry_zone[1] - self.roi[0], y + roi_h),
                 (255, 0, 0), 2)

        # Draw exit zone - Red
        cv2.line(debug_frame,
                 (x + self.exit_zone[0] - self.roi[0], y),
                 (x + self.exit_zone[0] - self.roi[0], y + roi_h),
                 (0, 0, 255), 2)

        # Draw system info panel
        panel_w = 300
        panel_h = 120
        panel_x = 10
        panel_y = 10

        # Panel background
        cv2.rectangle(debug_frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.rectangle(debug_frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)

        # Performance metrics
        cv2.putText(debug_frame, f"FPS: {self.frame_rate:.1f}",
                    (panel_x + 10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.thread_manager:
            queue_size = self.thread_manager.get_queue_size()
            cv2.putText(debug_frame, f"Queue: {queue_size}",
                        (panel_x + 10, panel_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Tracker stats
        active_count = sum(1 for p in self.tracked_pieces if not p.captured)
        processing_count = sum(1 for p in self.tracked_pieces if p.being_processed)

        cv2.putText(debug_frame, f"Active: {active_count} Processing: {processing_count}",
                    (panel_x + 10, panel_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Get current time for age calculation
        current_time = time.time()

        # Draw tracked pieces
        with self.lock:
            for piece in self.tracked_pieces:
                # Get bounding box
                x, y, w, h = piece.bounding_box

                # Apply ROI offset
                roi_x, roi_y = self.roi[0], self.roi[1]
                x += roi_x
                y += roi_y

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
                    # Show ID, age and update count for debugging
                    label_text = f"ID:{piece.id} ({age:.1f}s) U:{piece.update_count}"

                label_color = box_color
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(debug_frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 0, 0), -1)
                cv2.putText(debug_frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)

        # Draw background subtraction visualization if enabled
        if self.show_bg_subtraction:
            # Create small window for background subtraction preview
            mask = self._preprocess_frame(frame)
            mask_small = cv2.resize(mask, (320, 180))
            # Convert to color for visualization
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

            # Place in bottom right corner
            bg_x = w - 330
            bg_y = h - 190
            debug_frame[bg_y:bg_y + 180, bg_x:bg_x + 320] = mask_color
            cv2.rectangle(debug_frame, (bg_x, bg_y), (bg_x + 320, bg_y + 180), (255, 255, 255), 2)
            cv2.putText(debug_frame, "Background Mask", (bg_x + 10, bg_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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