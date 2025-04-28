"""
detector_module.py - Module for detecting and tracking Lego pieces on a conveyor belt

This module provides functionality to:
1. Define a Region of Interest (ROI) on a conveyor belt
2. Perform background subtraction optimized for moving conveyors
3. Detect and track Lego pieces as they move through the ROI
4. Determine when a piece should be captured
5. Provide visualization of the detection and tracking process
"""

import cv2
import numpy as np
import time
import os
import json
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# Set up module logger
logger = logging.getLogger(__name__)


@dataclass
class TrackedPiece:
    """A class representing a detected and tracked Lego piece"""
    id: int  # Unique identifier
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    contour: np.ndarray  # Contour points
    first_detected: float  # Timestamp when first detected
    last_updated: float  # Timestamp when last updated
    captured: bool = False  # Whether this piece has been captured
    update_count: int = 1  # Number of frames this piece has been tracked
    center: Tuple[float, float] = None  # Center point (calculated from bbox)
    being_processed: bool = False  # Flag for pieces being processed by worker thread
    processing_start_time: Optional[float] = None  # When processing started
    updated: bool = True  # Whether this piece was updated in the current frame
    in_exit_zone: bool = False  # Whether this piece is in the exit zone
    exit_zone_entry_time: Optional[float] = None  # When the piece entered the exit zone
    triggered_servo: bool = False  # Whether this piece has triggered servo movement

    def __post_init__(self):
        """Calculate derived properties"""
        x, y, w, h = self.bbox
        self.center = (x + w / 2, y + h / 2)

    def get_right_edge(self) -> float:
        """Get the x-coordinate of the right edge of the piece"""
        x, _, w, _ = self.bbox
        return x + w


class ConveyorDetector:
    """Detector for Lego pieces on a moving conveyor belt"""

    def __init__(self, config_manager=None, thread_manager=None):
        """Initialize the detector with optional configuration

        Args:
            config_manager: Optional configuration manager
            thread_manager: Optional thread manager for asynchronous processing
        """
        logger.info("Initializing ConveyorDetector")

        # Default configuration values
        self.config = {
            "min_piece_area": 1000,  # Minimum contour area to be considered a piece
            "max_piece_area": 50000,  # Maximum contour area to be considered a piece
            "bg_history": 500,  # Background subtractor history length
            "bg_threshold": 800,  # Background subtractor threshold
            "learn_rate": 0.005,  # Background learning rate
            "min_updates": 5,  # Minimum updates before a piece can be captured
            "track_timeout": 1.0,  # Time in seconds before removing stale tracks
            "entry_zone_percent": 0.1,  # Percent of ROI width to use as entry buffer
            "exit_zone_percent": 0.1,  # Percent of ROI width to use as exit buffer
            "morph_kernel_size": 5,  # Kernel size for morphological operations
            "capture_cooldown": 0.5,  # Minimum time between captures
            "match_threshold": 50,  # Maximum distance for matching tracks
            "roi_padding": 20,  # Padding for piece cropping
            "gaussian_blur": 7,  # Gaussian blur kernel size
            "processing_timeout": 60.0,  # Timeout for piece processing
            "load_roi_from_config": True,  # Whether to load ROI from config or use manual selection
            "text_margin_top": 40,  # Margin for text to avoid overlapping with pieces
            "default_roi_x": 125,  # Default ROI x position
            "default_roi_y": 200,  # Default ROI y position
            "default_roi_w": 1600,  # Default ROI width
            "default_roi_h": 500,  # Default ROI height
            "show_bg_subtraction": True,  # Whether to show background subtraction in debug view
        }

        # Update with provided config manager
        self.config_manager = config_manager
        if config_manager:
            detector_config = config_manager.get_section("detector")
            logger.info(f"Detector config from config_manager: {detector_config}")
            if detector_config:
                for key, value in detector_config.items():
                    if key in self.config:
                        self.config[key] = value
                        logger.info(f"Set config[{key}] = {value}")

            # Load exit zone trigger configuration
            self.exit_zone_trigger_config = config_manager.get_section("exit_zone_trigger")
            logger.info(f"Exit zone trigger config: {self.exit_zone_trigger_config}")

        else:
            # Set default exit zone trigger config if no config manager provided
            self.exit_zone_trigger_config = {
                "enabled": True,
                "fall_time": 1.0,
                "priority_method": "rightmost",
                "min_piece_spacing": 50,
                "cooldown_time": 0.5
            }

        # Store thread manager
        self.thread_manager = thread_manager

        # Initialize state with thread safety
        self.lock = threading.RLock()
        self.roi = None  # Region of interest (x, y, width, height)
        self.entry_zone = None  # Entry zone x coordinates (start, end)
        self.exit_zone = None  # Exit zone x coordinates (start, end)
        self.valid_zone = None  # Valid zone x coordinates (start, end)
        self.bg_subtractor = None  # Background subtractor
        self.tracked_pieces = []  # List of currently tracked pieces
        self.next_id = 1  # ID for the next detected piece
        self.last_capture_time = 0  # Time of last capture
        self.frame_count = 0  # Counter for processed frames
        self.latest_debug_frame = None  # Store latest debug frame

        # Exit zone tracking
        self.pieces_in_exit_zone = []  # List of pieces currently in exit zone
        self.current_priority_piece = None  # Current priority piece in exit zone
        self.last_servo_trigger_time = 0  # Time of last servo trigger
        self.last_exit_zone_check_time = 0  # Time of last exit zone check

        # Performance tracking
        self.start_time = time.time()
        self.fps = 0
        self.last_fps_update = 0

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

        # Check if we should load from config
        load_from_config = self.config.get("load_roi_from_config", True)
        logger.info(f"load_roi_from_config parameter value: {load_from_config}")

        if load_from_config and config_manager:
            # Try to load ROI from config
            roi_config = config_manager.get_section("detector_roi")
            logger.info(f"ROI config from config_manager: {roi_config}")

            # Validate ROI configuration
            if (roi_config and
                    "x" in roi_config and "y" in roi_config and
                    "w" in roi_config and "h" in roi_config and
                    roi_config["w"] > 0 and roi_config["h"] > 0):

                # ROI exists in config, use it
                roi = (roi_config["x"], roi_config["y"], roi_config["w"], roi_config["h"])
                logger.info(f"Using ROI from config: {roi}")

                with self.lock:
                    self.set_roi(frame, roi)

                return roi
            else:
                logger.warning(f"Invalid ROI configuration found: {roi_config}")

                # Use default ROI values
                default_roi = (
                    self.config["default_roi_x"],
                    self.config["default_roi_y"],
                    self.config["default_roi_w"],
                    self.config["default_roi_h"]
                )

                logger.info(f"Using default ROI: {default_roi}")
                with self.lock:
                    self.set_roi(frame, default_roi)

                # Save default ROI to config
                if config_manager:
                    roi_config = {
                        "x": default_roi[0],
                        "y": default_roi[1],
                        "w": default_roi[2],
                        "h": default_roi[3]
                    }
                    config_manager.update_section("detector_roi", roi_config)
                    config_manager.save_config()
                    logger.info("Default ROI saved to configuration")

                return default_roi
        else:
            logger.info(f"Manual calibration requested (load_from_config={load_from_config})")

        # If we get here, use manual calibration
        return self.calibrate(frame)

    def calibrate(self, frame):
        """Select ROI for the conveyor belt and initialize background model

        Args:
            frame: Video frame to use for ROI selection

        Returns:
            Tuple[int, int, int, int]: Selected ROI (x, y, width, height)
        """
        logger.info("Starting manual ROI calibration")
        print("\nSelect the conveyor belt region (ROI)...")
        print("Draw a rectangle with your mouse, then press ENTER or SPACE to confirm")
        roi = cv2.selectROI("Select Conveyor Belt ROI", frame, showCrosshair=False)
        cv2.destroyWindow("Select Conveyor Belt ROI")

        with self.lock:
            self.set_roi(frame, roi)

            # Save ROI to config if available
            if self.config_manager:
                roi_config = {
                    "x": roi[0],
                    "y": roi[1],
                    "w": roi[2],
                    "h": roi[3]
                }
                self.config_manager.update_section("detector_roi", roi_config)

                # Set load_roi_from_config to True to use this ROI in the future
                self.config["load_roi_from_config"] = True
                self.config_manager.set("detector", "load_roi_from_config", True)

                self.config_manager.save_config()
                logger.info("ROI saved to configuration and load_roi_from_config set to True")

        return roi

    def set_roi(self, frame, roi):
        """Set the ROI and initialize related zones and background model

        Args:
            frame: Video frame to use for initialization
            roi: Tuple of (x, y, width, height) defining the ROI
        """
        self.roi = roi
        x, y, w, h = roi

        # Calculate buffer zones with separate entry and exit zone percentages
        entry_buffer_width = int(w * self.config["entry_zone_percent"])
        exit_buffer_width = int(w * self.config["exit_zone_percent"])

        self.entry_zone = (x, x + entry_buffer_width)
        self.exit_zone = (x + w - exit_buffer_width, x + w)
        self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

        # Initialize background subtractor with KNN
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=self.config["bg_history"],
            dist2Threshold=self.config["bg_threshold"],
            detectShadows=False
        )

        # Extract ROI and initialize background model
        roi_frame = frame[y:y + h, x:x + w]
        # Apply several times to build initial model
        for _ in range(30):
            self.bg_subtractor.apply(roi_frame, learningRate=0.1)

        logger.info(f"ROI set to {roi}")
        logger.info(f"Entry zone: {self.entry_zone}")
        logger.info(f"Valid zone: {self.valid_zone}")
        logger.info(f"Exit zone: {self.exit_zone}")

        print(f"ROI set to {roi}")
        print(f"Entry zone: {self.entry_zone}")
        print(f"Valid zone: {self.valid_zone}")
        print(f"Exit zone: {self.exit_zone}")

    def preprocess_frame(self, frame):
        """Preprocess frame for background subtraction

        Args:
            frame: Input video frame

        Returns:
            Tuple[np.ndarray, np.ndarray]: ROI image and foreground mask
        """
        if self.roi is None:
            raise ValueError("ROI not set. Call calibrate() or set_roi() first.")

        # Extract ROI
        x, y, w, h = self.roi
        roi_img = frame[y:y + h, x:x + w].copy()

        # Apply preprocessing with optimized parameters
        blur_size = self.config["gaussian_blur"]
        blurred = cv2.GaussianBlur(roi_img, (blur_size, blur_size), 0)

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=self.config["learn_rate"])

        # Clean up the mask with morphological operations
        kernel = np.ones((self.config["morph_kernel_size"], self.config["morph_kernel_size"]), np.uint8)
        mask_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        return roi_img, mask_cleaned

    def detect_pieces(self, mask):
        """Detect pieces in the foreground mask

        Args:
            mask: Foreground mask from background subtraction

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int, int, int]]]: List of (contour, bounding_box) pairs
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_pieces = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.config["min_piece_area"] or area > self.config["max_piece_area"]:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            detected_pieces.append((contour, (x, y, w, h)))

        return detected_pieces

    def update_tracking(self, detected_pieces):
        """Update tracking with newly detected pieces and prevent duplicates after capture.

        Args:
            detected_pieces: List of (contour, bounding_box) pairs
        """
        current_time = time.time()

        # Mark all existing tracks as not updated
        for piece in self.tracked_pieces:
            piece.updated = False

        # Match detected pieces with existing tracks
        unmatched_detections = []
        for contour, bbox in detected_pieces:
            x, y, w, h = bbox
            center = (x + w / 2, y + h / 2)

            # Try to find matching track
            matched = False
            for piece in self.tracked_pieces:
                dist = np.sqrt((center[0] - piece.center[0]) ** 2 + (center[1] - piece.center[1]) ** 2)
                if dist < self.config["match_threshold"]:
                    # Update existing track
                    piece.bbox = bbox
                    piece.contour = contour
                    piece.last_updated = current_time
                    piece.center = center
                    piece.update_count += 1
                    piece.updated = True

                    # Check if the piece has entered or left the exit zone
                    self.update_exit_zone_status(piece, current_time)

                    matched = True
                    break

            if not matched:
                unmatched_detections.append((contour, bbox, center))

        # Create new tracks for unmatched detections, but avoid creating duplicates of captured pieces
        for contour, bbox, center in unmatched_detections:
            x, y, w, h = bbox

            # Only create tracks for pieces that have entered the ROI
            if x >= self.entry_zone[0]:
                # Check if this detection is close to any captured pieces
                # This prevents creating duplicate tracks when a piece is captured but continues moving
                close_to_captured = False
                for piece in self.tracked_pieces:
                    if piece.captured:
                        dist = np.sqrt((center[0] - piece.center[0]) ** 2 + (center[1] - piece.center[1]) ** 2)
                        # Use a slightly larger threshold for captured pieces to ensure we catch all potential duplicates
                        if dist < self.config["match_threshold"] * 1.5:
                            close_to_captured = True
                            # Update the captured piece's position since it's still moving
                            piece.bbox = bbox
                            piece.contour = contour
                            piece.last_updated = current_time
                            piece.center = center
                            piece.updated = True
                            # Update exit zone status for the captured piece
                            self.update_exit_zone_status(piece, current_time)
                            logger.debug(f"Updated captured piece ID {piece.id} instead of creating duplicate")
                            break

                # Only create a new track if it's not close to any captured piece
                if not close_to_captured:
                    new_piece = TrackedPiece(
                        id=self.next_id,
                        bbox=bbox,
                        contour=contour,
                        first_detected=current_time,
                        last_updated=current_time
                    )
                    new_piece.updated = True

                    # Check if the new piece is already in the exit zone
                    self.update_exit_zone_status(new_piece, current_time)

                    self.tracked_pieces.append(new_piece)
                    self.next_id += 1
                    logger.debug(f"Created new track ID {self.next_id - 1} at position {bbox}")

        # Remove stale tracks
        active_tracks = []
        for piece in self.tracked_pieces:
            # Keep if updated or recently updated
            if piece.updated or (current_time - piece.last_updated) < self.config["track_timeout"]:
                active_tracks.append(piece)
            # Reset processing status if timeout exceeded
            elif piece.being_processed:
                if piece.processing_start_time is not None and (
                        current_time - piece.processing_start_time > self.config["processing_timeout"]):
                    logger.warning(f"Processing timeout for piece ID {piece.id}, resetting status")
                    piece.being_processed = False
                    piece.processing_start_time = None
                    active_tracks.append(piece)
                else:
                    active_tracks.append(piece)
            # Keep captured pieces longer to prevent immediate duplicate creation
            elif piece.captured and (current_time - piece.last_updated) < (self.config["track_timeout"] * 2):
                active_tracks.append(piece)

        # Update tracked pieces list
        removed_count = len(self.tracked_pieces) - len(active_tracks)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} stale tracks")

        self.tracked_pieces = active_tracks

        # Update the list of pieces in the exit zone
        self.update_exit_zone_pieces()

    def update_exit_zone_status(self, piece, current_time):
        """Update whether a piece is in the exit zone

        Args:
            piece: The TrackedPiece to update
            current_time: Current timestamp
        """
        if self.exit_zone is None:
            return

        x, y, w, h = piece.bbox
        right_edge = x + w

        # Check if piece is in exit zone
        in_exit_zone = x >= self.exit_zone[0] and x < self.exit_zone[1]

        # If piece just entered the exit zone
        if in_exit_zone and not piece.in_exit_zone:
            piece.in_exit_zone = True
            piece.exit_zone_entry_time = current_time
            logger.info(f"Piece ID {piece.id} entered exit zone")

        # If piece just left the exit zone
        elif not in_exit_zone and piece.in_exit_zone:
            piece.in_exit_zone = False
            piece.exit_zone_entry_time = None
            logger.info(f"Piece ID {piece.id} left exit zone")

    def update_exit_zone_pieces(self):
        """Update the list of pieces currently in the exit zone"""
        # Clear the current list
        self.pieces_in_exit_zone = []

        # Add all pieces currently in the exit zone that haven't been processed
        # and haven't been captured by the traditional method
        for piece in self.tracked_pieces:
            if piece.in_exit_zone and not piece.triggered_servo and not piece.captured:
                self.pieces_in_exit_zone.append(piece)

        # Sort the pieces based on the priority method
        if self.pieces_in_exit_zone:
            priority_method = self.exit_zone_trigger_config.get("priority_method", "rightmost")

            if priority_method == "rightmost":
                # Sort by x position (rightmost first)
                self.pieces_in_exit_zone.sort(key=lambda p: p.get_right_edge(), reverse=True)
            elif priority_method == "first_in":
                # Sort by time entered exit zone (earliest first)
                self.pieces_in_exit_zone.sort(key=lambda p: p.exit_zone_entry_time or float('inf'))

            # Update the current priority piece
            if self.pieces_in_exit_zone:
                self.current_priority_piece = self.pieces_in_exit_zone[0]
            else:
                self.current_priority_piece = None

    def get_priority_exit_zone_piece(self):
        """Get the highest priority piece in the exit zone

        Returns:
            Optional[TrackedPiece]: The priority piece or None if no pieces in exit zone
        """
        if not self.pieces_in_exit_zone:
            return None

        # Return the highest priority piece (first in the sorted list)
        return self.pieces_in_exit_zone[0] if self.pieces_in_exit_zone else None

    def check_for_capture(self):
        """Check if any piece should be captured

        Returns:
            Optional[TrackedPiece]: Piece to capture or None
        """
        current_time = time.time()

        # Don't capture if it's too soon after last capture
        if current_time - self.last_capture_time < self.config["capture_cooldown"]:
            return None

        for piece in self.tracked_pieces:
            # Skip if already captured or being processed
            if piece.captured or piece.being_processed:
                continue

            # Check if the piece has been tracked long enough
            if piece.update_count < self.config["min_updates"]:
                continue

            # Check if the piece is in the valid zone
            x, y, w, h = piece.bbox
            if not (self.valid_zone[0] <= x <= self.valid_zone[1]):
                continue

            # If all checks pass, mark as captured and return
            piece.captured = True
            self.last_capture_time = current_time
            logger.info(f"Selected piece ID {piece.id} for capture after {piece.update_count} updates")
            return piece

        return None

    def check_for_servo_trigger(self):
        """Check if a piece in the exit zone should trigger the servo

        Returns:
            Tuple[Optional[TrackedPiece], bool]: (Priority piece, trigger_servo flag)
        """
        if not self.exit_zone_trigger_config.get("enabled", True):
            return None, False

        current_time = time.time()

        # Check cooldown time since last servo trigger
        cooldown_time = self.exit_zone_trigger_config.get("cooldown_time", 0.5)
        if current_time - self.last_servo_trigger_time < cooldown_time:
            return self.current_priority_piece, False

        # Get the piece with highest priority in the exit zone
        priority_piece = self.get_priority_exit_zone_piece()

        # If no pieces in exit zone, return None
        if not priority_piece:
            return None, False

        # If the piece has already triggered the servo or has been captured by traditional method, skip
        if priority_piece.triggered_servo or priority_piece.captured:
            return priority_piece, False

        # Check if the piece has been processed by the normal capture
        if priority_piece.being_processed:
            return priority_piece, False

        # Trigger the servo for this piece
        priority_piece.triggered_servo = True
        self.last_servo_trigger_time = current_time
        logger.info(f"Triggering servo for piece ID {priority_piece.id} in exit zone")

        return priority_piece, True

    def crop_piece_image(self, frame, piece):
        """Crop the piece from the frame with padding and added space for text

        Args:
            frame: Full video frame
            piece: TrackedPiece to crop

        Returns:
            np.ndarray: Cropped image of the piece with extra space for ID text
        """
        # Get bounding box and add ROI offset
        x, y, w, h = piece.bbox
        roi_x, roi_y = self.roi[0], self.roi[1]
        x += roi_x
        y += roi_y

        # Add padding and ensure within frame bounds
        pad = self.config["roi_padding"]
        text_margin = self.config["text_margin_top"]  # Extra space at the top for text

        x1 = max(0, x - pad)
        # Add extra space at the top for the number label
        y1 = max(0, y - pad - text_margin)
        x2 = min(frame.shape[1] - 1, x + w + pad)
        y2 = min(frame.shape[0] - 1, y + h + pad)

        # Create cropped image
        cropped = frame[y1:y2, x1:x2].copy()

        return cropped

    def process_frame(self, frame, current_count=None):
        """Process a video frame to detect and track pieces

        Args:
            frame: Input video frame
            current_count: Current image counter (for filename numbering)

        Returns:
            Tuple[List[TrackedPiece], Optional[np.ndarray], bool]:
                List of tracked pieces, cropped image if a piece was captured, and increment signal
        """
        with self.lock:
            if self.roi is None:
                raise ValueError("ROI not set. Call calibrate() or set_roi() first.")

            self.frame_count += 1

            # Update FPS calculation
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                elapsed = current_time - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                self.last_fps_update = current_time

            # Preprocess and detect pieces
            roi_img, mask = self.preprocess_frame(frame)
            detected_pieces = self.detect_pieces(mask)

            # Update tracking
            self.update_tracking(detected_pieces)

            # Check for servo trigger based on exit zone
            priority_piece, trigger_servo = self.check_for_servo_trigger()

            # If a piece should trigger the servo
            if trigger_servo and priority_piece and self.thread_manager:
                # Get cropped image
                cropped_image = self.crop_piece_image(frame, priority_piece)

                # Get the current image number
                image_number = current_count

                # Add the piece ID number at the top with increased margin
                text_y = self.config["text_margin_top"] - 10  # Position for text
                cv2.putText(cropped_image, f"{image_number}",
                            (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

                # Mark as being processed
                priority_piece.being_processed = True
                priority_piece.processing_start_time = current_time

                # Add to processing queue with the highest priority (0)
                self.thread_manager.add_message(
                    piece_id=priority_piece.id,
                    image=cropped_image.copy(),
                    frame_number=image_number,
                    position=priority_piece.bbox,
                    priority=0  # Highest priority for exit zone pieces
                )
                logger.info(f"Added exit zone piece ID {priority_piece.id} to processing queue with highest priority")

                # Return signal to increment camera count
                return self.tracked_pieces, None, True

            # Traditional capture method (now less important with exit zone trigger)
            piece_to_capture = self.check_for_capture()

            if piece_to_capture:
                # If using thread manager, mark as being processed
                if self.thread_manager:
                    piece_to_capture.being_processed = True
                    piece_to_capture.processing_start_time = current_time

                    # Get cropped image
                    cropped_image = self.crop_piece_image(frame, piece_to_capture)

                    # Add the piece ID number at the top with increased margin
                    text_y = self.config["text_margin_top"] - 10  # Position for text
                    image_number = current_count
                    cv2.putText(cropped_image, f"{image_number}",
                                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

                    # Add to processing queue
                    self.thread_manager.add_message(
                        piece_id=piece_to_capture.id,
                        image=cropped_image.copy(),
                        frame_number=image_number,
                        position=piece_to_capture.bbox,
                        priority=1  # Lower priority than exit zone pieces
                    )
                    logger.info(f"Added piece ID {piece_to_capture.id} to processing queue")

                    # Return signal to increment camera count
                    return self.tracked_pieces, None, True

            return self.tracked_pieces, None, False

    def draw_debug(self, frame):
        """Draw debug visualization on the frame

        Args:
            frame: Input video frame

        Returns:
            np.ndarray: Frame with debug visualization
        """
        with self.lock:
            if self.roi is None:
                return frame

            debug_frame = frame.copy()
            x, y, w, h = self.roi

            # Draw ROI rectangle
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw entry and exit zones
            cv2.line(debug_frame,
                     (self.entry_zone[1], y),
                     (self.entry_zone[1], y + h),
                     (255, 0, 0), 2)  # Blue line for entry

            cv2.line(debug_frame,
                     (self.exit_zone[0], y),
                     (self.exit_zone[0], y + h),
                     (0, 0, 255), 2)  # Red line for exit zone start

            # Highlight exit zone with semi-transparent overlay
            exit_zone_overlay = debug_frame.copy()
            cv2.rectangle(exit_zone_overlay,
                          (self.exit_zone[0], y),
                          (self.exit_zone[1], y + h),
                          (0, 0, 255), -1)  # Filled red rectangle
            cv2.addWeighted(exit_zone_overlay, 0.2, debug_frame, 0.8, 0, debug_frame)

            # Draw tracked pieces
            for piece in self.tracked_pieces:
                # Get bounding box coordinates with ROI offset
                px, py, pw, ph = piece.bbox
                px += x  # Add ROI x offset
                py += y  # Add ROI y offset

                # Different colors based on status
                if piece == self.current_priority_piece:
                    color = (255, 255, 0)  # Yellow for priority piece in exit zone
                elif piece.being_processed:
                    color = (255, 0, 255)  # Purple for being processed
                elif piece.in_exit_zone:
                    color = (0, 165, 255)  # Orange for in exit zone but not priority
                elif piece.captured:
                    color = (0, 255, 0)  # Green for captured
                else:
                    color = (0, 0, 255)  # Red for active but not captured

                # Draw bounding box
                cv2.rectangle(debug_frame, (px, py), (px + pw, py + ph), color, 2)

                # Add ID and update count
                if piece == self.current_priority_piece:
                    label = f"ID:{piece.id} PRIORITY"
                elif piece.in_exit_zone:
                    label = f"ID:{piece.id} EXIT"
                else:
                    label = f"ID:{piece.id} U:{piece.update_count}"

                cv2.putText(debug_frame, label, (px, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Add performance info
            cv2.putText(debug_frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(debug_frame, f"Tracked: {len(self.tracked_pieces)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Count pieces by status
            active_count = sum(1 for p in self.tracked_pieces if not p.captured and not p.being_processed)
            processing_count = sum(1 for p in self.tracked_pieces if p.being_processed)
            captured_count = sum(1 for p in self.tracked_pieces if p.captured and not p.being_processed)
            exit_zone_count = len(self.pieces_in_exit_zone)

            cv2.putText(debug_frame,
                        f"Active: {active_count} Processing: {processing_count} Captured: {captured_count}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Add exit zone piece count
            cv2.putText(debug_frame,
                        f"Exit Zone: {exit_zone_count}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # Show background subtraction result in corner if enabled
            if self.config.get("show_bg_subtraction", True):
                roi_img, mask = self.preprocess_frame(frame)
                mask_small = cv2.resize(mask, (160, 90))
                mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                mask_color = mask_color * np.array([0, 255, 0], dtype=np.uint8)  # Green mask

                # Place mask in top right corner
                h, w = debug_frame.shape[:2]
                x_offset = w - 170
                y_offset = 10
                debug_frame[y_offset:y_offset + 90, x_offset:x_offset + 160] = mask_color

            # Store latest debug frame
            self.latest_debug_frame = debug_frame

            return debug_frame

    def get_visualization_data(self):
        """Return data needed for visualization without drawing.

        Returns:
            dict: Data for UI visualization
        """
        with self.lock:
            if self.roi is None:
                return {}

            # Prepare data dictionary with all elements needed for visualization
            visualization_data = {
                "roi": self.roi,
                "entry_zone": self.entry_zone,
                "exit_zone": self.exit_zone,
                "fps": self.fps,
                "tracked_pieces": []
            }

            # Convert tracked pieces to serializable format
            for piece in self.tracked_pieces:
                piece_data = {
                    "id": piece.id,
                    "bbox": piece.bbox,
                    "update_count": piece.update_count,
                    "captured": piece.captured,
                    "being_processed": piece.being_processed,
                    "center": piece.center
                }
                visualization_data["tracked_pieces"].append(piece_data)

            return visualization_data

    def release(self):
        """Release resources used by the detector"""
        logger.info("Releasing detector resources")
        # Currently no resources that need explicit release
        # but adding for API consistency
        with self.lock:
            self.bg_subtractor = None


# Factory function
def create_detector(detector_type="conveyor", config_manager=None, thread_manager=None):
    """Create a detector of the specified type

    Args:
        detector_type: Type of detector ("conveyor" or future types)
        config_manager: Configuration manager
        thread_manager: Thread manager for asynchronous operation

    Returns:
        Detector instance
    """
    logger.info(f"Creating detector of type: {detector_type}")

    if detector_type == "conveyor":
        return ConveyorDetector(config_manager, thread_manager)
    else:
        error_msg = f"Unsupported detector type: {detector_type}. Currently only 'conveyor' is supported."
        logger.error(error_msg)
        raise ValueError(error_msg)


# Simple configuration manager class for standalone testing
class SimpleConfigManager:
    """A simplified configuration manager for standalone testing"""

    def __init__(self, config_path=None):
        """Initialize with optional path to config file"""
        self.config_path = config_path
        self.config = {}

        if config_path and os.path.exists(config_path):
            self.load_config()
        else:
            # Default empty config
            self.config = {
                "detector": {},
                "detector_roi": {}
            }

    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self.config = {
                "detector": {},
                "detector_roi": {}
            }

    def save_config(self):
        """Save configuration to file"""
        if self.config_path:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
                print(f"Saved configuration to {self.config_path}")
            except Exception as e:
                print(f"Error saving configuration: {e}")

    def get_section(self, section):
        """Get a configuration section"""
        return self.config.get(section, {})

    def update_section(self, section, values):
        """Update a section with new values"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)

    def set(self, section, key, value):
        """Set a specific configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
