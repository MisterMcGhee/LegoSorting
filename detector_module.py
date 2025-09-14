"""
detector_module.py - Module for detecting and tracking Lego pieces on a conveyor belt

This module is responsible for:
1. Defining a Region of Interest (ROI) where pieces are detected
2. Using background subtraction to identify moving pieces
3. Tracking pieces as they move through different zones
4. Determining when pieces should be captured or trigger sorting

The module does NOT handle visualization - that's the GUI's job.
It only provides data about what it detects.
"""

import cv2
import numpy as np
import time
import os
import json
import threading
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enhanced_config_manager import ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================
# These classes define the structure of data we work with

@dataclass
class TrackedPiece:
    """
    Represents a single Lego piece being tracked on the conveyor.

    This class now includes velocity tracking and motion prediction to improve
    tracking accuracy and enable better servo timing predictions.
    """
    # Basic identification
    id: int  # Unique ID for this piece
    bbox: Tuple[int, int, int, int]  # Bounding box (x, y, width, height)
    contour: np.ndarray  # Actual shape outline

    # Timing information
    first_detected: float  # When we first saw this piece
    last_updated: float  # When we last confirmed it's still there

    # Status flags
    captured: bool = False  # Has image been saved?
    being_processed: bool = False  # Is API currently analyzing it?
    fully_in_frame: bool = False  # Is entire piece visible?

    # Zone tracking
    in_exit_zone: bool = False  # Is it in the exit zone?
    exit_zone_entry_time: Optional[float] = None  # When did it enter exit zone?

    # Processing information
    processing_start_time: Optional[float] = None  # When did processing start?
    target_bin: Optional[int] = None  # Which bin should it go to?

    # Tracking metrics
    update_count: int = 1  # How many frames has it appeared in?
    center: Tuple[float, float] = None  # Center point of piece

    # NEW: Motion tracking fields for velocity and prediction
    velocity: Tuple[float, float] = (0.0, 0.0)  # Speed in pixels/second (x, y)
    trajectory: List[Tuple[float, float, float]] = None  # History of (x, y, timestamp)
    predicted_exit_time: Optional[float] = None  # When will piece fall off conveyor?

    # NEW: Cached calculations to avoid redundant math
    right_edge: float = 0.0  # Rightmost x-coordinate
    left_edge: float = 0.0  # Leftmost x-coordinate
    area: float = 0.0  # Area of bounding box

    def __post_init__(self):
        """
        Calculate derived properties after creation.
        This runs automatically when a TrackedPiece is created.
        """
        self.update_derived_properties()
        # Initialize trajectory history with first position
        self.trajectory = [(self.center[0], self.center[1], self.first_detected)]

    def update_derived_properties(self):
        """
        Recalculate all derived values from the bounding box.
        This centralizes calculations so we do them once instead of repeatedly.

        Call this whenever bbox changes to update all dependent values.
        """
        x, y, w, h = self.bbox
        # Calculate center point once
        self.center = (x + w / 2, y + h / 2)
        # Cache edge positions for quick access
        self.right_edge = x + w
        self.left_edge = x
        # Cache area for quick access
        self.area = w * h

    def update_motion(self, new_center: Tuple[float, float], timestamp: float):
        """
        Update velocity and trajectory based on new position.

        This tracks how fast and in what direction the piece is moving,
        allowing us to predict where it will be in future frames.

        Args:
            new_center: Current center position (x, y)
            timestamp: Current time in seconds
        """
        if self.trajectory is None:
            self.trajectory = []

        # Calculate velocity if we have previous positions
        if len(self.trajectory) > 0:
            last_x, last_y, last_time = self.trajectory[-1]
            dt = timestamp - last_time

            # Only update velocity if enough time has passed (avoid divide by zero)
            if dt > 0.001:  # At least 1 millisecond
                # Velocity = distance / time for each axis
                self.velocity = (
                    (new_center[0] - last_x) / dt,  # Horizontal speed
                    (new_center[1] - last_y) / dt  # Vertical speed
                )

        # Add current position to trajectory history
        self.trajectory.append((new_center[0], new_center[1], timestamp))

        # Keep only recent history (last 10 positions) to save memory
        if len(self.trajectory) > 10:
            self.trajectory.pop(0)

    def predict_position(self, future_time: float) -> Tuple[float, float]:
        """
        Predict where this piece will be at a future time.

        Uses the piece's current velocity to extrapolate position.
        This helps with matching pieces between frames and predicting
        when they'll reach the exit zone.

        Args:
            future_time: Time to predict position for (in seconds)

        Returns:
            Predicted (x, y) position
        """
        if self.trajectory and len(self.trajectory) > 0:
            # Get most recent position
            last_x, last_y, last_time = self.trajectory[-1]

            # Calculate time difference
            dt = future_time - last_time

            # Extrapolate position using velocity
            # New position = old position + velocity * time
            predicted_x = last_x + self.velocity[0] * dt
            predicted_y = last_y + self.velocity[1] * dt

            return (predicted_x, predicted_y)

        # If no trajectory, return current center
        return self.center

    def get_right_edge(self) -> float:
        """Get the rightmost x-coordinate of the piece (now using cached value)"""
        return self.right_edge

    def get_left_edge(self) -> float:
        """Get the leftmost x-coordinate of the piece (now using cached value)"""
        return self.left_edge

# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class ConveyorDetector:
    """
    Main detector class that tracks Lego pieces on a moving conveyor belt.

    The conveyor is divided into three zones:
    1. Entry Zone - Where pieces first appear (we wait until they're fully visible)
    2. Valid Zone - Where we can safely capture images
    3. Exit Zone - Where we trigger the sorting servo
    """

    def __init__(self, config_manager=None):
        """
        Initialize the detector with configuration settings.

        Args:
            config_manager: Configuration manager with all settings
        """
        logger.info("Initializing ConveyorDetector")

        # Load configuration
        self._load_configuration(config_manager)

        # Initialize ROI and zones (will be set during calibration)
        self.roi = None
        self.entry_zone = None
        self.valid_zone = None
        self.exit_zone = None

        # Initialize tracking structures
        self.tracked_pieces = []  # List of all pieces we're tracking
        self.piece_id_counter = 0  # Counter for generating unique IDs
        self.pieces_in_exit_zone = []  # Pieces currently in exit zone
        self.current_priority_piece = None  # Piece that should trigger servo next

        # Background subtraction for detecting movement
        self.bg_subtractor = None  # Will be initialized when ROI is set

        # Performance tracking
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()

        # Timing controls
        self.last_capture_time = 0
        self.last_servo_trigger_time = 0

        # Bin tracking for smart servo control
        self.current_bin_position = 0  # Which bin is servo currently at?
        self.bin_arrival_times = {}    # When did servo reach each bin?

        # Thread safety
        self.lock = threading.Lock()

        logger.info("ConveyorDetector initialized successfully")

    # ========================================================================
    # CONFIGURATION SECTION
    # ========================================================================
    # These methods handle loading and managing configuration

    def _load_configuration(self, config_manager):
        """
        Load all configuration values from the config manager.
        This centralizes all config loading in one place.

        Args:
            config_manager: Configuration manager instance
        """
        if config_manager:
            # Get module configurations
            detector_config = config_manager.get_module_config(ModuleConfig.DETECTOR.value)
            exit_zone_config = config_manager.get_module_config(ModuleConfig.EXIT_ZONE.value)

            # Store complete configs for reference
            self.config = detector_config
            self.exit_zone_trigger_config = exit_zone_config

            # Extract commonly used values - FIXED KEY NAMES
            self.min_piece_area = detector_config["min_area"]  # Changed from "min_piece_area"
            self.max_piece_area = detector_config["max_area"]  # Changed from "max_piece_area"
            self.min_updates = detector_config.get("min_updates", 5)  # Added .get() with default
            self.capture_cooldown = detector_config.get("capture_min_interval", 0.5)
            self.processing_timeout = detector_config.get("processing_timeout", 10.0)

            # Background subtraction parameters - ADD .get() WITH DEFAULTS
            self.bg_history = detector_config.get("bg_history", 500)
            self.bg_threshold = detector_config.get("bg_threshold", 800)
            self.learn_rate = detector_config.get("learn_rate", 0.005)

            # Morphological operation parameters - ADD .get() WITH DEFAULTS
            self.morph_kernel_size = detector_config.get("morph_kernel_size", 5)
            self.gaussian_blur = detector_config.get("gaussian_blur", 7)

            # Zone percentages - ADD .get() WITH DEFAULTS
            self.entry_zone_percent = detector_config.get("entry_zone_percent", 0.15)
            self.exit_zone_percent = detector_config.get("exit_zone_percent", 0.15)

            # Exit zone timing - ADD .get() WITH DEFAULTS
            self.fall_time = exit_zone_config.get("fall_time", 1.0)
            self.exit_zone_enabled = exit_zone_config.get("enabled", True)

        else:
            # Use defaults if no config manager
            logger.warning("No config manager provided, using defaults")
            self._set_default_configuration()

    def _set_default_configuration(self):
        """Set default configuration values when no config manager is available"""
        self.min_piece_area = 1000
        self.max_piece_area = 50000
        self.min_updates = 5
        self.capture_cooldown = 0.5
        self.processing_timeout = 10.0
        self.bg_history = 500
        self.bg_threshold = 800
        self.learn_rate = 0.005
        self.morph_kernel_size = 5
        self.gaussian_blur = 7
        self.entry_zone_percent = 0.15
        self.exit_zone_percent = 0.15
        self.fall_time = 1.0
        self.exit_zone_enabled = True

    # ========================================================================
    # ROI AND CALIBRATION SECTION
    # ========================================================================
    # These methods handle setting up the detection area

    def load_roi_from_config(self, frame, config_manager):
        """
        Load ROI from saved configuration if available.

        Args:
            frame: Sample frame to validate ROI against
            config_manager: Configuration manager with ROI settings

        Returns:
            bool: True if ROI was loaded successfully
        """
        if not config_manager:
            logger.warning("No config manager provided for ROI loading")
            return False

        try:
            roi_config = config_manager.get_module_config(ModuleConfig.DETECTOR_ROI.value)

            x = roi_config.get("x", 0)
            y = roi_config.get("y", 0)
            w = roi_config.get("w", 100)
            h = roi_config.get("h", 100)

            # Validate ROI is within frame bounds
            frame_h, frame_w = frame.shape[:2]
            if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                logger.error(f"ROI {(x, y, w, h)} is outside frame bounds {(frame_w, frame_h)}")
                return False

            # Set the ROI
            self.set_roi((x, y, w, h), frame)
            logger.info(f"ROI loaded from config: {(x, y, w, h)}")
            return True

        except Exception as e:
            logger.error(f"Failed to load ROI from config: {e}")
            return False

    def set_roi(self, roi: Tuple[int, int, int, int], frame: np.ndarray):
        """
        Set the Region of Interest and initialize zones.

        The ROI is divided into three zones:
        - Entry Zone: Where pieces first appear (we wait for full visibility)
        - Valid Zone: Where we can capture images
        - Exit Zone: Where we trigger sorting

        Args:
            roi: Tuple of (x, y, width, height)
            frame: Sample frame for initializing background subtraction
        """
        with self.lock:
            self.roi = roi
            x, y, w, h = roi

            # Calculate zone boundaries
            entry_width = int(w * self.entry_zone_percent)
            exit_width = int(w * self.exit_zone_percent)

            # Define zones as x-coordinate ranges
            self.entry_zone = (x, x + entry_width)
            self.exit_zone = (x + w - exit_width, x + w)
            self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

            # Initialize background subtractor
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.bg_history,
                dist2Threshold=self.bg_threshold,
                detectShadows=False
            )

            # Train background model with initial frame
            roi_frame = frame[y:y + h, x:x + w]
            for _ in range(30):  # Multiple passes to build good model
                self.bg_subtractor.apply(roi_frame, learningRate=0.1)

            logger.info(f"ROI set to {roi}")
            logger.info(f"Entry zone: x={self.entry_zone[0]} to x={self.entry_zone[1]}")
            logger.info(f"Valid zone: x={self.valid_zone[0]} to x={self.valid_zone[1]}")
            logger.info(f"Exit zone: x={self.exit_zone[0]} to x={self.exit_zone[1]}")

    # ========================================================================
    # FRAME PROCESSING SECTION
    # ========================================================================
    # These methods handle processing video frames to detect pieces

    def process_frame_for_consumer(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main entry point for the camera frame consumer system.
        This method is called by the camera module for each frame.

        Args:
            frame: Video frame from camera

        Returns:
            Dictionary with visualization data for GUI
        """
        with self.lock:
            if self.roi is None:
                return {"error": "ROI not configured"}

            # Update frame counter and FPS
            self._update_fps()

            # Process the frame to detect and track pieces
            roi_img, mask = self._preprocess_frame(frame)
            detected_pieces = self._detect_pieces_in_mask(mask)
            self._update_tracking(detected_pieces)

            # Check for any pieces that need processing
            self._check_and_trigger_captures(frame)

            # Return data for visualization (GUI will handle drawing)
            return self.get_visualization_data()

    def _update_fps(self):
        """Update FPS calculation for performance monitoring"""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.last_fps_update = current_time

    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare frame for piece detection using background subtraction.

        Args:
            frame: Full video frame

        Returns:
            Tuple of (ROI image, foreground mask)
        """
        # Extract ROI from frame
        x, y, w, h = self.roi
        roi_img = frame[y:y + h, x:x + w].copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(roi_img, (self.gaussian_blur, self.gaussian_blur), 0)

        # Apply background subtraction to detect moving objects
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=self.learn_rate)

        # Clean up the mask with morphological operations
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)  # Fill gaps

        return roi_img, mask_cleaned

    def _detect_pieces_in_mask(self, mask: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Find pieces in the foreground mask with optimized processing.

        This method now includes an early exit optimization: if the mask
        doesn't have enough white pixels to form even the smallest valid piece,
        we skip the expensive contour detection entirely.

        Args:
            mask: Binary mask from background subtraction (white = movement)

        Returns:
            List of (contour, bounding_box, area) tuples
            Note: Now includes area to avoid recalculating it later
        """
        # OPTIMIZATION: Quick check before expensive processing
        # Count white pixels in mask
        white_pixel_count = np.sum(mask > 0)

        # If there aren't enough white pixels to make a valid piece, skip everything
        if white_pixel_count < self.min_piece_area:
            # Return empty list - no need to search for contours
            return []

        # Find contours (piece outlines) in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_pieces = []
        for contour in contours:
            # Calculate area once and store it
            area = cv2.contourArea(contour)

            # Filter by area to remove noise and too-large objects
            if self.min_piece_area <= area <= self.max_piece_area:
                # Get bounding box for the piece
                x, y, w, h = cv2.boundingRect(contour)

                # Store area with the detection to avoid recalculating
                detected_pieces.append((contour, (x, y, w, h), area))

        return detected_pieces

    # ========================================================================
    # TRACKING SECTION
    # ========================================================================
    # These methods handle tracking pieces across frames

    def _update_tracking(self, detected_pieces: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]):
        """
        Update tracking information for all pieces using motion prediction.

        This enhanced version uses velocity-based prediction to better match
        pieces between frames, especially when pieces are moving quickly.

        Args:
            detected_pieces: List of newly detected pieces (contour, bbox, area)
        """
        current_time = time.time()

        # Mark all existing pieces as not updated yet this frame
        for piece in self.tracked_pieces:
            piece.updated = False

        # Process each detected piece
        for contour, bbox, area in detected_pieces:  # Now includes area
            x, y, w, h = bbox

            # Check if piece is fully in frame (not cut off at entry)
            piece_left_edge = x

            # Skip pieces that aren't fully past the entry zone
            if piece_left_edge < (self.entry_zone[1] - self.roi[0]):
                logger.debug(f"Skipping piece at x={piece_left_edge}, not fully in frame yet")
                continue

            # Check if piece is within ROI bounds (not cut off at edges)
            if x < 0 or y < 0 or x + w > self.roi[2] or y + h > self.roi[3]:
                logger.debug(f"Skipping piece at {bbox}, partially outside ROI")
                continue

            # Try to match to existing piece using motion prediction
            matched_piece = self._match_to_existing_piece_with_prediction(bbox, current_time)

            if matched_piece:
                # Update existing piece
                self._update_existing_piece(matched_piece, bbox, contour, current_time)
            else:
                # Create new tracked piece
                self._create_new_piece(bbox, contour, current_time)

        # Remove pieces that haven't been seen recently
        self._remove_stale_pieces(current_time)

        # Update zone status for all pieces
        self._update_zone_status()

        # NEW: Calculate exit predictions for pieces
        self._update_exit_predictions()

    def _match_to_existing_piece_with_prediction(self, bbox: Tuple[int, int, int, int],
                                                 current_time: float) -> Optional[TrackedPiece]:
        """
        Match a detected piece to an existing tracked piece using motion prediction.

        This improved version predicts where existing pieces SHOULD be based on
        their velocity, making tracking more accurate for fast-moving pieces.

        Args:
            bbox: Bounding box of detected piece
            current_time: Current timestamp

        Returns:
            Matched TrackedPiece or None if no match found
        """
        x, y, w, h = bbox
        new_center = (x + w / 2, y + h / 2)

        # Store candidates with their matching scores
        candidates = []

        for piece in self.tracked_pieces:
            if piece.updated:  # Skip if already matched in this frame
                continue

            # IMPROVEMENT: Use predicted position instead of last known position
            if piece.velocity != (0.0, 0.0) and len(piece.trajectory) > 1:
                # Piece is moving - predict where it should be now
                predicted_center = piece.predict_position(current_time)

                # Use larger search radius for moving pieces (they're less predictable)
                search_radius = self.config.get("match_threshold", 50) * 1.5
            else:
                # Piece just appeared or isn't moving - use last known position
                predicted_center = piece.center
                search_radius = self.config.get("match_threshold", 50)

            # Calculate distance between detected position and predicted position
            distance = np.sqrt(
                (new_center[0] - predicted_center[0]) ** 2 +
                (new_center[1] - predicted_center[1]) ** 2
            )

            # If within search radius, add as candidate
            if distance < search_radius:
                candidates.append((piece, distance))

        # Return the closest match if any candidates found
        if candidates:
            # Sort by distance and return closest
            candidates.sort(key=lambda x: x[1])
            best_match = candidates[0][0]

            logger.debug(f"Matched piece {best_match.id} with distance {candidates[0][1]:.1f}")
            return best_match

        return None

    def _update_existing_piece(self, piece: TrackedPiece, bbox: Tuple[int, int, int, int], contour: np.ndarray, current_time: float):
        """
        Update information for an existing tracked piece including motion data.

        This enhanced version also updates velocity and trajectory information
        for better prediction in future frames.

        Args:
            piece: Existing TrackedPiece to update
            bbox: New bounding box
            contour: New contour
            current_time: Current timestamp
        """
        # Update basic properties
        piece.bbox = bbox
        piece.contour = contour
        piece.last_updated = current_time
        piece.update_count += 1
        piece.updated = True

        # OPTIMIZATION: Update all derived properties at once
        piece.update_derived_properties()

        # NEW: Update motion tracking (velocity and trajectory)
        piece.update_motion(piece.center, current_time)

        # Check if piece is now fully in frame
        if not piece.fully_in_frame:
            if piece.left_edge >= (self.entry_zone[1] - self.roi[0]):
                piece.fully_in_frame = True
                logger.info(f"Piece {piece.id} is now fully in frame")

    def _create_new_piece(self, bbox: Tuple[int, int, int, int],
                         contour: np.ndarray, current_time: float):
        """
        Create a new tracked piece.

        Args:
            bbox: Bounding box of new piece
            contour: Contour of new piece
            current_time: Current timestamp
        """
        new_piece = TrackedPiece(
            id=self.piece_id_counter,
            bbox=bbox,
            contour=contour,
            first_detected=current_time,
            last_updated=current_time,
            fully_in_frame=True  # We only create pieces that are fully visible
        )

        self.tracked_pieces.append(new_piece)
        self.piece_id_counter += 1
        logger.info(f"New piece detected: ID {new_piece.id}")

    def _remove_stale_pieces(self, current_time: float):
        """
        Remove pieces that haven't been seen recently.

        Args:
            current_time: Current timestamp
        """
        timeout = self.config.get("track_timeout", 1.0)

        # Remove pieces that haven't been updated recently
        self.tracked_pieces = [
            p for p in self.tracked_pieces
            if (current_time - p.last_updated) < timeout
        ]

    def _update_zone_status(self):
        """
        Update which zone each piece is in.
        This determines if pieces are in entry, valid, or exit zones.
        """
        self.pieces_in_exit_zone = []

        for piece in self.tracked_pieces:
            # Get piece position relative to full frame
            piece_center_x = piece.center[0] + self.roi[0]

            # Check which zone the piece is in
            if piece_center_x >= self.exit_zone[0]:
                # Piece is in exit zone
                if not piece.in_exit_zone:
                    piece.in_exit_zone = True
                    piece.exit_zone_entry_time = time.time()
                    logger.info(f"Piece {piece.id} entered exit zone")
                self.pieces_in_exit_zone.append(piece)
            else:
                # Piece is not in exit zone
                piece.in_exit_zone = False
                piece.exit_zone_entry_time = None

    def _update_exit_predictions(self):
        """
        Calculate when each piece will reach the drop-off point.

        This allows the system to prepare the servo in advance rather than
        reacting at the last moment when a piece enters the exit zone.
        """
        for piece in self.tracked_pieces:
            # Only predict for pieces moving toward exit (positive x velocity)
            if piece.velocity[0] > 0:
                # Calculate distance to drop-off point
                exit_x = self.exit_zone[1]  # Rightmost edge of exit zone
                current_x = piece.center[0] + self.roi[0]  # Piece position in frame
                distance_to_exit = exit_x - current_x

                # Calculate time until piece reaches drop-off
                # Time = Distance / Speed
                time_to_exit = distance_to_exit / piece.velocity[0]

                # Store predicted exit time
                piece.predicted_exit_time = time.time() + time_to_exit

                # Log if piece will exit soon
                if time_to_exit < 2.0:  # Less than 2 seconds until drop
                    logger.debug(f"Piece {piece.id} will exit in {time_to_exit:.1f} seconds")
            else:
                # Piece not moving toward exit or stationary
                piece.predicted_exit_time = None

    def get_prioritized_exit_pieces(self) -> List[TrackedPiece]:
        """
        Get pieces sorted by urgency (closest to falling off conveyor).

        This method helps the servo controller plan movements by knowing
        which pieces will need sorting first.

        Returns:
            List of pieces sorted by predicted exit time (soonest first)
        """
        exit_pieces = []

        for piece in self.tracked_pieces:
            # Only include pieces with valid exit predictions
            if piece.predicted_exit_time is not None:
                # Check if piece will exit within reasonable timeframe (10 seconds)
                time_until_exit = piece.predicted_exit_time - time.time()
                if 0 < time_until_exit < 10.0:
                    exit_pieces.append(piece)

        # Sort by predicted exit time (soonest first)
        exit_pieces.sort(key=lambda p: p.predicted_exit_time)

        return exit_pieces

    def get_piece_time_to_exit(self, piece_id: int) -> Optional[float]:
        """
        Get the time remaining until a specific piece falls off the conveyor.

        This is useful for the servo controller to know exactly when to trigger.

        Args:
            piece_id: ID of the piece to check

        Returns:
            Time in seconds until piece exits, or None if piece not found
        """
        with self.lock:
            for piece in self.tracked_pieces:
                if piece.id == piece_id:
                    if piece.predicted_exit_time:
                        return piece.predicted_exit_time - time.time()
                    else:
                        return None
            return None

    # ========================================================================
    # CAPTURE AND PROCESSING SECTION
    # ========================================================================
    # These methods determine when to capture images and trigger processing

    def _check_and_trigger_captures(self, frame: np.ndarray):
        """
        Check if any pieces need to be captured or processed.

        This method handles two types of captures:
        1. Exit zone pieces (high priority for servo triggering)
        2. Valid zone pieces (normal capture for identification)

        Args:
            frame: Current video frame
        """
        current_time = time.time()

        # First, check exit zone for high-priority pieces
        if self.exit_zone_enabled:
            exit_piece = self._check_exit_zone_trigger()
            if exit_piece:
                self._queue_piece_for_processing(exit_piece, frame, priority=0)
                return  # Process one piece at a time

        # Then check for normal captures in valid zone
        if current_time - self.last_capture_time >= self.capture_cooldown:
            capture_piece = self._check_for_normal_capture()
            if capture_piece:
                self._queue_piece_for_processing(capture_piece, frame, priority=1)

    def _check_exit_zone_trigger(self) -> Optional[TrackedPiece]:
        """
        Check if a piece in exit zone should trigger processing.

        This enhanced version uses exit time predictions to make better
        decisions about when to trigger the servo.

        Returns:
            Piece to process or None
        """
        if not self.pieces_in_exit_zone:
            return None

        current_time = time.time()

        # Find pieces that need urgent processing
        urgent_pieces = []
        for piece in self.pieces_in_exit_zone:
            # Skip if already being handled
            if piece.being_processed or piece.captured:
                continue

            # Check if we have exit prediction
            if piece.predicted_exit_time:
                time_until_exit = piece.predicted_exit_time - current_time

                # If piece will fall off soon, mark as urgent
                # Use fall_time as the threshold for "urgent"
                if time_until_exit <= self.fall_time:
                    urgent_pieces.append((time_until_exit, piece))

        # If no urgent pieces, return None
        if not urgent_pieces:
            return None

        # Sort by urgency (soonest exit first)
        urgent_pieces.sort(key=lambda x: x[0])
        priority_piece = urgent_pieces[0][1]

        # Check if we've waited long enough since last trigger
        if current_time - self.last_servo_trigger_time < self.fall_time:
            # Still processing previous piece
            return None

        logger.info(f"Exit zone trigger for piece {priority_piece.id} "
                    f"(exits in {urgent_pieces[0][0]:.1f}s)")
        self.last_servo_trigger_time = current_time
        return priority_piece

    def _check_for_normal_capture(self) -> Optional[TrackedPiece]:
        """
        Check if any piece in the valid zone should be captured.

        Returns:
            Piece to capture or None
        """
        for piece in self.tracked_pieces:
            # Skip if already processed or not ready
            if piece.captured or piece.being_processed or not piece.fully_in_frame:
                continue

            # Need minimum number of updates to ensure stable tracking
            if piece.update_count < self.min_updates:
                continue

            # Check if in valid zone
            piece_center_x = piece.center[0] + self.roi[0]
            if self.valid_zone[0] <= piece_center_x <= self.valid_zone[1]:
                logger.info(f"Normal capture for piece {piece.id}")
                self.last_capture_time = time.time()
                return piece

        return None

    def _queue_piece_for_processing(self, piece: TrackedPiece, frame: np.ndarray, priority: int):
        """
        Queue a piece for processing by the thread manager.

        Args:
            piece: Piece to process
            frame: Current video frame
            priority: Processing priority (0 = highest)
        """
        # Mark as being processed
        piece.being_processed = True
        piece.processing_start_time = time.time()

        # Crop piece image from frame
        cropped_image = self._crop_piece_image(frame, piece)

        # Send to thread manager if available
        if hasattr(self, 'thread_manager') and self.thread_manager:
            self.thread_manager.add_message(
                piece_id=piece.id,
                image=cropped_image,
                frame_number=self.frame_count,
                position=piece.bbox,
                priority=priority,
                in_exit_zone=piece.in_exit_zone
            )
            logger.info(f"Queued piece {piece.id} with priority {priority}")

    def _crop_piece_image(self, frame: np.ndarray, piece: TrackedPiece) -> np.ndarray:
        """
        Extract the piece image from the full frame.

        Args:
            frame: Full video frame
            piece: Piece to crop

        Returns:
            Cropped image of the piece
        """
        # Get piece position in full frame coordinates
        x, y, w, h = piece.bbox
        roi_x, roi_y = self.roi[0], self.roi[1]

        # Convert to full frame coordinates
        abs_x = roi_x + x
        abs_y = roi_y + y

        # Add padding around piece
        padding = self.config.get("crop_padding", 20)
        x1 = max(0, abs_x - padding)
        y1 = max(0, abs_y - padding)
        x2 = min(frame.shape[1], abs_x + w + padding)
        y2 = min(frame.shape[0], abs_y + h + padding)

        # Crop and return
        return frame[y1:y2, x1:x2].copy()

    # ========================================================================
    # DATA OUTPUT SECTION
    # ========================================================================
    # These methods provide data to other modules (mainly GUI)

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get all detection data for visualization.
        This is what the GUI uses to draw the detection overlay.

        Returns:
            Dictionary with all visualization data
        """
        with self.lock:
            if self.roi is None:
                return {"error": "ROI not configured"}

            # Prepare tracked pieces data
            pieces_data = []
            for piece in self.tracked_pieces:
                pieces_data.append({
                    "id": piece.id,
                    "bbox": piece.bbox,  # In ROI coordinates
                    "fully_in_frame": piece.fully_in_frame,
                    "being_processed": piece.being_processed,
                    "captured": piece.captured,
                    "in_exit_zone": piece.in_exit_zone,
                    "update_count": piece.update_count,
                    "is_priority": piece == self.current_priority_piece,
                    "center": piece.center
                })

            return {
                "roi": self.roi,
                "entry_zone": self.entry_zone,
                "valid_zone": self.valid_zone,
                "exit_zone": self.exit_zone,
                "tracked_pieces": pieces_data,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "pieces_in_exit_zone": len(self.pieces_in_exit_zone),
                "active_pieces": sum(1 for p in self.tracked_pieces
                                   if not p.captured and not p.being_processed),
                "processing_pieces": sum(1 for p in self.tracked_pieces
                                       if p.being_processed)
            }

    def update_piece_result(self, piece_id: int, result: Dict[str, Any]):
        """
        Update a piece with processing results.
        Called when API identification is complete.

        Args:
            piece_id: ID of processed piece
            result: Processing results including bin assignment
        """
        with self.lock:
            for piece in self.tracked_pieces:
                if piece.id == piece_id:
                    piece.target_bin = result.get("bin_number")
                    piece.captured = True
                    piece.being_processed = False
                    logger.info(f"Updated piece {piece_id} with bin {piece.target_bin}")
                    break

    # ========================================================================
    # UTILITY SECTION
    # ========================================================================
    # Helper methods and cleanup

    def set_thread_manager(self, thread_manager):
        """
        Set the thread manager for queuing pieces.

        Args:
            thread_manager: ThreadManager instance
        """
        self.thread_manager = thread_manager
        logger.info("Thread manager connected to detector")

    def release(self):
        """
        Clean up resources when shutting down.
        """
        logger.info("Releasing detector resources")
        with self.lock:
            self.bg_subtractor = None
            self.tracked_pieces = []
            self.pieces_in_exit_zone = []


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
# This is the main way to create a detector instance

def create_detector(detector_type: str = "conveyor", config_manager=None) -> ConveyorDetector:
    """
    Create a detector instance of the specified type.

    Args:
        detector_type: Type of detector (currently only "conveyor" is supported)
        config_manager: Configuration manager with settings

    Returns:
        ConveyorDetector instance

    Raises:
        ValueError: If detector_type is not supported
    """
    logger.info(f"Creating detector of type: {detector_type}")

    if detector_type == "conveyor":
        return ConveyorDetector(config_manager)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}. Only 'conveyor' is supported.")


# ============================================================================
# STANDALONE TESTING
# ============================================================================
# This section only runs if the file is executed directly (not imported)

if __name__ == "__main__":
    """Test detector with real camera feed"""
    import sys

    logging.basicConfig(level=logging.INFO)

    # Import camera module for real testing
    from camera_module import create_camera

    print("Detector Module Camera Test")
    print("===========================")
    print("Press 'q' to quit, 's' to simulate a piece")

    # Create real camera and detector
    camera = create_camera()
    detector = create_detector("conveyor")

    # Get first frame to set ROI
    camera.initialize()
    ret, frame = camera.cap.read()

    if ret:
        # Set ROI (you might want to adjust these values)
        detector.set_roi((400, 200, 800, 400), frame)

        # Process frames from camera
        while True:
            ret, frame = camera.cap.read()
            if not ret:
                break

            # Process frame
            result = detector.process_frame_for_consumer(frame)

            # Show statistics
            print(f"\rFPS: {result.get('fps', 0):.1f} | "
                  f"Active: {result.get('active_pieces', 0)} | "
                  f"Processing: {result.get('processing_pieces', 0)} | "
                  f"Exit Zone: {result.get('pieces_in_exit_zone', 0)}",
                  end='')

            # Display frame with basic visualization
            display_frame = frame.copy()
            if 'roi' in result:
                x, y, w, h = result['roi']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Detector Test', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n[Simulating piece detection]")

    # Cleanup
    cv2.destroyAllWindows()
    camera.release()
    detector.release()
    print("\nTest complete!")