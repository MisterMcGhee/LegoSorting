"""
capture_controller.py - Controls image capture and preprocessing for piece identification

This module determines when TrackedPiece objects should have their images captured
and handles the preprocessing needed to create ready-to-identify image packages.

Key Responsibilities:
- Evaluate piece readiness for capture (fully visible, in valid zone, stable tracking)
- Apply capture timing controls and cooldowns
- Crop and preprocess images with piece ID overlay
- Create complete image packages ready for API identification
- Log capture events with position data

Piece Capture Criteria:
- Piece must be fully visible in frame (fully_in_frame = True)
- Piece must be in valid capture zone (via zone_manager filtering)
- Piece must have stable tracking (minimum update count)
- Piece must not already be captured or being processed

Image Processing:
- Crops frame to piece bounding box with configurable padding
- Adds space at top of image for piece ID number overlay
- Saves as JPG format with piece ID burned into the image
- Maintains original aspect ratio, no size standardization

This module does NOT:
- Communicate with identification APIs (threading module handles that)
- Perform piece identification (API module responsibility)
- Make sorting decisions (servo controller responsibility)
- Handle coordinate conversions (detector_coordinator provides frame coordinates)
"""

import cv2
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from detector.detector_data_models import TrackedPiece, CapturePackage
from pathlib import Path
from detector.detector_data_models import TrackedPiece
from detector.zone_manager import ZoneManager
from enhanced_config_manager import ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN CAPTURE CONTROLLER CLASS
# ============================================================================

class CaptureController:
    """
    Controls the capture and preprocessing of piece images for identification.

    This module bridges the gap between piece tracking and identification by
    determining when pieces should be captured and creating processed image
    packages ready for API analysis.

    The controller works with pieces in frame coordinates (provided by
    detector_coordinator) to ensure accurate position logging and image cropping.
    """

    def __init__(self, config_manager, zone_manager: ZoneManager):
        """
        Initialize the capture controller with configuration and zone manager.

        Args:
            config_manager: Enhanced config manager instance (required)
            zone_manager: Zone manager for filtering pieces by zone (required)

        Raises:
            ValueError: If config_manager or zone_manager is None
        """
        if not config_manager:
            raise ValueError("CaptureController requires a valid config_manager")
        if not zone_manager:
            raise ValueError("CaptureController requires a valid zone_manager")

        logger.info("Initializing CaptureController")

        self.config_manager = config_manager
        self.zone_manager = zone_manager

        # Capture state tracking
        self.last_capture_time = 0.0
        self.captured_piece_ids = set()  # Track which pieces have been captured
        self.capture_count = 0

        # Load configuration from unified system
        self._load_configuration()

        # Create capture directory if needed
        self._setup_capture_directory()

        logger.info("CaptureController initialized successfully")

    def _load_configuration(self):
        """
        Load capture control parameters from enhanced_config_manager.

        Configuration comes from the detector module since capture
        parameters are closely related to detection and tracking settings.
        """
        # Get detector configuration from unified system
        detector_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR.value)

        # Piece readiness criteria
        self.min_stability_updates = detector_config.get("min_stability_updates", 5)
        self.capture_valid_zone_only = detector_config.get("capture_valid_zone_only", True)

        # Capture timing controls
        self.capture_cooldown_seconds = detector_config.get("capture_cooldown_seconds", 0.5)
        self.max_concurrent_processing = detector_config.get("max_concurrent_processing", 3)

        # Image processing parameters
        self.crop_padding = detector_config.get("crop_padding", 20)
        self.id_label_height = detector_config.get("id_label_height", 40)
        self.id_font_scale = detector_config.get("id_font_scale", 1.5)
        self.id_font_thickness = detector_config.get("id_font_thickness", 2)

        # File management
        self.save_captured_images = detector_config.get("save_captured_images", True)
        self.capture_directory = detector_config.get("capture_directory", "captured_pieces")

        logger.info("Capture configuration loaded from enhanced_config_manager")
        logger.info(f"Capture settings: cooldown={self.capture_cooldown_seconds}s, "
                    f"min_updates={self.min_stability_updates}, padding={self.crop_padding}px")

    def _setup_capture_directory(self):
        """
        Create the capture directory if it doesn't exist and image saving is enabled.
        """
        if self.save_captured_images:
            capture_path = Path(self.capture_directory)
            capture_path.mkdir(exist_ok=True)
            logger.info(f"Capture directory ready: {capture_path.absolute()}")

    # ========================================================================
    # MAIN CAPTURE PROCESSING PIPELINE
    # ========================================================================

    def check_and_process_captures(self, frame: np.ndarray,
                                   tracked_pieces: List[TrackedPiece]) -> List[CapturePackage]:
        """
        Check for capture opportunities and process any eligible pieces.

        This is the main entry point for capture processing. It evaluates
        all tracked pieces and creates capture packages for eligible pieces.

        Args:
            frame: Current camera frame in frame coordinates
            tracked_pieces: List of TrackedPiece objects in frame coordinates

        Returns:
            List of CapturePackage objects for newly captured pieces
        """
        current_time = time.time()
        capture_packages = []

        # Check if we're within capture cooldown period
        if current_time - self.last_capture_time < self.capture_cooldown_seconds:
            logger.debug("Within capture cooldown period, skipping capture check")
            return capture_packages

        # Get pieces eligible for capture
        eligible_pieces = self._get_capture_eligible_pieces(tracked_pieces)

        if not eligible_pieces:
            logger.debug("No pieces eligible for capture")
            return capture_packages

        # Check processing capacity
        currently_processing = sum(1 for piece in tracked_pieces if piece.being_processed)
        available_capacity = self.max_concurrent_processing - currently_processing

        if available_capacity <= 0:
            logger.debug(f"Processing capacity full ({currently_processing}/{self.max_concurrent_processing})")
            return capture_packages

        # Process captures up to available capacity
        pieces_to_capture = eligible_pieces[:available_capacity]

        for piece in pieces_to_capture:
            try:
                # Create capture package
                capture_package = self._create_capture_package(frame, piece, current_time)

                if capture_package:
                    # Update piece status
                    piece.captured = True
                    piece.being_processed = True
                    piece.processing_start_time = current_time

                    # Track capture
                    self.captured_piece_ids.add(piece.id)
                    self.last_capture_time = current_time
                    self.capture_count += 1

                    capture_packages.append(capture_package)

                    logger.info(f"Captured piece {piece.id} at position {capture_package.capture_position}")

            except Exception as e:
                logger.error(f"Failed to capture piece {piece.id}: {e}")
                continue

        if capture_packages:
            logger.info(f"Processed {len(capture_packages)} captures this frame")

        return capture_packages

    # ========================================================================
    # PIECE ELIGIBILITY ASSESSMENT
    # ========================================================================

    def _get_capture_eligible_pieces(self, tracked_pieces: List[TrackedPiece]) -> List[TrackedPiece]:
        """
        Filter tracked pieces to find those eligible for capture.

        Applies all capture criteria:
        - Fully visible in frame
        - In valid capture zone (if zone filtering enabled)
        - Stable tracking (sufficient update count)
        - Not already captured or being processed

        Args:
            tracked_pieces: List of TrackedPiece objects to evaluate

        Returns:
            List of pieces eligible for capture, sorted by readiness
        """
        eligible_pieces = []

        for piece in tracked_pieces:
            # Check basic eligibility criteria
            if not self._is_piece_capture_ready(piece):
                continue

            # Check zone eligibility if zone filtering is enabled
            if self.capture_valid_zone_only:
                if not self._is_piece_in_valid_zone(piece):
                    continue

            eligible_pieces.append(piece)

        # Sort by readiness (prioritize pieces with more updates = more stable)
        eligible_pieces.sort(key=lambda p: p.update_count, reverse=True)

        logger.debug(f"Found {len(eligible_pieces)} capture-eligible pieces")
        return eligible_pieces

    def _is_piece_capture_ready(self, piece: TrackedPiece) -> bool:
        """
        Check if a piece meets basic capture readiness criteria.

        Args:
            piece: TrackedPiece to evaluate

        Returns:
            True if piece is ready for capture
        """
        # Must be fully visible
        if not piece.fully_in_frame:
            return False

        # Must have stable tracking
        if piece.update_count < self.min_stability_updates:
            return False

        # Must not already be captured or processing
        if piece.captured or piece.being_processed:
            return False

        # Must not have been captured before (in case flags were reset)
        if piece.id in self.captured_piece_ids:
            return False

        return True

    def _is_piece_in_valid_zone(self, piece: TrackedPiece) -> bool:
        """
        Check if a piece is in the valid capture zone.

        Uses the zone status flag that is maintained by the zone_manager
        during its zone update process. This eliminates the need for
        coordinate calculations in the capture controller.

        Args:
            piece: TrackedPiece to check

        Returns:
            True if piece is in valid zone
        """
        return piece.in_valid_zone

    # ========================================================================
    # IMAGE CAPTURE AND PROCESSING
    # ========================================================================

    def _create_capture_package(self, frame: np.ndarray, piece: TrackedPiece,
                                timestamp: float) -> Optional[CapturePackage]:
        """
        Create a complete capture package from a piece and frame.

        This handles all image processing steps:
        - Crop frame to piece bounding box with padding
        - Add space for piece ID label
        - Overlay piece ID number on image
        - Create capture package with metadata

        Args:
            frame: Current camera frame
            piece: TrackedPiece to capture (in frame coordinates)
            timestamp: Capture timestamp

        Returns:
            CapturePackage object or None if processing failed
        """
        try:
            # Extract piece information
            piece_bbox = piece.bbox  # Already in frame coordinates
            piece_center = piece.center  # Already in frame coordinates

            # Calculate crop region with padding and label space
            crop_region = self._calculate_crop_region(frame, piece_bbox)

            # Crop the image
            cropped_image = self._crop_piece_image(frame, crop_region)

            if cropped_image is None:
                logger.error(f"Failed to crop image for piece {piece.id}")
                return None

            # Add piece ID label to image
            labeled_image = self._add_piece_id_label(cropped_image, piece.id)

            # Create capture package
            capture_package = CapturePackage(
                piece_id=piece.id,
                processed_image=labeled_image,
                capture_timestamp=timestamp,
                capture_position=(int(piece_center[0]), int(piece_center[1])),
                original_bbox=piece_bbox
            )

            # Save image if enabled
            if self.save_captured_images:
                self._save_capture_image(capture_package)

            return capture_package

        except Exception as e:
            logger.error(f"Failed to create capture package for piece {piece.id}: {e}")
            return None

    def _calculate_crop_region(self, frame: np.ndarray,
                               piece_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate the crop region for a piece with padding and label space.

        Adds padding around the piece bounding box and extra space at the top
        for the piece ID label overlay.

        Args:
            frame: Camera frame for boundary checking
            piece_bbox: Piece bounding box (x, y, width, height)

        Returns:
            Crop region as (x, y, width, height) clamped to frame bounds
        """
        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = piece_bbox

        # Calculate crop bounds with padding
        crop_x1 = max(0, x - self.crop_padding)
        crop_y1 = max(0, y - self.crop_padding - self.id_label_height)  # Extra space for label
        crop_x2 = min(frame_width, x + w + self.crop_padding)
        crop_y2 = min(frame_height, y + h + self.crop_padding)

        # Convert to (x, y, width, height) format
        crop_x = crop_x1
        crop_y = crop_y1
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1

        logger.debug(f"Calculated crop region: ({crop_x}, {crop_y}, {crop_width}, {crop_height})")
        return (crop_x, crop_y, crop_width, crop_height)

    def _crop_piece_image(self, frame: np.ndarray,
                          crop_region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop the piece image from the frame.

        Args:
            frame: Full camera frame
            crop_region: Region to crop (x, y, width, height)

        Returns:
            Cropped image or None if cropping failed
        """
        try:
            x, y, w, h = crop_region

            # Validate crop region
            if w <= 0 or h <= 0:
                logger.error(f"Invalid crop dimensions: {w}x{h}")
                return None

            # Crop the image
            cropped = frame[y:y + h, x:x + w].copy()

            logger.debug(f"Cropped image size: {cropped.shape}")
            return cropped

        except Exception as e:
            logger.error(f"Error cropping image: {e}")
            return None

    def _add_piece_id_label(self, image: np.ndarray, piece_id: int) -> np.ndarray:
        """
        Add piece ID label overlay to the cropped image.

        Places the piece ID number at the top of the image in white text
        with black outline for good visibility.

        Args:
            image: Cropped piece image
            piece_id: Piece ID number to overlay

        Returns:
            Image with piece ID label overlay
        """
        try:
            # Create a copy to avoid modifying the original
            labeled_image = image.copy()

            # Prepare text
            text = str(piece_id)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, self.id_font_scale, self.id_font_thickness
            )

            # Position text at top center
            x = (image.shape[1] - text_width) // 2
            y = text_height + 10  # 10 pixels from top

            # Draw text with outline for visibility
            # Black outline
            cv2.putText(labeled_image, text, (x - 1, y - 1), font,
                        self.id_font_scale, (0, 0, 0), self.id_font_thickness + 1)
            cv2.putText(labeled_image, text, (x + 1, y + 1), font,
                        self.id_font_scale, (0, 0, 0), self.id_font_thickness + 1)

            # White text
            cv2.putText(labeled_image, text, (x, y), font,
                        self.id_font_scale, (255, 255, 255), self.id_font_thickness)

            logger.debug(f"Added piece ID {piece_id} label to image")
            return labeled_image

        except Exception as e:
            logger.error(f"Error adding piece ID label: {e}")
            return image  # Return original image if labeling fails

    def _save_capture_image(self, capture_package: CapturePackage):
        """
        Save a capture package image to disk.

        Creates filename with timestamp and piece ID for easy identification.

        Args:
            capture_package: CapturePackage to save
        """
        try:
            # Create filename with timestamp and piece ID
            timestamp_str = str(int(capture_package.capture_timestamp))
            filename = f"piece_{capture_package.piece_id}_{timestamp_str}.jpg"
            file_path = Path(self.capture_directory) / filename

            # Save the image
            success = capture_package.save_image(str(file_path))

            if success:
                logger.debug(f"Saved capture to {file_path}")
            else:
                logger.error(f"Failed to save capture to {file_path}")

        except Exception as e:
            logger.error(f"Error saving capture image: {e}")

    # ========================================================================
    # STATUS AND MONITORING
    # ========================================================================

    def get_capture_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about capture operations for monitoring.

        Returns:
            Dictionary with capture performance metrics
        """
        return {
            "total_captures": self.capture_count,
            "captured_piece_ids": len(self.captured_piece_ids),
            "last_capture_time": self.last_capture_time,
            "capture_cooldown": self.capture_cooldown_seconds,
            "max_concurrent_processing": self.max_concurrent_processing,
            "save_images_enabled": self.save_captured_images,
            "capture_directory": self.capture_directory
        }

    def reset_capture_state(self):
        """
        Reset capture state for system restart.

        Clears captured piece tracking but preserves configuration.
        """
        logger.info("Resetting capture controller state")
        self.last_capture_time = 0.0
        self.captured_piece_ids.clear()
        self.capture_count = 0
        logger.info("Capture state reset complete")

    def update_configuration(self):
        """
        Reload configuration from the config manager.

        Allows dynamic updates to capture parameters without restarting.
        """
        logger.info("Reloading capture configuration")
        self._load_configuration()
        self._setup_capture_directory()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_capture_controller(config_manager, zone_manager: ZoneManager) -> CaptureController:
    """
    Create a CaptureController instance with unified configuration.

    This is the standard way to create a capture controller using the
    enhanced_config_manager and zone_manager for consistent operation.

    Args:
        config_manager: Enhanced config manager instance (required)
        zone_manager: Zone manager for piece filtering (required)

    Returns:
        Configured CaptureController instance

    Raises:
        ValueError: If config_manager or zone_manager is None
    """
    if config_manager is None:
        raise ValueError("create_capture_controller requires a valid config_manager")
    if zone_manager is None:
        raise ValueError("create_capture_controller requires a valid zone_manager")

    logger.info("Creating CaptureController with enhanced_config_manager")
    return CaptureController(config_manager, zone_manager)


# ============================================================================
# TESTING AND UTILITIES
# ============================================================================

def create_test_frame_with_pieces(width: int = 640, height: int = 480,
                                  piece_positions: List[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a test frame with synthetic pieces for capture testing.

    Args:
        width: Frame width
        height: Frame height
        piece_positions: List of (x, y) positions for test pieces

    Returns:
        Synthetic frame with white rectangles representing pieces
    """
    if piece_positions is None:
        piece_positions = [(200, 200), (400, 250)]

    # Create black background frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add white rectangles as test pieces
    for i, (x, y) in enumerate(piece_positions):
        cv2.rectangle(frame, (x - 20, y - 15), (x + 20, y + 15), (255, 255, 255), -1)
        # Add some detail to make pieces more realistic
        cv2.rectangle(frame, (x - 15, y - 10), (x + 15, y + 10), (200, 200, 200), 1)

    return frame


if __name__ == "__main__":
    """
    Test the capture controller with synthetic data.

    This demonstrates capture eligibility assessment, image processing,
    and capture package creation.
    """
    import sys
    from detector_data_models import TrackedPiece, Detection, create_tracked_piece_from_detection
    from zone_manager import create_zone_manager, ZoneManager

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing CaptureController with synthetic data")


    # Create mock config manager
    class MockConfigManager:
        """Mock config manager with test configuration"""

        def get_module_config(self, module_name):
            configs = {
                "detector": {
                    "min_stability_updates": 5,
                    "capture_valid_zone_only": True,
                    "capture_cooldown_seconds": 0.5,
                    "max_concurrent_processing": 3,
                    "crop_padding": 20,
                    "id_label_height": 40,
                    "id_font_scale": 1.5,
                    "id_font_thickness": 2,
                    "save_captured_images": False,  # Disable for testing
                    "capture_directory": "test_captures",
                    "entry_zone_percent": 0.15,
                    "exit_zone_percent": 0.15,
                    "zone_transition_debounce": 0.1
                }
            }
            return configs.get(module_name, {})


    try:
        # Create dependencies
        mock_config = MockConfigManager()
        zone_manager = create_zone_manager(mock_config)

        # Create capture controller
        capture_controller = create_capture_controller(mock_config, zone_manager)

        # Create test frame
        test_frame = create_test_frame_with_pieces(640, 480, [(300, 200), (500, 250)])

        # Create test tracked pieces
        test_pieces = []
        for i, (x, y) in enumerate([(300, 200), (500, 250)]):
            # Create detection
            bbox = (x - 20, y - 15, 40, 30)
            contour = np.array([[[x - 20, y - 15]], [[x + 20, y - 15]],
                                [[x + 20, y + 15]], [[x - 20, y + 15]]])
            detection = Detection(contour=contour, bbox=bbox, area=1200, confidence=0.9)

            # Create tracked piece
            piece = create_tracked_piece_from_detection(detection, i + 1, time.time())
            piece.fully_in_frame = True
            piece.update_count = 10  # Make it stable
            test_pieces.append(piece)

        print(f"\nCaptureController Test Results:")
        print(f"=" * 50)

        # Test capture processing
        capture_packages = capture_controller.check_and_process_captures(test_frame, test_pieces)

        print(f"Capture packages created: {len(capture_packages)}")

        for package in capture_packages:
            print(f"  Piece {package.piece_id}:")
            print(f"    Position: {package.capture_position}")
            print(f"    Image size: {package.processed_image.shape}")
            print(f"    Original bbox: {package.original_bbox}")

        # Show capture statistics
        stats = capture_controller.get_capture_statistics()
        print(f"\nCapture Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        logger.info("CaptureController test completed successfully")

    except Exception as e:
        logger.error(f"CaptureController test failed: {e}")
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()