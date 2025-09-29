"""
vision_processor.py - Pure computer vision processing for moving object detection

This module is the "stateless black box" of the detection system. It has one job:
analyze a camera frame and report "something moved here."

Key characteristics:
- STATELESS: No memory between frames (except background model)
- FOCUSED: Only does computer vision, no tracking or decision-making
- PURE FUNCTION: Same input always produces same output
- SIMPLE INTERFACE: Takes frame + ROI, returns Detection objects
- UNIFIED CONFIG: Uses enhanced_config_manager for all settings

This module does NOT:
- Remember pieces between frames (that's the tracker's job)
- Assign piece IDs (that's the tracker's job)
- Make capture decisions (that's the capture controller's job)
- Handle zone logic (that's the zone manager's job)
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from data_models import Detection, RegionOfInterest
from enhanced_config_manager import ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN VISION PROCESSOR CLASS
# ============================================================================

class VisionProcessor:
    """
    Stateless computer vision processor for detecting moving objects.

    This class maintains only the background subtraction model between frames.
    All other processing is stateless - the same frame will always produce
    the same Detection results.

    Configuration is loaded from the unified enhanced_config_manager system.
    """

    def __init__(self, config_manager):
        """
        Initialize the vision processor with unified configuration.

        Args:
            config_manager: Enhanced config manager instance (required)

        Raises:
            ValueError: If config_manager is None or invalid
        """
        if not config_manager:
            raise ValueError("VisionProcessor requires a valid config_manager")

        logger.info("Initializing VisionProcessor")

        self.config_manager = config_manager
        self.bg_subtractor = None  # Will be initialized when ROI is set
        self._roi = None  # Current region of interest

        # Load all configuration from unified system
        self._load_configuration()

        logger.info("VisionProcessor initialized with unified configuration")

    def _load_configuration(self):
        """
        Load all vision processing parameters from enhanced_config_manager.

        This centralizes configuration loading and ensures consistency with
        the rest of the system. All vision parameters come from the detector
        module configuration.

        Raises:
            Exception: If configuration cannot be loaded
        """
        # Get detector configuration from unified system
        detector_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR.value)

        # Size filtering parameters
        self.min_piece_area = detector_config.get("min_area", 1000)
        self.max_piece_area = detector_config.get("max_area", 50000)
        self.min_aspect_ratio = detector_config.get("min_aspect_ratio", 0.3)
        self.max_aspect_ratio = detector_config.get("max_aspect_ratio", 3.0)

        # Background subtraction parameters
        self.bg_history = detector_config.get("bg_history", 500)
        self.bg_threshold = detector_config.get("bg_threshold", 800.0)
        self.learn_rate = detector_config.get("learn_rate", 0.005)

        # Noise reduction parameters
        self.gaussian_blur_size = detector_config.get("gaussian_blur_size", 7)
        self.morph_kernel_size = detector_config.get("morph_kernel_size", 5)

        # Quality filtering parameters
        self.min_contour_points = detector_config.get("min_contour_points", 5)

        logger.info("Vision configuration loaded from enhanced_config_manager")

    # ========================================================================
    # ROI SETUP AND MANAGEMENT
    # ========================================================================

    def set_roi(self, roi: RegionOfInterest, sample_frame: np.ndarray):
        """
        Set the region of interest and initialize background subtraction.

        This method trains the background model using the provided sample frame.
        The background model learns what the "empty" conveyor looks like.

        Args:
            roi: Region of interest where detection should occur
            sample_frame: Initial frame to train background model
        """
        logger.info(f"Setting ROI to {roi.to_tuple()}")

        self._roi = roi

        # Create background subtractor with configured parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=self.bg_history,
            dist2Threshold=self.bg_threshold,
            detectShadows=False  # Shadows can interfere with piece detection
        )

        # Train the background model with initial frame
        self._train_background_model(sample_frame)

        logger.info("ROI set and background model trained")

    def _train_background_model(self, frame: np.ndarray):
        """
        Train the background subtraction model with an initial frame.

        This builds the model of what the "empty" conveyor belt looks like.
        We process the same frame multiple times to build a stable background.

        Args:
            frame: Sample frame showing empty conveyor
        """
        if self._roi is None or self.bg_subtractor is None:
            raise ValueError("ROI must be set before training background model")

        # Extract ROI from frame
        roi_frame = self._extract_roi_from_frame(frame)

        # Process frame multiple times to build stable background model
        # Higher learning rate initially to build model quickly
        for i in range(30):
            learning_rate = 0.1 if i < 10 else 0.05  # Start fast, then slow down
            self.bg_subtractor.apply(roi_frame, learningRate=learning_rate)

        logger.info("Background model training complete")

    # ========================================================================
    # MAIN PROCESSING PIPELINE
    # ========================================================================

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Process a single frame to detect moving objects.

        This is the main entry point for vision processing. It takes a raw
        camera frame and returns a list of detected moving objects.

        Args:
            frame: Raw camera frame as numpy array

        Returns:
            List of Detection objects for moving objects found in frame

        Raises:
            ValueError: If ROI is not configured
        """
        if self._roi is None or self.bg_subtractor is None:
            raise ValueError("ROI must be set before processing frames")

        # Step 1: Extract and preprocess the ROI from the frame
        roi_frame, foreground_mask = self._preprocess_roi_frame(frame)

        # Step 2: Find contours of moving objects in the mask
        contours = self._find_object_contours(foreground_mask)

        # Step 3: Filter and convert contours to Detection objects
        detections = self._convert_contours_to_detections(contours)

        logger.debug(f"Processed frame: found {len(detections)} valid detections")
        return detections

    # ========================================================================
    # FRAME PREPROCESSING
    # ========================================================================

    def _extract_roi_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the region of interest from the full camera frame.

        Args:
            frame: Full camera frame

        Returns:
            Cropped frame containing only the ROI area
        """
        x, y, w, h = self._roi.to_tuple()
        return frame[y:y + h, x:x + w].copy()

    def _preprocess_roi_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the ROI frame to detect moving objects.

        This applies background subtraction and noise reduction to create
        a clean binary mask where white pixels represent moving objects.

        Args:
            frame: Full camera frame

        Returns:
            Tuple of (roi_frame, foreground_mask)
        """
        # Extract ROI from full frame
        roi_frame = self._extract_roi_from_frame(frame)

        # Apply Gaussian blur to reduce camera noise and improve detection
        blurred = cv2.GaussianBlur(
            roi_frame,
            (self.gaussian_blur_size, self.gaussian_blur_size),
            0
        )

        # Apply background subtraction to detect moving objects
        # This creates a mask where white = foreground, black = background
        foreground_mask = self.bg_subtractor.apply(
            blurred,
            learningRate=self.learn_rate
        )

        # Clean up the mask with morphological operations
        cleaned_mask = self._clean_foreground_mask(foreground_mask)

        return roi_frame, cleaned_mask

    def _clean_foreground_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up the foreground mask to remove noise and fill gaps.

        Morphological operations help create cleaner object boundaries:
        - Opening removes small noise specks
        - Closing fills small gaps within objects

        Args:
            mask: Raw foreground mask from background subtraction

        Returns:
            Cleaned binary mask
        """
        # Create morphological kernel
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)

        # Opening: erosion followed by dilation (removes noise)
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Closing: dilation followed by erosion (fills gaps)
        mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

        return mask_cleaned

    # ========================================================================
    # CONTOUR DETECTION AND FILTERING
    # ========================================================================

    def _find_object_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contours of objects in the cleaned foreground mask.

        Args:
            mask: Cleaned binary mask (white = moving objects)

        Returns:
            List of contour arrays representing object boundaries
        """
        # Early exit optimization: if mask is mostly empty, skip expensive contour detection
        white_pixel_count = np.sum(mask > 0)
        if white_pixel_count < self.min_piece_area:
            logger.debug("Insufficient white pixels for any valid pieces, skipping contour detection")
            return []

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        logger.debug(f"Found {len(contours)} raw contours in mask")
        return contours

    def _convert_contours_to_detections(self, contours: List[np.ndarray]) -> List[Detection]:
        """
        Convert raw contours to Detection objects with filtering.

        This applies size and quality filters to remove noise and invalid detections.
        Only contours that pass all filters become Detection objects.

        Args:
            contours: List of contour arrays from OpenCV

        Returns:
            List of Detection objects for valid moving objects
        """
        detections = []

        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            bbox = cv2.boundingRect(contour)

            # Apply size filter
            if not self._is_valid_size(area, bbox):
                continue

            # Apply quality filter
            if not self._is_valid_quality(contour):
                continue

            # Calculate confidence based on contour properties
            confidence = self._calculate_detection_confidence(contour, area, bbox)

            # Create Detection object
            detection = Detection(
                contour=contour,
                bbox=bbox,
                area=area,
                confidence=confidence
            )

            detections.append(detection)

        logger.debug(f"Converted {len(contours)} contours to {len(detections)} valid detections")
        return detections

    # ========================================================================
    # FILTERING LOGIC
    # ========================================================================

    def _is_valid_size(self, area: float, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if detection meets size requirements.

        Args:
            area: Contour area in pixels
            bbox: Bounding box (x, y, width, height)

        Returns:
            True if size is within acceptable range
        """
        # Check area bounds
        if not (self.min_piece_area <= area <= self.max_piece_area):
            return False

        # Check aspect ratio to filter out long thin noise
        x, y, w, h = bbox
        if w == 0 or h == 0:  # Avoid division by zero
            return False

        aspect_ratio = max(w / h, h / w)  # Always >= 1.0
        if aspect_ratio > self.max_aspect_ratio:
            logger.debug(f"Filtered detection with aspect ratio {aspect_ratio:.1f}")
            return False

        return True

    def _is_valid_quality(self, contour: np.ndarray) -> bool:
        """
        Check if contour meets quality requirements.

        Args:
            contour: Contour array from OpenCV

        Returns:
            True if contour quality is acceptable
        """
        # Check if contour has enough points to be meaningful
        if len(contour) < self.min_contour_points:
            return False

        # Additional quality checks could go here:
        # - Contour complexity analysis
        # - Shape regularity checks
        # - Boundary smoothness evaluation

        return True

    def _calculate_detection_confidence(self, contour: np.ndarray, area: float,
                                        bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate confidence score for a detection.

        This analyzes various properties of the detected object to estimate
        how likely it is to be a real Lego piece rather than noise.

        Args:
            contour: Object contour
            area: Contour area in pixels
            bbox: Bounding box

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0

        # Reduce confidence for very small pieces (might be noise)
        size_ratio = area / self.max_piece_area
        if size_ratio < 0.1:  # Very small relative to max size
            confidence *= 0.7

        # Reduce confidence for pieces with extreme aspect ratios
        x, y, w, h = bbox
        if w > 0 and h > 0:
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > 5.0:  # Very elongated
                confidence *= 0.8

        # Additional confidence factors could include:
        # - Contour regularity (smooth vs jagged edges)
        # - Fill ratio (contour area vs bounding box area)
        # - Position within ROI (center vs edges)

        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_background_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current background model.

        This is useful for debugging and monitoring the vision system.

        Returns:
            Dictionary with background model statistics
        """
        if self.bg_subtractor is None:
            return {"error": "Background subtractor not initialized"}

        return {
            "history_length": self.bg_history,
            "threshold": self.bg_threshold,
            "learning_rate": self.learn_rate,
            "roi_set": self._roi is not None
        }

    def update_background_model(self, frame: np.ndarray):
        """
        Manually update the background model with a frame.

        This can be used to retrain the background when lighting conditions
        change or when you know the conveyor is empty.

        Args:
            frame: Frame showing empty conveyor belt
        """
        if self._roi is None or self.bg_subtractor is None:
            raise ValueError("ROI must be set before updating background model")

        roi_frame = self._extract_roi_from_frame(frame)

        # Apply with higher learning rate for faster adaptation
        self.bg_subtractor.apply(roi_frame, learningRate=0.1)

        logger.info("Background model manually updated")

    def reset_background_model(self, sample_frame: np.ndarray):
        """
        Completely reset and retrain the background model.

        Use this when lighting conditions have changed significantly
        or when the current background model is performing poorly.

        Args:
            sample_frame: Frame showing empty conveyor for training
        """
        if self._roi is None:
            raise ValueError("ROI must be set before resetting background model")

        logger.info("Resetting background model")

        # Recreate background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=self.bg_history,
            dist2Threshold=self.bg_threshold,
            detectShadows=False
        )

        # Retrain with sample frame
        self._train_background_model(sample_frame)

        logger.info("Background model reset complete")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_vision_processor(config_manager) -> VisionProcessor:
    """
    Create a VisionProcessor instance with unified configuration.

    This is the standard way to create a vision processor using the
    enhanced_config_manager system for consistent configuration management.

    Args:
        config_manager: Enhanced config manager instance (required)

    Returns:
        Configured VisionProcessor instance

    Raises:
        ValueError: If config_manager is None or invalid
    """
    if config_manager is None:
        raise ValueError("create_vision_processor requires a valid config_manager")

    logger.info("Creating VisionProcessor with enhanced_config_manager")
    return VisionProcessor(config_manager)


# ============================================================================
# TESTING AND UTILITIES
# ============================================================================

def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a synthetic test frame for testing vision processing.

    This generates a simple frame with geometric shapes that can be used
    to test detection algorithms without needing a real camera.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        Synthetic test frame as numpy array
    """
    # Create black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some geometric shapes as "pieces"
    # Rectangle (simulates brick)
    cv2.rectangle(frame, (100, 200), (180, 240), (255, 255, 255), -1)

    # Circle (simulates round piece)
    cv2.circle(frame, (300, 250), 25, (255, 255, 255), -1)

    # Another rectangle
    cv2.rectangle(frame, (450, 180), (520, 220), (255, 255, 255), -1)

    return frame


if __name__ == "__main__":
    """
    Test the vision processor with synthetic data.

    This demonstrates how to use the vision processor with a proper
    config manager setup.
    """
    import sys

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing VisionProcessor with synthetic data")


    # Create a mock config manager for testing
    class MockConfigManager:
        """Simple mock config manager for testing purposes"""

        def get_module_config(self, module_name):
            if module_name == "detector":
                return {
                    "min_area": 1000,
                    "max_area": 50000,
                    "min_aspect_ratio": 0.3,
                    "max_aspect_ratio": 3.0,
                    "bg_history": 500,
                    "bg_threshold": 800.0,
                    "learn_rate": 0.005,
                    "gaussian_blur_size": 7,
                    "morph_kernel_size": 5,
                    "min_contour_points": 5
                }
            return {}


    try:
        # Create vision processor with mock config manager
        mock_config = MockConfigManager()
        processor = create_vision_processor(mock_config)

        # Create test ROI and frame
        test_frame = create_test_frame(640, 480)
        roi = RegionOfInterest(x=50, y=150, width=500, height=200)

        # Set ROI and train background (using empty frame)
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame
        processor.set_roi(roi, empty_frame)

        # Process test frame
        detections = processor.process_frame(test_frame)

        # Report results
        print(f"\nVision Processing Results:")
        print(f"Detections found: {len(detections)}")

        for i, detection in enumerate(detections):
            print(f"Detection {i + 1}:")
            print(f"  - Bounding box: {detection.bbox}")
            print(f"  - Area: {detection.area:.0f} pixels")
            print(f"  - Center: ({detection.get_center()[0]:.1f}, {detection.get_center()[1]:.1f})")
            print(f"  - Confidence: {detection.confidence:.2f}")

        logger.info("VisionProcessor test complete")

    except Exception as e:
        logger.error(f"VisionProcessor test failed: {e}")
        print(f"\nTest failed: {e}")
        print("This demonstrates the requirement for a valid config_manager")