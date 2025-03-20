"""
camera_module.py - A simple module for camera functionality
"""

import os
import cv2
import time
import logging
from typing import Tuple, Optional, Any

from error_module import CameraError

# Set up module logger
logger = logging.getLogger(__name__)


class CameraModule:
    """A simplified camera module for image acquisition"""

    def __init__(self, config_manager=None):
        """Initialize camera with configuration

        Args:
            config_manager: Configuration manager object, or None to use defaults
        """
        # Use config if provided, otherwise use defaults
        if config_manager:
            device_id = config_manager.get("camera", "device_id", 0)
            self.directory = os.path.abspath(config_manager.get("camera", "directory", "LegoPictures"))
            self.filename_prefix = config_manager.get("camera", "filename_prefix", "Lego")
        else:
            device_id = 0
            self.directory = os.path.abspath("LegoPictures")
            self.filename_prefix = "Lego"

        logger.info(f"Initializing camera with device ID {device_id}")
        logger.info(f"Image directory: {self.directory}")

        # Create directory if it doesn't exist
        if not os.path.exists(self.directory):
            try:
                os.makedirs(self.directory)
                logger.info(f"Created directory: {self.directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {self.directory}: {str(e)}")
                raise CameraError(f"Failed to create directory: {str(e)}")

        # Initialize count by checking existing files
        self.count = 1  # Start at 1 by default
        if os.path.exists(self.directory):
            existing_files = [f for f in os.listdir(self.directory)
                              if f.startswith(self.filename_prefix) and f.endswith(".jpg")]
            if existing_files:
                # Extract numbers from filenames and find the highest
                numbers = []
                for f in existing_files:
                    try:
                        # Get number from filename (e.g., Lego001.jpg -> 1)
                        num_str = f[len(self.filename_prefix):len(self.filename_prefix) + 3]
                        numbers.append(int(num_str))
                    except (ValueError, IndexError):
                        continue

                if numbers:
                    self.count = max(numbers) + 1

        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(device_id)
            if not self.cap.isOpened():
                error_msg = f"Failed to initialize camera with device ID {device_id}"
                logger.error(error_msg)
                raise CameraError(error_msg)

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering for real-time

            # Camera is now initialized but may need warm-up
            self.is_initialized = True
            self.is_warmed_up = False  # Flag to track warm-up state
            logger.info(f"Camera initialized with starting count: {self.count}")
        except Exception as e:
            logger.error(f"Error during camera initialization: {str(e)}")
            raise CameraError(f"Camera initialization error: {str(e)}")

    def capture_image(self) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Captures and saves an image

        Returns:
            tuple: (filename, image_number, error_message)
                - filename: Path to saved image, or None if failed
                - image_number: Current image number, or None if failed
                - error_message: Error message if failed, or None if successful
        """
        if not self.is_initialized:
            error_msg = "Camera not initialized"
            logger.error(error_msg)
            return None, None, error_msg

        ret, frame = self.cap.read()
        if not ret:
            error_msg = "Failed to capture image"
            logger.error(error_msg)
            return None, None, error_msg

        filename = os.path.join(self.directory, f"{self.filename_prefix}{self.count:03d}.jpg")

        try:
            cv2.imwrite(filename, frame)
            current_count = self.count  # Store current count before incrementing
            self.count += 1
            logger.info(f"Image captured and saved to {filename}")
            return filename, current_count, None
        except Exception as e:
            error_msg = f"Failed to save image: {str(e)}"
            logger.error(error_msg)
            return None, None, error_msg

    def get_preview_frame(self):
        """Gets a frame for preview without saving

        Returns:
            numpy.ndarray or None: Frame if successful, None otherwise
        """
        if not self.is_initialized:
            logger.error("Attempted to get preview frame, but camera is not initialized")
            return None

        # If camera hasn't been warmed up yet, discard some frames to ensure we get a valid one
        if not self.is_warmed_up:
            logger.info("Warming up camera...")
            self._warm_up_camera()

        # Try to get a valid frame
        max_attempts = 3
        for attempt in range(max_attempts):
            ret, frame = self.cap.read()
            if ret and frame is not None and not self._is_empty_frame(frame):
                return frame
            logger.warning(f"Got invalid frame, attempt {attempt + 1}/{max_attempts}")
            time.sleep(0.1)  # Small delay between attempts

        logger.error("Failed to get valid preview frame after multiple attempts")
        return None

    def _warm_up_camera(self, num_frames=10, delay=0.1):
        """Warm up the camera by capturing and discarding initial frames.

        Args:
            num_frames: Number of frames to discard
            delay: Delay between frame captures in seconds
        """
        logger.info(f"Warming up camera by discarding {num_frames} frames")
        for i in range(num_frames):
            ret, _ = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame during warm-up ({i + 1}/{num_frames})")
            time.sleep(delay)  # Small delay between frames
        self.is_warmed_up = True
        logger.info("Camera warm-up complete")

    def _is_empty_frame(self, frame):
        """Check if a frame is empty or just black.

        Args:
            frame: The frame to check

        Returns:
            bool: True if frame is empty/black, False otherwise
        """
        if frame is None:
            return True

        # Check if frame is all zeros (black) or very close to it
        # Using mean value across all channels as a simple heuristic
        mean_value = cv2.mean(frame)[0]
        return mean_value < 5.0  # Threshold for "blackness"

    def release(self) -> None:
        """Releases camera resources"""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            self.cap.release()
            self.is_initialized = False
            logger.info("Camera resources released")
        else:
            logger.debug("Release called on non-initialized camera")


class VideoFileModule:
    """
    Placeholder for future implementation of a video file source.

    This class would allow using pre-recorded videos as an input source,
    which can be useful for:
    - Testing the pipeline with consistent inputs
    - Debugging image processing issues
    - Batch processing previously recorded sessions
    - Creating demonstrations with known inputs

    Future implementation would mimic the CameraModule's interface
    but read frames from a video file instead of a live camera.
    """

    def release(self) -> None:
        """Placeholder for resource release method"""
        logger.debug("VideoFileModule.release() called")
        pass


# Simple factory function
def create_camera(camera_type="webcam", config_manager=None):
    """Create a camera module of the specified type

    Args:
        camera_type: Type of camera ("webcam" or in the future, "video_file")
        config_manager: Configuration manager

    Returns:
        Camera module instance

    Raises:
        CameraError: When camera initialization fails or an unsupported type is requested
    """
    logger.info(f"Creating camera of type: {camera_type}")

    try:
        if camera_type == "webcam":
            return CameraModule(config_manager)
        else:
            error_msg = f"Unsupported camera type: {camera_type}. Currently only 'webcam' is supported."
            logger.error(error_msg)
            raise CameraError(error_msg)
    except Exception as e:
        if isinstance(e, CameraError):
            raise
        logger.error(f"Error creating camera: {str(e)}")
        raise CameraError(f"Failed to create camera: {str(e)}")
