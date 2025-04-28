"""
camera_module.py - Simplified camera module with single responsibility
"""

import cv2
import logging
import time
import numpy as np
from typing import Optional
from error_module import CameraError

# Initialize module-specific logger
logger = logging.getLogger(__name__)


class CameraModule:
    """
    Camera module that purely handles image acquisition.

    This module is responsible ONLY for:
    1. Initializing camera hardware
    2. Retrieving frames from the camera
    3. Managing camera resource lifecycle

    It does NOT handle:
    - File system operations (saving images)
    - Image numbering or tracking
    - Directory management
    - Image processing

    This separation ensures the module has a single responsibility,
    making it easier to test, maintain, and extend.
    """

    def __init__(self, config_manager):
        """
        Initialize the camera hardware.

        Args:
            config_manager: Configuration manager to get camera settings

        The constructor only sets up initial state. Actual camera
        initialization happens in _initialize_camera() for better
        error handling and potential retry logic.
        """
        # Get device ID from config (default to device 0)
        self.device_id = config_manager.get("camera", "device_id", 0)

        # Camera handle - will be set during initialization
        self.cap = None

        # Flag to track if camera is properly initialized
        self.is_initialized = False

        # Flag to track if camera has been warmed up
        # (some cameras need time to adjust exposure/focus)
        self.is_warmed_up = False

        # Attempt to initialize the camera
        self._initialize_camera()

    def _initialize_camera(self):
        """
        Initialize the camera hardware.

        This private method handles the actual camera initialization,
        separated from the constructor to allow for:
        1. Better error handling
        2. Potential retry logic
        3. Clean separation of concerns

        Raises:
            CameraError: If camera initialization fails
        """
        try:
            # Create VideoCapture object with specified device ID
            self.cap = cv2.VideoCapture(self.device_id)

            # Check if camera opened successfully
            if not self.cap.isOpened():
                raise CameraError(f"Failed to open camera {self.device_id}")

            # Configure camera properties
            # Minimize buffering for real-time performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # You could add more camera settings here if needed:
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Mark as initialized
            self.is_initialized = True
            logger.info(f"Camera {self.device_id} initialized successfully")

            # Warm up camera to ensure stable images
            self._warm_up_camera()

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            # Convert any exception to CameraError for consistent error handling
            raise CameraError(f"Camera initialization failed: {e}")

    def _warm_up_camera(self, num_frames=10, delay=0.1):
        """
        Warm up the camera by capturing and discarding initial frames.

        Many cameras need time to adjust settings like exposure and focus.
        This method ensures the camera is ready before we start using it.

        Args:
            num_frames: Number of frames to discard (default: 10)
            delay: Delay between frames in seconds (default: 0.1)
        """
        if not self.is_initialized:
            logger.warning("Attempted to warm up non-initialized camera")
            return

        logger.info(f"Warming up camera with {num_frames} frames")

        # Capture and discard frames
        for i in range(num_frames):
            ret, _ = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame during warm-up ({i + 1}/{num_frames})")
            time.sleep(delay)

        self.is_warmed_up = True
        logger.info("Camera warm-up complete")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a single frame from the camera.

        This is the main interface method for retrieving camera data.
        It handles all the error checking and returns None if anything fails,
        allowing the caller to handle errors gracefully.

        Returns:
            np.ndarray: Captured frame, or None if capture failed
        """
        # Check if camera is ready
        if not self.is_initialized:
            logger.error("Attempted to get frame from non-initialized camera")
            return None

        # Ensure camera is warmed up
        if not self.is_warmed_up:
            logger.warning("Camera not warmed up, warming up now")
            self._warm_up_camera()

        try:
            # Attempt to capture a frame
            ret, frame = self.cap.read()

            # Check if capture was successful
            if not ret:
                logger.error("Failed to capture frame")
                return None

            # Validate frame is not empty
            if frame is None or self._is_empty_frame(frame):
                logger.error("Captured frame is empty or invalid")
                return None

            return frame

        except Exception as e:
            logger.error(f"Exception while capturing frame: {e}")
            return None

    def _is_empty_frame(self, frame) -> bool:
        """
        Check if a frame is empty or invalid.

        This method helps detect when the camera returns invalid data,
        which can happen due to hardware issues or timing problems.

        Args:
            frame: Frame to check

        Returns:
            bool: True if frame is empty/invalid, False otherwise
        """
        if frame is None:
            return True

        # Check if frame is all black (common failure mode)
        # Using a small threshold to account for sensor noise
        mean_value = np.mean(frame)
        return mean_value < 5.0  # Threshold for "blackness"

    def is_initialized(self) -> bool:
        """
        Check if camera is properly initialized.

        This method provides a simple way for other modules to check
        if the camera is ready before attempting to use it.

        Returns:
            bool: True if camera is initialized, False otherwise
        """
        return self.is_initialized

    def release(self):
        """
        Release camera resources.

        This method ensures proper cleanup of camera resources.
        It should be called when the application is shutting down
        or when the camera is no longer needed.
        """
        if self.cap and self.is_initialized:
            self.cap.release()
            self.is_initialized = False
            self.is_warmed_up = False
            logger.info("Camera resources released")
        else:
            logger.debug("Camera release called but camera not initialized")


# Factory function for consistent creation pattern
def create_camera(camera_type="webcam", config_manager=None):
    """
    Factory function to create camera instances.

    This provides a consistent interface for creating cameras,
    allowing for future extension to different camera types.

    Args:
        camera_type: Type of camera to create (currently only "webcam")
        config_manager: Configuration manager instance

    Returns:
        CameraModule: Initialized camera instance

    Raises:
        CameraError: If camera creation fails
    """
    logger.info(f"Creating camera of type: {camera_type}")

    try:
        if camera_type == "webcam":
            return CameraModule(config_manager)
        else:
            error_msg = f"Unsupported camera type: {camera_type}"
            logger.error(error_msg)
            raise CameraError(error_msg)
    except Exception as e:
        logger.error(f"Error creating camera: {e}")
        raise CameraError(f"Failed to create camera: {e}")
