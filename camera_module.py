"""
camera_module.py - A simple module for camera functionality
"""

import os
import cv2
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

            self.is_initialized = True
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

    def get_preview_frame(self) -> Optional[Any]:
        """Gets a frame for preview without saving

        Returns:
            numpy.ndarray or None: Frame if successful, None otherwise
        """
        if not self.is_initialized:
            logger.error("Attempted to get preview frame, but camera is not initialized")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to get preview frame")
            return None

        return frame

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


# Example usage
if __name__ == "__main__":
    from error_module import setup_logging
    setup_logging()

    try:
        # Create camera
        camera = create_camera("webcam")

        # Show preview
        frame = camera.get_preview_frame()
        if frame is not None:
            cv2.imshow("Preview", frame)
            cv2.waitKey(1000)  # Wait for 1 second

        # Capture image
        filename, count, error = camera.capture_image()
        if error:
            logger.error(f"Error: {error}")
        else:
            logger.info(f"Image saved to: {filename}")

    except CameraError as e:
        logger.error(f"Camera error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Clean up
        if 'camera' in locals():
            camera.release()
            cv2.destroyAllWindows()