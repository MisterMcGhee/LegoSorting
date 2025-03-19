"""
camera_module.py - A simple module for camera functionality
"""

import os
import cv2
from typing import Dict, Any, Tuple, Optional


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

        # Create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

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
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to initialize camera with device ID {device_id}")

        self.is_initialized = True
        print(f"Camera initialized with starting count: {self.count}")

    def capture_image(self):
        """Captures and saves an image

        Returns:
            tuple: (filename, image_number, error_message)
                - filename: Path to saved image, or None if failed
                - image_number: Current image number, or None if failed
                - error_message: Error message if failed, or None if successful
        """
        if not self.is_initialized:
            return None, None, "Camera not initialized"

        ret, frame = self.cap.read()
        if not ret:
            return None, None, "Failed to capture image"

        filename = os.path.join(self.directory, f"{self.filename_prefix}{self.count:03d}.jpg")

        try:
            cv2.imwrite(filename, frame)
            current_count = self.count  # Store current count before incrementing
            self.count += 1
            return filename, current_count, None
        except Exception as e:
            return None, None, f"Failed to save image: {str(e)}"

    def get_preview_frame(self):
        """Gets a frame for preview without saving

        Returns:
            numpy.ndarray or None: Frame if successful, None otherwise
        """
        if not self.is_initialized:
            return None

        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """Releases camera resources"""
        if self.is_initialized:
            self.cap.release()
            self.is_initialized = False
            print("Camera resources released")


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
    pass


# Simple factory function
def create_camera(camera_type="webcam", config_manager=None):
    """Create a camera module of the specified type

    Args:
        camera_type: Type of camera ("webcam" or in the future, "video_file")
        config_manager: Configuration manager

    Returns:
        Camera module instance
    """
    if camera_type == "webcam":
        return CameraModule(config_manager)
    else:
        raise ValueError(f"Unsupported camera type: {camera_type}. Currently only 'webcam' is supported.")


# Example usage
if __name__ == "__main__":
    # Create camera
    camera = create_camera("webcam")

    try:
        # Show preview
        frame = camera.get_preview_frame()
        if frame is not None:
            cv2.imshow("Preview", frame)
            cv2.waitKey(1000)  # Wait for 1 second

        # Capture image
        filename, count, error = camera.capture_image()
        if error:
            print(f"Error: {error}")
        else:
            print(f"Image saved to: {filename}")

    finally:
        # Clean up
        camera.release()
        cv2.destroyAllWindows()