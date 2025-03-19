import json
import os
from typing import Any, Dict

import cv2
import requests

from camera_module import create_camera
from detector_module import create_detector
from sorting_module import create_sorting_manager


class ConfigManager:
    """Manages configuration settings for the Lego sorting application."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file or create default if not exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
                self._create_default_config()
        else:
            print(f"Configuration file not found at {self.config_path}")
            self._create_default_config()

    # Make sure the ConfigManager has the necessary fields for the camera
    def _create_default_config(self) -> None:
        """Create default configuration."""
        self.config = {
            "camera": {
                "device_id": 0,
                "directory": "LegoPictures",
                "filename_prefix": "Lego"  # Add this for the new camera module
            },
            "detector": {
                "min_piece_area": 300,
                "max_piece_area": 100000,
                "buffer_percent": 0.1,
                "crop_padding": 50,
                "history_length": 2,
                "grid_size": [10, 10],
                "color_margin": 40
            },
            "piece_identifier": {
                "csv_path": "Lego_Categories.csv",
                "confidence_threshold": 0.7
            },
            "sorting": {
                "strategy": "primary",
                "target_primary_category": "Basic",
                "target_secondary_category": "Brick",
                "max_bins": 9,
                "overflow_bin": 9
            }
        }
        self.save_config()

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found

        Returns:
            The configuration value or default
        """
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section.

        Args:
            section: Configuration section

        Returns:
            Dictionary containing the section or empty dict if not found
        """
        return self.config.get(section, {})

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update an entire section with new values.

        Args:
            section: Configuration section
            values: Dictionary of values to update
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)


class CameraManager:
    def __init__(self, config_manager=None):
        """Initialize camera and directory for storing images.

        Args:
            config_manager: Optional ConfigManager instance
        """
        # Use config if provided, otherwise use defaults
        if config_manager:
            device_id = config_manager.get("camera", "device_id", 0)
            self.directory = os.path.abspath(config_manager.get("camera", "directory", "LegoPictures"))
        else:
            device_id = 0
            self.directory = os.path.abspath("LegoPictures")

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Initialize count by checking existing files
        self.count = 1  # Start at 1 by default
        if os.path.exists(self.directory):
            existing_files = [f for f in os.listdir(self.directory)
                              if f.startswith("Lego") and f.endswith(".jpg")]
            if existing_files:
                # Extract numbers from filenames and find the highest
                numbers = [int(f[4:7]) for f in existing_files]  # Lego001.jpg -> 1
                self.count = max(numbers) + 1

        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to initialize camera with device ID {device_id}")

    def capture_image(self):
        """Captures and saves an image. Returns filename, current count, and any error."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, "Failed to capture image"

        filename = os.path.join(self.directory, f"Lego{self.count:03}.jpg")
        try:
            cv2.imwrite(filename, frame)
            current_count = self.count  # Store current count before incrementing
            self.count += 1
            return filename, current_count, None
        except Exception as e:
            return None, None, f"Failed to save image: {str(e)}"

    def get_preview_frame(self):
        """Gets a frame for preview/detection without saving."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def cleanup(self):
        """Releases camera resources."""
        self.cap.release()


class BrickognizeAPI:
    """Handles communication with the Brickognize API."""

    @staticmethod
    def send_to_api(image_path: str) -> Dict[str, Any]:
        """Send image to Brickognize API for identification.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with API response or error
        """
        url = "https://api.brickognize.com/predict/"
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']

        if not os.path.isfile(image_path):
            return {"error": f"File not found: {image_path}"}
        if os.path.getsize(image_path) == 0:
            return {"error": "File is empty"}
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            return {"error": f"Invalid file type. Allowed: {', '.join(valid_extensions)}"}

        try:
            with open(image_path, "rb") as image_file:
                files = {"query_image": (image_path, image_file, "image/jpeg")}
                response = requests.post(url, files=files)

            if response.status_code != 200:
                return {"error": f"API request failed: {response.status_code}"}

            data = response.json()
            if "items" in data and data["items"]:
                item = data["items"][0]
                return {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "score": item.get("score")
                }

            return {"error": "No items found in response"}

        except Exception as e:
            return {"error": f"Request error: {str(e)}"}


class SortingManager:
    def __init__(self, config_path="config.json"):
        """Initialize sorting manager with configuration.

        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)

        # Initialize components with config
        self.camera = create_camera("webcam", self.config_manager)

        # Create detector with config parameters
        self.detector = create_detector("tracked", self.config_manager)

        # Initialize the sorting module
        self.sorting_manager = create_sorting_manager(self.config_manager)

        # API interface
        self.api = BrickognizeAPI()

        print(f"Starting with image count: {self.camera.count}")
        print(f"Sorting strategy: {self.sorting_manager.get_strategy_description()}")

    def run(self):
        """Main sorting loop."""
        try:
            print("\nInitializing sorting system...")

            print("\nOpening camera preview for belt region selection.")
            print("Position the camera to view the belt clearly.")
            print("Select the region to monitor by dragging a rectangle:")
            print("Press SPACE or ENTER to confirm selection")
            print("Press ESC to exit")

            # Get initial frame for ROI selection
            frame = self.camera.get_preview_frame()
            if frame is None:
                raise RuntimeError("Failed to get preview frame")

            # Initialize detector with ROI
            self.detector.calibrate(frame)

            print("\nSetup complete!")
            print("\nSorting system ready. Press ESC to exit.")

            while True:
                frame = self.camera.get_preview_frame()
                if frame is None:
                    print("Failed to get preview frame")
                    break

                # Process frame and get any pieces to capture
                tracked_pieces, piece_image = self.detector.process_frame(
                    frame=frame,
                    current_count=self.camera.count
                )
                # If we have a piece to capture
                if piece_image is not None:
                    # Save the cropped image
                    file_path, image_number, error = self.camera.capture_image()

                    if error:
                        print(f"Error capturing image: {error}")
                        continue

                    # Save the cropped piece image
                    cv2.imwrite(file_path, piece_image)

                    # Send to API for identification
                    api_result = self.api.send_to_api(file_path)

                    if "error" in api_result:
                        print(f"API error: {api_result['error']}")
                        continue

                    # Process with sorting manager
                    result = self.sorting_manager.identify_piece(api_result)

                    if "error" in result:
                        print(f"Sorting error: {result['error']}")
                    else:
                        print(f"\nPiece #{image_number:03} identified:")
                        print(f"Image: {os.path.basename(file_path)}")
                        print(f"Element ID: {result.get('element_id', 'Unknown')}")
                        print(f"Name: {result.get('name', 'Unknown')}")
                        print(f"Primary Category: {result.get('primary_category', 'Unknown')}")
                        print(f"Secondary Category: {result.get('secondary_category', 'Unknown')}")
                        print(f"Bin Number: {result.get('bin_number', 9)}")

                # Show debug view
                debug_frame = self.detector.draw_debug(frame)
                cv2.imshow("Capture", debug_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.camera.cleanup()
        cv2.destroyAllWindows()


# Main program that runs
if __name__ == "__main__":
    import argparse

    # Add command-line argument for config file
    parser = argparse.ArgumentParser(description='Lego Sorting System')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Initialize with specified config file
    sorter = SortingManager(config_path=args.config)
    sorter.run()
