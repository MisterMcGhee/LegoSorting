"""
Lego Sorting 003 - Main application for Lego piece sorting

This script serves as the entry point for the Lego sorting application,
coordinating between the various modules.
"""

import os
import argparse
import cv2
import logging

# Import modules
from camera_module import create_camera
from detector_module import create_detector
from sorting_module import create_sorting_manager
from config_management_module import create_config_manager
from api_module import create_api_client
from error_module import setup_logging, CameraError, APIError, DetectorError


class LegoSortingApplication:
    """Main application controller for the Lego sorting system."""

    def __init__(self, config_path="config.json"):
        """Initialize the Lego sorting application.

        Args:
            config_path: Path to configuration file
        """
        # Get logger
        self.logger = logging.getLogger(__name__)

        # Initialize configuration
        self.logger.info("Loading configuration")
        self.config_manager = create_config_manager(config_path)

        # Initialize components with config
        try:
            self.logger.info("Initializing camera")
            self.camera = create_camera("webcam", self.config_manager)

            self.logger.info("Initializing detector")
            self.detector = create_detector("tracked", self.config_manager)

            self.logger.info("Initializing sorting manager")
            self.sorting_manager = create_sorting_manager(self.config_manager)

            self.logger.info("Initializing API client")
            self.api_client = create_api_client("brickognize", self.config_manager)

            self.logger.info(f"Sorting strategy: {self.sorting_manager.get_strategy_description()}")
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise

    def run(self):
        """Run the main sorting application loop."""
        try:
            self.logger.info("Initializing sorting system")

            self.logger.info("Opening camera preview for belt region selection")
            print("\nOpening camera preview for belt region selection.")
            print("Position the camera to view the belt clearly.")
            print("Select the region to monitor by dragging a rectangle:")
            print("Press SPACE or ENTER to confirm selection")
            print("Press ESC to exit")

            # Get initial frame for ROI selection
            frame = self.camera.get_preview_frame()
            if frame is None:
                error_msg = "Failed to get preview frame"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Initialize detector with ROI
            self.detector.calibrate(frame)

            self.logger.info("Setup complete")
            print("\nSetup complete!")
            print("\nSorting system ready. Press ESC to exit.")

            while True:
                frame = self.camera.get_preview_frame()
                if frame is None:
                    self.logger.warning("Failed to get preview frame")
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
                        self.logger.error(f"Error capturing image: {error}")
                        print(f"Error capturing image: {error}")
                        continue

                    # Save the cropped piece image
                    cv2.imwrite(file_path, piece_image)
                    self.logger.info(f"Saved piece image to {file_path}")

                    # Send to API for identification
                    self.logger.info(f"Sending image to API for identification")
                    api_result = self.api_client.send_to_api(file_path)

                    if "error" in api_result:
                        self.logger.error(f"API error: {api_result['error']}")
                        print(f"API error: {api_result['error']}")
                        continue

                    # Process with sorting manager
                    result = self.sorting_manager.identify_piece(api_result)

                    if "error" in result:
                        self.logger.error(f"Sorting error: {result['error']}")
                        print(f"Sorting error: {result['error']}")
                    else:
                        self.logger.info(f"Piece identified: {result.get('element_id', 'Unknown')}, "
                                         f"bin: {result.get('bin_number', 9)}")

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
                    self.logger.info("ESC pressed, exiting application")
                    break

        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
            raise
        finally:
            self.release()

    def release(self):
        """Releases all resources used by the application.

        This method ensures a clean shutdown by releasing all hardware resources
        and closing any open windows.
        """
        self.logger.info("Releasing application resources")

        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()

        cv2.destroyAllWindows()
        self.logger.info("Application shutdown complete")


# Main program
if __name__ == "__main__":
    # Add command-line argument for config file
    parser = argparse.ArgumentParser(description='Lego Sorting System')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Lego Sorting Application")

    try:
        # Initialize with specified config file
        application = LegoSortingApplication(config_path=args.config)
        application.run()
    except Exception as e:
        logger.critical(f"Application failed: {e}")
        print(f"Application error: {e}")
    finally:
        logger.info("Application terminated")
