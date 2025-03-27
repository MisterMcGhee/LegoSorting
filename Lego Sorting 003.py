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
from servo_module import create_servo_module
from error_module import setup_logging, CameraError, APIError, DetectorError


class LegoSortingApplication:
    """Main application controller for the Lego sorting system."""

    def __init__(self, config_path="config.json"):
        """Initialize the Lego sorting application.

        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config_manager = create_config_manager(config_path)

        # Initialize components with config
        self.camera = create_camera("webcam", self.config_manager)
        self.detector = create_detector("tracked", self.config_manager)
        self.sorting_manager = create_sorting_manager(self.config_manager)
        self.api_client = create_api_client("brickognize", self.config_manager)
        self.servo = create_servo_module(self.config_manager)

        print(f"Sorting strategy: {self.sorting_manager.get_strategy_description()}")

    def run(self):
        """Run the main sorting application loop."""
        try:
            print("\nInitializing sorting system...")

            # Get initial frame for ROI selection
            frame = self.camera.get_preview_frame()
            if frame is None:
                raise RuntimeError("Failed to get preview frame")

            # Check if we're drawing the ROI or using default
            draw_roi = self.config_manager.get("detector", "draw_roi", True)
            show_visualizer = self.config_manager.get("detector", "show_visualizer", True)

            if draw_roi:
                print("\nOpening camera preview for belt region selection.")
                print("Position the camera to view the belt clearly.")
                print("Select the region to monitor by dragging a rectangle:")
                print("Press SPACE or ENTER to confirm selection")
                print("Press ESC to exit")
            else:
                print("\nUsing default ROI from configuration.")

            if show_visualizer:
                print("Visualizer is enabled - debug view will be shown.")
            else:
                print("Visualizer is disabled - running in headless mode.")

            # Initialize detector with ROI
            self.detector.calibrate(frame)

            print("\nSetup complete!")
            print("\nSorting system ready. Press ESC to exit (if visualizer is enabled).")

            # Move servo to default position (overflow bin) initially
            self.servo.move_to_bin(self.config_manager.get("sorting", "overflow_bin", 9))

            try:
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
                        try:
                            # Save the cropped image
                            file_path, image_number, error = self.camera.capture_image()

                            if error:
                                print(f"Error capturing image: {error}")
                                continue

                            # Save the cropped piece image
                            cv2.imwrite(file_path, piece_image)

                            # Send to API for identification
                            api_result = self.api_client.send_to_api(file_path)

                            if "error" in api_result:
                                print(f"API error: {api_result['error']}")
                                continue

                            # Process with sorting manager
                            result = self.sorting_manager.identify_piece(api_result)

                            if "error" in result:
                                print(f"Sorting error: {result['error']}")
                                # If sorting error, direct to overflow bin
                                self.servo.move_to_bin(self.config_manager.get("sorting", "overflow_bin", 9))
                            else:
                                # Get bin number and move servo
                                bin_number = result.get("bin_number", 9)
                                self.servo.move_to_bin(bin_number)

                                print(f"\nPiece #{image_number:03} identified:")
                                print(f"Image: {os.path.basename(file_path)}")
                                print(f"Element ID: {result.get('element_id', 'Unknown')}")
                                print(f"Name: {result.get('name', 'Unknown')}")
                                print(f"Primary Category: {result.get('primary_category', 'Unknown')}")
                                print(f"Secondary Category: {result.get('secondary_category', 'Unknown')}")
                                print(f"Bin Number: {bin_number}")
                        except Exception as e:
                            print(f"Error processing piece: {str(e)}")
                            logger = logging.getLogger(__name__)
                            logger.error(f"Error processing piece: {str(e)}")
                            # Move to overflow bin in case of error
                            self.servo.move_to_bin(self.config_manager.get("sorting", "overflow_bin", 9))

                    # Only prepare and show debug frame if visualizer is enabled
                    if show_visualizer:
                        # Show debug view
                        debug_frame = self.detector.draw_debug(frame)

                        # Add servo information to debug frame
                        if self.servo.current_bin is not None:
                            bin_text = f"Current Bin: {self.servo.current_bin}"
                            cv2.putText(debug_frame, bin_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Always show visualizer if enabled in config, regardless of display detection
                        cv2.imshow("Capture", debug_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC key
                            break
                    else:
                        # Brief pause to prevent CPU overutilization in headless mode
                        import time
                        time.sleep(0.01)
            except KeyboardInterrupt:
                print("\nOperation interrupted by user")
            except Exception as e:
                print(f"\nError in main processing loop: {str(e)}")
                logger = logging.getLogger(__name__)
                logger.error(f"Error in main processing loop: {str(e)}")
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            logger = logging.getLogger(__name__)
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        self.camera.release()
        self.servo.release()
        cv2.destroyAllWindows()


# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lego Sorting System')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--calibrate-servo', action='store_true',
                        help='Run servo calibration at startup')
    parser.add_argument('--draw-roi', action='store_true',
                        help='Force drawing ROI at startup regardless of config')
    parser.add_argument('--use-default-roi', action='store_true',
                        help='Force using default ROI at startup regardless of config')
    parser.add_argument('--set-default-roi', type=str,
                        help='Set new default ROI (format: "x,y,w,h")')
    parser.add_argument('--show-visualizer', action='store_true',
                        help='Enable visualizer/debug view regardless of config')
    parser.add_argument('--hide-visualizer', action='store_true',
                        help='Disable visualizer/debug view regardless of config')
    args = parser.parse_args()

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Lego Sorting Application")

    # Initialize config manager for updating settings
    config_manager = create_config_manager(args.config)

    # If servo calibration flag is set, update config
    if args.calibrate_servo:
        config_manager.set("servo", "calibration_mode", True)
        config_manager.save_config()
        logger.info("Servo calibration mode activated")

    # Handle ROI override arguments
    if args.draw_roi and args.use_default_roi:
        logger.warning("Both --draw-roi and --use-default-roi specified. Using --draw-roi.")
        args.use_default_roi = False

    if args.draw_roi:
        config_manager.set("detector", "draw_roi", True)
        config_manager.save_config()
        logger.info("ROI drawing mode activated")

    if args.use_default_roi:
        config_manager.set("detector", "draw_roi", False)
        config_manager.save_config()
        logger.info("Using default ROI from config")

    # Handle visualizer override arguments
    if args.show_visualizer and args.hide_visualizer:
        logger.warning("Both --show-visualizer and --hide-visualizer specified. Using --show-visualizer.")
        args.hide_visualizer = False

    if args.show_visualizer:
        config_manager.set("detector", "show_visualizer", True)
        config_manager.save_config()
        logger.info("Visualizer enabled")

    if args.hide_visualizer:
        config_manager.set("detector", "show_visualizer", False)
        config_manager.save_config()
        logger.info("Visualizer disabled")

    if args.set_default_roi:
        try:
            # Parse the ROI string format "x,y,w,h"
            roi_values = [int(x) for x in args.set_default_roi.split(',')]
            if len(roi_values) != 4:
                raise ValueError("ROI must have exactly 4 values")

            config_manager.set("detector", "default_roi", roi_values)
            config_manager.save_config()
            logger.info(f"Default ROI updated to {roi_values}")
        except Exception as e:
            logger.error(f"Failed to set default ROI: {str(e)}")
            logger.error("ROI should be specified as 'x,y,w,h' (e.g., '100,100,400,300')")
            print(f"Error setting default ROI: {str(e)}")
            print("ROI should be specified as 'x,y,w,h' (e.g., '100,100,400,300')")

    try:
        # Create and run application
        application = LegoSortingApplication(config_path=args.config)
        application.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        logger.info("Application terminated by user")
    except Exception as e:
        print(f"\nApplication error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
    finally:
        print("Shutting down Lego Sorting Application")