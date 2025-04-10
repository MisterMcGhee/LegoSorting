"""
Lego Sorting 003 - Main application for Lego piece sorting

This script serves as the entry point for the Lego sorting application,
coordinating between the various modules. This version includes multithreaded
processing for improved performance and reduced stuttering.
"""

import os
import argparse
import cv2
import time
import logging
import threading

# Import modules
from camera_module import create_camera
from detector_module import create_detector
from sorting_module import create_sorting_manager
from config_management_module import create_config_manager
from api_module import create_api_client
from servo_module import create_servo_module
from thread_management_module import create_thread_manager
from processing_module import processing_worker_thread
from error_module import setup_logging, get_logger, CameraError, APIError, DetectorError, ThreadingError


# Initialize logger
logger = get_logger(__name__)


class LegoSortingApplication:
    """Main application controller for the Lego sorting system."""

    def __init__(self, config_path="config.json"):
        """Initialize the Lego sorting application.

        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Lego Sorting Application")

        # Initialize configuration
        self.config_manager = create_config_manager(config_path)

        # Check if threading is enabled
        self.threading_enabled = self.config_manager.is_threading_enabled()
        logger.info("Threading enabled: %s", self.threading_enabled)

        # Initialize components that don't depend on threading first
        try:
            self.camera = create_camera("webcam", self.config_manager)
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            raise RuntimeError(f"Camera initialization failed: {str(e)}")

        # Initialize thread manager if threading is enabled
        self.thread_manager = None
        if self.threading_enabled:
            try:
                self.thread_manager = create_thread_manager(self.config_manager)
                # Register callbacks for processing events
                self.thread_manager.register_callback("piece_processed", self._on_piece_processed)
                self.thread_manager.register_callback("piece_error", self._on_piece_error)
                logger.info("Thread manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize thread manager: {str(e)}")
                self.threading_enabled = False
                logger.warning("Disabling threading due to initialization failure")

        # Now initialize detector with thread manager (or None if threading disabled)
        try:
            self.detector = create_detector("tracked", self.config_manager, self.thread_manager)
            logger.info("Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {str(e)}")
            raise RuntimeError(f"Detector initialization failed: {str(e)}")

        # Initialize sorting manager
        try:
            self.sorting_manager = create_sorting_manager(self.config_manager)
            logger.info("Sorting manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sorting manager: {str(e)}")
            raise RuntimeError(f"Sorting manager initialization failed: {str(e)}")

        # Only initialize API client and servo in the main thread if threading is disabled
        # Otherwise, these will be initialized in the worker thread
        if not self.threading_enabled:
            try:
                self.api_client = create_api_client("brickognize", self.config_manager)
                self.servo = create_servo_module(self.config_manager)
                logger.info("API client and servo initialized in main thread")
            except Exception as e:
                logger.error(f"Failed to initialize API client or servo: {str(e)}")
                raise RuntimeError(f"API client or servo initialization failed: {str(e)}")
        else:
            self.api_client = None
            self.servo = None

        logger.info("Sorting strategy: %s", self.sorting_manager.get_strategy_description())

        # Status flags
        self.running = False
        self.processing_threads = []
        self.last_status_update = time.time()
        self.status_update_interval = 5.0  # seconds

        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
        self.last_fps_update = 0
        self.processed_pieces = 0
    def _start_processing_thread(self):
        """Start the processing worker thread."""
        if not self.threading_enabled:
            return

        logger.info("Starting processing worker thread")

        # Create worker thread
        thread_name = "processing_worker"
        target_func = processing_worker_thread
        args = (self.thread_manager, self.config_manager)

        # Start the thread
        if self.thread_manager.start_worker(thread_name, target_func, args):
            logger.info("Processing worker thread started")
            self.processing_threads.append(thread_name)
        else:
            logger.error("Failed to start processing worker thread")

    def _on_piece_processed(self, result):
        """Callback for when a piece is processed.

        Args:
            result: Dictionary with processing result
        """
        self.processed_pieces += 1

        # Log the result
        logger.info("Piece processed: Element %s (%s) to bin %d",
                   result.get("element_id", "Unknown"),
                   result.get("name", "Unknown"),
                   result.get("bin_number", 9))

        # Print to console for visibility
        print(f"\nPiece #{result.get('image_number', 0):03} identified:")
        print(f"Image: {os.path.basename(result.get('file_path', 'Unknown'))}")
        print(f"Element ID: {result.get('element_id', 'Unknown')}")
        print(f"Name: {result.get('name', 'Unknown')}")
        print(f"Primary Category: {result.get('primary_category', 'Unknown')}")
        print(f"Secondary Category: {result.get('secondary_category', 'Unknown')}")
        print(f"Bin Number: {result.get('bin_number', 9)}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")

    def _on_piece_error(self, piece_id, error):
        """Callback for when a piece processing errors.

        Args:
            piece_id: ID of the piece that errored
            error: Error message
        """
        logger.error("Error processing piece ID %d: %s", piece_id, error)
        print(f"\nError processing piece ID {piece_id}: {error}")

    def _update_status(self, frame, debug_frame):
        """Add status information to the debug frame.

        Args:
            frame: Original frame for performance calculation
            debug_frame: Debug frame to add status to

        Returns:
            Frame with status information
        """
        current_time = time.time()

        # Update FPS calculation every second
        if current_time - self.last_fps_update >= 1.0:
            if self.start_time and self.frame_count > 0:
                elapsed = current_time - self.start_time
                self.fps = self.frame_count / elapsed
            self.last_fps_update = current_time

        # Skip if not time to update yet
        if current_time - self.last_status_update < self.status_update_interval:
            # Just add FPS
            cv2.putText(debug_frame, f"FPS: {self.fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return debug_frame

        self.last_status_update = current_time

        # Add performance information
        cv2.putText(debug_frame, f"FPS: {self.fps:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.threading_enabled and self.thread_manager:
            # Get queue size
            queue_size = self.thread_manager.get_queue_size()

            # Add queue information
            cv2.putText(debug_frame, f"Queue: {queue_size}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Add processed count
            cv2.putText(debug_frame, f"Processed: {self.processed_pieces}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Add threading status
            thread_status = "Threaded" if self.threading_enabled else "Synchronous"
            cv2.putText(debug_frame, thread_status, (frame.shape[1] - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return debug_frame

    """
    Modified run method for LegoSortingApplication class to improve
    ROI loading and error handling.
    """

    def run(self):
        """Run the main sorting application loop."""
        try:
            logger.info("Initializing sorting system")
            print("\nInitializing sorting system...")

            # Get initial frame for ROI selection
            frame = None
            max_attempts = 5
            for attempt in range(max_attempts):
                frame = self.camera.get_preview_frame()
                if frame is not None:
                    break
                print(f"Attempt {attempt + 1}/{max_attempts} to get preview frame failed, retrying...")
                time.sleep(1)

            if frame is None:
                logger.error("Failed to get preview frame after multiple attempts")
                raise RuntimeError("Failed to get preview frame after multiple attempts")

            # Initialize detector with ROI (from config if available)
            try:
                # Initialize detector with ROI
                self.detector.load_roi_from_config(frame, self.config_manager)
                logger.info("Detector ROI initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize detector ROI: {str(e)}")
                print(f"\nError initializing detector ROI: {str(e)}")
                print("Attempting manual calibration...")
                # Try manual calibration as fallback
                self.detector.calibrate(frame)

            print("\nSetup complete!")

            # Start processing thread if threading is enabled
            if self.threading_enabled:
                self._start_processing_thread()
                print("Started processing thread")

                # Initialize overflow bin position in async mode
                # This will be controlled by the worker thread for specific bins
                overflow_bin = self.config_manager.get("sorting", "overflow_bin", 9)
                print(f"Default overflow bin: {overflow_bin}")
            else:
                # In synchronous mode, move servo to default position initially
                if hasattr(self, 'servo') and self.servo:
                    self.servo.move_to_bin(self.config_manager.get("sorting", "overflow_bin", 9))
                    print("Moved servo to default position")

            print("\nSorting system ready. Press ESC to exit.")

            # Set running flag and reset counters
            self.running = True
            self.frame_count = 0
            self.start_time = time.time()
            self.last_fps_update = self.start_time
            self.processed_pieces = 0

            while self.running:
                # Get frame from camera
                frame = self.camera.get_preview_frame()
                if frame is None:
                    logger.warning("Failed to get preview frame")
                    time.sleep(0.1)  # Short delay to avoid tight loop if camera is failing
                    continue

                # Increment frame counter
                self.frame_count += 1

                # Process frame based on threading mode
                try:
                    if self.threading_enabled:
                        # Asynchronous mode - detection only, processing happens in worker thread
                        tracked_pieces = self.detector.process_frame_async(
                            frame=frame,
                            current_count=self.camera.count
                        )
                        piece_image = None  # No immediate image in async mode
                    else:
                        # Synchronous mode - detection and immediate processing
                        tracked_pieces, piece_image = self.detector.process_frame(
                            frame=frame,
                            current_count=self.camera.count
                        )
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    print(f"Error processing frame: {str(e)}")
                    time.sleep(0.1)  # Short delay to avoid tight loop in case of errors
                    continue

                # In synchronous mode, process detected pieces immediately
                if not self.threading_enabled and piece_image is not None:
                    # Save the cropped image
                    file_path, image_number, error = self.camera.capture_image()

                    if error:
                        logger.error("Error capturing image: %s", error)
                        continue

                    # Save the cropped piece image
                    cv2.imwrite(file_path, piece_image)

                    # Send to API for identification
                    api_result = self.api_client.send_to_api(file_path)

                    if "error" in api_result:
                        logger.error("API error: %s", api_result["error"])
                        continue

                    # Process with sorting manager
                    result = self.sorting_manager.identify_piece(api_result)

                    if "error" in result:
                        logger.error("Sorting error: %s", result["error"])
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

                        # Update counter
                        self.processed_pieces += 1

                # Show debug view
                try:
                    debug_frame = self.detector.draw_debug(frame)

                    # Add status information
                    debug_frame = self._update_status(frame, debug_frame)

                    # Add servo information to debug frame
                    if not self.threading_enabled and hasattr(self,
                                                              'servo') and self.servo and self.servo.current_bin is not None:
                        bin_text = f"Current Bin: {self.servo.current_bin}"
                        cv2.putText(debug_frame, bin_text, (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Show frame
                    cv2.imshow("Lego Sorting System", debug_frame)
                except Exception as e:
                    logger.error(f"Error displaying debug frame: {str(e)}")
                    # Show original frame as fallback
                    cv2.imshow("Lego Sorting System", frame)

                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    logger.info("ESC key pressed, exiting")
                    self.running = False
                    break

        except Exception as e:
            logger.exception("Error in main loop: %s", str(e))
            print(f"Error: {str(e)}")
        finally:
            self.cleanup()
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources")

        # Stop threads if threading is enabled
        if self.threading_enabled and self.thread_manager:
            logger.info("Stopping worker threads")
            timeout = self.config_manager.get("threading", "shutdown_timeout", 5.0)
            self.thread_manager.stop_all_workers(timeout=timeout)

        # Release camera
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()

        # Release servo in synchronous mode (worker thread handles it in async mode)
        if not self.threading_enabled and hasattr(self, 'servo') and self.servo:
            self.servo.release()

        # Close windows
        cv2.destroyAllWindows()

        # Calculate and log final statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            logger.info("Final statistics: Processed %d pieces, %d frames in %.2f seconds (%.2f FPS)",
                       self.processed_pieces, self.frame_count, elapsed, fps)
            print(f"\nFinal statistics:")
            print(f"Processed {self.processed_pieces} pieces")
            print(f"Processed {self.frame_count} frames in {elapsed:.2f} seconds ({fps:.2f} FPS)")

        logger.info("Application shutdown complete")


# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lego Sorting System')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--calibrate-servo', action='store_true',
                        help='Run servo calibration at startup')
    parser.add_argument('--no-threading', action='store_true',
                        help='Disable multithreaded processing')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level')
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(console_level=log_level)
    logger.info("Starting Lego Sorting Application")

    # Load configuration
    config_manager = create_config_manager(args.config)

    # If servo calibration flag is set, update config
    if args.calibrate_servo:
        config_manager.set("servo", "calibration_mode", True)
        config_manager.save_config()
        logger.info("Servo calibration mode activated")

    # If threading is disabled, update config
    if args.no_threading:
        config_manager.set("threading", "enabled", False)
        config_manager.save_config()
        logger.info("Threading disabled via command line")

    # Create and run application
    application = LegoSortingApplication(config_path=args.config)
    application.run()