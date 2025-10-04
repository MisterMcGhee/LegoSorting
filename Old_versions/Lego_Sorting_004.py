"""
Lego_sorting_004.py - Enhanced main application for Lego piece sorting

This script serves as the entry point for the improved Lego sorting application,
integrating all modules with better coordination, error handling, and monitoring.
"""

import os
import argparse
import cv2
import time
import logging
import signal
import sys
from typing import Dict, Any

# Import modules
from camera_module import create_camera
from Old_versions.detector_module import create_detector
from Old_versions.sorting_module import create_sorting_manager
from config_management_module import create_config_manager
from thread_management_module import create_thread_manager
from processing_module import processing_worker_thread
from error_module import setup_logging, get_logger
from ui_module import create_ui_manager

# Initialize logger
logger = get_logger(__name__)


class SystemMonitor:
    """Monitors system health and resource usage"""

    def __init__(self, thread_manager, update_interval=5.0):
        """Initialize system monitor

        Args:
            thread_manager: Thread manager instance
            update_interval: Update interval in seconds
        """
        self.thread_manager = thread_manager
        self.update_interval = update_interval
        self.last_update_time = 0
        self.metrics = {
            "queue_size": 0,
            "fps": 0,
            "processed_count": 0,
            "error_count": 0,
            "camera_status": "unknown",
            "worker_status": "unknown",
            "api_response_time": 0
        }

    def update(self, current_time: float, fps: float, processed_count: int) -> None:
        """Update system metrics

        Args:
            current_time: Current time
            fps: Current FPS
            processed_count: Processed piece count
        """
        # Update time-dependent metrics on every call
        self.metrics["fps"] = fps
        self.metrics["processed_count"] = processed_count

        # Only refresh other metrics periodically to avoid overhead
        if current_time - self.last_update_time >= self.update_interval:
            self.metrics["queue_size"] = self.thread_manager.get_queue_size()
            self.last_update_time = current_time

            # Log periodic system status
            logger.debug(f"System status: FPS={fps:.1f}, Queue={self.metrics['queue_size']}, "
                         f"Processed={processed_count}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics

        Returns:
            Dict with system metrics
        """
        return self.metrics.copy()


class LegoSorting004:
    """Enhanced controller for the Lego sorting system"""

    def __init__(self, config_path="config.json"):
        """Initialize the Lego sorting application

        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Lego Sorting Application 004")

        # Flag to track initialization state
        self.initialized = False

        try:
            # Initialize configuration
            self.config_manager = create_config_manager(config_path)

            # Initialize thread manager
            self.thread_manager = create_thread_manager(self.config_manager)

            # Register callbacks for processing events
            self.thread_manager.register_callback("piece_processed", self._on_piece_processed)
            self.thread_manager.register_callback("piece_error", self._on_piece_error)

            # Create system monitor
            self.system_monitor = SystemMonitor(self.thread_manager)

            # Initialize components with config
            logger.info("Initializing camera...")
            self.camera = create_camera("webcam", self.config_manager)

            logger.info("Initializing detector...")
            self.detector = create_detector("conveyor", self.config_manager, self.thread_manager)

            logger.info("Initializing sorting manager...")
            self.sorting_manager = create_sorting_manager(self.config_manager)

            # Initialize UI manager
            self.ui_manager = create_ui_manager(self.config_manager)
            logger.info("UI manager initialized")
            # Only initialize API client and servo in the main thread if threading is disabled
            # Otherwise, these will be initialized in the worker thread
            self.api_client = None
            self.servo = None

            logger.info(f"Sorting strategy: {self.sorting_manager.get_strategy_description()}")

            # Status flags
            self.running = False
            self.processing_threads = []

            # Performance tracking
            self.frame_count = 0
            self.start_time = None
            self.fps = 0
            self.last_fps_update = 0
            self.processed_pieces = 0

            # Display options
            self.show_help = True  # Show help overlay initially
            self.show_metrics = True  # Show system metrics

            # Set initialized flag
            self.initialized = True

            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            logger.info("Initialization complete")

        except Exception as e:
            logger.exception(f"Initialization failed: {str(e)}")
            print(f"Error during initialization: {str(e)}")
            self.cleanup()
            raise

    def _signal_handler(self, sig, frame):
        """Handle termination signals

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {sig}, initiating shutdown")
        print("\nShutting down... Please wait.")
        self.running = False

    def _start_processing_thread(self):
        """Start the processing worker thread"""
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
            print("Warning: Failed to start processing thread. System may operate more slowly.")

    def _on_piece_processed(self, result):
        """Callback for when a piece is processed

        Args:
            result: Dictionary with processing result
        """
        self.processed_pieces += 1

        # Check if bin assignments have changed
        current_bin_mapping = self.sorting_manager.get_bin_mapping()
        if not hasattr(self, 'sorting_data') or current_bin_mapping != self.sorting_data.get("bin_assignments", {}):
            # Only update if different from current data
            self.sorting_data = {
                "strategy": self.sorting_manager.get_strategy_description(),
                "bin_assignments": current_bin_mapping
            }
            logger.debug("Sorting information updated with new bin assignments")

        # Log the result (existing code)
        logger.info(f"Piece processed: Element {result.get('element_id', 'Unknown')} "
                    f"({result.get('name', 'Unknown')}) to bin {result.get('bin_number', 9)}")

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
        """Callback for when a piece processing errors

        Args:
            piece_id: ID of the piece that errored
            error: Error message
        """
        logger.error(f"Error processing piece ID {piece_id}: {error}")
        print(f"\nError processing piece ID {piece_id}: {error}")

    def _add_system_overlay(self, frame):
        """Add system information overlay to the frame

        Args:
            frame: Input frame

        Returns:
            Frame with overlay
        """
        # Get current metrics
        metrics = self.system_monitor.get_metrics()

        # Create a semi-transparent overlay for the metrics panel
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Create a black rectangle for the top status bar
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)

        # Add FPS and processing status
        status_text = f"FPS: {self.fps:.1f} | Queue: {metrics['queue_size']} | Processed: {self.processed_pieces}"
        cv2.putText(overlay, status_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Only show detailed metrics if enabled
        if self.show_metrics:
            # Create a semi-transparent panel for detailed metrics
            cv2.rectangle(overlay, (w - 250, 40), (w - 10, 160), (0, 0, 0), -1)

            # Add metrics text
            cv2.putText(overlay, f"Queue Size: {metrics['queue_size']}", (w - 240, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, f"Processed: {self.processed_pieces}", (w - 240, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, f"Frame Count: {self.frame_count}", (w - 240, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add run time
            if self.start_time:
                run_time = time.time() - self.start_time
                time_text = f"Run Time: {int(run_time // 3600):02d}:{int((run_time % 3600) // 60):02d}:{int(run_time % 60):02d}"
                cv2.putText(overlay, time_text, (w - 240, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add help overlay if enabled
        if self.show_help:
            # Create a semi-transparent panel for help
            cv2.rectangle(overlay, (10, h - 110), (300, h - 10), (0, 0, 0), -1)

            # Add help text
            cv2.putText(overlay, "Controls:", (20, h - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(overlay, "ESC - Exit Program", (20, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, "H - Toggle Help", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay, "M - Toggle Metrics", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend the overlay with the original frame
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def _process_keyboard_input(self, key: int) -> bool:
        """Process keyboard input

        Args:
            key: Key code

        Returns:
            True if processing should continue, False to exit
        """
        if key == 27:  # ESC key
            logger.info("ESC key pressed, exiting")
            return False
        elif key == ord('h') or key == ord('H'):  # Toggle help
            self.show_help = not self.show_help
            logger.debug(f"Help overlay toggled: {self.show_help}")
        elif key == ord('m') or key == ord('M'):  # Toggle metrics
            self.show_metrics = not self.show_metrics
            logger.debug(f"Metrics overlay toggled: {self.show_metrics}")

        return True

    def initialize_system(self):
        """Initialize the system for operation"""
        if not self.initialized:
            logger.error("Cannot initialize system - application not properly initialized")
            return False

        try:
            logger.info("Initializing sorting system")
            print("\nInitializing sorting system...")

            # Get initial frame for ROI selection
            frame = self.camera.get_preview_frame()
            if frame is None:
                error_msg = "Failed to get preview frame from camera"
                logger.error(error_msg)
                print(f"Error: {error_msg}")
                return False

            # Initialize detector with ROI (from config if available)
            self.detector.load_roi_from_config(frame, self.config_manager)

            print("\nSetup complete!")

            # Always start processing thread. Previously this was conditional.
            self._start_processing_thread()
            print("Started processing thread")

            # Initialize overflow bin position in async mode
            # This will be controlled by the worker thread for specific bins
            overflow_bin = self.config_manager.get("sorting", "overflow_bin", 9)
            print(f"Default overflow bin: {overflow_bin}")

            # Initialize sorting information for UI
            self.sorting_data = {
                "strategy": self.sorting_manager.get_strategy_description(),
                "bin_assignments": self.sorting_manager.get_bin_mapping()
            }
            logger.info(f"Initialized sorting information: {self.sorting_data['strategy']}")

            print("\nSorting system ready. Press ESC to exit.")
            return True

        except Exception as e:
            logger.exception(f"System initialization failed: {str(e)}")
            print(f"Error: {str(e)}")
            return False

    def run(self):
        """Run the main sorting application loop"""
        if not self.initialized:
            logger.error("Cannot run - application not properly initialized")
            return

        # Initialize the system
        if not self.initialize_system():
            return

        try:
            # Set running flag and reset counters
            self.running = True
            self.frame_count = 0
            self.start_time = time.time()
            self.last_fps_update = self.start_time
            self.processed_pieces = 0

            # Main processing loop
            while self.running:
                loop_start_time = time.time()

                # Get frame from camera
                frame = self.camera.get_preview_frame()
                if frame is None:
                    logger.warning("Failed to get preview frame")
                    time.sleep(0.1)  # Small delay to prevent CPU spinning
                    continue

                # Increment frame counter
                self.frame_count += 1

                # Update FPS calculation
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    elapsed = current_time - self.start_time
                    self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                    self.last_fps_update = current_time

                # Update system monitor
                self.system_monitor.update(current_time, self.fps, self.processed_pieces)

                # Process frame - using the unified process_frame method
                tracked_pieces, piece_image, increment_signal = self.detector.process_frame(
                    frame=frame,
                    current_count=self.camera.count
                )

                # Then use increment_signal appropriately (probably to increment camera.count)
                if increment_signal:
                    self.camera.count += 1

                # Get debug visualization from detector
                debug_frame = self.detector.draw_debug(frame)

                # Add system information overlay
                display_frame = self._add_system_overlay(debug_frame)

                # Pass sorting information to UI manager
                if hasattr(self, 'ui_manager') and hasattr(self, 'sorting_data'):
                    # If you have a dedicated UI manager:
                    display_frame = self.ui_manager.add_sorting_info(display_frame, self.sorting_data)
                elif hasattr(self, 'sorting_data'):
                    # If you're directly rendering in the system overlay function:
                    # You'll need to modify _add_system_overlay to handle sorting_data
                    pass

                # Show frame
                cv2.imshow("Lego Sorting System 004", display_frame)

                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._process_keyboard_input(key):
                    break

                # Control loop timing if needed
                loop_time = time.time() - loop_start_time
                if loop_time < 0.01:  # Target ~100 FPS max
                    time.sleep(0.01 - loop_time)

        except Exception as e:
            logger.exception(f"Error in main loop: {str(e)}")
            print(f"Error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")

        # Stop threads
        if hasattr(self, 'thread_manager') and self.thread_manager:
            logger.info("Stopping worker threads")

            # Get timeout from config or use default
            timeout = 5.0
            if hasattr(self, 'config_manager') and self.config_manager:
                timeout = self.config_manager.get("threading", "shutdown_timeout", 5.0)

            self.thread_manager.stop_all_workers(timeout=timeout)

        # Release camera
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()

        # Close windows
        cv2.destroyAllWindows()

        # Calculate and log final statistics
        if hasattr(self, 'start_time') and self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0

            logger.info(f"Final statistics: Processed {self.processed_pieces} pieces, "
                        f"{self.frame_count} frames in {elapsed:.2f} seconds ({fps:.2f} FPS)")

            print(f"\nFinal statistics:")
            print(f"Processed {self.processed_pieces} pieces")
            print(f"Processed {self.frame_count} frames in {elapsed:.2f} seconds ({fps:.2f} FPS)")

        logger.info("Application shutdown complete")


def configure_logging(log_level_name: str) -> None:
    """Configure logging with the specified level

    Args:
        log_level_name: Logging level name (DEBUG, INFO, etc.)
    """
    # Convert string level to logging constant
    log_level = getattr(logging, log_level_name)

    # Configure file logging with DEBUG level
    setup_logging(console_level=log_level)

    # Configure console to be less verbose
    console_handler = logging.getLogger().handlers[1]  # The second handler is the console
    console_handler.setLevel(max(log_level, logging.INFO))  # Make sure console is at least INFO level

    logger.info(f"Logging initialized with file level=DEBUG, console level={log_level_name}")


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lego Sorting System 004')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--calibrate-servo', action='store_true',
                        help='Run servo calibration at startup')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level')
    args = parser.parse_args()

    # Configure logging
    configure_logging(args.log_level)

    print("\n========================================")
    print("    Lego Sorting System 004")
    print("========================================")

    try:
        # Load configuration
        config_manager = create_config_manager(args.config)

        # Update config based on command line arguments
        if args.calibrate_servo:
            config_manager.set("servo", "calibration_mode", True)
            config_manager.save_config()
            logger.info("Servo calibration mode activated")
            print("Servo calibration mode activated")

        # Create and run application
        application = LegoSorting004(config_path=args.config)
        application.run()

    except Exception as e:
        logger.exception(f"Application error: {str(e)}")
        print(f"\nCritical error: {str(e)}")
        print("Please check the log file for more details.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
