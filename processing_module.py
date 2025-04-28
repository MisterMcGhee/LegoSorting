"""
processing_module.py - Asynchronous processing for the Lego sorting application

This module implements the consumer thread functionality for processing detected
Lego pieces. It handles image saving, API calls, and sorting decisions asynchronously
from the main detection and UI thread.
"""

import os
import time
import threading
import cv2
import logging
from typing import Dict, Any, Optional, Tuple, Callable

# Import from other modules
from thread_management_module import ThreadManager, PieceMessage
from error_module import get_logger, retry_on_error, APIError, SortingError, TimeoutError
from api_module import create_api_client
from sorting_module import create_sorting_manager
from arduino_servo_module import create_arduino_servo_module

# Initialize module logger
logger = get_logger(__name__)


class ProcessingWorker:
    """Worker that processes detected pieces asynchronously."""

    def __init__(self, thread_manager: ThreadManager, config_manager: Any,
                 save_directory: str = "LegoPictures", filename_prefix: str = "Lego",
                 piece_history=None):
        """Initialize processing worker.

        Args:
            thread_manager: Thread manager instance for message passing
            config_manager: Configuration manager
            save_directory: Directory to save piece images
            filename_prefix: Prefix for saved image filenames
            piece_history: PieceHistory instance for tracking processed pieces
        """
        logger.info("Initializing processing worker")

        self.thread_manager = thread_manager
        self.config_manager = config_manager
        self.save_directory = save_directory
        self.filename_prefix = filename_prefix
        self.piece_history = piece_history  # Store piece_history instance

        # Ensure save directory exists
        self.save_directory = os.path.abspath(save_directory)
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Initialize components
        self.api_client = create_api_client("brickognize", config_manager)
        self.sorting_manager = create_sorting_manager(config_manager)

        # Initialize Arduino servo control
        self.servo = create_arduino_servo_module(config_manager)
        logger.info("Processing worker using Arduino-based servo control")

        # Get configuration values
        threading_config = config_manager.get_section("threading")
        self.api_timeout = threading_config.get("api_timeout", 30.0)
        self.processing_timeout = threading_config.get("processing_timeout", 60.0)
        self.polling_interval = threading_config.get("polling_interval", 0.01)

        # Exit zone trigger configuration
        self.exit_zone_trigger_config = config_manager.get_section("exit_zone_trigger")
        self.exit_zone_enabled = self.exit_zone_trigger_config.get("enabled", True)
        self.fall_time = self.exit_zone_trigger_config.get("fall_time", 1.0)
        self.cooldown_time = self.exit_zone_trigger_config.get("cooldown_time", 0.5)

        logger.info(f"Exit zone trigger enabled: {self.exit_zone_enabled}")
        logger.info(f"Fall time: {self.fall_time} seconds")
        logger.info(f"Cooldown time: {self.cooldown_time} seconds")

        # Status tracking
        self.should_exit = threading.Event()
        self.running = False
        self.current_message = None
        self.last_servo_move_time = 0  # Track when the servo was last moved

        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None

        logger.info("Processing worker initialized")

    def start(self) -> None:
        """Start the processing worker."""
        if self.running:
            logger.warning("Processing worker already running")
            return

        logger.info("Starting processing worker")
        self.running = True
        self.should_exit.clear()
        self.start_time = time.time()
        self.processed_count = 0
        self.error_count = 0

    def stop(self) -> None:
        """Stop the processing worker."""
        logger.info("Stopping processing worker")
        self.should_exit.set()
        self.running = False

    def _save_image(self, message: PieceMessage) -> Tuple[str, int]:
        """Save piece image to disk.

        Args:
            message: Message containing piece information

        Returns:
            tuple: (file_path, image_number)
        """
        # Generate unique number for this image
        number = self._get_next_image_number()

        # Create filename with padded number
        filename = f"{self.filename_prefix}{number:03d}.jpg"
        file_path = os.path.join(self.save_directory, filename)

        # Save image to disk
        cv2.imwrite(file_path, message.image)
        logger.debug("Saved image to %s", file_path)

        return file_path, number

    def _get_next_image_number(self) -> int:
        """Get the next available image number.

        Returns:
            int: Next available image number
        """
        # Find existing files to determine next number
        existing_files = [f for f in os.listdir(self.save_directory)
                          if f.startswith(self.filename_prefix) and f.endswith(".jpg")]

        if not existing_files:
            return 1

        # Extract numbers from filenames and find the highest
        numbers = []
        for f in existing_files:
            try:
                # Get number from filename (e.g., Lego001.jpg -> 1)
                num_str = f[len(self.filename_prefix):len(self.filename_prefix) + 3]
                numbers.append(int(num_str))
            except (ValueError, IndexError):
                continue

        if not numbers:
            return 1

        return max(numbers) + 1

    @retry_on_error(max_attempts=3, delay=1.0, exceptions=(APIError,))
    def _call_api(self, file_path: str) -> Dict[str, Any]:
        """Send image to API with retry capability.

        Args:
            file_path: Path to the image file

        Returns:
            dict: API response

        Raises:
            APIError: If API call fails after all retries
            TimeoutError: If API call exceeds timeout
        """
        logger.info("Sending image to API: %s", os.path.basename(file_path))

        # Use lock to prevent multiple simultaneous API calls
        def api_call():
            return self.api_client.send_to_api(file_path)

        # Execute with timeout
        start_time = time.time()
        result = self.thread_manager.with_lock("api", api_call)

        # Check for timeout
        elapsed = time.time() - start_time
        if elapsed > self.api_timeout:
            logger.warning("API call took %.2f seconds (exceeds timeout of %.2f)",
                           elapsed, self.api_timeout)

        # Check for error in result
        if "error" in result:
            raise APIError(result["error"])

        return result

    def _identify_and_sort(self, api_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process API result and determine sorting bin.

        Args:
            api_result: Result from Brickognize API

        Returns:
            dict: Sorting result with bin number

        Raises:
            SortingError: If sorting fails
        """
        logger.debug("Processing API result for sorting")

        # Use lock to prevent conflicts in sorting manager
        def sort_piece():
            return self.sorting_manager.identify_piece(api_result)

        result = self.thread_manager.with_lock("sorting", sort_piece)

        # Check for error in result
        if "error" in result:
            raise SortingError(result["error"])

        return result

    def _move_servo(self, bin_number: int, is_exit_zone_piece: bool = False) -> bool:
        """Move servo to the specified bin position.

        Args:
            bin_number: Target bin number
            is_exit_zone_piece: Whether this is a piece from the exit zone

        Returns:
            bool: True if successful, False otherwise
        """
        current_time = time.time()

        # If this is an exit zone piece, apply fall time delay
        if is_exit_zone_piece and self.exit_zone_enabled:
            # Calculate remaining time to wait based on fall time
            wait_time = self.cooldown_time
            if current_time - self.last_servo_move_time < wait_time:
                # Need to wait for cooldown from previous piece
                wait_remaining = wait_time - (current_time - self.last_servo_move_time)
                logger.info(
                    f"Waiting {wait_remaining:.2f} seconds for servo cooldown before moving to bin {bin_number}")
                time.sleep(wait_remaining)

        logger.debug("Moving servo to bin %d", bin_number)

        # Use lock to prevent conflicts in servo control
        def move():
            return self.servo.move_to_bin(bin_number)

        result = self.thread_manager.with_lock("servo", move)

        # Update last servo move time
        self.last_servo_move_time = time.time()

        return result

    def _process_message(self, message: PieceMessage) -> Optional[Dict[str, Any]]:
        """Process a piece message through the complete pipeline.

        Args:
            message: Message containing piece information

        Returns:
            dict: Processing result or None if failed
        """
        try:
            # Update message status
            message.status = "processing"
            self.current_message = message

            # Check if this is a high-priority exit zone piece
            is_exit_zone_piece = message.priority == 0  # Priority 0 is set for exit zone pieces

            if is_exit_zone_piece:
                logger.info(f"Processing exit zone piece (ID: {message.piece_id})")

            # Start timing
            start_time = time.time()

            # Step 1: Save piece image
            file_path, image_number = self._save_image(message)

            # Step 2: Call API for identification
            api_result = self._call_api(file_path)

            # Step 3: Determine sorting bin
            api_result['image_number'] = image_number
            sorting_result = self._identify_and_sort(api_result)

            # Step 4: Move servo to bin position
            bin_number = sorting_result.get("bin_number",
                                            self.config_manager.get("sorting", "overflow_bin", 9))

            # Move the servo, indicating if this is an exit zone piece
            servo_success = self._move_servo(bin_number, is_exit_zone_piece)

            # Create result dictionary
            result = {
                "piece_id": message.piece_id,
                "image_number": image_number,
                "file_path": file_path,
                "element_id": sorting_result.get("element_id", "Unknown"),
                "name": sorting_result.get("name", "Unknown"),
                "primary_category": sorting_result.get("primary_category", "Unknown"),
                "secondary_category": sorting_result.get("secondary_category", "Unknown"),
                "bin_number": bin_number,
                "servo_success": servo_success,
                "processing_time": time.time() - start_time,
                "is_exit_zone_piece": is_exit_zone_piece
            }

            # Add piece to history if piece_history is available
            if self.piece_history:
                self.piece_history.add_piece(result)

            # Update message status and result
            message.status = "completed"
            message.result = result

            # Update statistics
            self.processed_count += 1

            # Trigger callback
            self.thread_manager.trigger_callback("piece_processed", result=result)

            logger.info("Processed piece #%d (%s) to bin %d in %.2f seconds%s",
                        image_number, result["element_id"], bin_number, result["processing_time"],
                        " (exit zone piece)" if is_exit_zone_piece else "")

            return result

        except Exception as e:
            # Update statistics
            self.error_count += 1

            # Log error
            logger.error("Error processing piece: %s", str(e))

            # Update message status
            message.status = "failed"
            message.result = {"error": str(e), "error_type": type(e).__name__}

            # Trigger error callback with result data
            self.thread_manager.trigger_callback("piece_error",
                                                 piece_id=message.piece_id,
                                                 error=str(e),
                                                 result=message.result)

            # Send to overflow bin on error
            overflow_bin = self.config_manager.get("sorting", "overflow_bin", 9)
            self._move_servo(overflow_bin)

            return None

    def run(self) -> None:
        """Main processing loop for the worker thread."""
        logger.info("Processing worker running")

        try:
            while self.running and not self.should_exit.is_set():
                # Get next message from queue with timeout
                message = self.thread_manager.get_next_message(timeout=self.polling_interval)

                # Skip if no message
                if message is None:
                    continue

                try:
                    # Process message
                    self._process_message(message)

                except Exception as e:
                    # Log unexpected errors
                    logger.exception("Unexpected error processing message: %s", str(e))

                finally:
                    # Mark task as done
                    self.thread_manager.task_done()

        except Exception as e:
            logger.exception("Fatal error in processing worker: %s", str(e))

        finally:
            # Clean up
            logger.info("Processing worker stopped (processed: %d, errors: %d)",
                        self.processed_count, self.error_count)
            self.running = False
            self.current_message = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics.

        Returns:
            dict: Statistics dictionary
        """
        stats = {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "running": self.running,
            "queue_size": self.thread_manager.get_queue_size(),
            "uptime": time.time() - self.start_time if self.start_time else 0
        }

        if self.processed_count > 0 and stats["uptime"] > 0:
            stats["pieces_per_second"] = self.processed_count / stats["uptime"]
        else:
            stats["pieces_per_second"] = 0

        return stats


# Factory function
def create_processing_worker(thread_manager: ThreadManager, config_manager: Any,
                             piece_history=None) -> ProcessingWorker:
    """Create a processing worker instance.

    Args:
        thread_manager: Thread manager instance
        config_manager: Configuration manager instance
        piece_history: PieceHistory instance for tracking processed pieces

    Returns:
        ProcessingWorker instance
    """
    # Get camera configuration for save directory and filename prefix
    save_directory = config_manager.get("camera", "directory", "LegoPictures")
    filename_prefix = config_manager.get("camera", "filename_prefix", "Lego")

    return ProcessingWorker(thread_manager, config_manager, save_directory,
                            filename_prefix, piece_history)


# Worker thread function
def processing_worker_thread(thread_manager: ThreadManager, config_manager: Any,
                             piece_history=None) -> None:
    """Worker thread function for processing pieces.

    This function is meant to be run in a separate thread and handles
    the complete processing pipeline for detected pieces.

    Args:
        thread_manager: Thread manager instance
        config_manager: Configuration manager instance
        piece_history: PieceHistory instance for tracking processed pieces
    """
    # Create worker
    worker = create_processing_worker(thread_manager, config_manager, piece_history)

    try:
        # Start worker
        worker.start()

        # Run processing loop
        worker.run()

    except Exception as e:
        logger.exception("Error in processing worker thread: %s", str(e))

    finally:
        # Stop worker
        worker.stop()
