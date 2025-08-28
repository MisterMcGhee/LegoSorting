"""
processing_module.py - Asynchronous processing for the Lego sorting application

This module implements the consumer thread functionality for processing detected
Lego pieces. It handles image saving, API calls, and sorting decisions asynchronously
from the main detection and UI thread.

ARCHITECTURE NOTES:
- This module runs in a separate thread from detection
- It consumes pieces from PieceQueueManager (not ThreadManager)
- It uses ThreadManager only for thread-safe locks
- PieceHistory is now a required dependency (not optional)

MAIN RESPONSIBILITIES:
1. Get pieces from the priority queue
2. Save piece images with sequential numbering
3. Call API for piece identification
4. Determine sorting bin via sorting module
5. Control servo movement
6. Record piece history
7. Handle errors and retries
"""

import os
import time
import threading
import cv2
import logging
from typing import Dict, Any, Optional, Tuple, Callable

# Import from other modules - NOTE THE CHANGES HERE
from piece_queue_manager import PieceQueueManager, PieceMessage
from thread_manager import ThreadManager
from piece_history_module import PieceHistory
from error_module import get_logger, retry_on_error, APIError, SortingError, TimeoutError
from api_module import create_api_client
from sorting_module import create_sorting_manager
from arduino_servo_module import create_arduino_servo_module
from enhanced_config_manager import ModuleConfig

# Initialize module logger
logger = get_logger(__name__)


# ============================================================================
# MAIN PROCESSING WORKER CLASS
# ============================================================================

class ProcessingWorker:
    """
    Worker that processes detected pieces asynchronously.

    This class orchestrates the entire processing pipeline for each piece:
    - Image saving
    - API identification
    - Sorting decisions
    - Servo control
    - History recording

    It runs in a separate thread and pulls pieces from a priority queue.
    """

    def __init__(self,
                 queue_manager: PieceQueueManager,  # CHANGED: Now uses PieceQueueManager
                 thread_manager: ThreadManager,  # ADDED: Separate ThreadManager for locks
                 config_manager: Any,
                 piece_history: PieceHistory,
                 save_directory: str = "LegoPictures",
                 filename_prefix: str = "Lego"):
        """
        Initialize processing worker with all required dependencies.

        Args:
            queue_manager: Manages the piece queue (get pieces, mark complete/failed)
            thread_manager: Provides thread-safe locks for shared resources
            config_manager: Configuration manager for all settings
            save_directory: Directory to save piece images
            filename_prefix: Prefix for saved image filenames
            piece_history: REQUIRED - Tracks processed pieces for inventory

        Raises:
            ValueError: If piece_history is None (it's required now)
        """
        logger.info("Initializing processing worker")

        # ====== DEPENDENCY VALIDATION SECTION ======
        # Ensure all required dependencies are provided
        if piece_history is None:
            raise ValueError("piece_history is required and cannot be None")

        # ====== DEPENDENCY STORAGE SECTION ======
        # Store all the managers we'll use
        self.queue_manager = queue_manager  # For queue operations
        self.thread_manager = thread_manager  # For thread safety
        self.config_manager = config_manager  # For configuration
        self.piece_history = piece_history  # For history tracking

        # ====== IMAGE STORAGE CONFIGURATION SECTION ======
        # Set up where and how to save images
        self.save_directory = save_directory
        self.filename_prefix = filename_prefix

        # Ensure save directory exists
        self.save_directory = os.path.abspath(save_directory)
        logger.info(f"Saving images to directory: {os.path.abspath(self.save_directory)}")

        if not os.path.exists(self.save_directory):
            try:
                os.makedirs(self.save_directory)
                logger.info(f"Created directory: {self.save_directory}")
            except Exception as e:
                logger.error(f"Failed to create directory: {e}")
        else:
            logger.info(f"Directory already exists: {self.save_directory}")

        # ====== MODULE INITIALIZATION SECTION ======
        # Initialize the modules we'll use for processing
        self.api_client = create_api_client("brickognize", config_manager)
        self.sorting_manager = create_sorting_manager(config_manager)
        self.servo = create_arduino_servo_module(config_manager)

        logger.info("Processing worker using Arduino-based servo control")

        # ====== CONFIGURATION LOADING SECTION ======
        # Load all configuration values we'll need
        threading_config = config_manager.get_module_config(ModuleConfig.THREADING.value)
        self.api_timeout = threading_config["api_timeout"]
        self.processing_timeout = threading_config["processing_timeout"]
        self.polling_interval = threading_config["polling_interval"]

        # Get exit zone trigger configuration
        exit_zone_config = config_manager.get_module_config(ModuleConfig.EXIT_ZONE.value)
        self.exit_zone_enabled = exit_zone_config["enabled"]
        self.fall_time = exit_zone_config["fall_time"]
        self.cooldown_time = exit_zone_config["cooldown_time"]

        logger.info(f"Exit zone trigger enabled: {self.exit_zone_enabled}")
        logger.info(f"Fall time: {self.fall_time} seconds")
        logger.info(f"Cooldown time: {self.cooldown_time} seconds")

        # ====== WORKER STATE MANAGEMENT SECTION ======
        # Variables to track the state of this worker
        self.should_exit = threading.Event()  # Signal to stop processing
        self.running = False  # Is the worker running?
        self.current_message = None  # What piece are we processing?
        self.last_servo_move_time = 0  # When did servo last move?

        # ====== STATISTICS TRACKING SECTION ======
        # Track performance metrics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None

        logger.info("Processing worker initialized")

    # ========================================================================
    # LIFECYCLE MANAGEMENT SECTION
    # ========================================================================
    # Methods that control the worker's lifecycle (start, stop, run)

    def start(self) -> None:
        """
        Start the processing worker.

        This prepares the worker to begin processing but doesn't start
        the main loop - that happens in run().
        """
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
        """
        Stop the processing worker gracefully.

        This signals the worker to stop after completing the current piece.
        """
        logger.info("Stopping processing worker")
        self.should_exit.set()
        self.running = False

    def run(self) -> None:
        """
        Main processing loop for the worker thread.

        This is the heart of the processing module. It:
        1. Gets pieces from the queue
        2. Processes each piece through the pipeline
        3. Handles errors and updates statistics
        4. Runs until told to stop
        """
        logger.info("Processing worker running")
        last_empty_log_time = 0  # Track when we last logged an empty queue

        try:
            # ====== MAIN PROCESSING LOOP ======
            # Keep running until we're told to stop
            while self.running and not self.should_exit.is_set():

                # CHANGED: Use queue_manager instead of thread_manager
                # Get next piece from the priority queue
                message = self.queue_manager.get_next_piece(timeout=self.polling_interval)

                # Handle empty queue
                if message is None:
                    current_time = time.time()
                    # Only log periodically to avoid spam
                    if current_time - last_empty_log_time > 5.0:
                        logger.debug(f"Queue empty, waiting for messages (poll interval: {self.polling_interval}s)")
                        last_empty_log_time = current_time
                    continue

                # Reset the empty log timer when we get a message
                last_empty_log_time = 0

                # Process the piece
                logger.debug(f"Processing message for piece ID {message.piece_id}")

                try:
                    # ====== PIECE PROCESSING ======
                    # Run the piece through the entire pipeline
                    result = self._process_message(message)

                    if result:
                        # Success - mark piece as completed
                        self.queue_manager.mark_piece_completed(message.piece_id, result)

                        # Trigger success callback
                        self.queue_manager.trigger_external_callback(
                            "piece_processed",
                            piece_id=message.piece_id,
                            result=result
                        )
                    else:
                        # Failure - mark piece as failed
                        self.queue_manager.mark_piece_failed(
                            message.piece_id,
                            "Processing failed",
                            retry=True  # Allow retry
                        )

                        # Trigger failure callback
                        self.queue_manager.trigger_external_callback(
                            "piece_failed",
                            piece_id=message.piece_id,
                            error="Processing failed"
                        )

                except Exception as e:
                    # Unexpected error - log and mark as failed
                    logger.exception(f"Unexpected error processing piece {message.piece_id}: {e}")

                    self.queue_manager.mark_piece_failed(
                        message.piece_id,
                        str(e),
                        retry=False  # Don't retry on unexpected errors
                    )

                    # Trigger error callback
                    self.queue_manager.trigger_external_callback(
                        "piece_error",
                        piece_id=message.piece_id,
                        error=str(e)
                    )

        except Exception as e:
            logger.exception(f"Fatal error in processing worker: {e}")

        finally:
            # ====== CLEANUP SECTION ======
            # Clean up when the worker stops
            logger.info(f"Processing worker stopped (processed: {self.processed_count}, errors: {self.error_count})")
            self.running = False
            self.current_message = None

    # ========================================================================
    # MAIN PROCESSING PIPELINE SECTION
    # ========================================================================
    # The core method that processes a single piece through all stages

    def _process_message(self, message: PieceMessage) -> Optional[Dict[str, Any]]:
        """
        Process a piece message through the complete pipeline.

        This is the main orchestration method that:
        1. Saves the piece image
        2. Calls the API for identification
        3. Determines the sorting bin
        4. Moves the servo
        5. Records history

        Args:
            message: Message containing piece information

        Returns:
            Dictionary with all processing results, or None if failed
        """
        try:
            # ====== PREPARATION SECTION ======
            # Update message status and check priority
            message.status = "processing"
            self.current_message = message

            # Check if this is a high-priority exit zone piece
            is_exit_zone_piece = message.priority == 0  # Priority 0 = urgent!

            if is_exit_zone_piece:
                logger.info(f"Processing exit zone piece (ID: {message.piece_id})")

            # Start timing for performance tracking
            start_time = time.time()

            # ====== STEP 1: SAVE IMAGE ======
            # Save piece image with sequential numbering
            file_path, image_number = self._save_image(message)

            # ====== STEP 2: API IDENTIFICATION ======
            # Send image to Brickognize API for identification
            api_result = self._call_api(file_path)

            # ====== STEP 3: SORTING DECISION ======
            # Determine which bin this piece should go to
            api_result['image_number'] = image_number  # Add image number to result
            sorting_result = self._identify_and_sort(api_result)

            # ====== STEP 4: SERVO MOVEMENT ======
            # Move the servo to direct piece to the correct bin
            bin_number = sorting_result.get("bin_number",
                                            self.config_manager.get("sorting", "overflow_bin", 9))

            servo_success = self._move_servo(bin_number, is_exit_zone_piece)

            # ====== STEP 5: CREATE RESULT DICTIONARY ======
            # Compile all the information about this piece
            result = {
                "piece_id": message.piece_id,
                "image_number": image_number,
                "file_path": file_path,
                "element_id": sorting_result.get("element_id", "Unknown"),
                "name": sorting_result.get("name", "Unknown"),
                "primary_category": sorting_result.get("primary_category", ""),
                "secondary_category": sorting_result.get("secondary_category", ""),
                "tertiary_category": sorting_result.get("tertiary_category", ""),
                "bin_number": bin_number,
                "servo_success": servo_success,
                "processing_time": time.time() - start_time,
                "is_exit_zone_piece": is_exit_zone_piece
            }

            # Log the result for debugging
            logger.debug(f"Created result dictionary for piece {message.piece_id}:")
            logger.debug(f"  Image: {image_number}, Element ID: {result['element_id']}")
            logger.debug(f"  Name: {result['name']}, Bin: {bin_number}")

            # ====== STEP 6: RECORD HISTORY ======
            # Add piece to history (now guaranteed to exist)
            try:
                logger.info(f"Adding piece ID {message.piece_id} to piece history")
                self.piece_history.add_piece(result)
                logger.info(f"Successfully added piece ID {message.piece_id} to piece history")
            except Exception as e:
                logger.error(f"Error adding piece to history: {e}", exc_info=True)
                # Don't fail the whole process if history fails

            # ====== SUCCESS TRACKING ======
            # Update statistics
            self.processed_count += 1

            logger.info(f"Processed piece #{image_number} ({result['element_id']}) to bin {bin_number} "
                        f"in {result['processing_time']:.2f} seconds"
                        f"{' (exit zone piece)' if is_exit_zone_piece else ''}")

            return result

        except Exception as e:
            # ====== ERROR HANDLING SECTION ======
            # Handle any errors that occur during processing
            self.error_count += 1
            logger.error(f"Error processing piece: {e}")

            # Try to move servo to overflow bin on error
            try:
                overflow_bin = self.config_manager.get("sorting", "overflow_bin", 9)
                # CHANGED: Use thread_manager for thread-safe servo move
                self.thread_manager.with_lock("servo", self.servo.move_to_bin, overflow_bin)
            except Exception as servo_error:
                logger.error(f"Failed to move servo to overflow bin: {servo_error}")

            return None

    # ========================================================================
    # IMAGE HANDLING SECTION
    # ========================================================================
    # Methods for saving and managing piece images

    def _save_image(self, message: PieceMessage) -> Tuple[str, int]:
        """
        Save the piece image with a sequential number label.

        This method:
        1. Determines the next available image number
        2. Adds a visible red number to the image
        3. Saves the image to disk

        Args:
            message: Message containing the piece image

        Returns:
            Tuple of (file_path, image_number)
        """
        # Get the next sequential image number
        number = self._get_next_image_number()

        # Create a copy of the image to avoid modifying the original
        labeled_image = message.image.copy()

        # Add visible red number to top-left corner
        text_y = 30
        cv2.putText(labeled_image, f"{number}",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)  # Red color, thickness 2

        # Create filename with padded number (e.g., Lego001.jpg)
        filename = f"{self.filename_prefix}{number:03d}.jpg"
        file_path = os.path.join(self.save_directory, filename)

        # Save the labeled image to disk
        cv2.imwrite(file_path, labeled_image)
        logger.debug(f"Saved image to {file_path}")

        return file_path, number

    def _get_next_image_number(self) -> int:
        """
        Get the next available image number by scanning existing files.

        This ensures we don't overwrite existing images if the program
        is restarted.

        Returns:
            Next available image number (starting from 1)
        """
        # Find all existing image files
        existing_files = [f for f in os.listdir(self.save_directory)
                          if f.startswith(self.filename_prefix) and f.endswith(".jpg")]

        if not existing_files:
            return 1

        # Extract numbers from filenames
        numbers = []
        for f in existing_files:
            try:
                # Extract the number part (e.g., "001" from "Lego001.jpg")
                num_str = f[len(self.filename_prefix):len(self.filename_prefix) + 3]
                numbers.append(int(num_str))
            except (ValueError, IndexError):
                continue

        if not numbers:
            return 1

        # Return one more than the highest existing number
        return max(numbers) + 1

    # ========================================================================
    # API INTERACTION SECTION
    # ========================================================================
    # Methods for calling the Brickognize API

    @retry_on_error(max_attempts=3, delay=1.0, exceptions=(APIError,))
    def _call_api(self, file_path: str) -> Dict[str, Any]:
        """
        Send image to API with retry capability and thread safety.

        This method uses the @retry_on_error decorator to automatically
        retry failed API calls up to 3 times.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with API response

        Raises:
            APIError: If API call fails after all retries
            TimeoutError: If API call exceeds timeout
        """
        logger.info(f"Sending image to API: {os.path.basename(file_path)}")

        # CHANGED: Use thread_manager.with_lock for thread safety
        # This ensures only one API call happens at a time
        def api_call():
            return self.api_client.send_to_api(file_path)

        # Execute with thread safety
        start_time = time.time()
        result = self.thread_manager.with_lock("api", api_call)

        # Check for timeout
        elapsed = time.time() - start_time
        if elapsed > self.api_timeout:
            logger.warning(f"API call took {elapsed:.2f} seconds (exceeds timeout of {self.api_timeout:.2f})")

        # Check for error in result
        if "error" in result:
            raise APIError(result["error"])

        return result

    # ========================================================================
    # SORTING LOGIC SECTION
    # ========================================================================
    # Methods for determining piece sorting

    def _identify_and_sort(self, api_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process API result and determine sorting bin.

        This method uses the sorting_manager to:
        1. Look up piece categories from the database
        2. Apply sorting strategy to determine bin

        Args:
            api_result: Result from Brickognize API

        Returns:
            Dictionary with sorting result including bin number

        Raises:
            SortingError: If sorting fails
        """
        logger.debug("Processing API result for sorting")

        # CHANGED: Use thread_manager.with_lock for thread safety
        def sort_piece():
            return self.sorting_manager.identify_piece(api_result)

        result = self.thread_manager.with_lock("sorting", sort_piece)

        # Check for error in result
        if "error" in result:
            raise SortingError(result["error"])

        return result

    # ========================================================================
    # SERVO CONTROL SECTION
    # ========================================================================
    # Methods for controlling the physical servo

    def _move_servo(self, bin_number: int, is_exit_zone_piece: bool = False) -> bool:
        """
        Move servo to the specified bin position with timing control.

        This method handles:
        1. Cooldown timing between moves
        2. Special timing for exit zone pieces
        3. Thread-safe servo control

        Args:
            bin_number: Target bin number (0-9)
            is_exit_zone_piece: Whether this piece is from exit zone

        Returns:
            True if servo moved successfully
        """
        current_time = time.time()

        # ====== TIMING CONTROL SECTION ======
        # Handle cooldown and fall time for exit zone pieces
        if is_exit_zone_piece and self.exit_zone_enabled:
            # Calculate remaining time to wait based on cooldown
            wait_time = self.cooldown_time
            if current_time - self.last_servo_move_time < wait_time:
                # Need to wait for cooldown from previous piece
                wait_remaining = wait_time - (current_time - self.last_servo_move_time)
                logger.info(
                    f"Waiting {wait_remaining:.2f} seconds for servo cooldown before moving to bin {bin_number}")
                time.sleep(wait_remaining)

        logger.debug(f"Moving servo to bin {bin_number}")

        # CHANGED: Use thread_manager.with_lock for thread safety
        def move():
            return self.servo.move_to_bin(bin_number)

        result = self.thread_manager.with_lock("servo", move)

        # Update last servo move time for cooldown tracking
        self.last_servo_move_time = time.time()

        return result

    # ========================================================================
    # STATISTICS AND MONITORING SECTION
    # ========================================================================
    # Methods for tracking performance

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get worker statistics for monitoring.

        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "running": self.running,
            "queue_size": self.queue_manager.get_statistics()["current_queue_size"],
            "uptime": time.time() - self.start_time if self.start_time else 0
        }

        if self.processed_count > 0 and stats["uptime"] > 0:
            stats["pieces_per_second"] = self.processed_count / stats["uptime"]
        else:
            stats["pieces_per_second"] = 0

        return stats


# ============================================================================
# FACTORY FUNCTIONS SECTION
# ============================================================================
# Functions to create and manage processing workers

def create_processing_worker(queue_manager: PieceQueueManager,
                             thread_manager: ThreadManager,
                             config_manager: Any,
                             piece_history: PieceHistory) -> ProcessingWorker:
    """
    Create a processing worker instance with proper dependencies.

    This is the recommended way to create a ProcessingWorker.

    Args:
        queue_manager: Manages the piece queue
        thread_manager: Provides thread safety
        config_manager: Configuration settings
        piece_history: REQUIRED - Piece history tracker

    Returns:
        ProcessingWorker instance

    Raises:
        ValueError: If piece_history is None
    """
    # Get camera configuration for save directory and filename prefix
    camera_config = config_manager.get_module_config(ModuleConfig.CAMERA.value)
    save_directory = camera_config.get("directory", "LegoPictures")
    filename_prefix = camera_config.get("filename_prefix", "Lego")

    return ProcessingWorker(
        queue_manager=queue_manager,
        thread_manager=thread_manager,
        config_manager=config_manager,
        save_directory=save_directory,
        filename_prefix=filename_prefix,
        piece_history=piece_history
    )


def processing_worker_thread(queue_manager: PieceQueueManager,
                             thread_manager: ThreadManager,
                             config_manager: Any,
                             piece_history: PieceHistory) -> None:
    """
    Worker thread function for processing pieces.

    This function is meant to be run in a separate thread and handles
    the complete processing pipeline for detected pieces.

    Args:
        queue_manager: Manages the piece queue
        thread_manager: Provides thread safety
        config_manager: Configuration settings
        piece_history: REQUIRED - Piece history tracker
    """
    # Create worker with proper dependencies
    worker = create_processing_worker(
        queue_manager=queue_manager,
        thread_manager=thread_manager,
        config_manager=config_manager,
        piece_history=piece_history
    )

    try:
        # Start and run the worker
        worker.start()
        worker.run()

    except Exception as e:
        logger.exception(f"Error in processing worker thread: {e}")

    finally:
        # Always stop the worker cleanly
        worker.stop()


# ============================================================================
# MODULE TESTING SECTION
# ============================================================================

if __name__ == "__main__":
    """
    Test the processing module in isolation.

    This creates mock dependencies and tests the processing pipeline.
    """
    import numpy as np
    from enhanced_config_manager import create_config_manager
    from piece_queue_manager import create_piece_queue_manager
    from thread_manager import create_thread_manager
    from piece_history_module import create_piece_history

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Testing processing module...")

    # Create all required dependencies
    config_manager = create_config_manager()
    thread_manager = create_thread_manager(config_manager)
    queue_manager = create_piece_queue_manager(thread_manager, config_manager)
    piece_history = create_piece_history(config_manager)

    # Add a test piece to the queue
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Black test image
    queue_manager.add_piece(
        piece_id=1,
        image=test_image,
        frame_number=100,
        position=(10, 20, 30, 40),
        in_exit_zone=False
    )

    logger.info("Added test piece to queue")

    # Create and run the processing worker
    worker = create_processing_worker(
        queue_manager=queue_manager,
        thread_manager=thread_manager,
        config_manager=config_manager,
        piece_history=piece_history
    )

    # Process one piece
    worker.start()
    message = queue_manager.get_next_piece(timeout=1.0)

    if message:
        logger.info(f"Processing test piece {message.piece_id}")
        result = worker._process_message(message)

        if result:
            logger.info(f"Successfully processed: {result}")
        else:
            logger.error("Processing failed")

    # Get statistics
    stats = worker.get_statistics()
    logger.info(f"Worker statistics: {stats}")

    # Cleanup
    worker.stop()
    thread_manager.shutdown()
    logger.info("Test complete")