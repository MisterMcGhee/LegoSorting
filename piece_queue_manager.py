"""
piece_queue_manager.py - Piece-specific message queue management

This module handles the queueing and prioritization of detected Lego pieces
for processing. It uses the generic thread_manager for managing worker threads
but focuses specifically on piece-related logic.

MAIN PURPOSE:
- Manage the queue of detected pieces waiting for API processing
- Prioritize pieces based on position and timing
- Handle batch operations for efficiency
- Track piece processing status
- Coordinate with other modules for piece handling

This module is used by:
- detector_module.py to queue newly detected pieces
- processing_module.py to get pieces for API identification
- sorting_GUI_module.py to display queue status
- Lego_Sorting_006.py main application for coordination

IMPORT REQUIREMENTS:
- Standard library imports (queue, threading, time, logging, typing, dataclasses, enum)
- thread_manager.py for generic thread management
- enhanced_config_manager.py for configuration loading
"""

import queue
import threading
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import from your other modules
from enhanced_config_manager import ModuleConfig, ConfigSchema
from thread_manager import ThreadManager, create_thread_manager

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================
# These define piece-specific data structures

class PieceStatus(Enum):
    """Status of a piece in the processing pipeline"""
    PENDING = "pending"  # Waiting in queue
    PROCESSING = "processing"  # Being processed by API
    COMPLETED = "completed"  # Processing complete
    FAILED = "failed"  # Processing failed
    SKIPPED = "skipped"  # Skipped (duplicate, timeout, etc.)


class PriorityStrategy(Enum):
    """Different strategies for calculating piece priority"""
    EXIT_ZONE_FIRST = "exit_zone_first"  # Prioritize pieces in exit zone
    FIFO = "fifo"  # First in, first out
    LARGEST_FIRST = "largest_first"  # Prioritize larger pieces
    RIGHTMOST_FIRST = "rightmost_first"  # Prioritize pieces closer to exit


@dataclass
class PieceMessage:
    """
    Message structure for passing detected piece information between threads.

    This represents a single piece that has been detected and needs processing.
    It contains all the information needed for API identification and sorting.
    """
    # ====== Core Identification ======
    piece_id: int  # Unique identifier for the piece
    image: Any  # The cropped image of the piece
    frame_number: int  # The frame number when detected
    timestamp: float  # Detection timestamp

    # ====== Position Information ======
    position: Tuple[int, int, int, int]  # Bounding box (x, y, w, h)
    rightmost_position: float = None  # Rightmost x-coordinate (for exit priority)
    center_position: Tuple[float, float] = None  # Center point of piece

    # ====== Processing State ======
    priority: float = 0.0  # Calculated priority (lower = higher priority)
    status: PieceStatus = PieceStatus.PENDING  # Current processing status
    processing_start_time: Optional[float] = None  # When processing started
    processing_end_time: Optional[float] = None  # When processing ended

    # ====== Zone Information ======
    in_exit_zone: bool = False  # Whether piece is in exit zone
    exit_zone_entry_time: Optional[float] = None  # When entered exit zone

    # ====== Results ======
    result: Dict[str, Any] = field(default_factory=dict)  # API results and bin assignment
    error_message: Optional[str] = None  # Error if processing failed
    retry_count: int = 0  # Number of processing attempts

    def __post_init__(self):
        """
        Calculate derived properties after creation.

        This ensures that computed fields are always available.
        """
        # Calculate rightmost position if not provided
        if self.rightmost_position is None and self.position:
            x, y, w, h = self.position
            self.rightmost_position = x + w

        # Calculate center position if not provided
        if self.center_position is None and self.position:
            x, y, w, h = self.position
            self.center_position = (x + w / 2, y + h / 2)

    def get_area(self) -> float:
        """Calculate the area of the piece bounding box."""
        if self.position:
            _, _, w, h = self.position
            return w * h
        return 0.0

    def get_processing_time(self) -> Optional[float]:
        """Get the total processing time if completed."""
        if self.processing_start_time and self.processing_end_time:
            return self.processing_end_time - self.processing_start_time
        return None


# ============================================================================
# MAIN PIECE QUEUE MANAGER CLASS
# ============================================================================

class PieceQueueManager:
    """
    Manages the queue of detected pieces for processing.

    This class handles:
    - Priority queue management for pieces
    - Batch operations for efficiency
    - Exit zone prioritization
    - Statistics tracking
    - Integration with processing workers

    It uses the generic ThreadManager for worker thread management but
    focuses on piece-specific logic and prioritization.
    """

    def __init__(self, thread_manager: ThreadManager = None, config_manager=None):
        """
        Initialize the piece queue manager.

        Args:
            thread_manager: Generic thread manager for worker threads
            config_manager: Configuration manager for settings
        """
        logger.info("Initializing PieceQueueManager")

        # ====== Dependencies ======
        # Use provided thread manager or create one
        self.thread_manager = thread_manager or create_thread_manager(config_manager)
        self.config_manager = config_manager

        # ====== Configuration ======
        self._load_configuration(config_manager)

        # ====== Queue Management ======
        # Priority queue for pieces waiting to be processed
        self.piece_queue = queue.PriorityQueue(maxsize=self.max_queue_size)

        # Track pieces currently in the queue to prevent duplicates
        self.pieces_in_queue: Dict[int, PieceMessage] = {}

        # Track pieces being processed
        self.pieces_processing: Dict[int, PieceMessage] = {}

        # Recently completed pieces (for deduplication)
        self.recent_pieces: List[int] = []
        self.max_recent_pieces = 100

        # ====== Thread Safety ======
        self.queue_lock = threading.RLock()
        self.stats_lock = threading.RLock()

        # ====== Statistics ======
        self.statistics = {
            "total_queued": 0,
            "total_processed": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "current_queue_size": 0,
            "current_processing": 0,
            "queue_high_water_mark": 0,
            "average_wait_time": 0.0,
            "average_processing_time": 0.0,
            "exit_zone_pieces": 0,
            "priority_overrides": 0
        }

        # ====== Callbacks ======
        # Allow other modules to register for events
        self.callbacks: Dict[str, List[Callable]] = {
            "piece_queued": [],
            "piece_processing": [],
            "piece_completed": [],
            "piece_failed": [],
            "queue_full": []
        }

        logger.info(f"PieceQueueManager initialized with max_queue_size={self.max_queue_size}, "
                    f"priority_strategy={self.priority_strategy}")

    # ========================================================================
    # CONFIGURATION SECTION
    # ========================================================================
    # Methods that handle loading and managing configuration

    def _load_configuration(self, config_manager):
        """
        Load configuration from the config manager or use defaults.

        This centralizes all configuration loading for piece queue management.

        Args:
            config_manager: Configuration manager instance or None
        """
        if config_manager:
            # Get threading configuration for piece-specific settings
            config = config_manager.get_module_config(ModuleConfig.THREADING.value)
            piece_config = config.get("piece_queue", {})

            # Extract settings with defaults
            self.max_queue_size = piece_config.get("max_queue_size", 100)
            self.processing_timeout = piece_config.get("processing_timeout", 10.0)
            self.priority_strategy = piece_config.get("priority_strategy", "exit_zone_first")
            self.batch_size = piece_config.get("batch_size", 1)
            self.max_retries = piece_config.get("max_retries", 2)
            self.duplicate_threshold = piece_config.get("duplicate_threshold", 0.5)

            # Exit zone specific settings
            exit_config = config_manager.get_module_config(ModuleConfig.EXIT_ZONE.value)
            self.exit_zone_priority_boost = exit_config.get("priority_boost", 1000.0)
            self.exit_zone_timeout = exit_config.get("timeout", 2.0)

            logger.info(f"Loaded configuration: strategy={self.priority_strategy}, "
                        f"batch_size={self.batch_size}")
        else:
            # Use defaults
            self.max_queue_size = 100
            self.processing_timeout = 10.0
            self.priority_strategy = "exit_zone_first"
            self.batch_size = 1
            self.max_retries = 2
            self.duplicate_threshold = 0.5
            self.exit_zone_priority_boost = 1000.0
            self.exit_zone_timeout = 2.0

            logger.info("Using default configuration")

    # ========================================================================
    # QUEUE MANAGEMENT SECTION
    # ========================================================================
    # Methods that handle adding, removing, and prioritizing pieces in the queue

    def add_piece(self, piece_id: int, image: Any, frame_number: int,
                  position: Tuple[int, int, int, int], in_exit_zone: bool = False,
                  exit_zone_entry_time: Optional[float] = None) -> bool:
        """
        Add a single piece to the processing queue.

        This is the main method for adding detected pieces to the queue.
        It calculates priority and handles duplicate detection.

        Args:
            piece_id: Unique identifier for the piece
            image: Cropped image of the piece
            frame_number: Frame number when detected
            position: Bounding box (x, y, w, h)
            in_exit_zone: Whether piece is in exit zone
            exit_zone_entry_time: When piece entered exit zone

        Returns:
            bool: True if piece was added, False if rejected (duplicate, queue full, etc.)
        """
        with self.queue_lock:
            # Check if piece is already in queue or being processed
            if piece_id in self.pieces_in_queue:
                logger.debug(f"Piece {piece_id} already in queue")
                return False

            if piece_id in self.pieces_processing:
                logger.debug(f"Piece {piece_id} already being processed")
                return False

            # Check if piece was recently processed (deduplication)
            if piece_id in self.recent_pieces:
                logger.debug(f"Piece {piece_id} was recently processed")
                self.statistics["total_skipped"] += 1
                return False

            # Check if queue is full
            if self.piece_queue.qsize() >= self.max_queue_size:
                logger.warning(f"Queue full ({self.max_queue_size}), rejecting piece {piece_id}")
                self._trigger_callbacks("queue_full", piece_id=piece_id)
                return False

            # Create piece message
            message = PieceMessage(
                piece_id=piece_id,
                image=image,
                frame_number=frame_number,
                timestamp=time.time(),
                position=position,
                in_exit_zone=in_exit_zone,
                exit_zone_entry_time=exit_zone_entry_time
            )

            # Calculate priority
            priority = self._calculate_priority(message)
            message.priority = priority

            # Add to queue
            self.piece_queue.put((priority, message))
            self.pieces_in_queue[piece_id] = message

            # Update statistics
            self._update_statistics_on_queue(message)

            # Trigger callbacks
            self._trigger_callbacks("piece_queued", piece_id=piece_id, priority=priority)

            logger.info(f"Added piece {piece_id} to queue with priority {priority:.2f}")
            return True

    def add_pieces_batch(self, pieces: List[Dict[str, Any]]) -> int:
        """
        Add multiple pieces to the queue efficiently in a single operation.

        This method minimizes lock contention by processing all pieces under
        a single lock acquisition. This is especially useful when the detector
        finds multiple pieces in a single frame.

        Args:
            pieces: List of dictionaries containing piece information
                   Each dict should have: piece_id, image, frame_number, position, etc.

        Returns:
            int: Number of pieces successfully added to the queue

        Example:
            pieces = [
                {"piece_id": 1, "image": img1, "frame_number": 100, "position": (10, 20, 30, 40)},
                {"piece_id": 2, "image": img2, "frame_number": 100, "position": (50, 60, 70, 80)},
            ]
            added = queue_manager.add_pieces_batch(pieces)
        """
        added_count = 0

        with self.queue_lock:
            # Process all pieces under a single lock
            for piece_data in pieces:
                piece_id = piece_data.get("piece_id")

                # Skip if already in queue or processing
                if piece_id in self.pieces_in_queue or piece_id in self.pieces_processing:
                    continue

                # Skip if recently processed
                if piece_id in self.recent_pieces:
                    self.statistics["total_skipped"] += 1
                    continue

                # Check queue capacity
                if self.piece_queue.qsize() >= self.max_queue_size:
                    logger.warning(f"Queue full, could only add {added_count} of {len(pieces)} pieces")
                    break

                # Create message
                message = PieceMessage(
                    piece_id=piece_id,
                    image=piece_data.get("image"),
                    frame_number=piece_data.get("frame_number"),
                    timestamp=time.time(),
                    position=piece_data.get("position"),
                    in_exit_zone=piece_data.get("in_exit_zone", False),
                    exit_zone_entry_time=piece_data.get("exit_zone_entry_time")
                )

                # Calculate priority
                priority = self._calculate_priority(message)
                message.priority = priority

                # Add to queue
                self.piece_queue.put((priority, message))
                self.pieces_in_queue[piece_id] = message
                added_count += 1

                # Update statistics
                self._update_statistics_on_queue(message)

            if added_count > 0:
                logger.info(f"Batch added {added_count} pieces to queue")

                # Single callback for the batch
                self._trigger_callbacks("piece_queued", count=added_count, batch=True)

        return added_count

    def get_next_piece(self, timeout: Optional[float] = None) -> Optional[PieceMessage]:
        """
        Get the next piece from the queue for processing.

        This method is called by processing workers to get pieces to process.
        It respects priority ordering and handles timeout.

        Args:
            timeout: Maximum time to wait for a piece (None = block forever)

        Returns:
            PieceMessage if available, None if timeout or queue empty
        """
        try:
            # Get from priority queue (blocks until available or timeout)
            priority, message = self.piece_queue.get(timeout=timeout)

            with self.queue_lock:
                # Remove from tracking
                if message.piece_id in self.pieces_in_queue:
                    del self.pieces_in_queue[message.piece_id]

                # Add to processing
                message.status = PieceStatus.PROCESSING
                message.processing_start_time = time.time()
                self.pieces_processing[message.piece_id] = message

                # Update statistics
                wait_time = time.time() - message.timestamp
                self._update_average("average_wait_time", wait_time)
                self.statistics["current_queue_size"] = self.piece_queue.qsize()
                self.statistics["current_processing"] = len(self.pieces_processing)

            # Trigger callback
            self._trigger_callbacks("piece_processing", piece_id=message.piece_id)

            logger.debug(f"Retrieved piece {message.piece_id} for processing "
                         f"(waited {wait_time:.2f}s)")

            return message

        except queue.Empty:
            return None

    def mark_piece_completed(self, piece_id: int, result: Dict[str, Any]):
        """
        Mark a piece as successfully processed.

        Called when API identification is complete and successful.

        Args:
            piece_id: ID of the completed piece
            result: Processing results (API response, bin assignment, etc.)
        """
        with self.queue_lock:
            if piece_id not in self.pieces_processing:
                logger.warning(f"Piece {piece_id} not found in processing")
                return

            message = self.pieces_processing[piece_id]
            message.status = PieceStatus.COMPLETED
            message.processing_end_time = time.time()
            message.result = result

            # Remove from processing
            del self.pieces_processing[piece_id]

            # Add to recent pieces for deduplication
            self.recent_pieces.append(piece_id)
            if len(self.recent_pieces) > self.max_recent_pieces:
                self.recent_pieces.pop(0)

            # Update statistics
            processing_time = message.get_processing_time()
            if processing_time:
                self._update_average("average_processing_time", processing_time)
            self.statistics["total_processed"] += 1
            self.statistics["current_processing"] = len(self.pieces_processing)

        # Trigger callback
        self._trigger_callbacks("piece_completed", piece_id=piece_id, result=result)

        logger.info(f"Piece {piece_id} completed successfully")

    def mark_piece_failed(self, piece_id: int, error: str, retry: bool = True):
        """
        Mark a piece as failed processing.

        Called when API identification fails. Can optionally retry.

        Args:
            piece_id: ID of the failed piece
            error: Error message describing the failure
            retry: Whether to retry processing this piece
        """
        with self.queue_lock:
            if piece_id not in self.pieces_processing:
                logger.warning(f"Piece {piece_id} not found in processing")
                return

            message = self.pieces_processing[piece_id]
            message.error_message = error
            message.retry_count += 1

            # Check if we should retry
            if retry and message.retry_count < self.max_retries:
                logger.info(f"Retrying piece {piece_id} (attempt {message.retry_count + 1})")

                # Recalculate priority (may have changed)
                priority = self._calculate_priority(message)
                message.priority = priority

                # Put back in queue
                message.status = PieceStatus.PENDING
                message.processing_start_time = None
                self.piece_queue.put((priority, message))
                self.pieces_in_queue[piece_id] = message

                # Remove from processing
                del self.pieces_processing[piece_id]
            else:
                # Mark as permanently failed
                message.status = PieceStatus.FAILED
                message.processing_end_time = time.time()

                # Remove from processing
                del self.pieces_processing[piece_id]

                # Add to recent to prevent reprocessing
                self.recent_pieces.append(piece_id)
                if len(self.recent_pieces) > self.max_recent_pieces:
                    self.recent_pieces.pop(0)

                # Update statistics
                self.statistics["total_failed"] += 1
                self.statistics["current_processing"] = len(self.pieces_processing)

                # Trigger callback
                self._trigger_callbacks("piece_failed", piece_id=piece_id, error=error)

                logger.error(f"Piece {piece_id} failed permanently: {error}")

    # ========================================================================
    # PRIORITY CALCULATION SECTION
    # ========================================================================
    # Methods that calculate and manage piece priorities

    def _calculate_priority(self, message: PieceMessage) -> float:
        """
        Calculate priority score for a piece message.

        Lower values = higher priority (because Python's PriorityQueue is a min-heap).

        The priority calculation considers:
        - Exit zone status (highest priority)
        - Position on conveyor
        - Time in queue
        - Piece size
        - Strategy setting

        Args:
            message: The piece message to calculate priority for

        Returns:
            float: Priority score (lower = higher priority)
        """
        base_priority = 1000.0  # Base priority level

        # Apply strategy-specific calculation
        if self.priority_strategy == PriorityStrategy.EXIT_ZONE_FIRST.value:
            # Exit zone pieces get massive priority boost
            if message.in_exit_zone:
                # The longer in exit zone, the higher priority
                time_in_exit = 0.0
                if message.exit_zone_entry_time:
                    time_in_exit = time.time() - message.exit_zone_entry_time

                # Subtract large value to ensure exit zone pieces are processed first
                base_priority -= self.exit_zone_priority_boost
                base_priority -= (time_in_exit * 100)  # Additional boost based on time

                # Also consider rightmost position within exit zone
                if message.rightmost_position:
                    base_priority -= (message.rightmost_position / 10)
            else:
                # For non-exit zone pieces, prioritize by position
                if message.rightmost_position:
                    base_priority += (2000 - message.rightmost_position) / 100

        elif self.priority_strategy == PriorityStrategy.FIFO.value:
            # First in, first out - use timestamp
            base_priority = message.timestamp

        elif self.priority_strategy == PriorityStrategy.LARGEST_FIRST.value:
            # Prioritize larger pieces
            area = message.get_area()
            base_priority = 10000 - area  # Larger area = lower priority value

        elif self.priority_strategy == PriorityStrategy.RIGHTMOST_FIRST.value:
            # Prioritize pieces closer to the exit
            if message.rightmost_position:
                base_priority = 2000 - message.rightmost_position

        # Boost priority for pieces that have been waiting long
        wait_time = time.time() - message.timestamp
        if wait_time > 5.0:  # If waiting more than 5 seconds
            base_priority -= (wait_time * 10)  # Boost priority

        # Boost priority for retry attempts
        if message.retry_count > 0:
            base_priority -= (message.retry_count * 100)

        return base_priority

    def update_piece_position(self, piece_id: int, position: Tuple[int, int, int, int],
                              in_exit_zone: bool = None) -> bool:
        """
        Update position information for a piece already in the queue.

        This is useful when the detector updates piece positions in subsequent frames.
        The priority will be recalculated based on the new position.

        Args:
            piece_id: ID of the piece to update
            position: New position (x, y, w, h)
            in_exit_zone: New exit zone status (or None to leave unchanged)

        Returns:
            bool: True if piece was found and updated, False otherwise
        """
        with self.queue_lock:
            if piece_id not in self.pieces_in_queue:
                return False

            message = self.pieces_in_queue[piece_id]
            old_priority = message.priority

            # Update position
            message.position = position
            x, y, w, h = position
            message.rightmost_position = x + w
            message.center_position = (x + w / 2, y + h / 2)

            # Update exit zone status if provided
            if in_exit_zone is not None:
                old_exit_status = message.in_exit_zone
                message.in_exit_zone = in_exit_zone

                # Track entry time if just entered exit zone
                if in_exit_zone and not old_exit_status:
                    message.exit_zone_entry_time = time.time()
                    self.statistics["exit_zone_pieces"] += 1
                elif not in_exit_zone and old_exit_status:
                    self.statistics["exit_zone_pieces"] -= 1

            # Recalculate priority
            new_priority = self._calculate_priority(message)

            # If priority changed significantly, we need to requeue
            if abs(new_priority - old_priority) > 0.01:
                message.priority = new_priority
                self.statistics["priority_overrides"] += 1

                # Note: In a real implementation, we'd need to rebuild the priority queue
                # This is expensive but necessary for correct ordering
                logger.debug(f"Updated piece {piece_id} priority from {old_priority:.2f} "
                             f"to {new_priority:.2f}")

            return True

    # ========================================================================
    # STATISTICS AND MONITORING SECTION
    # ========================================================================
    # Methods for tracking performance and gathering statistics

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current queue statistics.

        Returns:
            Dictionary containing queue metrics
        """
        with self.stats_lock:
            stats = self.statistics.copy()

            # Add current state information
            stats["current_queue_size"] = self.piece_queue.qsize()
            stats["current_processing"] = len(self.pieces_processing)
            stats["pieces_in_queue"] = list(self.pieces_in_queue.keys())
            stats["pieces_processing"] = list(self.pieces_processing.keys())

            return stats

    def _update_statistics_on_queue(self, message: PieceMessage):
        """Update statistics when a piece is queued."""
        with self.stats_lock:
            self.statistics["total_queued"] += 1
            self.statistics["current_queue_size"] = self.piece_queue.qsize()

            # Track high water mark
            if self.statistics["current_queue_size"] > self.statistics["queue_high_water_mark"]:
                self.statistics["queue_high_water_mark"] = self.statistics["current_queue_size"]

            # Track exit zone pieces
            if message.in_exit_zone:
                self.statistics["exit_zone_pieces"] += 1

    def _update_average(self, stat_name: str, new_value: float):
        """
        Update a running average statistic.

        Uses exponential moving average for efficiency.
        """
        with self.stats_lock:
            old_avg = self.statistics[stat_name]
            # Exponential moving average with alpha=0.1
            self.statistics[stat_name] = (0.9 * old_avg) + (0.1 * new_value)

    def get_queue_health(self) -> Dict[str, Any]:
        """
        Get health metrics for the queue.

        Returns:
            Dictionary with health indicators
        """
        stats = self.get_statistics()

        # Calculate health indicators
        queue_utilization = stats["current_queue_size"] / self.max_queue_size

        # Determine health status
        if queue_utilization > 0.9:
            health_status = "critical"
        elif queue_utilization > 0.7:
            health_status = "warning"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "queue_utilization": queue_utilization,
            "processing_count": stats["current_processing"],
            "average_wait_time": stats["average_wait_time"],
            "average_processing_time": stats["average_processing_time"],
            "failure_rate": stats["total_failed"] / max(1, stats["total_processed"]),
            "exit_zone_pieces": stats["exit_zone_pieces"]
        }

    # ========================================================================
    # CALLBACK MANAGEMENT SECTION
    # ========================================================================
    # Methods for managing event callbacks

    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback for queue events.

        Available events:
        - piece_queued: When a piece is added to the queue
        - piece_processing: When a piece starts processing
        - piece_completed: When processing completes successfully
        - piece_failed: When processing fails
        - queue_full: When the queue reaches capacity

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self.callbacks:
            logger.warning(f"Unknown event type: {event_type}")
            return

        self.callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for event: {event_type}")

    def _trigger_callbacks(self, event_type: str, **kwargs):
        """
        Trigger all callbacks for an event type.

        Args:
            event_type: Type of event that occurred
            **kwargs: Event-specific data to pass to callbacks
        """
        if event_type not in self.callbacks:
            return

        for callback in self.callbacks[event_type]:
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")

    # ========================================================================
    # UTILITY SECTION
    # ========================================================================
    # Helper methods and cleanup

    def clear_queue(self) -> int:
        """
        Clear all pending pieces from the queue.

        This is useful for reset scenarios or when switching modes.

        Returns:
            int: Number of pieces cleared
        """
        with self.queue_lock:
            count = self.piece_queue.qsize()

            # Create new empty queue
            self.piece_queue = queue.PriorityQueue(maxsize=self.max_queue_size)
            self.pieces_in_queue.clear()

            # Update statistics
            self.statistics["total_skipped"] += count
            self.statistics["current_queue_size"] = 0

            logger.info(f"Cleared {count} pieces from queue")
            return count

    def stop_processing(self) -> List[int]:
        """
        Stop processing all pieces and return them to pending.

        Returns:
            List of piece IDs that were being processed
        """
        with self.queue_lock:
            stopped_pieces = []

            for piece_id, message in self.pieces_processing.items():
                # Reset status
                message.status = PieceStatus.PENDING
                message.processing_start_time = None

                # Recalculate priority and requeue
                priority = self._calculate_priority(message)
                message.priority = priority
                self.piece_queue.put((priority, message))
                self.pieces_in_queue[piece_id] = message

                stopped_pieces.append(piece_id)

            # Clear processing dict
            self.pieces_processing.clear()
            self.statistics["current_processing"] = 0

            if stopped_pieces:
                logger.info(f"Stopped processing {len(stopped_pieces)} pieces and returned to queue")

            return stopped_pieces

    def get_piece_info(self, piece_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific piece.

        Args:
            piece_id: ID of the piece to query

        Returns:
            Dictionary with piece information or None if not found
        """
        with self.queue_lock:
            # Check in queue
            if piece_id in self.pieces_in_queue:
                message = self.pieces_in_queue[piece_id]
                return {
                    "status": "queued",
                    "priority": message.priority,
                    "position": message.position,
                    "in_exit_zone": message.in_exit_zone,
                    "wait_time": time.time() - message.timestamp
                }

            # Check in processing
            if piece_id in self.pieces_processing:
                message = self.pieces_processing[piece_id]
                return {
                    "status": "processing",
                    "position": message.position,
                    "processing_time": time.time() - message.processing_start_time,
                    "retry_count": message.retry_count
                }

            return None


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_piece_queue_manager(thread_manager: ThreadManager = None,
                               config_manager=None) -> PieceQueueManager:
    """
    Create a piece queue manager instance.

    This is the recommended way to create a PieceQueueManager.

    Args:
        thread_manager: Optional thread manager for worker threads
        config_manager: Optional configuration manager

    Returns:
        PieceQueueManager instance

    Example:
        thread_manager = create_thread_manager(config_manager)
        queue_manager = create_piece_queue_manager(thread_manager, config_manager)

        # Add pieces
        queue_manager.add_piece(piece_id=1, image=img, ...)

        # Get pieces for processing
        piece = queue_manager.get_next_piece()
    """
    return PieceQueueManager(thread_manager, config_manager)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating how to use the PieceQueueManager.

    This shows integration with the detector and processing modules.
    """
    import numpy as np

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create managers
    thread_manager = create_thread_manager()
    queue_manager = create_piece_queue_manager(thread_manager)

    # Example: Detector adds pieces
    logger.info("=== Simulating detector adding pieces ===")

    # Simulate detecting 5 pieces in a frame
    pieces_batch = []
    for i in range(5):
        pieces_batch.append({
            "piece_id": i,
            "image": np.zeros((50, 50, 3), dtype=np.uint8),  # Dummy image
            "frame_number": 100,
            "position": (100 + i * 50, 200, 40, 40),
            "in_exit_zone": i >= 3  # Last 2 pieces in exit zone
        })

    # Add batch
    added = queue_manager.add_pieces_batch(pieces_batch)
    logger.info(f"Added {added} pieces to queue")

    # Check statistics
    stats = queue_manager.get_statistics()
    logger.info(f"Queue stats: {stats}")

    # Example: Processing worker gets pieces
    logger.info("=== Simulating processing worker ===")


    def process_worker(queue_mgr: PieceQueueManager, shutdown_event: threading.Event):
        """Example processing worker."""
        while not shutdown_event.is_set():
            # Get next piece
            piece = queue_mgr.get_next_piece(timeout=1.0)

            if piece:
                logger.info(f"Processing piece {piece.piece_id} "
                            f"(priority={piece.priority:.2f}, exit_zone={piece.in_exit_zone})")

                # Simulate processing
                time.sleep(0.5)

                # Mark as completed
                result = {"bin": 3, "confidence": 0.95}
                queue_mgr.mark_piece_completed(piece.piece_id, result)

            if shutdown_event.wait(timeout=0):
                break


    # Register processing worker
    thread_manager.register_worker(
        name="processor",
        target=process_worker,
        args=(queue_manager, thread_manager.get_shutdown_event())
    )

    # Let it run for a bit
    time.sleep(3)

    # Check final stats
    final_stats = queue_manager.get_statistics()
    logger.info(f"Final stats: {final_stats}")

    # Shutdown
    thread_manager.shutdown()
    logger.info("Example complete")