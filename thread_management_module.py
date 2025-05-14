"""
thread_management_module.py - Thread management for the Lego sorting application

This module provides a multithreaded architecture for the Lego sorting application,
separating detection from processing to improve performance and reduce stuttering.
It includes thread-safe queues, worker thread management, and message passing.
"""

import queue
import threading
import time
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple
from dataclasses import dataclass, field

# Get module logger
logger = logging.getLogger(__name__)


@dataclass
class PieceMessage:
    """Message structure for passing detected piece information between threads."""
    piece_id: int  # Unique identifier for the piece
    image: Any  # The cropped image of the piece
    frame_number: int  # The frame number when detected
    timestamp: float  # Detection timestamp
    position: Tuple[int, int, int, int]  # x, y, w, h of the bounding box
    priority: int = 0  # Message priority (lower values = higher priority)
    status: str = "pending"  # Status of processing (pending, processing, completed, failed)
    result: Dict[str, Any] = None  # Results of API call and sorting
    in_exit_zone: bool = False  # Whether the piece is in the exit zone
    rightmost_position: float = None  # Rightmost position of the piece (for exit zone priority)

    def __post_init__(self):
        """Initialize derived properties if not set explicitly."""
        if self.rightmost_position is None and self.position:
            x, y, w, h = self.position
            self.rightmost_position = x + w


class ThreadManager:
    """Manages worker threads and message passing for the Lego sorting application."""

    def __init__(self, config_manager=None, max_queue_size: int = 100):
        """Initialize thread manager.

        Args:
            config_manager: Optional configuration manager for thread settings
            max_queue_size: Maximum number of items in the queue before blocking
        """
        # Set up configuration
        self.config_manager = config_manager

        # Use config if provided, otherwise use defaults
        if config_manager:
            max_queue_size = config_manager.get("threading", "max_queue_size", max_queue_size)

            # Get exit zone trigger configuration
            self.exit_zone_trigger_config = config_manager.get_section("exit_zone_trigger")
            self.exit_zone_enabled = self.exit_zone_trigger_config.get("enabled", True)
            self.priority_method = self.exit_zone_trigger_config.get("priority_method", "rightmost")
        else:
            self.exit_zone_enabled = True
            self.priority_method = "rightmost"
            self.exit_zone_trigger_config = {}

        # Initialize message queue with priority
        self.message_queue = queue.PriorityQueue(maxsize=max_queue_size)

        # Keep track of messages in the queue for reprioritization
        self.messages_in_queue = {}  # piece_id -> PieceMessage
        self.queue_lock = threading.RLock()

        # Thread control flags
        self.running = False
        self.should_exit = threading.Event()

        # Thread registry
        self.workers = {}

        # Callback registry
        self.callbacks = {}

        # Synchronization primitives
        self.locks = {
            "camera": threading.Lock(),
            "servo": threading.Lock(),
            "api": threading.Lock()
        }

        logger.info("Thread manager initialized with max queue size: %d", max_queue_size)
        logger.info(f"Exit zone trigger enabled: {self.exit_zone_enabled}")
        logger.info(f"Priority method: {self.priority_method}")

    def start_worker(self, name: str, target: Callable, args: Tuple = (), daemon: bool = True) -> bool:
        """Start a new worker thread.

        Args:
            name: Name of the worker thread
            target: Target function to run in the thread
            args: Arguments to pass to the target function
            daemon: Whether the thread should be a daemon thread

        Returns:
            bool: True if worker started successfully, False otherwise
        """
        if name in self.workers and self.workers[name].is_alive():
            logger.warning("Worker '%s' already running", name)
            return False

        logger.info("Starting worker thread: %s", name)
        thread = threading.Thread(target=target, args=args, name=name, daemon=daemon)
        thread.start()
        self.workers[name] = thread
        return True

    def stop_all_workers(self, timeout: float = 5.0) -> bool:
        """Stop all worker threads gracefully.

        Args:
            timeout: Maximum time to wait for threads to exit

        Returns:
            bool: True if all workers stopped successfully, False otherwise
        """
        logger.info("Stopping all worker threads")
        self.should_exit.set()
        self.running = False

        # Wait for all threads to exit
        end_time = time.time() + timeout
        alive_threads = []

        for name, thread in self.workers.items():
            remaining_time = max(0.1, end_time - time.time())
            thread.join(timeout=remaining_time)
            if thread.is_alive():
                alive_threads.append(name)

        if alive_threads:
            logger.warning("The following threads did not exit: %s", ", ".join(alive_threads))
            return False

        logger.info("All worker threads stopped successfully")
        return True

    def add_message(self, piece_id: int, image: Any, frame_number: int,
                    position: Tuple[int, int, int, int], priority: int = 0,
                    in_exit_zone: bool = False) -> bool:
        """Add a message to the queue for processing.

        Args:
            piece_id: Unique identifier for the piece
            image: The cropped image of the piece
            frame_number: The frame number when detected
            position: Bounding box (x, y, w, h) of the piece
            priority: Message priority (lower values = higher priority)
            in_exit_zone: Whether the piece is in the exit zone

        Returns:
            bool: True if message added successfully, False if queue is full
        """
        try:
            # Calculate rightmost position for priority determination
            x, y, w, h = position
            rightmost_position = x + w

            message = PieceMessage(
                piece_id=piece_id,
                image=image,
                frame_number=frame_number,
                timestamp=time.time(),
                position=position,
                priority=priority,
                in_exit_zone=in_exit_zone,
                rightmost_position=rightmost_position
            )

            # Determine final queue priority based on message properties
            queue_priority = self._calculate_priority(message)

            with self.queue_lock:
                # Add to queue with priority
                self.message_queue.put((queue_priority, message.piece_id, message), block=False)
                # Store reference to message
                self.messages_in_queue[piece_id] = message

            if in_exit_zone:
                logger.debug(f"Added exit zone piece {piece_id} to queue with priority {queue_priority}")
            else:
                logger.debug(f"Added piece {piece_id} to queue with priority {queue_priority}")

            return True

        except queue.Full:
            logger.warning(f"Message queue is full, dropping message for piece {piece_id}")
            return False

    def _calculate_priority(self, message: PieceMessage) -> float:
        """Calculate queue priority based on message properties.

        Lower values = higher priority. Uses a multi-level approach:
        1. Base priority from the message (0 for exit zone, higher for normal)
        2. For exit zone pieces, additional sub-priority based on position

        Args:
            message: The message to calculate priority for

        Returns:
            float: Priority value for queue (lower = higher priority)
        """
        # Start with the base priority
        base_priority = float(message.priority)

        # If exit zone is enabled and this is an exit zone piece, refine the priority
        if self.exit_zone_enabled and message.in_exit_zone:
            if self.priority_method == "rightmost":
                # Rightmost pieces get highest priority
                # Use negative rightmost position so higher value = higher priority
                position_priority = -message.rightmost_position / 10000.0

                # Combine: Main level is base priority, sub-level is position
                # This ensures all exit zone pieces are higher priority than non-exit zone,
                # but within exit zone, they're ordered by position
                return base_priority + position_priority

            elif self.priority_method == "first_in":
                # First detected pieces get highest priority
                # Use negative timestamp so earlier time = higher priority
                time_priority = -message.timestamp / 10000.0
                return base_priority + time_priority

        # For non-exit zone pieces, just use the base priority
        return base_priority

    def get_next_message(self, timeout: float = None) -> Optional[PieceMessage]:
        """Get the next message from the queue.

        Args:
            timeout: Maximum time to wait for a message

        Returns:
            PieceMessage or None if queue is empty or timeout occurs
        """
        try:
            with self.queue_lock:
                # Get item from priority queue (unpack the priority, message tuple)
                _, message = self.message_queue.get(block=True, timeout=timeout)
                message.status = "processing"
                # Remove from tracked messages
                if message.piece_id in self.messages_in_queue:
                    del self.messages_in_queue[message.piece_id]
                return message

        except queue.Empty:
            return None

    def task_done(self):
        """Mark a task as done in the queue."""
        self.message_queue.task_done()

    def reprioritize_messages(self):
        """Re-evaluate priorities for all messages in the queue.

        This is useful when the priority criteria change (e.g., pieces move position).
        This operation is expensive and should be used sparingly.
        """
        with self.queue_lock:
            if not self.messages_in_queue:
                return  # Nothing to reprioritize

            # Create a new queue
            new_queue = queue.PriorityQueue(self.message_queue.maxsize)

            # Copy all items to the new queue with recalculated priorities
            messages_to_reprioritize = list(self.messages_in_queue.values())
            for message in messages_to_reprioritize:
                new_priority = self._calculate_priority(message)
                new_queue.put((new_priority, message))

            # Replace the old queue with the new one
            self.message_queue = new_queue

            logger.debug(f"Reprioritized {len(messages_to_reprioritize)} messages in queue")

    def update_message_position(self, piece_id: int, position: Tuple[int, int, int, int],
                                in_exit_zone: bool = None) -> bool:
        """Update position information for a piece already in the queue.

        Args:
            piece_id: ID of the piece to update
            position: New position (x, y, w, h)
            in_exit_zone: New exit zone status (or None to leave unchanged)

        Returns:
            bool: True if message was found and updated, False otherwise
        """
        with self.queue_lock:
            if piece_id not in self.messages_in_queue:
                return False

            message = self.messages_in_queue[piece_id]
            message.position = position

            # Calculate rightmost position
            x, y, w, h = position
            message.rightmost_position = x + w

            # Update exit zone status if provided
            if in_exit_zone is not None:
                message.in_exit_zone = in_exit_zone

            return True

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback function for specific events.

        Args:
            event_type: Type of event to trigger callback
            callback: Function to call when event occurs
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []

        self.callbacks[event_type].append(callback)
        logger.debug("Registered callback for event type: %s", event_type)

    def trigger_callback(self, event_type: str, **kwargs) -> None:
        """Trigger all callbacks registered for an event type.

        Args:
            event_type: Type of event that occurred
            **kwargs: Arguments to pass to the callback functions
        """
        if event_type not in self.callbacks:
            return

        for callback in self.callbacks[event_type]:
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error("Error in callback for event %s: %s", event_type, str(e))

    def get_queue_size(self) -> int:
        """Get current size of the message queue.

        Returns:
            int: Number of messages in the queue
        """
        return self.message_queue.qsize()

    def with_lock(self, lock_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with a named lock.

        Args:
            lock_name: Name of the lock to acquire
            func: Function to execute with the lock
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function call
        """
        if lock_name not in self.locks:
            self.locks[lock_name] = threading.Lock()

        with self.locks[lock_name]:
            return func(*args, **kwargs)

    def clear_queue(self) -> int:
        """Clear all pending messages from the queue.

        Returns:
            int: Number of messages cleared
        """
        with self.queue_lock:
            count = self.message_queue.qsize()
            # Create a new empty queue with the same size
            self.message_queue = queue.PriorityQueue(self.message_queue.maxsize)
            self.messages_in_queue.clear()
            logger.info(f"Cleared {count} messages from queue")
            return count


# Factory function
def create_thread_manager(config_manager=None) -> ThreadManager:
    """Create a thread manager instance.

    Args:
        config_manager: Configuration manager

    Returns:
        ThreadManager instance
    """
    return ThreadManager(config_manager)