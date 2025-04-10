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
from dataclasses import dataclass

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
    priority: int = 0  # Message priority (higher values = higher priority)
    status: str = "pending"  # Status of processing (pending, processing, completed, failed)
    result: Dict[str, Any] = None  # Results of API call and sorting


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

        # Initialize message queue with priority
        self.message_queue = queue.PriorityQueue(maxsize=max_queue_size)

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
                    position: Tuple[int, int, int, int], priority: int = 0) -> bool:
        """Add a message to the queue for processing.

        Args:
            piece_id: Unique identifier for the piece
            image: The cropped image of the piece
            frame_number: The frame number when detected
            position: Bounding box (x, y, w, h) of the piece
            priority: Message priority (lower values = higher priority)

        Returns:
            bool: True if message added successfully, False if queue is full
        """
        try:
            message = PieceMessage(
                piece_id=piece_id,
                image=image,
                frame_number=frame_number,
                timestamp=time.time(),
                position=position,
                priority=priority
            )

            # Add to queue with priority (negative so lower values = higher priority)
            self.message_queue.put((priority, message), block=False)
            logger.debug("Added message for piece %d to queue (priority=%d)", piece_id, priority)
            return True

        except queue.Full:
            logger.warning("Message queue is full, dropping message for piece %d", piece_id)
            return False

    def get_next_message(self, timeout: float = None) -> Optional[PieceMessage]:
        """Get the next message from the queue.

        Args:
            timeout: Maximum time to wait for a message

        Returns:
            PieceMessage or None if queue is empty or timeout occurs
        """
        try:
            # Get item from priority queue (unpack the priority, message tuple)
            _, message = self.message_queue.get(block=True, timeout=timeout)
            message.status = "processing"
            return message

        except queue.Empty:
            return None

    def task_done(self):
        """Mark a task as done in the queue."""
        self.message_queue.task_done()

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


# Factory function
def create_thread_manager(config_manager=None) -> ThreadManager:
    """Create a thread manager instance.

    Args:
        config_manager: Configuration manager

    Returns:
        ThreadManager instance
    """
    return ThreadManager(config_manager)