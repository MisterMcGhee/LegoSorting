# processing/processing_worker.py
"""
processing_worker.py - Worker threads that process pieces from the queue

This module implements worker threads that bridge the processing queue and
the processing coordinator. Workers continuously pull pieces from the queue,
process them through the coordinator, and report results back.

RESPONSIBILITIES:
- Run in separate thread (multiple workers can run in parallel)
- Pull CapturePackages from ProcessingQueueManager
- Pass packages to ProcessingCoordinator for processing
- Report results (completed/failed) back to queue
- Track worker-specific statistics
- Clean shutdown when signaled

DOES NOT:
- Process pieces itself (that's coordinator's job)
- Manage the queue (that's queue_manager's job)
- Control hardware or servos

THREADING MODEL:
- Each worker runs in its own thread
- Multiple workers can run simultaneously (default: 3)
- Workers share the same queue (thread-safe)
- Workers share the same coordinator (stateless, thread-safe)
"""

import logging
import threading
import time
from typing import Optional
from processing_queue_manager import ProcessingQueueManager
from processing.processing_coordinator import ProcessingCoordinator
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

logger = logging.getLogger(__name__)


class ProcessingWorker:
    """
    Worker thread that processes pieces from the queue.

    This class runs in its own thread, continuously pulling pieces from
    the queue and processing them through the coordinator pipeline.

    Multiple workers can run in parallel to process pieces concurrently.
    """

    def __init__(self,
                 worker_id: int,
                 queue_manager: ProcessingQueueManager,
                 coordinator: ProcessingCoordinator,
                 config_manager: EnhancedConfigManager):
        """
        Initialize processing worker.

        Called during application startup for each worker thread.

        Args:
            worker_id: Unique identifier for this worker (0, 1, 2, ...)
            queue_manager: Queue manager to pull pieces from
            coordinator: Coordinator to process pieces with
            config_manager: Configuration manager for settings
        """
        self.worker_id = worker_id
        self.queue_manager = queue_manager
        self.coordinator = coordinator
        self.config_manager = config_manager

        # Get worker configuration
        worker_config = config_manager.get_module_config(
            ModuleConfig.PROCESSING_WORKERS.value
        )
        self.timeout = worker_config["worker_timeout"]  # Queue timeout in seconds

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False  # Flag to signal thread to stop

        # Statistics (per-worker)
        self.pieces_processed = 0
        self.pieces_failed = 0

        logger.info(f"Worker {self.worker_id} initialized (timeout={self.timeout}s)")

    def start(self) -> None:
        """
        Start the worker thread.

        Called during application startup after all workers are created.
        Launches the thread that runs _run() method.

        Example:
            worker = ProcessingWorker(0, queue_manager, coordinator, config_manager)
            worker.start()  # Thread begins running
        """
        if self.running:
            logger.warning(f"Worker {self.worker_id} already running")
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            name=f"ProcessingWorker-{self.worker_id}",
            daemon=True  # Thread will stop when main program exits
        )
        self.thread.start()
        logger.info(f"Worker {self.worker_id} thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the worker thread gracefully.

        Called during application shutdown. Signals the thread to stop
        and waits for it to finish current work.

        Args:
            timeout: How long to wait for thread to stop (seconds)

        Example:
            worker.stop()  # Thread finishes current piece and exits
        """
        if not self.running:
            logger.debug(f"Worker {self.worker_id} already stopped")
            return

        logger.info(f"Stopping worker {self.worker_id}...")
        self.running = False

        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

            if self.thread.is_alive():
                logger.warning(
                    f"Worker {self.worker_id} did not stop within {timeout}s"
                )
            else:
                logger.info(f"Worker {self.worker_id} stopped cleanly")

    def _run(self) -> None:
        """
        Main worker loop - runs in separate thread.

        This is the core of the worker. It continuously:
        1. Pulls pieces from queue (with timeout)
        2. Processes through coordinator
        3. Reports success/failure back to queue
        4. Repeats until running=False

        Called by: Thread started in start() method
        Runs until: self.running is set to False
        """
        logger.info(f"Worker {self.worker_id} entering main loop")

        while self.running:
            try:
                # Pull next piece from queue (blocks up to timeout seconds)
                capture_package = self.queue_manager.get_next_piece(
                    timeout=self.timeout
                )

                # If no piece available (timeout), continue loop
                if capture_package is None:
                    continue

                # Got a piece - process it
                self._process_piece(capture_package)

            except Exception as e:
                # Catch any unexpected errors to prevent worker crash
                logger.error(
                    f"Worker {self.worker_id} encountered unexpected error: {e}",
                    exc_info=True
                )
                # Continue running despite error

        logger.info(f"Worker {self.worker_id} exiting main loop")

    def _process_piece(self, capture_package) -> None:
        """
        Process a single piece through the coordinator.

        This wraps the coordinator call with error handling and result reporting.
        Even if processing fails, we report back to the queue so it doesn't
        think the piece is still processing.

        Called by: _run() method for each piece pulled from queue

        Args:
            capture_package: Captured piece to process

        Side effects:
            - Calls queue_manager.mark_completed() on success
            - Calls queue_manager.mark_failed() on failure
            - Updates worker statistics
        """
        piece_id = capture_package.piece_id

        logger.debug(f"Worker {self.worker_id} processing piece {piece_id}")

        try:
            # Process through coordinator pipeline
            identified_piece = self.coordinator.process_piece(capture_package)

            # Check if processing completed successfully
            if identified_piece.complete:
                # Success - report to queue manager
                self.queue_manager.mark_completed(piece_id, identified_piece)
                self.pieces_processed += 1

                logger.info(
                    f"Worker {self.worker_id} completed piece {piece_id}: "
                    f"{identified_piece.name} → Bin {identified_piece.bin_number}"
                )
            else:
                # Processing incomplete (API failed, category not found, etc.)
                error_msg = "Processing incomplete - piece has missing data"
                self.queue_manager.mark_failed(piece_id, error_msg)
                self.pieces_failed += 1

                logger.warning(
                    f"Worker {self.worker_id} piece {piece_id} incomplete: {error_msg}"
                )

        except Exception as e:
            # Unexpected error during processing
            error_msg = f"Processing error: {str(e)}"
            self.queue_manager.mark_failed(piece_id, error_msg)
            self.pieces_failed += 1

            logger.error(
                f"Worker {self.worker_id} failed to process piece {piece_id}: {e}",
                exc_info=True
            )

    def get_statistics(self) -> dict:
        """
        Get statistics for this worker.

        Returns per-worker statistics. Useful for monitoring worker health
        and load balancing.

        Returns:
            Dictionary with worker statistics
        """
        return {
            "worker_id": self.worker_id,
            "pieces_processed": self.pieces_processed,
            "pieces_failed": self.pieces_failed,
            "is_running": self.running,
            "is_alive": self.thread.is_alive() if self.thread else False
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_processing_worker(
        worker_id: int,
        queue_manager: ProcessingQueueManager,
        coordinator: ProcessingCoordinator,
        config_manager: EnhancedConfigManager
) -> ProcessingWorker:
    """
    Factory function to create a processing worker.

    Called during application startup to create worker instances.
    Typically called in a loop to create multiple workers.

    Args:
        worker_id: Unique ID for this worker
        queue_manager: Shared queue manager
        coordinator: Shared coordinator
        config_manager: Configuration manager

    Returns:
        Initialized ProcessingWorker (not started yet)

    Example:
        # Create 3 workers
        workers = []
        for i in range(3):
            worker = create_processing_worker(i, queue_manager, coordinator, config_manager)
            workers.append(worker)

        # Start all workers
        for worker in workers:
            worker.start()
    """
    return ProcessingWorker(worker_id, queue_manager, coordinator, config_manager)
