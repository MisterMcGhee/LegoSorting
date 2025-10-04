# processing_queue_manager.py
"""
processing_queue_manager.py - Queue layer between detection and processing pipelines

This module manages the buffer of captured pieces waiting for processing.
It provides a thread-safe priority queue that allows:
- Detection pipeline to submit pieces asynchronously
- Processing workers to pull pieces for identification
- Status tracking through the processing lifecycle
- Basic statistics for monitoring

Priority Levels:
- 0: Exit zone pieces (urgent - process first)
- 10: Normal pieces (standard priority)

This is infrastructure that sits between pipelines but doesn't perform
any business logic - it just manages the queue and tracks status.
"""

import queue
import threading
import logging
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum

from detector.detector_data_models import CapturePackage
from processing.processing_data_models import IdentifiedPiece
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class PieceStatus(Enum):
    """Status of a piece in the processing pipeline"""
    QUEUED = "queued"  # Waiting in queue
    PROCESSING = "processing"  # Being processed by worker
    COMPLETED = "completed"  # Successfully processed
    FAILED = "failed"  # Processing failed


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QueuedPiece:
    """
    Wrapper around CapturePackage with queue tracking information.

    This adds minimal metadata needed to track pieces through the queue
    and processing pipeline.
    """
    capture_package: CapturePackage
    priority: int  # 0 = exit zone (urgent), 10 = normal
    status: PieceStatus = PieceStatus.QUEUED
    result: Optional[IdentifiedPiece] = None
    error_message: Optional[str] = None

    def __lt__(self, other):
        """
        Comparison operator for PriorityQueue ordering.

        Lower priority number = processed first.
        This means exit zone pieces (priority=0) are pulled before
        normal pieces (priority=10).

        Args:
            other: Another QueuedPiece to compare against

        Returns:
            True if this piece has higher priority (lower number)
        """
        return self.priority < other.priority


# ============================================================================
# MAIN QUEUE MANAGER CLASS
# ============================================================================

class ProcessingQueueManager:
    """Manages queue between detection and processing pipelines"""

    def __init__(self, config_manager: EnhancedConfigManager):
        """
        Initialize queue manager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

        # Get complete config for this module (with all defaults)
        module_config = config_manager.get_module_config(
            ModuleConfig.PROCESSING_QUEUE.value
        )

        # Load configuration values
        self.max_queue_size = module_config["max_queue_size"]

        # Priority queue (thread-safe, blocks when full/empty)
        self.queue = queue.PriorityQueue(maxsize=self.max_queue_size)

        # Track all pieces by piece_id
        self.pieces: Dict[int, QueuedPiece] = {}
        self.pieces_lock = threading.Lock()

        # Simple statistics
        self.total_queued = 0
        self.total_completed = 0
        self.total_failed = 0

        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "piece_completed": [],
            "piece_failed": []
        }

        logger.info(
            f"Processing queue manager initialized "
            f"(max size: {self.max_queue_size})"
        )

    # ========================================================================
    # SUBMISSION (Detection → Queue)
    # ========================================================================

    def submit_piece(self, capture_package: CapturePackage) -> bool:
        """
        Submit a captured piece to the processing queue.

        This is called by the detection pipeline when a piece has been
        captured and is ready for identification.

        Args:
            capture_package: Captured piece from detection pipeline

        Returns:
            True if successfully queued, False if queue is full
        """
        try:
            # Determine priority based on piece location
            # Exit zone pieces get priority 0 (processed first)
            # Normal pieces get priority 10 (standard)
            priority = 0 if getattr(capture_package, 'is_priority', False) else 10

            # Create queued piece wrapper
            queued_piece = QueuedPiece(
                capture_package=capture_package,
                priority=priority,
                status=PieceStatus.QUEUED
            )

            # Try to add to queue (non-blocking - fail if full)
            self.queue.put(queued_piece, block=False)

            # Track in dictionary
            with self.pieces_lock:
                self.pieces[capture_package.piece_id] = queued_piece
                self.total_queued += 1

            # Log submission
            priority_str = "exit zone" if priority == 0 else "normal"
            logger.info(
                f"Piece {capture_package.piece_id} queued "
                f"(priority: {priority_str})"
            )

            return True

        except queue.Full:
            logger.error(
                f"Queue full ({self.max_queue_size}), "
                f"cannot add piece {capture_package.piece_id}"
            )
            return False

        except Exception as e:
            logger.error(f"Failed to queue piece: {e}")
            return False

    # ========================================================================
    # RETRIEVAL (Queue → Processing Worker)
    # ========================================================================

    def get_next_piece(self, timeout: float = 1.0) -> Optional[CapturePackage]:
        """
        Get next highest-priority piece for processing.

        This is called by processing workers to get work. It blocks until
        a piece is available or the timeout expires.

        Args:
            timeout: How long to wait for a piece (seconds)

        Returns:
            CapturePackage to process, or None if timeout
        """
        try:
            # Get highest priority piece (blocks until available or timeout)
            queued_piece = self.queue.get(block=True, timeout=timeout)

            # Update status to processing
            with self.pieces_lock:
                queued_piece.status = PieceStatus.PROCESSING

            logger.debug(
                f"Piece {queued_piece.capture_package.piece_id} "
                f"pulled from queue for processing"
            )

            return queued_piece.capture_package

        except queue.Empty:
            # Timeout - no pieces available
            return None

        except Exception as e:
            logger.error(f"Error getting piece from queue: {e}")
            return None

    # ========================================================================
    # STATUS UPDATES (Processing → Queue)
    # ========================================================================

    def mark_completed(self, piece_id: int, result: IdentifiedPiece):
        """
        Mark a piece as successfully processed.

        This is called by processing workers after a piece has been
        successfully identified and assigned a bin.

        Args:
            piece_id: ID of the piece that was processed
            result: Complete IdentifiedPiece with all data
        """
        with self.pieces_lock:
            if piece_id in self.pieces:
                self.pieces[piece_id].status = PieceStatus.COMPLETED
                self.pieces[piece_id].result = result
                self.total_completed += 1

                logger.info(
                    f"Piece {piece_id} completed: "
                    f"{result.name} → Bin {result.bin_number} "
                    f"(confidence: {result.confidence:.2f})"
                )
            else:
                logger.warning(f"Piece {piece_id} not found in tracking")

        # Trigger callbacks
        self._trigger_callbacks("piece_completed", piece_id=piece_id, result=result)

    def mark_failed(self, piece_id: int, error: str):
        """
        Mark a piece as failed processing.

        This is called by processing workers after a piece fails to process
        (API error, lookup failure, etc.).

        Args:
            piece_id: ID of the piece that failed
            error: Error message describing the failure
        """
        with self.pieces_lock:
            if piece_id in self.pieces:
                self.pieces[piece_id].status = PieceStatus.FAILED
                self.pieces[piece_id].error_message = error
                self.total_failed += 1

                logger.error(f"Piece {piece_id} failed: {error}")
            else:
                logger.warning(f"Piece {piece_id} not found in tracking")

        # Trigger callbacks
        self._trigger_callbacks("piece_failed", piece_id=piece_id, error=error)

    # ========================================================================
    # QUERY METHODS (Monitoring)
    # ========================================================================

    def get_queue_size(self) -> int:
        """
        Get number of pieces currently waiting in queue.

        Returns:
            Number of pieces waiting to be processed
        """
        return self.queue.qsize()

    def get_statistics(self) -> Dict[str, int]:
        """
        Get simple statistics for monitoring queue health.

        Returns:
            Dictionary with current queue size and lifetime totals
        """
        with self.pieces_lock:
            return {
                "queued": self.get_queue_size(),
                "total_completed": self.total_completed,
                "total_failed": self.total_failed,
                "total_queued": self.total_queued
            }

    def is_queue_full(self) -> bool:
        """
        Check if queue is at capacity.

        Returns:
            True if queue is full and cannot accept more pieces
        """
        return self.queue.full()

    # ========================================================================
    # CALLBACK SYSTEM
    # ========================================================================

    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback function for queue events.

        Available events:
        - "piece_completed": Called when piece successfully processed
          Signature: callback(piece_id: int, result: IdentifiedPiece)

        - "piece_failed": Called when piece fails processing
          Signature: callback(piece_id: int, error: str)

        Args:
            event: Event name to register for
            callback: Function to call when event occurs
        """
        if event not in self.callbacks:
            logger.error(f"Unknown event type: {event}")
            return

        self.callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")

    def _trigger_callbacks(self, event: str, **kwargs):
        """
        Trigger all callbacks registered for an event.

        This catches and logs any exceptions from callbacks to prevent
        one bad callback from breaking the queue system.

        Args:
            event: Event name to trigger
            **kwargs: Arguments to pass to callbacks
        """
        for callback in self.callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error(
                    f"Error in callback for event '{event}': {e}",
                    exc_info=True
                )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_processing_queue_manager(config_manager: EnhancedConfigManager) -> ProcessingQueueManager:
    """
    Factory function to create a processing queue manager.

    This provides a consistent interface for creating queue managers
    across the application.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Initialized ProcessingQueueManager

    Example:
        queue_manager = create_processing_queue_manager(config_manager)
        queue_manager.submit_piece(capture_package)
    """
    return ProcessingQueueManager(config_manager)


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the processing queue manager with mock data.

    This verifies:
    - Queue creation and initialization
    - Piece submission
    - Priority ordering
    - Piece retrieval
    - Status updates
    - Statistics tracking
    """
    import numpy as np
    import time
    from enhanced_config_manager import create_config_manager

    # Set up logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Testing Processing Queue Manager...\n")

    # Create config manager
    config_manager = create_config_manager()

    # Create queue manager
    queue_manager = create_processing_queue_manager(config_manager)

    # Create mock CapturePackage
    mock_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test 1: Submit normal priority piece
    logger.info("TEST 1: Submit normal priority piece")
    normal_package = CapturePackage(
        piece_id=1,
        processed_image=mock_image,
        capture_timestamp=time.time(),
        capture_position=(100, 200),
        original_bbox=(100, 200, 50, 50)
    )
    normal_package.is_priority = False

    success = queue_manager.submit_piece(normal_package)
    assert success, "Failed to submit normal piece"
    assert queue_manager.get_queue_size() == 1, "Queue size should be 1"
    logger.info("✓ Normal piece queued successfully\n")

    # Test 2: Submit exit zone (priority) piece
    logger.info("TEST 2: Submit exit zone priority piece")
    priority_package = CapturePackage(
        piece_id=2,
        processed_image=mock_image,
        capture_timestamp=time.time(),
        capture_position=(200, 300),
        original_bbox=(200, 300, 50, 50)
    )
    priority_package.is_priority = True

    success = queue_manager.submit_piece(priority_package)
    assert success, "Failed to submit priority piece"
    assert queue_manager.get_queue_size() == 2, "Queue size should be 2"
    logger.info("✓ Priority piece queued successfully\n")

    # Test 3: Verify priority ordering (priority piece should come first)
    logger.info("TEST 3: Verify priority ordering")
    first_piece = queue_manager.get_next_piece(timeout=1.0)
    assert first_piece is not None, "Should get a piece"
    assert first_piece.piece_id == 2, "Priority piece should be retrieved first"
    assert queue_manager.get_queue_size() == 1, "Queue size should be 1"
    logger.info("✓ Priority piece retrieved first (correct ordering)\n")

    # Test 4: Get second piece
    logger.info("TEST 4: Get second piece")
    second_piece = queue_manager.get_next_piece(timeout=1.0)
    assert second_piece is not None, "Should get second piece"
    assert second_piece.piece_id == 1, "Should get normal piece"
    assert queue_manager.get_queue_size() == 0, "Queue should be empty"
    logger.info("✓ Second piece retrieved successfully\n")

    # Test 5: Mark piece as completed
    logger.info("TEST 5: Mark piece as completed")
    mock_result = IdentifiedPiece(
        piece_id=2,
        image_path="/test/path.jpg",
        capture_timestamp=time.time()
    )
    mock_result.element_id = "3001"
    mock_result.name = "Brick 2x4"
    mock_result.identification_confidence = 0.95
    mock_result.primary_category = "Basic"
    mock_result.bin_number = 3
    mock_result.complete = True

    queue_manager.mark_completed(2, mock_result)
    stats = queue_manager.get_statistics()
    assert stats["total_completed"] == 1, "Should have 1 completed"
    logger.info("✓ Piece marked as completed\n")

    # Test 6: Mark piece as failed
    logger.info("TEST 6: Mark piece as failed")
    queue_manager.mark_failed(1, "API timeout")
    stats = queue_manager.get_statistics()
    assert stats["total_failed"] == 1, "Should have 1 failed"
    logger.info("✓ Piece marked as failed\n")

    # Test 7: Test timeout on empty queue
    logger.info("TEST 7: Test timeout on empty queue")
    empty_result = queue_manager.get_next_piece(timeout=0.5)
    assert empty_result is None, "Should return None on timeout"
    logger.info("✓ Timeout works correctly\n")

    # Test 8: Test callbacks
    logger.info("TEST 8: Test callbacks")
    callback_triggered = {"completed": False, "failed": False}


    def on_completed(piece_id, result):
        callback_triggered["completed"] = True
        logger.info(f"  Callback: Piece {piece_id} completed")


    def on_failed(piece_id, error):
        callback_triggered["failed"] = True
        logger.info(f"  Callback: Piece {piece_id} failed")


    queue_manager.register_callback("piece_completed", on_completed)
    queue_manager.register_callback("piece_failed", on_failed)

    # Trigger callbacks
    queue_manager.mark_completed(99, mock_result)
    queue_manager.mark_failed(100, "Test error")

    assert callback_triggered["completed"], "Completed callback should trigger"
    assert callback_triggered["failed"], "Failed callback should trigger"
    logger.info("✓ Callbacks work correctly\n")

    # Final statistics
    logger.info("FINAL STATISTICS:")
    final_stats = queue_manager.get_statistics()
    for key, value in final_stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n✓ ALL TESTS PASSED!")
