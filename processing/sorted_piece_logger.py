# processing/sorted_piece_logger.py
"""
sorted_piece_logger.py - Log successfully identified pieces to CSV for collection analysis

This module maintains a database of all sorted pieces encountered during operation,
tracking how many times each unique element has been seen.

RESPONSIBILITIES:
- Maintain sorted_pieces.csv with encounter counts for all identified pieces
- Update encounter counts for repeated pieces
- Provide thread-safe logging from multiple worker threads
- Track collection statistics

PATTERN:
Follows CategoryLookup's unknown piece logging pattern with thread-safe CSV operations.

DOES NOT:
- Log unknown/failed pieces (those go to unknown_pieces.csv via CategoryLookup)
- Make decisions about sorting or bin assignment
- Modify piece processing pipeline
"""

import os
import csv
import threading
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from processing.processing_data_models import IdentifiedPiece
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


class SortedPieceLogger:
    """
    Logs successfully identified pieces to CSV database.

    Maintains sorted_pieces.csv with all pieces that have been successfully
    identified and sorted, tracking encounter counts for collection analysis.
    """

    def __init__(self, config_manager: EnhancedConfigManager):
        """
        Initialize the sorted piece logger with configuration.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

        # Get configuration
        try:
            module_config = config_manager.get_module_config(
                ModuleConfig.PIECE_IDENTIFIER.value
            )

            self.logging_enabled = module_config.get("log_sorted_pieces", True)
            self.log_path = module_config.get("sorted_pieces_path", "sorted_pieces.csv")
        except Exception as e:
            # If config missing, use sensible defaults
            logger.warning(f"Sorted piece logger config missing, using defaults: {e}")
            self.logging_enabled = True
            self.log_path = "sorted_pieces.csv"

        # Thread safety for CSV operations
        self.log_lock = threading.Lock()

        # Initialize log file
        if self.logging_enabled:
            self._initialize_log()
            logger.info(f"Sorted piece logger initialized (enabled, path: {self.log_path})")
        else:
            logger.info("Sorted piece logger initialized (disabled by config)")

    def _initialize_log(self):
        """
        Create the CSV file with headers if it doesn't exist.

        Called during initialization to ensure the log file is ready.
        """
        if not os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        'element_id',
                        'name',
                        'primary_category',
                        'secondary_category',
                        'tertiary_category',
                        'encounter_count'
                    ])
                logger.info(f"Created sorted pieces log: {self.log_path}")
            except Exception as e:
                logger.error(f"Failed to create sorted pieces log: {e}")

    def log_piece(self, identified_piece: IdentifiedPiece):
        """
        Log a successfully sorted piece to the database.

        This is the main entry point called by ProcessingCoordinator after
        piece processing completes. Only pieces with valid element_ids are logged.

        WHEN CALLED:
        - After identified_piece.finalize()
        - Before GUI callbacks
        - Only for pieces with element_id (filters out unknowns/failures)

        THREAD SAFETY:
        Uses self.log_lock to ensure concurrent logging is safe.

        Args:
            identified_piece: Successfully identified piece to log
        """
        # Pre-check: Return early if logging disabled
        if not self.logging_enabled:
            logger.debug("Sorted piece logging disabled by config")
            return

        # Validation: Check if element_id exists
        if not identified_piece.element_id:
            # Can't log without element_id - this filters out unknown pieces
            logger.debug(f"Skipping log for piece {identified_piece.piece_id} (no element_id)")
            return

        # Thread-safe operation
        with self.log_lock:
            try:
                # Read existing data
                existing_entries = {}

                if os.path.exists(self.log_path):
                    with open(self.log_path, 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            element_id = row.get('element_id', '').strip()
                            if element_id:
                                existing_entries[element_id] = row

                element_id = identified_piece.element_id

                # Update or create entry
                if element_id in existing_entries:
                    # Increment encounter count for existing piece
                    existing_entries[element_id]['encounter_count'] = str(
                        int(existing_entries[element_id]['encounter_count']) + 1
                    )
                    count = existing_entries[element_id]['encounter_count']
                    logger.debug(
                        f"Piece {element_id} encountered again (count: {count})"
                    )
                else:
                    # Create new entry for first encounter
                    existing_entries[element_id] = {
                        'element_id': element_id,
                        'name': identified_piece.name or "Unknown",
                        'primary_category': identified_piece.primary_category or "",
                        'secondary_category': identified_piece.secondary_category or "",
                        'tertiary_category': identified_piece.tertiary_category or "",
                        'encounter_count': '1'
                    }
                    logger.info(
                        f"New piece logged: {element_id} ({identified_piece.name})"
                    )

                # Write updated data
                with open(self.log_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=[
                        'element_id',
                        'name',
                        'primary_category',
                        'secondary_category',
                        'tertiary_category',
                        'encounter_count'
                    ])
                    writer.writeheader()
                    for entry in existing_entries.values():
                        writer.writerow(entry)

            except Exception as e:
                # Failed logging shouldn't break the sorting pipeline
                logger.error(f"Failed to log sorted piece: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Provide summary statistics for debugging/monitoring.

        Returns:
            Dictionary with logging statistics:
            - total_unique_pieces: Count of unique element_ids in log
            - total_encounters: Sum of all encounter_counts
            - logging_enabled: Boolean indicating if logging is active
            - log_file_path: Path to the CSV file

        Example:
            stats = logger.get_statistics()
            print(f"Unique pieces: {stats['total_unique_pieces']}")
            print(f"Total encounters: {stats['total_encounters']}")
        """
        stats = {
            "total_unique_pieces": 0,
            "total_encounters": 0,
            "logging_enabled": self.logging_enabled,
            "log_file_path": self.log_path
        }

        if not self.logging_enabled:
            return stats

        # Thread-safe read
        with self.log_lock:
            try:
                if os.path.exists(self.log_path):
                    with open(self.log_path, 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            stats["total_unique_pieces"] += 1
                            count = int(row.get('encounter_count', 0))
                            stats["total_encounters"] += count

            except Exception as e:
                logger.error(f"Error reading statistics: {e}")

        return stats


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_sorted_piece_logger(
        config_manager: EnhancedConfigManager
) -> SortedPieceLogger:
    """
    Factory function to create a sorted piece logger instance.

    This follows the standard factory pattern used throughout the codebase
    for module instantiation.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Initialized SortedPieceLogger ready to use

    Example:
        config_manager = create_config_manager()
        logger = create_sorted_piece_logger(config_manager)
        logger.log_piece(identified_piece)
    """
    return SortedPieceLogger(config_manager)
