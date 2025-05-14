"""
piece_history_module.py - Centralized piece history management for the Lego sorting application

This module maintains both an in-memory history of recently processed pieces (for quick access)
and a persistent CSV record (for inventory and data retention purposes).
"""

import csv
import os
import logging
import threading
import time
from typing import Dict, Any, Optional, List

from error_module import get_logger

# Initialize module logger
logger = get_logger(__name__)


class PieceHistory:
    """Manages piece processing history with both in-memory and persistent storage"""

    def __init__(self, config_manager=None, csv_path: str = "piece_history.csv"):
        """Initialize piece history manager

        Args:
            config_manager: Configuration manager for getting settings
            csv_path: Path to the CSV file for persistent storage
        """
        # Thread-safe access
        self.lock = threading.RLock()

        # In-memory history (limited size)
        self.history = []

        # Configuration
        if config_manager:
            history_config = config_manager.get_section("piece_history")
            self.max_entries = history_config.get("max_entries", 10)
            self.csv_path = history_config.get("csv_path", csv_path)
            self.include_timestamp = history_config.get("include_timestamp", True)
        else:
            # Default configuration
            self.max_entries = 10
            self.csv_path = csv_path
            self.include_timestamp = True

        # Initialize CSV file
        self._initialize_csv()

        logger.info(f"PieceHistory initialized with max_entries={self.max_entries}, csv_path={self.csv_path}")

    def _initialize_csv(self) -> None:
        """Create CSV file with headers if it doesn't exist"""
        try:
            # Check if file exists
            if not os.path.exists(self.csv_path):
                # Create directory if needed
                directory = os.path.dirname(self.csv_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)

                # Create CSV with headers
                headers = [
                    'piece_id',
                    'image_number',
                    'file_path',
                    'element_id',
                    'name',
                    'primary_category',
                    'secondary_category',
                    'bin_number',
                    'processing_time',
                    'is_exit_zone_piece'
                ]

                if self.include_timestamp:
                    headers.append('timestamp')

                with open(self.csv_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writeheader()

                logger.info(f"Created new piece history CSV at {self.csv_path}")
            else:
                logger.info(f"Using existing piece history CSV at {self.csv_path}")

        except Exception as e:
            logger.error(f"Error initializing CSV file: {e}")

    def add_piece(self, piece_data: Dict[str, Any]) -> None:
        """Add a piece to both in-memory history and CSV file

        Args:
            piece_data: Dictionary containing piece information
        """
        with self.lock:
            try:
                # Add timestamp if configured
                if self.include_timestamp:
                    piece_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

                # Add to in-memory history
                self.history.append(piece_data.copy())

                # Maintain max size limit
                if len(self.history) > self.max_entries:
                    self.history.pop(0)  # Remove oldest entry

                # Append to CSV file
                self._append_to_csv(piece_data)

                logger.debug(f"Added piece {piece_data.get('piece_id')} to history")

            except Exception as e:
                logger.error(f"Error adding piece to history: {e}")

    def _append_to_csv(self, piece_data: Dict[str, Any]) -> None:
        """Append a single piece to the CSV file

        Args:
            piece_data: Dictionary containing piece information
        """
        try:
            # Log what we're trying to write
            logger.info(f"Writing piece data to CSV: {piece_data.get('piece_id')}")

            # Define standard fieldnames directly
            fieldnames = [
                'piece_id',
                'image_number',
                'file_path',
                'element_id',
                'name',
                'primary_category',
                'secondary_category',
                'bin_number',
                'processing_time',
                'is_exit_zone_piece'
            ]

            # Add timestamp field if configured
            if self.include_timestamp:
                fieldnames.append('timestamp')

            # Create a clean copy of the data that we'll write to CSV
            csv_data = {}

            # Explicitly copy each expected field, converting None to empty string
            for field in fieldnames:
                if field in piece_data and piece_data[field] is not None:
                    csv_data[field] = piece_data[field]
                else:
                    csv_data[field] = ""  # Empty string for missing or None values

            logger.debug(f"CSV data prepared: {csv_data}")

            # Check if file exists - if not or empty, create with headers
            file_exists = os.path.exists(self.csv_path)
            file_empty = not file_exists or os.path.getsize(self.csv_path) == 0

            if file_empty:
                # Create file with headers
                with open(self.csv_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    logger.info(f"Created CSV file with headers: {self.csv_path}")

            # Append the data
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(csv_data)
                logger.info(f"Successfully wrote piece {csv_data.get('piece_id')} to CSV")

        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")
            logger.error(f"CSV path: {self.csv_path}")
            logger.error(f"Piece data: {piece_data}")
            logger.exception("Stack trace:")

    def get_latest_piece(self) -> Optional[Dict[str, Any]]:
        """Get the most recently processed piece

        Returns:
            Dictionary containing piece data, or None if history is empty
        """
        with self.lock:
            if self.history:
                return self.history[-1].copy()
            return None

    def get_piece_count(self) -> int:
        """Get the total number of pieces in current in-memory history

        Returns:
            Number of pieces in history
        """
        with self.lock:
            return len(self.history)

    def has_new_piece_since(self, last_piece_id: int) -> bool:
        """Check if a new piece has been processed since the given ID

        Args:
            last_piece_id: Last piece ID that was displayed

        Returns:
            True if there is a new piece, False otherwise
        """
        with self.lock:
            if not self.history:
                return False
            return self.history[-1].get('piece_id') != last_piece_id

    def clear_memory_history(self) -> None:
        """Clear the in-memory history (does not affect CSV file)"""
        with self.lock:
            self.history.clear()
            logger.info("In-memory piece history cleared")


# Factory function
def create_piece_history(config_manager=None, csv_path: str = "piece_history.csv") -> PieceHistory:
    """Create a piece history manager instance

    Args:
        config_manager: Configuration manager instance
        csv_path: Path to the CSV file

    Returns:
        PieceHistory instance
    """
    return PieceHistory(config_manager, csv_path)