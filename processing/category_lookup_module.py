# processing/category_lookup_module.py
"""
category_lookup_module.py - Category database lookup for Lego pieces

This module handles looking up piece categories from the Lego_Categories.csv database.

RESPONSIBILITIES:
- Load and cache Lego_Categories.csv database
- Look up design_id → category information
- Return CategoryInfo dataclass
- Log unknown design_ids for manual database updates
- Track encounter counts for unknown pieces

DOES NOT:
- Assign bins (that's bin_assignment_module)
- Apply sorting strategy
- Make business logic decisions
"""

import os
import csv
import threading
import logging
from typing import Dict, Optional
from datetime import datetime

from processing.processing_data_models import CategoryInfo
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig

# Set up module logger
logger = logging.getLogger(__name__)


class CategoryLookup:
    """
    Looks up piece categories from CSV database.

    Loads the Lego_Categories.csv file once at initialization and caches
    all category data in memory for fast lookups.
    """

    def __init__(self, config_manager: EnhancedConfigManager):
        """
        Initialize category lookup with database.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

        # Get configuration
        module_config = config_manager.get_module_config(
            ModuleConfig.PIECE_IDENTIFIER.value
        )

        self.csv_path = module_config["csv_path"]
        self.save_unknown = module_config["save_unknown"]
        self.unknown_log_path = "data/unknown_pieces.csv"

        # Category database cache (design_id -> category info)
        self.categories_data: Dict[str, Dict[str, str]] = {}

        # Thread safety for unknown logging
        self.unknown_log_lock = threading.Lock()

        # Load the categories database
        self._load_categories_database()

        # Initialize unknown pieces log if needed
        if self.save_unknown:
            self._initialize_unknown_log()

        logger.info(
            f"Category lookup initialized with {len(self.categories_data)} pieces"
        )

    def _load_categories_database(self):
        """Load the Lego_Categories.csv file into memory."""
        if not os.path.exists(self.csv_path):
            error_msg = f"Categories database not found: {self.csv_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)

                for row in csv_reader:
                    design_id = row.get('design_id', '').strip()

                    if not design_id:
                        continue  # Skip empty rows

                    # Store complete piece information
                    self.categories_data[design_id] = {
                        'name': row.get('name', '').strip(),
                        'primary_category': row.get('primary_category', '').strip(),
                        'secondary_category': row.get('secondary_category', '').strip(),
                        'tertiary_category': row.get('tertiary_category', '').strip()
                    }

            logger.info(f"Loaded {len(self.categories_data)} pieces from {self.csv_path}")

        except Exception as e:
            error_msg = f"Error loading categories database: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _initialize_unknown_log(self):
        """Initialize the unknown pieces log file if it doesn't exist."""
        if not os.path.exists(self.unknown_log_path):
            try:
                with open(self.unknown_log_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        'design_id',
                        'name',
                        'first_seen',
                        'last_seen',
                        'encounter_count'
                    ])
                logger.info(f"Created unknown pieces log: {self.unknown_log_path}")
            except Exception as e:
                logger.error(f"Failed to create unknown pieces log: {e}")

    def get_categories(self, design_id: str,
                       piece_name: Optional[str] = None) -> CategoryInfo:
        """
        Look up categories for a design_id.

        If the design_id is not found in the database:
        - Logs it to unknown_pieces.csv for manual addition
        - Returns CategoryInfo with "Unknown" primary category
        - Sets found_in_database=False

        Args:
            design_id: Design ID from Brickognize (e.g., "3001")
            piece_name: Optional piece name from API (for logging unknowns)

        Returns:
            CategoryInfo with category hierarchy (or "Unknown" if not found)
        """
        # Check if design exists in database
        if design_id in self.categories_data:
            piece_data = self.categories_data[design_id]

            # Extract categories (handle empty strings as None)
            primary = piece_data.get('primary_category') or "Unknown"
            secondary = piece_data.get('secondary_category') or None
            tertiary = piece_data.get('tertiary_category') or None

            # Convert empty strings to None
            if secondary == "":
                secondary = None
            if tertiary == "":
                tertiary = None

            logger.debug(
                f"Found categories for {design_id}: "
                f"{primary}/{secondary}/{tertiary}"
            )

            return CategoryInfo(
                design_id=design_id,
                primary_category=primary,
                secondary_category=secondary,
                tertiary_category=tertiary,
                found_in_database=True
            )

        else:
            # Design not found - log it as unknown
            logger.warning(f"Design ID not found in database: {design_id}")

            if self.save_unknown:
                self._log_unknown_piece(design_id, piece_name)

            return CategoryInfo(
                design_id=design_id,
                primary_category="Unknown",
                secondary_category=None,
                tertiary_category=None,
                found_in_database=False
            )

    def design_exists(self, design_id: str) -> bool:
        """
        Check if a design_id exists in the database.

        Args:
            design_id: Design ID to check

        Returns:
            True if design exists in database
        """
        return design_id in self.categories_data

    def _log_unknown_piece(self, design_id: str, piece_name: Optional[str] = None):
        """
        Log an unknown piece to unknown_pieces.csv.

        If the piece has been seen before, updates last_seen and increments count.
        If it's new, creates a new entry.

        Args:
            design_id: Unknown design ID
            piece_name: Optional name from API (helps identify the piece)
        """
        with self.unknown_log_lock:
            try:
                # Read existing log to check if we've seen this before
                existing_entries = {}

                if os.path.exists(self.unknown_log_path):
                    with open(self.unknown_log_path, 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            existing_entries[row['design_id']] = row

                # Current timestamp
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Check if we've seen this design before
                if design_id in existing_entries:
                    # Update existing entry
                    existing_entries[design_id]['last_seen'] = now
                    existing_entries[design_id]['encounter_count'] = str(
                        int(existing_entries[design_id]['encounter_count']) + 1
                    )
                    logger.info(
                        f"Unknown piece {design_id} encountered again "
                        f"(count: {existing_entries[design_id]['encounter_count']})"
                    )
                else:
                    # New unknown piece
                    existing_entries[design_id] = {
                        'design_id': design_id,
                        'name': piece_name or "Unknown",
                        'first_seen': now,
                        'last_seen': now,
                        'encounter_count': '1'
                    }
                    logger.info(f"New unknown piece logged: {design_id} ({piece_name})")

                # Write updated log
                with open(self.unknown_log_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=[
                        'design_id',
                        'name',
                        'first_seen',
                        'last_seen',
                        'encounter_count'
                    ])
                    writer.writeheader()
                    for entry in existing_entries.values():
                        writer.writerow(entry)

            except Exception as e:
                logger.error(f"Failed to log unknown piece: {e}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_category_lookup(
        config_manager: EnhancedConfigManager
) -> CategoryLookup:
    """
    Factory function to create a category lookup instance.

    Args:
        config_manager: Configuration manager instance

    Returns:
        Initialized CategoryLookup

    Example:
        config_manager = create_config_manager()
        category_lookup = create_category_lookup(config_manager)
        categories = category_lookup.get_categories("3001", "Brick 2x4")
    """
    return CategoryLookup(config_manager)