"""
CSV handler for loading and validating LEGO set inventory

Handles CSV files with format:
Quantity, ElementID, Name
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class InventoryCSVHandler:
    """
    Handler for LEGO set inventory CSV files.

    Loads and validates inventory CSV with columns: Quantity, ElementID, Name
    """

    def __init__(self, csv_path: str):
        """
        Initialize CSV handler.

        Args:
            csv_path: Path to inventory CSV file
        """
        self.csv_path = csv_path
        self.inventory: Dict[str, Dict] = {}
        self._load_inventory()

    def _load_inventory(self) -> None:
        """Load inventory from CSV file."""
        try:
            # Load CSV with pandas
            df = pd.read_csv(self.csv_path, sep='\t')  # Try tab-separated first

            # Check if we got the right columns
            if 'ElementID' not in df.columns:
                # Try comma-separated
                df = pd.read_csv(self.csv_path, sep=',')

            # Validate required columns
            required_columns = ['Quantity', 'ElementID', 'Name']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Build inventory dictionary
            for _, row in df.iterrows():
                element_id = str(row['ElementID']).strip()
                quantity = int(row['Quantity'])
                name = str(row['Name']).strip()

                self.inventory[element_id] = {
                    'quantity': quantity,
                    'name': name
                }

            logger.info(f"Loaded {len(self.inventory)} pieces from {self.csv_path}")

        except Exception as e:
            logger.error(f"Error loading inventory CSV: {e}")
            raise

    def get_element_ids(self) -> Set[str]:
        """
        Get set of all element IDs in inventory.

        Returns:
            Set of element ID strings
        """
        return set(self.inventory.keys())

    def validate_element_id(self, element_id: str) -> bool:
        """
        Check if an element ID exists in the inventory.

        Args:
            element_id: Element ID to validate

        Returns:
            True if element ID exists in inventory
        """
        return element_id in self.inventory

    def get_piece_info(self, element_id: str) -> Optional[Dict]:
        """
        Get information about a piece.

        Args:
            element_id: Element ID to look up

        Returns:
            Dictionary with 'quantity' and 'name', or None if not found
        """
        return self.inventory.get(element_id)

    def get_piece_name(self, element_id: str) -> str:
        """
        Get the name of a piece.

        Args:
            element_id: Element ID to look up

        Returns:
            Piece name, or "Unknown" if not found
        """
        info = self.get_piece_info(element_id)
        return info['name'] if info else "Unknown"

    def get_total_pieces(self) -> int:
        """
        Get total number of unique pieces in inventory.

        Returns:
            Count of unique pieces
        """
        return len(self.inventory)

    def get_total_quantity(self) -> int:
        """
        Get total quantity of all pieces combined.

        Returns:
            Total piece count
        """
        return sum(info['quantity'] for info in self.inventory.values())

    def get_inventory_summary(self) -> Dict:
        """
        Get summary statistics about the inventory.

        Returns:
            Dictionary with summary statistics
        """
        return {
            'csv_path': self.csv_path,
            'unique_pieces': self.get_total_pieces(),
            'total_quantity': self.get_total_quantity(),
            'sample_element_ids': list(self.get_element_ids())[:5]
        }

    def __len__(self) -> int:
        """Return number of unique pieces in inventory"""
        return len(self.inventory)

    def __contains__(self, element_id: str) -> bool:
        """Check if element_id is in inventory"""
        return element_id in self.inventory

    def __str__(self) -> str:
        """String representation for logging"""
        return (f"InventoryCSV: {self.csv_path}\n"
                f"  Unique pieces: {self.get_total_pieces()}\n"
                f"  Total quantity: {self.get_total_quantity()}")
