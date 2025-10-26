# tools/categorization_tool/categorization_logic.py
"""
categorization_logic.py - Business logic for piece categorization

This module contains all business logic for categorizing LEGO pieces,
with NO GUI dependencies. Pure Python logic that can be tested independently.

RESPONSIBILITIES:
- Load/save unknown pieces from CSV
- Add/update categories in database
- Validate categorizations
- Manage category hierarchy reloading
- CSV file operations

DOES NOT:
- Create any GUI components
- Import PyQt5
- Handle user interaction directly
"""

import os
import csv
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from enhanced_config_manager import EnhancedConfigManager, ModuleConfig
from processing.category_hierarchy_service import CategoryHierarchyService

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class UnknownPiece:
    """Data model for an unknown piece."""
    
    def __init__(self, element_id: str, name: str, first_seen: str,
                 last_seen: str, encounter_count: str):
        self.element_id = element_id
        self.name = name
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.encounter_count = encounter_count
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for CSV writing."""
        return {
            'element_id': self.element_id,
            'name': self.name,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'encounter_count': self.encounter_count
        }


class Categorization:
    """Data model for a piece categorization."""
    
    def __init__(self, element_id: str, name: str, primary: str,
                 secondary: str, tertiary: Optional[str] = None):
        self.element_id = element_id
        self.name = name
        self.primary = primary
        self.secondary = secondary
        self.tertiary = tertiary or ""
    
    def to_csv_row(self) -> List[str]:
        """Convert to CSV row format."""
        return [
            self.element_id,
            self.name,
            self.primary,
            self.secondary,
            self.tertiary
        ]
    
    def __repr__(self) -> str:
        return (f"Categorization({self.element_id}, {self.primary}/"
                f"{self.secondary}/{self.tertiary})")


# ============================================================================
# UNKNOWN PIECES MANAGER
# ============================================================================

class UnknownPiecesManager:
    """
    Manages the unknown_pieces.csv file.
    
    Handles loading, removing, and tracking unknown pieces that need
    categorization.
    """
    
    def __init__(self, csv_path: str = "unknown_pieces.csv"):
        """
        Initialize the unknown pieces manager.
        
        Args:
            csv_path: Path to unknown_pieces.csv file
        """
        self.csv_path = csv_path
        logger.info(f"UnknownPiecesManager initialized: {csv_path}")
    
    def load_pieces(self) -> List[UnknownPiece]:
        """
        Load all unknown pieces from CSV.
        
        Returns:
            List of UnknownPiece objects in CSV order
        
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV format is invalid
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Unknown pieces file not found: {self.csv_path}")
        
        pieces = []
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Skip empty rows
                    element_id = row.get('element_id', '').strip()
                    if not element_id:
                        continue
                    
                    piece = UnknownPiece(
                        element_id=element_id,
                        name=row.get('name', '').strip(),
                        first_seen=row.get('first_seen', '').strip(),
                        last_seen=row.get('last_seen', '').strip(),
                        encounter_count=row.get('encounter_count', '0').strip()
                    )
                    pieces.append(piece)
            
            logger.info(f"Loaded {len(pieces)} unknown pieces")
            return pieces
            
        except Exception as e:
            logger.error(f"Error loading unknown pieces: {e}", exc_info=True)
            raise ValueError(f"Failed to load unknown pieces: {e}")
    
    def remove_piece(self, element_id: str) -> bool:
        """
        Remove a piece from the unknown pieces CSV.
        
        Args:
            element_id: Element ID to remove
        
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            # Read all pieces except the one to remove
            pieces = []
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames
                
                for row in reader:
                    if row.get('element_id', '').strip() != element_id:
                        pieces.append(row)
            
            # Write back to file
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(pieces)
            
            logger.info(f"Removed {element_id} from unknown pieces")
            return True
            
        except Exception as e:
            logger.error(f"Error removing piece from unknown CSV: {e}", exc_info=True)
            return False
    
    def get_count(self) -> int:
        """
        Get count of unknown pieces without loading full data.
        
        Returns:
            Number of pieces in the file
        """
        if not os.path.exists(self.csv_path):
            return 0
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                count = sum(1 for row in reader if row.get('element_id', '').strip())
            return count
        except Exception as e:
            logger.error(f"Error counting unknown pieces: {e}")
            return 0


# ============================================================================
# CATEGORIES DATABASE MANAGER
# ============================================================================

class CategoriesDatabaseManager:
    """
    Manages the Lego_Categories.csv database.
    
    Handles adding new categorizations and updating existing entries.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the categories database manager.
        
        Args:
            csv_path: Path to Lego_Categories.csv file
        """
        self.csv_path = csv_path
        logger.info(f"CategoriesDatabaseManager initialized: {csv_path}")
    
    def element_exists(self, element_id: str) -> bool:
        """
        Check if an element ID already exists in the database.
        
        Args:
            element_id: Element ID to check
        
        Returns:
            True if exists, False otherwise
        """
        if not os.path.exists(self.csv_path):
            return False
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('element_id', '').strip() == element_id:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking element existence: {e}")
            return False
    
    def add_categorization(self, categorization: Categorization) -> bool:
        """
        Add a new categorization to the database.
        
        Args:
            categorization: Categorization object to add
        
        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Append to CSV file
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(categorization.to_csv_row())
            
            logger.info(f"Added categorization: {categorization}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding categorization: {e}", exc_info=True)
            return False
    
    def update_categorization(self, categorization: Categorization) -> bool:
        """
        Update an existing categorization in the database.
        
        Args:
            categorization: Updated categorization
        
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Read all rows
            rows = []
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames
                
                for row in reader:
                    if row.get('element_id', '').strip() == categorization.element_id:
                        # Update this row
                        row['name'] = categorization.name
                        row['primary_category'] = categorization.primary
                        row['secondary_category'] = categorization.secondary
                        row['tertiary_category'] = categorization.tertiary
                    rows.append(row)
            
            # Write all rows back
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Updated categorization: {categorization}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating categorization: {e}", exc_info=True)
            return False


# ============================================================================
# CATEGORY VALIDATOR
# ============================================================================

class CategoryValidator:
    """
    Validates categorization inputs.
    
    Ensures categories meet requirements before saving.
    """
    
    @staticmethod
    def validate_categorization(primary: str, secondary: str,
                               tertiary: Optional[str] = None) -> tuple[bool, str]:
        """
        Validate a categorization.
        
        Args:
            primary: Primary category
            secondary: Secondary category
            tertiary: Optional tertiary category
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check primary (required)
        if not primary or not primary.strip():
            return False, "Primary category is required"
        
        # Check secondary (required)
        if not secondary or not secondary.strip():
            return False, "Secondary category is required"
        
        # Tertiary is optional, no validation needed
        
        return True, ""
    
    @staticmethod
    def sanitize_category(category: str) -> str:
        """
        Sanitize a category name.
        
        Args:
            category: Raw category name
        
        Returns:
            Sanitized category name
        """
        if not category:
            return ""
        
        # Strip whitespace
        category = category.strip()
        
        # Title case for consistency
        category = category.title()
        
        return category


# ============================================================================
# CATEGORIZATION CONTROLLER
# ============================================================================

class CategorizationController:
    """
    Main controller that coordinates business logic.
    
    This is the primary interface that the GUI should use.
    Orchestrates the managers and validator.
    """
    
    def __init__(self, config_manager: EnhancedConfigManager,
                 category_service: CategoryHierarchyService):
        """
        Initialize the categorization controller.
        
        Args:
            config_manager: Configuration manager
            category_service: Category hierarchy service
        """
        self.config_manager = config_manager
        self.category_service = category_service
        
        # Get file paths from config
        piece_config = config_manager.get_module_config(
            ModuleConfig.PIECE_IDENTIFIER.value
        )
        categories_path = piece_config["csv_path"]
        unknown_path = "unknown_pieces.csv"
        
        # Create managers
        self.unknown_manager = UnknownPiecesManager(unknown_path)
        self.database_manager = CategoriesDatabaseManager(categories_path)
        self.validator = CategoryValidator()
        
        logger.info("CategorizationController initialized")
    
    def load_unknown_pieces(self) -> List[UnknownPiece]:
        """
        Load all unknown pieces.
        
        Returns:
            List of UnknownPiece objects
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        return self.unknown_manager.load_pieces()
    
    def save_categorization(self, element_id: str, name: str,
                           primary: str, secondary: str,
                           tertiary: Optional[str] = None,
                           allow_update: bool = False) -> tuple[bool, str]:
        """
        Save a piece categorization.
        
        Args:
            element_id: Element ID
            name: Piece name
            primary: Primary category
            secondary: Secondary category
            tertiary: Optional tertiary category
            allow_update: Allow updating if element already exists
        
        Returns:
            Tuple of (success, message)
        """
        # Validate inputs
        valid, error = self.validator.validate_categorization(primary, secondary, tertiary)
        if not valid:
            logger.warning(f"Validation failed: {error}")
            return False, error
        
        # Sanitize categories
        primary = self.validator.sanitize_category(primary)
        secondary = self.validator.sanitize_category(secondary)
        tertiary = self.validator.sanitize_category(tertiary) if tertiary else ""
        
        # Create categorization object
        categorization = Categorization(element_id, name, primary, secondary, tertiary)
        
        # Check if element already exists
        if self.database_manager.element_exists(element_id):
            if allow_update:
                # Update existing entry
                success = self.database_manager.update_categorization(categorization)
                if success:
                    return True, f"Updated categorization for {element_id}"
                else:
                    return False, "Failed to update database"
            else:
                return False, f"Element {element_id} already exists in database"
        
        # Add new categorization
        success = self.database_manager.add_categorization(categorization)
        if not success:
            return False, "Failed to save to database"
        
        # Remove from unknown pieces
        self.unknown_manager.remove_piece(element_id)
        
        return True, f"Categorized {element_id} successfully"
    
    def reload_category_hierarchy(self):
        """Reload the category hierarchy service."""
        try:
            logger.info("Reloading category hierarchy")
            self.category_service._load_hierarchy()
            logger.info("✓ Category hierarchy reloaded")
        except Exception as e:
            logger.error(f"Error reloading category hierarchy: {e}", exc_info=True)
    
    def get_primary_categories(self) -> List[str]:
        """Get list of primary categories."""
        return self.category_service.get_primary_categories()
    
    def get_secondary_categories(self, primary: str) -> List[str]:
        """Get list of secondary categories under a primary."""
        return self.category_service.get_secondary_categories(primary)
    
    def get_tertiary_categories(self, primary: str, secondary: str) -> List[str]:
        """Get list of tertiary categories under primary+secondary."""
        return self.category_service.get_tertiary_categories(primary, secondary)
    
    def get_unknown_count(self) -> int:
        """Get count of unknown pieces remaining."""
        return self.unknown_manager.get_count()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_categorization_controller(
        config_manager: EnhancedConfigManager,
        category_service: CategoryHierarchyService) -> CategorizationController:
    """
    Factory function to create a categorization controller.
    
    Args:
        config_manager: Configuration manager instance
        category_service: Category hierarchy service instance
    
    Returns:
        Configured CategorizationController
    """
    return CategorizationController(config_manager, category_service)


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the categorization logic independently."""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Testing Categorization Logic")
    print("=" * 60)
    
    # Test validator
    print("\nTesting CategoryValidator...")
    validator = CategoryValidator()
    
    test_cases = [
        ("Basic", "Brick", "2x4", True, ""),
        ("Basic", "Brick", None, True, ""),
        ("", "Brick", "2x4", False, "Primary category is required"),
        ("Basic", "", "2x4", False, "Secondary category is required"),
    ]
    
    for primary, secondary, tertiary, expected_valid, expected_msg in test_cases:
        valid, msg = validator.validate_categorization(primary, secondary, tertiary)
        status = "✓" if valid == expected_valid else "✗"
        print(f"{status} validate({primary!r}, {secondary!r}, {tertiary!r}) -> {valid}, {msg!r}")
    
    print("\n" + "=" * 60)
    print("Logic tests complete!")
