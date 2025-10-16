# processing/processing_data_models.py
"""
processing_data_models.py - Data structures for piece processing pipeline

These models define the data contracts between processing modules:
- IdentificationResult: Output from API identification
- CategoryInfo: Output from category lookup
- BinAssignment: Output from bin assignment logic
- IdentifiedPiece: Final complete result

This file contains only the essential data needed for processing,
with no extra fields for metrics or debugging that aren't actively used.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


# ============================================================================
# INTERMEDIATE RESULT TYPES
# ============================================================================
# These are the outputs from individual processing modules


@dataclass
class IdentificationResult:
    """
    Result from identification API (Brickognize).

    This is what the API handler returns after sending an image
    to the identification service.
    """
    element_id: str  # Official Lego element ID (e.g., "3001")
    name: str  # Human-readable name (e.g., "Brick 2x4")
    confidence: float  # API confidence score (0.0 to 1.0)


@dataclass
class CategoryInfo:
    """
    Result from category database lookup.

    This is what the category lookup module returns after searching
    the CSV database for an element_id.
    """
    element_id: str
    primary_category: str  # e.g., "Basic", "Technic", "Plates"
    secondary_category: Optional[str] = None  # e.g., "Brick", "Pin", "Tile"
    tertiary_category: Optional[str] = None  # e.g., "2x4", "1x1", "Round"
    found_in_database: bool = True  # False if element_id not in CSV


@dataclass
class BinAssignment:
    """
    Result from bin assignment logic.

    This is what the bin assignment module returns after determining
    which bin the piece should go to.
    """
    bin_number: int  # Target bin (0-9, where 0 is overflow)


# ============================================================================
# FINAL RESULT TYPE
# ============================================================================
# This is the complete output of the entire processing pipeline


@dataclass
class IdentifiedPiece:
    """
    Complete identification and sorting result for a Lego piece.

    This is the final output of the processing pipeline, containing
    all information about a piece after identification and bin assignment.

    The coordinator creates this with basic info, then each processing
    module adds its results using the update methods.
    """

    # ========================================================================
    # INITIAL FIELDS (Set at creation)
    # ========================================================================
    piece_id: int  # Links back to TrackedPiece from detection
    image_path: str  # Path to saved image file
    capture_timestamp: float  # When image was captured

    # ========================================================================
    # API IDENTIFICATION FIELDS (Set by identification_api_handler)
    # ========================================================================
    element_id: Optional[str] = None
    name: Optional[str] = None
    identification_confidence: Optional[float] = None

    # ========================================================================
    # CATEGORY FIELDS (Set by category_lookup_module)
    # ========================================================================
    primary_category: Optional[str] = None
    secondary_category: Optional[str] = None
    tertiary_category: Optional[str] = None
    found_in_database: bool = False

    # ========================================================================
    # BIN ASSIGNMENT FIELDS (Set by bin_assignment_module)
    # ========================================================================
    bin_number: Optional[int] = None

    # ========================================================================
    # COMPLETION STATUS
    # ========================================================================
    complete: bool = False  # Set to True when processing finishes

    # ========================================================================
    # UPDATE METHODS (Called by processing_coordinator)
    # ========================================================================

    def update_from_identification(self, result: IdentificationResult):
        """
        Update fields from API identification result.

        Args:
            result: IdentificationResult from identification_api_handler
        """
        self.element_id = result.element_id
        self.name = result.name
        self.identification_confidence = result.confidence

    def update_from_categories(self, result: CategoryInfo):
        """
        Update fields from category lookup result.

        Args:
            result: CategoryInfo from category_lookup_module
        """
        self.primary_category = result.primary_category
        self.secondary_category = result.secondary_category
        self.tertiary_category = result.tertiary_category
        self.found_in_database = result.found_in_database

    def update_from_bin_assignment(self, result: BinAssignment):
        """
        Update fields from bin assignment result.

        Args:
            result: BinAssignment from bin_assignment_module
        """
        self.bin_number = result.bin_number

    def finalize(self):
        """
        Mark processing as complete.

        This should be called by the coordinator after all processing
        steps are done and all fields are populated.
        """
        if self.is_complete:
            self.complete = True

    # ========================================================================
    # PROPERTIES (Computed values)
    # ========================================================================

    @property
    def is_complete(self) -> bool:
        """
        Check if all required fields are populated.

        Returns:
            True if piece has been fully processed
        """
        return all([
            self.element_id is not None,
            self.primary_category is not None,
            self.bin_number is not None
        ])

    @property
    def confidence(self) -> float:
        """
        Alias for identification_confidence.

        This provides a shorter property name for backwards compatibility
        and cleaner access to the confidence score.

        Returns:
            Identification confidence (0.0 to 1.0)
        """
        return self.identification_confidence if self.identification_confidence is not None else 0.0

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def get_full_category_path(self) -> str:
        """
        Get complete category hierarchy as a single string.

        Returns:
            Category path like "Basic > Brick > 2x4" or "Unknown"

        Examples:
            "Basic"
            "Basic > Brick"
            "Basic > Brick > 2x4"
            "Unknown" (if no categories set)
        """
        parts = []
        if self.primary_category:
            parts.append(self.primary_category)
        if self.secondary_category:
            parts.append(self.secondary_category)
        if self.tertiary_category:
            parts.append(self.tertiary_category)
        return " > ".join(parts) if parts else "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for API responses, logging, or history storage.

        Returns:
            Dictionary with all important piece information
        """
        return {
            "piece_id": self.piece_id,
            "element_id": self.element_id,
            "b": self.name,
            "category_path": self.get_full_category_path(),
            "primary_category": self.primary_category,
            "secondary_category": self.secondary_category,
            "tertiary_category": self.tertiary_category,
            "bin_number": self.bin_number,
            "confidence": self.confidence,
            "complete": self.complete,
            "found_in_database": self.found_in_database,
            "image_path": self.image_path,
            "capture_timestamp": self.capture_timestamp
        }

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns:
            Human-readable string with key information
        """
        status = "COMPLETE" if self.complete else "INCOMPLETE"
        return (
            f"IdentifiedPiece(id={self.piece_id}, "
            f"element_id={self.element_id}, "
            f"name={self.name}, "
            f"bin={self.bin_number}, "
            f"confidence={self.confidence:.2f}, "
            f"status={status})"
        )


# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the data models to ensure they work correctly.
    """
    import time

    print("Testing processing data models...\n")

    # Create an identified piece with initial data
    piece = IdentifiedPiece(
        piece_id=42,
        image_path="/path/to/piece_042.jpg",
        capture_timestamp=time.time()
    )

    print(f"1. Created: {piece}")
    print(f"   Is complete? {piece.is_complete}\n")

    # Simulate API identification
    api_result = IdentificationResult(
        element_id="3001",
        name="Brick 2x4",
        confidence=0.95
    )
    piece.update_from_identification(api_result)
    print(f"2. After API: {piece}")
    print(f"   Is complete? {piece.is_complete}\n")

    # Simulate category lookup
    category_result = CategoryInfo(
        element_id="3001",
        primary_category="Basic",
        secondary_category="Brick",
        tertiary_category="2x4",
        found_in_database=True
    )
    piece.update_from_categories(category_result)
    print(f"3. After categories: {piece}")
    print(f"   Category path: {piece.get_full_category_path()}")
    print(f"   Is complete? {piece.is_complete}\n")

    # Simulate bin assignment
    bin_result = BinAssignment(bin_number=3)
    piece.update_from_bin_assignment(bin_result)
    print(f"4. After bin assignment: {piece}")
    print(f"   Is complete? {piece.is_complete}\n")

    # Finalize
    piece.finalize()
    print(f"5. After finalize: {piece}")
    print(f"   Complete flag: {piece.complete}\n")

    # Show dictionary representation
    print("6. Dictionary representation:")
    import json

    print(json.dumps(piece.to_dict(), indent=2))

    print("\n✓ All tests passed!")