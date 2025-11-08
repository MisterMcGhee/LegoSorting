"""
Reference library data structure

Manages the complete collection of piece references extracted from manual
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from .piece_reference import PieceReference


class ReferenceLibrary:
    """
    Complete reference library for a LEGO set.

    Stores all piece references extracted from the instruction manual
    along with metadata and validation status.
    """

    def __init__(self, set_number: str = "", set_name: str = ""):
        """
        Initialize empty reference library.

        Args:
            set_number: LEGO set number (e.g., "31129")
            set_name: LEGO set name (e.g., "Majestic Tiger")
        """
        self.set_number = set_number
        self.set_name = set_name
        self.pieces: Dict[str, PieceReference] = {}
        self.extraction_timestamp: Optional[str] = None
        self.reference_pages: List[int] = []
        self.validation_status: Dict[str, bool] = {}

    def add_piece(self, piece: PieceReference) -> None:
        """
        Add a piece reference to the library.

        Args:
            piece: PieceReference to add
        """
        self.pieces[piece.element_id] = piece
        self.validation_status[piece.element_id] = piece.validated

    def get_piece(self, element_id: str) -> Optional[PieceReference]:
        """
        Retrieve a piece by element_id.

        Args:
            element_id: Element ID to look up

        Returns:
            PieceReference if found, None otherwise
        """
        return self.pieces.get(element_id)

    def get_all_element_ids(self) -> List[str]:
        """
        Get list of all element IDs in library.

        Returns:
            List of element ID strings
        """
        return list(self.pieces.keys())

    def get_validated_count(self) -> int:
        """
        Count how many pieces passed CSV validation.

        Returns:
            Number of validated pieces
        """
        return sum(1 for piece in self.pieces.values() if piece.validated)

    def get_unvalidated_pieces(self) -> List[PieceReference]:
        """
        Get list of pieces that failed validation.

        Returns:
            List of unvalidated PieceReferences
        """
        return [piece for piece in self.pieces.values() if not piece.validated]

    def get_statistics(self) -> Dict:
        """
        Get library statistics for reporting.

        Returns:
            Dictionary with statistics
        """
        total = len(self.pieces)
        validated = self.get_validated_count()
        unvalidated = total - validated

        avg_confidence = 0.0
        if total > 0:
            avg_confidence = sum(p.ocr_confidence for p in self.pieces.values()) / total

        return {
            'total_pieces': total,
            'validated_pieces': validated,
            'unvalidated_pieces': unvalidated,
            'validation_rate': (validated / total * 100) if total > 0 else 0.0,
            'average_ocr_confidence': avg_confidence,
            'reference_pages': self.reference_pages,
            'extraction_timestamp': self.extraction_timestamp
        }

    def save_metadata(self, output_path: str) -> None:
        """
        Save library metadata to JSON (without large numpy arrays).

        Args:
            output_path: Path to save JSON file
        """
        metadata = {
            'set_number': self.set_number,
            'set_name': self.set_name,
            'extraction_timestamp': self.extraction_timestamp or datetime.now().isoformat(),
            'reference_pages': self.reference_pages,
            'statistics': self.get_statistics(),
            'pieces': {
                element_id: piece.to_dict(include_images=False)
                for element_id, piece in self.pieces.items()
            }
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def __len__(self) -> int:
        """Return number of pieces in library"""
        return len(self.pieces)

    def __str__(self) -> str:
        """String representation for logging"""
        stats = self.get_statistics()
        return (f"ReferenceLibrary: {self.set_number} - {self.set_name}\n"
                f"  Total pieces: {stats['total_pieces']}\n"
                f"  Validated: {stats['validated_pieces']} ({stats['validation_rate']:.1f}%)\n"
                f"  Avg OCR confidence: {stats['average_ocr_confidence']:.2f}")
