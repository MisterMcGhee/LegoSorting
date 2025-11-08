"""
Data models for piece references

Represents individual LEGO pieces extracted from reference pages
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import numpy as np


@dataclass
class PieceReference:
    """
    Represents a single LEGO piece from the reference inventory.

    Attributes:
        element_id: Unique element ID (design_id + color_id, e.g., "3034", "4081b")
        name: Human-readable piece name
        template_grayscale: Grayscale image template for matching (numpy array)
        original_image: Original color image from reference page (numpy array)
        bounding_box: (x, y, width, height) location on reference page
        ocr_confidence: Confidence score from OCR (0.0-1.0)
        validated: Whether element_id exists in inventory CSV
    """
    element_id: str
    name: str
    template_grayscale: np.ndarray
    original_image: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    ocr_confidence: float = 1.0
    validated: bool = False

    def to_dict(self, include_images: bool = False) -> dict:
        """
        Convert to dictionary for serialization.

        Args:
            include_images: Whether to include numpy arrays (large)

        Returns:
            Dictionary representation
        """
        data = {
            'element_id': self.element_id,
            'name': self.name,
            'bounding_box': self.bounding_box,
            'ocr_confidence': self.ocr_confidence,
            'validated': self.validated
        }

        if include_images:
            data['template_shape'] = self.template_grayscale.shape
            data['original_shape'] = self.original_image.shape

        return data

    def __str__(self) -> str:
        """String representation for logging"""
        status = "✓" if self.validated else "✗"
        return f"[{status}] {self.element_id}: {self.name} (conf: {self.ocr_confidence:.2f})"
