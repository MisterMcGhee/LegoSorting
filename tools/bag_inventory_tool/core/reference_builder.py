"""
Reference builder for extracting piece library from manual reference pages

Main processing logic for Phase 1: extracting, OCR'ing, and validating piece references
"""

import cv2
import numpy as np
import logging
import os
from typing import List, Tuple, Optional
from datetime import datetime

from ..models.piece_reference import PieceReference
from ..models.reference_library import ReferenceLibrary
from ..utils.csv_handler import InventoryCSVHandler
from ..utils.ocr_engine import OCREngine
from ..utils import image_processing as img_proc

logger = logging.getLogger(__name__)


class ReferenceBuilder:
    """
    Builds reference library from instruction manual reference pages.

    Extracts piece images, OCRs element IDs, validates against CSV,
    and creates templates for future matching.
    """

    def __init__(self, inventory_csv: InventoryCSVHandler, config: dict):
        """
        Initialize reference builder.

        Args:
            inventory_csv: CSV handler with set inventory (ground truth)
            config: Reference extraction configuration
        """
        self.inventory = inventory_csv
        self.config = config
        self.ocr_config = config  # Will be passed from main config

        # Initialize OCR engine
        self.ocr = None  # Lazy initialization

    def _init_ocr(self, ocr_config: dict) -> None:
        """Initialize OCR engine (lazy loading)."""
        if self.ocr is None:
            self.ocr = OCREngine(ocr_config)

    def build_reference_library(self, reference_pages: List[np.ndarray],
                                page_numbers: List[int],
                                ocr_config: dict,
                                set_number: str = "",
                                set_name: str = "") -> ReferenceLibrary:
        """
        Main processing function to build reference library.

        Args:
            reference_pages: List of reference page images
            page_numbers: Corresponding page numbers
            ocr_config: OCR configuration
            set_number: LEGO set number
            set_name: LEGO set name

        Returns:
            Complete ReferenceLibrary
        """
        logger.info(f"Building reference library from {len(reference_pages)} page(s)")

        # Initialize OCR
        self._init_ocr(ocr_config)

        # Create library
        library = ReferenceLibrary(set_number, set_name)
        library.reference_pages = page_numbers
        library.extraction_timestamp = datetime.now().isoformat()

        # Process each reference page
        for idx, page_image in enumerate(reference_pages):
            page_num = page_numbers[idx] if idx < len(page_numbers) else idx + 1
            logger.info(f"Processing reference page {page_num}...")

            pieces = self._extract_pieces_from_page(page_image, page_num)

            # Add to library
            for piece in pieces:
                library.add_piece(piece)

        logger.info(f"Reference library complete: {len(library)} pieces extracted")

        return library

    def _extract_pieces_from_page(self, page_image: np.ndarray,
                                  page_num: int) -> List[PieceReference]:
        """
        Extract all pieces from a single reference page.

        Args:
            page_image: Reference page image
            page_num: Page number for logging

        Returns:
            List of PieceReference objects
        """
        # Detect piece regions
        piece_regions = self.detect_piece_regions(page_image)
        logger.info(f"  Found {len(piece_regions)} potential pieces on page {page_num}")

        pieces = []

        for idx, bbox in enumerate(piece_regions):
            # Extract piece information
            piece = self._process_piece_region(page_image, bbox, page_num, idx)

            if piece is not None:
                pieces.append(piece)

        logger.info(f"  Successfully processed {len(pieces)} pieces")

        return pieces

    def _process_piece_region(self, page_image: np.ndarray,
                             bbox: Tuple[int, int, int, int],
                             page_num: int,
                             piece_idx: int) -> Optional[PieceReference]:
        """
        Process a single piece region: extract image, OCR element ID, validate.

        Args:
            page_image: Full page image
            bbox: (x, y, w, h) of piece
            page_num: Page number
            piece_idx: Piece index on page

        Returns:
            PieceReference if successful, None otherwise
        """
        x, y, w, h = bbox

        # Crop piece image
        piece_image = page_image[y:y+h, x:x+w]

        # OCR element ID (search near piece)
        element_id, ocr_confidence = self.ocr.find_element_id_near_piece(
            page_image, bbox, search_margin=100
        )

        # Check if we got a valid element ID
        if not element_id:
            logger.debug(f"  Piece {piece_idx}: No valid element ID found")
            return None

        # Validate against CSV
        validated = self.inventory.validate_element_id(element_id)
        piece_name = self.inventory.get_piece_name(element_id)

        if not validated:
            logger.warning(f"  Piece {piece_idx}: Element ID '{element_id}' not in inventory CSV")

        # Create template
        template = img_proc.create_template(piece_image, self.config)

        # Create PieceReference
        piece_ref = PieceReference(
            element_id=element_id,
            name=piece_name,
            template_grayscale=template,
            original_image=piece_image,
            bounding_box=bbox,
            ocr_confidence=ocr_confidence,
            validated=validated
        )

        logger.debug(f"  {piece_ref}")

        return piece_ref

    def detect_piece_regions(self, page_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect individual piece bounding boxes on reference page.

        Uses contour detection to find pieces arranged in columns.

        Args:
            page_image: Reference page image

        Returns:
            List of (x, y, w, h) bounding boxes
        """
        # Convert to grayscale
        gray = img_proc.convert_to_grayscale(page_image)

        # Apply Gaussian blur
        blur_size = self.config.get('gaussian_blur_size', 5)
        blurred = img_proc.apply_gaussian_blur(gray, blur_size)

        # Apply binary threshold to separate pieces from background
        threshold = self.config.get('binary_threshold', 200)
        binary = img_proc.apply_binary_threshold(blurred, threshold)

        # Find contours
        min_area = self.config.get('min_piece_area', 500)
        max_area = self.config.get('max_piece_area', 50000)
        contours = img_proc.find_contours(binary, min_area, max_area)

        # Get bounding boxes
        boxes = img_proc.get_bounding_boxes(contours)

        # Filter by aspect ratio
        min_aspect = self.config.get('min_aspect_ratio', 0.2)
        max_aspect = self.config.get('max_aspect_ratio', 5.0)
        filtered_boxes = []

        for (x, y, w, h) in boxes:
            aspect_ratio = w / h if h > 0 else 0
            if min_aspect <= aspect_ratio <= max_aspect:
                filtered_boxes.append((x, y, w, h))

        logger.debug(f"  Detected {len(contours)} contours, {len(filtered_boxes)} after filtering")

        return filtered_boxes

    def save_debug_visualization(self, page_image: np.ndarray,
                                 piece_regions: List[Tuple[int, int, int, int]],
                                 output_path: str) -> None:
        """
        Save visualization of detected pieces for debugging.

        Args:
            page_image: Reference page image
            piece_regions: List of detected piece bounding boxes
            output_path: Path to save image
        """
        # Draw bounding boxes
        vis_image = img_proc.draw_bounding_boxes(page_image, piece_regions)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        logger.info(f"  Saved debug visualization to {output_path}")

    def save_reference_library(self, library: ReferenceLibrary, output_dir: str) -> None:
        """
        Save reference library to disk.

        Saves:
        - Templates (grayscale) for matching
        - Original images (color)
        - Metadata JSON

        Args:
            library: ReferenceLibrary to save
            output_dir: Output directory
        """
        logger.info(f"Saving reference library to {output_dir}...")

        # Create directories
        templates_dir = os.path.join(output_dir, "templates")
        originals_dir = os.path.join(output_dir, "originals")
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(originals_dir, exist_ok=True)

        # Save each piece
        for element_id, piece in library.pieces.items():
            # Save template
            template_path = os.path.join(templates_dir, f"{element_id}.png")
            cv2.imwrite(template_path, piece.template_grayscale)

            # Save original
            original_path = os.path.join(originals_dir, f"{element_id}.png")
            cv2.imwrite(original_path, cv2.cvtColor(piece.original_image, cv2.COLOR_RGB2BGR))

        # Save metadata
        metadata_path = os.path.join(output_dir, "library.json")
        library.save_metadata(metadata_path)

        logger.info(f"  Saved {len(library)} pieces")
        logger.info(f"  Templates: {templates_dir}")
        logger.info(f"  Originals: {originals_dir}")
        logger.info(f"  Metadata: {metadata_path}")
