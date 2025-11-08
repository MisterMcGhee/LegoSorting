"""
Reference builder for extracting piece library from manual reference pages

REFACTORED: CSV-first approach
- CSV inventory is the source of truth
- OCR entire reference page to get all text
- For each CSV element ID, find it in OCR results using fuzzy matching
- Extract piece image near the matched text
"""

import cv2
import numpy as np
import logging
import os
import sys
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.piece_reference import PieceReference
from models.reference_library import ReferenceLibrary
from utils.csv_handler import InventoryCSVHandler
from utils.ocr_engine import OCREngine
from utils import image_processing as img_proc

logger = logging.getLogger(__name__)


class ReferenceBuilder:
    """
    Builds reference library from instruction manual reference pages.

    NEW APPROACH (CSV-first):
    1. Load CSV inventory (source of truth)
    2. OCR entire reference page to get all text with positions
    3. For each element ID in CSV:
       - Find it in OCR results (fuzzy matching)
       - Extract piece image near that text
       - Create template
    4. Report found/missing pieces
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
        Main processing function to build reference library using CSV-first approach.

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
        logger.info("Using CSV-first approach: CSV inventory is source of truth")

        # Initialize OCR
        self._init_ocr(ocr_config)

        # Create library
        library = ReferenceLibrary(set_number, set_name)
        library.reference_pages = page_numbers
        library.extraction_timestamp = datetime.now().isoformat()

        # Get all element IDs from CSV (source of truth)
        csv_element_ids = list(self.inventory.get_element_ids())
        logger.info(f"CSV contains {len(csv_element_ids)} unique element IDs")

        # OCR all reference pages to get text map
        logger.info("Scanning reference pages with OCR...")
        all_ocr_results = []
        for idx, page_image in enumerate(reference_pages):
            page_num = page_numbers[idx] if idx < len(page_numbers) else idx + 1
            logger.info(f"  Scanning page {page_num}...")

            ocr_results = self.ocr.scan_page(page_image)
            all_ocr_results.extend([
                {**result, 'page_num': page_num, 'page_image': page_image}
                for result in ocr_results
            ])

        logger.info(f"OCR found {len(all_ocr_results)} text regions total")
        likely_ids = [r for r in all_ocr_results if r['is_likely_element_id']]
        logger.info(f"  {len(likely_ids)} look like element IDs")

        # Now match each CSV element ID to OCR results
        logger.info("Matching CSV element IDs to OCR results...")
        found_count = 0
        missing_element_ids = []

        for element_id in csv_element_ids:
            piece_ref = self._find_and_extract_piece(
                element_id,
                all_ocr_results,
                ocr_config.get('fuzzy_match_threshold', 0.8)
            )

            if piece_ref:
                library.add_piece(piece_ref)
                found_count += 1
            else:
                missing_element_ids.append(element_id)
                logger.warning(f"  Could not find element ID: {element_id}")

        # Log results
        logger.info(f"\nReference library extraction complete:")
        logger.info(f"  Found: {found_count}/{len(csv_element_ids)} pieces ({found_count/len(csv_element_ids)*100:.1f}%)")
        logger.info(f"  Missing: {len(missing_element_ids)} pieces")

        if missing_element_ids:
            logger.info(f"  Missing element IDs: {', '.join(missing_element_ids[:10])}")
            if len(missing_element_ids) > 10:
                logger.info(f"    ... and {len(missing_element_ids) - 10} more")

        return library

    def _find_and_extract_piece(self,
                                element_id: str,
                                ocr_results: List[Dict],
                                fuzzy_threshold: float) -> Optional[PieceReference]:
        """
        Find element ID in OCR results and extract piece image.

        Args:
            element_id: Element ID from CSV to find
            ocr_results: All OCR results from reference pages
            fuzzy_threshold: Minimum similarity for fuzzy matching

        Returns:
            PieceReference if found, None otherwise
        """
        # Find this element ID in OCR results
        match = self.ocr.find_element_id_in_results(element_id, ocr_results, fuzzy_threshold)

        if not match:
            logger.debug(f"  Element ID {element_id}: Not found in OCR results")
            return None

        # Extract piece image near the matched text
        page_image = match['page_image']
        text_bbox = match['bbox']

        search_radius = self.config.get('piece_search_radius', 150)
        piece_result = self.ocr.get_piece_image_near_text(page_image, text_bbox, search_radius)

        if not piece_result:
            logger.debug(f"  Element ID {element_id}: Found text but couldn't extract piece image")
            return None

        piece_image, piece_bbox = piece_result

        # Get piece info from CSV
        piece_info = self.inventory.get_piece_info(element_id)
        piece_name = piece_info['name'] if piece_info else "Unknown"

        # Create template
        template = img_proc.create_template(piece_image, self.config)

        # Create PieceReference
        piece_ref = PieceReference(
            element_id=element_id,
            name=piece_name,
            template_grayscale=template,
            original_image=piece_image,
            bounding_box=piece_bbox,
            ocr_confidence=match['confidence'],
            validated=True  # Always true in CSV-first approach!
        )

        logger.debug(f"  ✓ {element_id}: {piece_name} (OCR conf: {match['confidence']:.2f})")

        return piece_ref

    def save_debug_visualization(self, page_image: np.ndarray,
                                 ocr_results: List[Dict],
                                 output_path: str) -> None:
        """
        Save visualization of OCR results for debugging.

        Args:
            page_image: Reference page image
            ocr_results: OCR results with bounding boxes
            output_path: Path to save image
        """
        vis_image = page_image.copy()

        for result in ocr_results:
            if result['is_likely_element_id']:
                x, y, w, h = result['bbox']
                # Draw green box for likely element IDs
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add text
                cv2.putText(vis_image, result['text'], (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Draw gray box for other text
                x, y, w, h = result['bbox']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (128, 128, 128), 1)

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
