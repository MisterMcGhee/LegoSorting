"""
OCR engine wrapper

Provides OCR functionality for extracting element IDs from piece images
"""

import re
import logging
import numpy as np
from typing import Optional, Tuple
from .image_processing import enhance_contrast, convert_to_grayscale

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR engine wrapper for extracting text from images.

    Uses EasyOCR for text recognition.
    """

    def __init__(self, config: dict):
        """
        Initialize OCR engine.

        Args:
            config: OCR configuration dictionary
        """
        self.config = config
        self.engine = config.get('engine', 'easyocr')
        self.languages = config.get('languages', ['en'])
        self.gpu = config.get('gpu', False)
        self.confidence_threshold = config.get('confidence_threshold', 0.3)

        # Initialize EasyOCR reader (lazy initialization)
        self.reader = None
        self._initialize_reader()

    def _initialize_reader(self) -> None:
        """Initialize the OCR reader (lazy loading)."""
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR with languages: {self.languages}")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR initialized successfully")
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {e}")
            raise

    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text from image using OCR.

        Args:
            image: Input image (grayscale or BGR)
            preprocess: Whether to preprocess image before OCR

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if self.reader is None:
            logger.error("OCR reader not initialized")
            return "", 0.0

        try:
            # Preprocess image if requested
            if preprocess:
                image = self._preprocess_for_ocr(image)

            # Run OCR
            results = self.reader.readtext(image)

            if not results:
                return "", 0.0

            # Get best result (highest confidence)
            best_result = max(results, key=lambda x: x[2])
            text = best_result[1]
            confidence = best_result[2]

            # Clean the extracted text
            cleaned_text = self._clean_element_id(text)

            logger.debug(f"OCR result: '{text}' -> '{cleaned_text}' (confidence: {confidence:.2f})")

            return cleaned_text, confidence

        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return "", 0.0

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = convert_to_grayscale(image)

        # Enhance contrast if configured
        if self.config.get('contrast_enhancement', True):
            gray = enhance_contrast(gray)

        # Additional denoising if configured
        if self.config.get('denoise', True):
            import cv2
            gray = cv2.fastNlMeansDenoising(gray, h=10)

        return gray

    def _clean_element_id(self, text: str) -> str:
        """
        Clean extracted text to get valid element ID.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned element ID
        """
        # Remove whitespace
        text = text.strip()

        # Remove common OCR mistakes
        text = text.replace(' ', '')
        text = text.replace('O', '0')  # Letter O -> digit 0
        text = text.replace('o', '0')
        text = text.replace('I', '1')  # Letter I -> digit 1
        text = text.replace('l', '1')  # Letter l -> digit 1

        # Keep only allowed characters
        allowed = self.config.get('allowed_characters',
                                  '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        text = ''.join(c for c in text if c in allowed)

        return text

    def validate_element_id(self, element_id: str) -> bool:
        """
        Validate that extracted text looks like a valid element ID.

        Args:
            element_id: Extracted element ID

        Returns:
            True if element ID appears valid
        """
        min_length = self.config.get('min_element_id_length', 4)
        max_length = self.config.get('max_element_id_length', 10)

        # Check length
        if not (min_length <= len(element_id) <= max_length):
            return False

        # Must contain at least some digits
        if not any(c.isdigit() for c in element_id):
            return False

        return True

    def extract_element_id_from_region(self, image: np.ndarray,
                                       region: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Extract element ID from specific region of image.

        Args:
            image: Full image
            region: (x, y, w, h) region to extract from

        Returns:
            Tuple of (element_id, confidence)
        """
        x, y, w, h = region

        # Crop to region
        cropped = image[y:y+h, x:x+w]

        # Extract text
        element_id, confidence = self.extract_text(cropped)

        # Validate
        if not self.validate_element_id(element_id):
            logger.debug(f"Invalid element ID format: '{element_id}'")
            return "", 0.0

        return element_id, confidence

    def find_element_id_near_piece(self, image: np.ndarray,
                                   piece_bbox: Tuple[int, int, int, int],
                                   search_margin: int = 50) -> Tuple[str, float]:
        """
        Find element ID text near a piece bounding box.

        Searches below and to the right of the piece for element ID text.

        Args:
            image: Full reference page image
            piece_bbox: (x, y, w, h) of piece
            search_margin: How far to search (pixels)

        Returns:
            Tuple of (element_id, confidence)
        """
        x, y, w, h = piece_bbox
        img_h, img_w = image.shape[:2]

        # Define search regions (below and to the right of piece)
        search_regions = []

        # Region below piece
        below_y = y + h
        below_h = min(search_margin, img_h - below_y)
        if below_h > 10:  # Minimum height
            search_regions.append((x, below_y, w, below_h))

        # Region to the right
        right_x = x + w
        right_w = min(search_margin, img_w - right_x)
        if right_w > 10:  # Minimum width
            search_regions.append((right_x, y, right_w, h))

        # Try each search region
        best_element_id = ""
        best_confidence = 0.0

        for region in search_regions:
            element_id, confidence = self.extract_element_id_from_region(image, region)

            if confidence > best_confidence:
                best_element_id = element_id
                best_confidence = confidence

        return best_element_id, best_confidence
