"""
PDF processor for extracting pages from instruction manuals

Uses PyMuPDF (fitz) for PDF rendering
"""

import fitz  # PyMuPDF
import numpy as np
import logging
from typing import List, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF processor for extracting pages as images.

    Handles PDF loading and page-by-page image extraction.
    """

    def __init__(self, pdf_path: str, config: dict):
        """
        Initialize PDF processor.

        Args:
            pdf_path: Path to PDF file
            config: PDF configuration dictionary
        """
        self.pdf_path = pdf_path
        self.config = config
        self.dpi = config.get('dpi', 300)
        self.color_mode = config.get('color_mode', 'RGB')

        # Open PDF
        self.doc = None
        self._open_pdf()

    def _open_pdf(self) -> None:
        """Open the PDF file."""
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"Opened PDF: {self.pdf_path} ({self.doc.page_count} pages)")
        except Exception as e:
            logger.error(f"Error opening PDF: {e}")
            raise

    def get_page_count(self) -> int:
        """
        Get total number of pages in PDF.

        Returns:
            Number of pages
        """
        if self.doc is None:
            return 0
        return self.doc.page_count

    def get_page_image(self, page_num: int) -> Optional[np.ndarray]:
        """
        Extract a single page as an image.

        Args:
            page_num: Page number (1-indexed)

        Returns:
            Image as numpy array (RGB), or None if error
        """
        if self.doc is None:
            logger.error("PDF not opened")
            return None

        # Convert to 0-indexed
        page_index = page_num - 1

        if page_index < 0 or page_index >= self.doc.page_count:
            logger.error(f"Invalid page number: {page_num} (total pages: {self.doc.page_count})")
            return None

        try:
            # Get page
            page = self.doc[page_index]

            # Calculate zoom factor for desired DPI
            # PyMuPDF default is 72 DPI
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to numpy array
            img_array = np.array(img)

            logger.debug(f"Extracted page {page_num}: {img_array.shape}")

            return img_array

        except Exception as e:
            logger.error(f"Error extracting page {page_num}: {e}")
            return None

    def get_pages(self, page_numbers: List[int]) -> List[np.ndarray]:
        """
        Extract multiple pages as images.

        Args:
            page_numbers: List of page numbers (1-indexed)

        Returns:
            List of images as numpy arrays
        """
        images = []

        for page_num in page_numbers:
            img = self.get_page_image(page_num)
            if img is not None:
                images.append(img)
            else:
                logger.warning(f"Failed to extract page {page_num}")

        logger.info(f"Extracted {len(images)}/{len(page_numbers)} pages")

        return images

    def get_page_range(self, start: int, end: int) -> List[np.ndarray]:
        """
        Extract a range of pages.

        Args:
            start: First page number (1-indexed, inclusive)
            end: Last page number (1-indexed, inclusive)

        Returns:
            List of images
        """
        page_numbers = list(range(start, end + 1))
        return self.get_pages(page_numbers)

    def get_last_n_pages(self, n: int = 3) -> List[np.ndarray]:
        """
        Get the last N pages (typically reference inventory).

        Args:
            n: Number of pages to extract

        Returns:
            List of images
        """
        if self.doc is None:
            return []

        total_pages = self.doc.page_count
        start_page = max(1, total_pages - n + 1)

        return self.get_page_range(start_page, total_pages)

    def close(self) -> None:
        """Close the PDF document."""
        if self.doc is not None:
            self.doc.close()
            logger.info(f"Closed PDF: {self.pdf_path}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()
