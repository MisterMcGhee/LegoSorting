"""
Image processing utilities

Common image manipulation functions for piece detection and template creation
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: Input image (BGR or RGB)

    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.

    Args:
        image: Input image
        kernel_size: Kernel size (must be odd)

    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_binary_threshold(image: np.ndarray, threshold: int = 200) -> np.ndarray:
    """
    Apply binary threshold to separate foreground from background.

    Args:
        image: Input grayscale image
        threshold: Threshold value (0-255)

    Returns:
        Binary image
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.

    Args:
        image: Input grayscale image

    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def remove_background(image: np.ndarray, threshold: int = 200) -> np.ndarray:
    """
    Remove white background from image.

    Args:
        image: Input BGR image
        threshold: Threshold for background detection

    Returns:
        Image with white background removed (made transparent)
    """
    # Convert to grayscale
    gray = convert_to_grayscale(image)

    # Create mask (background is white/bright)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Apply mask to original image
    result = image.copy()
    result[mask == 0] = [255, 255, 255]  # Set background to white

    return result


def crop_to_content(image: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Crop image to content with padding.

    Args:
        image: Input image
        padding: Padding around content (pixels)

    Returns:
        Cropped image
    """
    # Convert to grayscale
    gray = convert_to_grayscale(image)

    # Find non-white pixels
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Get bounding box of all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Add padding
    h, w = image.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    return image[y_min:y_max, x_min:x_max]


def resize_with_aspect_ratio(image: np.ndarray, target_height: int) -> np.ndarray:
    """
    Resize image maintaining aspect ratio.

    Args:
        image: Input image
        target_height: Desired height in pixels

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    if h == 0:
        return image

    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)

    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def create_template(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Create a normalized template from piece image for matching.

    Args:
        image: Input piece image
        config: Configuration dictionary with template settings

    Returns:
        Normalized grayscale template
    """
    # Convert to grayscale
    gray = convert_to_grayscale(image)

    # Crop to content
    padding = config.get('template_padding', 10)
    cropped = crop_to_content(image, padding)

    # Convert cropped to grayscale if needed
    if len(cropped.shape) == 3:
        cropped = convert_to_grayscale(cropped)

    # Normalize size if configured
    if config.get('normalize_size', True):
        target_height = config.get('target_template_height', 200)
        cropped = resize_with_aspect_ratio(cropped, target_height)

    # Enhance contrast
    enhanced = enhance_contrast(cropped)

    return enhanced


def find_contours(image: np.ndarray, min_area: int = 500, max_area: int = 50000) -> List:
    """
    Find contours in image with size filtering.

    Args:
        image: Input binary image
        min_area: Minimum contour area
        max_area: Maximum contour area

    Returns:
        List of filtered contours
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered.append(contour)

    return filtered


def get_bounding_boxes(contours: List) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes from contours.

    Args:
        contours: List of contours

    Returns:
        List of (x, y, w, h) tuples
    """
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    return boxes


def draw_bounding_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image for visualization.

    Args:
        image: Input image
        boxes: List of (x, y, w, h) tuples
        color: Box color (BGR)
        thickness: Line thickness

    Returns:
        Image with boxes drawn
    """
    result = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def sort_boxes_into_columns(boxes: List[Tuple[int, int, int, int]],
                            num_columns: int = 3,
                            tolerance: float = 0.15) -> List[List[Tuple[int, int, int, int]]]:
    """
    Sort bounding boxes into vertical columns.

    Args:
        boxes: List of (x, y, w, h) tuples
        num_columns: Expected number of columns
        tolerance: Tolerance for column alignment (as fraction of image width)

    Returns:
        List of columns, each containing sorted boxes
    """
    if not boxes:
        return []

    # Sort by x-coordinate first
    sorted_boxes = sorted(boxes, key=lambda b: b[0])

    # Group into columns by x-coordinate
    columns = []
    current_column = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        x, y, w, h = box
        prev_x = current_column[-1][0]

        # Check if this box is in the same column (similar x-coordinate)
        if abs(x - prev_x) < (w * tolerance * 10):  # Similar x position
            current_column.append(box)
        else:
            # New column
            columns.append(current_column)
            current_column = [box]

    # Add last column
    if current_column:
        columns.append(current_column)

    # Sort each column by y-coordinate (top to bottom)
    for column in columns:
        column.sort(key=lambda b: b[1])

    return columns
