import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any, Dict


@dataclass
class TrackedPiece:
    """Represents a piece being tracked through the ROI"""
    id: int  # Unique identifier for this piece
    contour: np.ndarray  # Current contour
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    entry_time: float  # When the piece entered tracking
    captured: bool = False  # Whether this piece has been photographed
    image_number: Optional[int] = None  # The number used when saving the image
    updated: bool = True  # Whether this piece was updated in the current frame


class TrackedLegoDetector:
    """Detects and tracks Lego pieces moving through a region of interest"""

    def __init__(self, config_manager=None):
        """Initialize the detector with configuration parameters

        Args:
            config_manager: Optional configuration manager
        """
        # Set default values
        self.min_piece_area = 1000
        self.max_piece_area = 100000
        self.buffer_percent = 0.01
        self.crop_padding = 20
        self.history_length = 2
        self.grid_size = (10, 10)
        self.color_margin = 40
        self.show_grid = False

        # Initialize from config if provided
        if config_manager:
            self.min_piece_area = config_manager.get("detector", "min_piece_area", self.min_piece_area)
            self.max_piece_area = config_manager.get("detector", "max_piece_area", self.max_piece_area)
            self.buffer_percent = config_manager.get("detector", "buffer_percent", self.buffer_percent)
            self.crop_padding = config_manager.get("detector", "crop_padding", self.crop_padding)
            self.history_length = config_manager.get("detector", "history_length", self.history_length)
            grid_size = config_manager.get("detector", "grid_size", self.grid_size)
            if isinstance(grid_size, list):
                self.grid_size = tuple(grid_size)
            self.color_margin = config_manager.get("detector", "color_margin", self.color_margin)
            self.show_grid = config_manager.get("detector", "show_grid", self.show_grid)

        # Initialize background model
        self.background_model = None

        # Initialize tracking-related variables
        self.last_detection_time = time.time()
        self.tracked_pieces: List[TrackedPiece] = []
        self.next_piece_id = 1
        self.roi = None
        self.entry_zone = None
        self.exit_zone = None
        self.valid_zone = None

    def calibrate(self, frame):
        """Get user-selected ROI and calculate buffer zones

        Args:
            frame: The video frame to use for calibration

        Returns:
            tuple: The selected ROI as (x, y, w, h)
        """
        # Get ROI selection from user
        roi = cv2.selectROI("Select Belt Region", frame, False)
        cv2.destroyWindow("Select Belt Region")

        self.roi = roi
        roi_width = roi[2]

        # Calculate buffer zones
        buffer_width = int(roi_width * self.buffer_percent)
        self.entry_zone = (roi[0], roi[0] + buffer_width)
        self.exit_zone = (roi[0] + roi[2] - buffer_width, roi[0] + roi[2])
        self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

        print(f"ROI: {roi}")
        print(f"Entry buffer: {self.entry_zone}")
        print(f"Valid zone: {self.valid_zone}")
        print(f"Exit buffer: {self.exit_zone}")

        # Initialize background model
        x, y, w, h = self.roi
        roi_frame = frame[y:y + h, x:x + w]
        self._initialize_background_model(roi_frame)

        return roi

    def _initialize_background_model(self, roi_frame):
        """Initialize the background model by computing average colors for each grid cell

        Args:
            roi_frame: The ROI portion of the frame
        """
        h, w = roi_frame.shape[:2]
        grid_h, grid_w = self.grid_size

        # Create storage for grid cell averages
        self.background_model = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

        # Calculate average color for each grid cell
        for i in range(grid_h):
            for j in range(grid_w):
                cell_y1 = i * (h // grid_h)
                cell_y2 = (i + 1) * (h // grid_h) if i < grid_h - 1 else h
                cell_x1 = j * (w // grid_w)
                cell_x2 = (j + 1) * (w // grid_w) if j < grid_w - 1 else w

                cell = roi_frame[cell_y1:cell_y2, cell_x1:cell_x2]
                self.background_model[i, j] = np.mean(cell, axis=(0, 1))

        print(f"Background model initialized with {grid_h}x{grid_w} grid")

    def _preprocess_frame(self, frame):
        """Extract and preprocess the ROI using color-based detection with efficient NumPy operations"""
        if self.roi is None:
            raise ValueError("Detector not calibrated. Call calibrate() first.")

        # Extract ROI
        x, y, w, h = self.roi
        roi = frame[y:y + h, x:x + w]

        # Create grid
        grid_h, grid_w = self.grid_size

        # Initialize the mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Process each grid cell with vectorized operations
        for i in range(grid_h):
            for j in range(grid_w):
                # Get cell coordinates
                cell_y1 = i * (h // grid_h)
                cell_y2 = (i + 1) * (h // grid_h) if i < grid_h - 1 else h
                cell_x1 = j * (w // grid_w)
                cell_x2 = (j + 1) * (w // grid_w) if j < grid_w - 1 else w

                # Extract cell
                cell = roi[cell_y1:cell_y2, cell_x1:cell_x2]

                # Get background model for this cell
                avg_color = self.background_model[i, j]

                # Calculate color difference for all pixels in the cell at once
                diff = np.abs(cell - avg_color)

                # Find pixels where any channel differs by more than color_margin
                color_mask = np.max(diff, axis=2) > self.color_margin

                # Copy to the main mask
                mask[cell_y1:cell_y2, cell_x1:cell_x2] = color_mask.astype(np.uint8) * 255

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _find_new_contours(self, mask):
        """Find contours in the current frame

        Args:
            mask: The binary mask to find contours in

        Returns:
            list: List of (contour, bounding_box) tuples
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_piece_area or area > self.max_piece_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            if self.roi[0] <= x < self.roi[0] + self.roi[2]:  # Entire ROI
                valid_contours.append((contour, (x, y, w, h)))

        return valid_contours

    def _match_and_update_tracks(self, new_contours):
        """Match new contours to existing tracks or create new tracks"""

        # First, mark all existing tracks as not updated
        for piece in self.tracked_pieces:
            piece.updated = False

        # Try to match new contours to existing tracks
        for contour, bbox in new_contours:
            x, y, w, h = bbox

            # Skip pieces that haven't entered the ROI yet
            if x < self.entry_zone[0]:
                continue

            # Check if this contour matches any existing piece
            matched = False
            for piece in self.tracked_pieces:
                old_x, old_y, old_w, old_h = piece.bounding_box

                # Calculate center points
                new_center_x = x + w / 2
                new_center_y = y + h / 2
                old_center_x = old_x + old_w / 2
                old_center_y = old_y + old_h / 2

                # Calculate distance between centers
                distance = ((new_center_x - old_center_x) ** 2 +
                            (new_center_y - old_center_y) ** 2) ** 0.5

                # If centers are close enough, update the existing track
                if distance < max(w, h) * 0.5:  # Adjust threshold as needed
                    piece.contour = contour
                    piece.bounding_box = bbox
                    piece.updated = True
                    matched = True
                    break

            # If no match found, create a new track
            if not matched:
                new_piece = TrackedPiece(
                    id=self.next_piece_id,
                    contour=contour,
                    bounding_box=bbox,
                    entry_time=time.time()
                )
                self.tracked_pieces.append(new_piece)
                self.next_piece_id += 1

        # Remove pieces that have left the ROI or weren't updated
        self.tracked_pieces = [p for p in self.tracked_pieces if
                               (p.updated or p.bounding_box[0] < self.exit_zone[0])]

    def _should_capture(self, piece: TrackedPiece) -> bool:
        """Determine if the piece should be captured."""
        if piece.captured:
            return False  # Already captured

        # Don't capture if it's too soon after the last capture
        current_time = time.time()
        if hasattr(self, 'last_capture_time') and current_time - self.last_capture_time < 1.0:
            return False

        x, y, w, h = piece.bounding_box
        piece_left = x
        piece_right = x + w

        # Capture if the piece is entirely within the valid zone
        if (self.valid_zone[0] <= piece_left and piece_right <= self.valid_zone[1]):
            self.last_capture_time = current_time
            return True
        return False

    def _crop_piece_image(self, frame, piece: TrackedPiece) -> np.ndarray:
        """Crop the frame to just the piece with padding

        Args:
            frame: The video frame to crop
            piece: The TrackedPiece to crop around

        Returns:
            np.ndarray: The cropped image
        """
        x, y, w, h = piece.bounding_box
        roi_x, roi_y = self.roi[0], self.roi[1]

        # Add padding and ensure within frame bounds
        pad = self.crop_padding
        x1 = max(0, x + roi_x - pad)
        y1 = max(0, y + roi_y - pad)
        x2 = min(frame.shape[1], x + roi_x + w + pad)
        y2 = min(frame.shape[0], y + roi_y + h + pad)

        return frame[y1:y2, x1:x2]

    def process_frame(self, frame, current_count=None) -> Tuple[List[TrackedPiece], Optional[np.ndarray]]:
        """Process a frame and return any pieces to capture.

        Args:
            frame: The video frame to process
            current_count: The current image count (optional)

        Returns:
            tuple: (tracked_pieces, cropped_image)
                - tracked_pieces: List of all currently tracked pieces
                - cropped_image: Cropped image of a piece to capture, or None
        """
        if self.roi is None:
            raise ValueError("Detector not calibrated. Call calibrate() first.")

        mask = self._preprocess_frame(frame)
        new_contours = self._find_new_contours(mask)

        # Update tracking
        self._match_and_update_tracks(new_contours)

        # Check for pieces to capture
        for piece in self.tracked_pieces:
            if self._should_capture(piece) and not piece.captured:
                cropped_image = self._crop_piece_image(frame, piece)
                piece.captured = True
                piece.image_number = current_count
                return self.tracked_pieces, cropped_image

        return self.tracked_pieces, None

    def draw_debug(self, frame):
        """Draw debug visualization with piece tracking, capture status, and grid overlay.

        Args:
            frame: The video frame to draw on

        Returns:
            np.ndarray: The debug frame with visualization
        """
        if self.roi is None:
            return frame

        debug_frame = frame.copy()

        # Draw ROI - Green boundary
        x, y, w, h = self.roi
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw entry zone - Blue
        cv2.line(debug_frame,
                 (self.entry_zone[1], y),
                 (self.entry_zone[1], y + h),
                 (255, 0, 0), 2)

        # Draw exit zone - Red
        cv2.line(debug_frame,
                 (self.exit_zone[0], y),
                 (self.exit_zone[0], y + h),
                 (0, 0, 255), 2)

        # Draw grid if enabled
        if self.show_grid:
            grid_h, grid_w = self.grid_size
            cell_h, cell_w = h // grid_h, w // grid_w

            # Draw horizontal grid lines
            for i in range(1, grid_h):
                y_pos = y + i * cell_h
                cv2.line(debug_frame, (x, y_pos), (x + w, y_pos), (200, 200, 200), 1)

            # Draw vertical grid lines
            for j in range(1, grid_w):
                x_pos = x + j * cell_w
                cv2.line(debug_frame, (x_pos, y), (x_pos, y + h), (200, 200, 200), 1)

        for piece in self.tracked_pieces:
            x, y, w, h = piece.bounding_box
            roi_x, roi_y = self.roi[0], self.roi[1]

            # Adjust coordinates to the full frame
            x += roi_x
            y += roi_y

            # Determine box color
            box_color = (0, 255, 0) if piece.captured else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), box_color, 2)

            # Add label
            label_text = f"#{piece.image_number}" if piece.captured else f"ID:{piece.id}"
            label_color = (0, 255, 0) if piece.captured else (0, 0, 255)
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(debug_frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 0, 0), -1)
            cv2.putText(debug_frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        return debug_frame


# Factory function to create a detector
def create_detector(detector_type="tracked", config_manager=None):
    """Create a detector of the specified type

    Args:
        detector_type: Type of detector ("tracked" or future types)
        config_manager: Configuration manager

    Returns:
        Detector instance
    """
    if detector_type == "tracked":
        return TrackedLegoDetector(config_manager)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}. Currently only 'tracked' is supported.")


# Example usage
if __name__ == "__main__":
    # Simple test to demonstrate the module
    import sys

    # Create a detector
    detector = create_detector("tracked")

    # Initialize a camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Get a frame for calibration
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read from camera")
        sys.exit(1)

    # Calibrate the detector
    detector.calibrate(frame)

    try:
        while True:
            # Get a frame
            ret, frame = camera.read()
            if not ret:
                break

            # Process the frame
            tracked_pieces, cropped_image = detector.process_frame(frame)

            # Draw debug visualization
            debug_frame = detector.draw_debug(frame)

            # Display the frame
            cv2.imshow("Lego Detector", debug_frame)

            # If we have a cropped image, display it
            if cropped_image is not None:
                cv2.imshow("Cropped Piece", cropped_image)

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # Clean up
        camera.release()
        cv2.destroyAllWindows()
