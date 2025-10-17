"""
camera_view.py - Camera view widgets for displaying live camera feeds

FILE LOCATION: GUI/widgets/camera_view.py

This module provides camera view widgets that can display live camera feeds with
various overlays for configuration and operation monitoring. All widgets integrate
with the camera module's consumer pattern for thread-safe frame delivery.

Widget Types:
- CameraViewRaw: Plain camera feed without overlays
- CameraViewROI: Camera feed with ROI rectangle and entry/exit zones
- CameraViewTracking: Camera feed with tracked piece bounding boxes
- CameraViewUnited: Camera feed with both ROI and tracking overlays

Usage:
    # For configuration GUI (camera tab)
    view = CameraViewRaw()
    camera.register_consumer("config_preview", view.receive_frame, "async", 20)

    # For detector tab with ROI configuration
    view = CameraViewROI()
    view.set_roi(100, 50, 1720, 980)
    view.set_zones(0.15, 0.15)

    # For sorting GUI with piece tracking
    view = CameraViewTracking()
    view.set_tracked_pieces(active_pieces)

    # For complete monitoring view
    view = CameraViewUnited()
    view.set_roi(100, 50, 1720, 980)
    view.set_tracked_pieces(active_pieces)
    view.set_fps_visible(True)
"""

from abc import ABC, abstractmethod, ABCMeta
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QImage, QPixmap, QPen, QColor, QFont
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import time
import logging
from typing import Optional, List, Dict, Any, Tuple

# Initialize module logger
logger = logging.getLogger(__name__)


# ============================================================================
# METACLASS RESOLUTION
# ============================================================================

class CombinedMeta(type(QWidget), ABCMeta):
    """
    Combined metaclass to resolve conflict between QWidget and ABC.

    QWidget uses Qt's metaclass (accessed via type(QWidget)),
    and ABC uses ABCMeta. This combined metaclass allows us to inherit from both.

    This approach is compatible across different PyQt5 versions.
    """
    pass


# ============================================================================
# DEFAULT STYLING CONSTANTS
# ============================================================================

class ViewStyles:
    """Default styling constants for camera view widgets."""

    # ROI and Zone Colors
    COLOR_ROI = QColor(255, 255, 0)          # Yellow - ROI border
    COLOR_ENTRY_ZONE = QColor(0, 255, 0, 80) # Green with transparency
    COLOR_EXIT_ZONE = QColor(255, 0, 0, 80)  # Red with transparency

    # Piece Tracking Colors (by processing stage)
    COLOR_DETECTED = QColor(255, 255, 255)   # White - newly detected
    COLOR_PROCESSING = QColor(0, 255, 255)   # Cyan - being processed
    COLOR_CAPTURED = QColor(0, 255, 0)       # Green - successfully captured
    COLOR_ERROR = QColor(255, 0, 0)          # Red - error/overflow

    # UI Element Colors
    COLOR_TEXT = QColor(255, 255, 255)       # White text
    COLOR_BACKGROUND = QColor(28, 28, 28)    # Dark background
    COLOR_FPS_BACKGROUND = QColor(0, 0, 0, 180)  # Semi-transparent black

    # Line Widths
    LINE_WIDTH_ROI = 2
    LINE_WIDTH_ZONE = 1
    LINE_WIDTH_PIECE = 2

    # Font Settings
    FONT_FAMILY = "Arial"
    FONT_SIZE_LABEL = 10
    FONT_SIZE_FPS = 12
    FONT_SIZE_ID = 8


# ============================================================================
# BASE CAMERA VIEW WIDGET
# ============================================================================

class BaseCameraViewWidget(QWidget, metaclass=CombinedMeta):
    """
    Abstract base class for all camera view widgets.

    Provides common functionality for:
    - Frame reception and display
    - Thread-safe frame updates via signals
    - Aspect ratio preservation
    - FPS tracking and display
    - Camera consumer integration

    Signals:
        frame_received(np.ndarray): Emitted when a new frame is received

    Subclasses must implement:
        - process_frame(): Apply specific overlays to the frame
    """

    # Signal emitted when frame is received from camera
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        """
        Initialize the base camera view widget.

        Args:
            parent: Parent widget (typically None or a tab)
        """
        super().__init__(parent)

        # Frame storage
        self.current_frame = None
        self.display_frame = None

        # FPS tracking
        self.fps_visible = False
        self.fps_counter = 0
        self.fps_value = 0.0
        self.last_fps_time = time.time()

        # Widget configuration
        self.setMinimumSize(640, 480)
        self.setStyleSheet(f"background-color: rgb({ViewStyles.COLOR_BACKGROUND.red()}, "
                          f"{ViewStyles.COLOR_BACKGROUND.green()}, "
                          f"{ViewStyles.COLOR_BACKGROUND.blue()});")

        # Connect signal to update slot
        self.frame_received.connect(self._on_frame_received)

        logger.debug(f"{self.__class__.__name__} initialized")

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    def receive_frame(self, frame: np.ndarray):
        """
        Camera consumer callback - receives frames from camera module.

        This method is called by the camera module's frame distributor.
        It emits a signal to ensure thread-safe GUI updates.

        Args:
            frame: Camera frame in BGR format (OpenCV standard)
        """
        try:
            # Emit signal for thread-safe processing
            self.frame_received.emit(frame.copy())
        except Exception as e:
            logger.error(f"Error receiving frame in {self.__class__.__name__}: {e}")

    def set_fps_visible(self, visible: bool):
        """
        Toggle FPS counter visibility.

        Args:
            visible: True to show FPS counter, False to hide
        """
        self.fps_visible = visible
        self.update()
        logger.debug(f"FPS visibility set to {visible}")

    def clear_display(self):
        """Clear the current display and show no signal placeholder."""
        self.current_frame = None
        self.display_frame = None
        self.update()

    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame and apply overlays specific to this widget type.

        Args:
            frame: Input frame in BGR format

        Returns:
            Processed frame with overlays applied
        """
        pass

    # ========================================================================
    # PRIVATE METHODS - Frame Processing
    # ========================================================================

    def _on_frame_received(self, frame: np.ndarray):
        """
        Slot for frame_received signal - processes and displays frame.

        This runs in the main GUI thread, ensuring thread-safe updates.

        Args:
            frame: Camera frame in BGR format
        """
        try:
            # Update FPS counter
            self._update_fps()

            # Store original frame
            self.current_frame = frame

            # Process frame with subclass-specific overlays
            self.display_frame = self.process_frame(frame.copy())

            # Trigger repaint
            self.update()

        except Exception as e:
            logger.error(f"Error processing frame in {self.__class__.__name__}: {e}")

    def _update_fps(self):
        """Update FPS counter based on frame timing."""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            self.fps_value = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_fps_time = current_time

    # ========================================================================
    # PAINTING METHODS
    # ========================================================================

    def paintEvent(self, event):
        """
        Qt paint event - draws the camera frame and overlays.

        Args:
            event: Qt paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.display_frame is None:
            self._draw_no_signal(painter)
            return

        try:
            # Draw the camera frame
            self._draw_frame(painter, self.display_frame)

            # Draw FPS counter if enabled
            if self.fps_visible:
                self._draw_fps(painter)

        except Exception as e:
            logger.error(f"Error in paintEvent: {e}")
            self._draw_no_signal(painter)

    def _draw_frame(self, painter: QPainter, frame: np.ndarray):
        """
        Draw camera frame scaled to widget size.

        Args:
            painter: Qt painter object
            frame: Frame to draw in BGR format
        """
        # Convert BGR to RGB
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create QImage
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height,
                        bytes_per_line, QImage.Format_RGB888)

        # Convert to pixmap and scale
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)

        # Center the image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2

        painter.drawPixmap(x, y, scaled_pixmap)

    def _draw_no_signal(self, painter: QPainter):
        """
        Draw placeholder when no camera feed is available.

        Args:
            painter: Qt painter object
        """
        painter.fillRect(self.rect(), ViewStyles.COLOR_BACKGROUND)
        painter.setPen(ViewStyles.COLOR_TEXT)

        font = QFont(ViewStyles.FONT_FAMILY, 16)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter,
                        "No Camera Signal\nWaiting for frames...")

    def _draw_fps(self, painter: QPainter):
        """
        Draw FPS counter in top-right corner.

        Args:
            painter: Qt painter object
        """
        # Background rectangle
        fps_text = f"FPS: {self.fps_value:.1f}"
        font = QFont(ViewStyles.FONT_FAMILY, ViewStyles.FONT_SIZE_FPS, QFont.Bold)
        painter.setFont(font)

        # Calculate text size
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(fps_text)
        text_height = metrics.height()

        # Draw background
        padding = 8
        bg_rect_x = self.width() - text_width - padding * 2 - 10
        bg_rect_y = 10
        bg_rect_width = text_width + padding * 2
        bg_rect_height = text_height + padding

        painter.fillRect(bg_rect_x, bg_rect_y, bg_rect_width, bg_rect_height,
                        ViewStyles.COLOR_FPS_BACKGROUND)

        # Draw text
        painter.setPen(ViewStyles.COLOR_TEXT)
        text_x = bg_rect_x + padding
        text_y = bg_rect_y + padding + metrics.ascent()
        painter.drawText(text_x, text_y, fps_text)


# ============================================================================
# RAW CAMERA VIEW WIDGET
# ============================================================================

class CameraViewRaw(BaseCameraViewWidget):
    """
    Raw camera view widget - displays camera feed without any overlays.

    Use this widget when you need a simple camera display without any
    additional visual elements. Ideal for basic camera testing and the
    camera configuration tab.

    Example:
        view = CameraViewRaw()
        view.set_fps_visible(True)
        camera.register_consumer("preview", view.receive_frame, "async", 20)
    """

    def __init__(self, parent=None):
        """
        Initialize raw camera view widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        logger.debug("CameraViewRaw initialized")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Return frame without any modifications.

        Args:
            frame: Input frame

        Returns:
            Unmodified frame
        """
        return frame


# ============================================================================
# ROI CAMERA VIEW WIDGET
# ============================================================================

class CameraViewROI(BaseCameraViewWidget):
    """
    ROI camera view widget - displays camera feed with ROI rectangle and zones.

    Overlays:
    - Yellow ROI rectangle border
    - Green semi-transparent entry zone
    - Red semi-transparent exit zone
    - Zone labels

    Use this widget for detector configuration where the user needs to
    visualize the detection region and entry/exit zones.

    Example:
        view = CameraViewROI()
        view.set_roi(100, 50, 1720, 980)
        view.set_zones(0.15, 0.15)
    """

    def __init__(self, parent=None):
        """
        Initialize ROI camera view widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # ROI configuration
        self.roi = None  # (x, y, width, height)
        self.entry_zone_pct = 0.15
        self.exit_zone_pct = 0.15

        logger.debug("CameraViewROI initialized")

    # ========================================================================
    # PUBLIC CONFIGURATION METHODS
    # ========================================================================

    def set_roi(self, x: int, y: int, width: int, height: int):
        """
        Set the Region of Interest coordinates.

        Args:
            x: ROI top-left x coordinate
            y: ROI top-left y coordinate
            width: ROI width
            height: ROI height
        """
        self.roi = (x, y, width, height)
        logger.debug(f"ROI set to: ({x}, {y}, {width}, {height})")

    def set_zones(self, entry_pct: float, exit_pct: float):
        """
        Set entry and exit zone percentages.

        Args:
            entry_pct: Entry zone percentage (0.0 to 1.0)
            exit_pct: Exit zone percentage (0.0 to 1.0)
        """
        self.entry_zone_pct = entry_pct
        self.exit_zone_pct = exit_pct
        logger.debug(f"Zones set to: entry={entry_pct}, exit={exit_pct}")

    # ========================================================================
    # FRAME PROCESSING
    # ========================================================================

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply ROI and zone overlays to frame.

        Args:
            frame: Input frame

        Returns:
            Frame with ROI overlays
        """
        if self.roi is None:
            return frame

        x, y, w, h = self.roi

        # Draw ROI rectangle (yellow border)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                     (0, 255, 255), ViewStyles.LINE_WIDTH_ROI)  # Yellow in BGR

        # Draw zones
        self._draw_entry_zone(frame, x, y, w, h)
        self._draw_exit_zone(frame, x, y, w, h)

        return frame

    # ========================================================================
    # HELPER METHODS - Zone Drawing
    # ========================================================================

    def _draw_entry_zone(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """
        Draw entry zone overlay on frame.

        Args:
            frame: Frame to draw on
            x, y, w, h: ROI coordinates
        """
        entry_width = int(w * self.entry_zone_pct)

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + entry_width, y + h),
                     (0, 255, 0), -1)  # Green filled in BGR
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x, y), (x + entry_width, y + h),
                     (0, 255, 0), ViewStyles.LINE_WIDTH_ZONE)

        # Add label
        cv2.putText(frame, "ENTRY", (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _draw_exit_zone(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """
        Draw exit zone overlay on frame.

        Args:
            frame: Frame to draw on
            x, y, w, h: ROI coordinates
        """
        exit_width = int(w * self.exit_zone_pct)
        exit_x = x + w - exit_width

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (exit_x, y), (x + w, y + h),
                     (0, 0, 255), -1)  # Red filled in BGR
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw border
        cv2.rectangle(frame, (exit_x, y), (x + w, y + h),
                     (0, 0, 255), ViewStyles.LINE_WIDTH_ZONE)

        # Add label
        cv2.putText(frame, "EXIT", (exit_x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ============================================================================
# TRACKING CAMERA VIEW WIDGET
# ============================================================================

class CameraViewTracking(BaseCameraViewWidget):
    """
    Tracking camera view widget - displays camera feed with tracked piece overlays.

    Overlays:
    - Color-coded bounding boxes by processing stage
    - Piece IDs
    - Processing status indicators

    Color Coding:
    - White: Newly detected piece
    - Cyan: Piece being processed
    - Green: Successfully captured piece
    - Red: Error or overflow

    Use this widget for monitoring active piece tracking during sorting operations.

    Example:
        view = CameraViewTracking()
        view.set_tracked_pieces([piece1, piece2, piece3])
    """

    def __init__(self, parent=None):
        """
        Initialize tracking camera view widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Tracked pieces list
        self.tracked_pieces = []

        logger.debug("CameraViewTracking initialized")

    # ========================================================================
    # PUBLIC CONFIGURATION METHODS
    # ========================================================================

    def set_tracked_pieces(self, pieces: List[Any]):
        """
        Update the list of tracked pieces to display.

        Args:
            pieces: List of piece objects with bbox and status attributes
        """
        self.tracked_pieces = pieces if pieces else []

    # ========================================================================
    # FRAME PROCESSING
    # ========================================================================

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply piece tracking overlays to frame.

        Args:
            frame: Input frame

        Returns:
            Frame with tracking overlays
        """
        if not self.tracked_pieces:
            return frame

        for piece in self.tracked_pieces:
            self._draw_tracked_piece(frame, piece)

        return frame

    # ========================================================================
    # HELPER METHODS - Piece Drawing
    # ========================================================================

    def _draw_tracked_piece(self, frame: np.ndarray, piece: Any):
        """
        Draw overlay for a single tracked piece.

        Args:
            frame: Frame to draw on
            piece: Piece object to draw
        """
        # Get bounding box
        if not hasattr(piece, 'bbox') or piece.bbox is None:
            return

        x, y, w, h = piece.bbox

        # Determine color based on piece status
        color = self._get_piece_color(piece)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                     color, ViewStyles.LINE_WIDTH_PIECE)

        # Draw piece ID if available
        if hasattr(piece, 'id'):
            label = f"ID: {piece.id}"
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw additional status info
        self._draw_piece_status(frame, piece, x, y, w, h, color)

    def _get_piece_color(self, piece: Any) -> Tuple[int, int, int]:
        """
        Determine bounding box color based on piece processing status.

        Args:
            piece: Piece object

        Returns:
            BGR color tuple
        """
        # Check various status attributes
        if hasattr(piece, 'being_processed') and piece.being_processed:
            return (255, 255, 0)  # Cyan in BGR
        elif hasattr(piece, 'captured') and piece.captured:
            return (0, 255, 0)  # Green in BGR
        elif hasattr(piece, 'error') and piece.error:
            return (0, 0, 255)  # Red in BGR
        else:
            return (255, 255, 255)  # White in BGR (detected)

    def _draw_piece_status(self, frame: np.ndarray, piece: Any,
                          x: int, y: int, w: int, h: int, color: Tuple[int, int, int]):
        """
        Draw additional status information for a piece.

        Args:
            frame: Frame to draw on
            piece: Piece object
            x, y, w, h: Bounding box coordinates
            color: Color to use for text
        """
        # Draw zone indicator if available
        if hasattr(piece, 'in_entry_zone') and piece.in_entry_zone:
            cv2.putText(frame, "ENTRY", (x, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        elif hasattr(piece, 'in_exit_zone') and piece.in_exit_zone:
            cv2.putText(frame, "EXIT", (x, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


# ============================================================================
# UNITED CAMERA VIEW WIDGET
# ============================================================================

class CameraViewUnited(BaseCameraViewWidget):
    """
    United camera view widget - displays camera feed with both ROI and tracking overlays.

    Combines all overlay features:
    - ROI rectangle and zones (from CameraViewROI)
    - Tracked piece bounding boxes (from CameraViewTracking)

    Use this widget when you need complete monitoring during sorting operations,
    showing both the detection region configuration and active piece tracking.

    Example:
        view = CameraViewUnited()
        view.set_roi(100, 50, 1720, 980)
        view.set_zones(0.15, 0.15)
        view.set_tracked_pieces([piece1, piece2])
        view.set_fps_visible(True)
    """

    def __init__(self, parent=None):
        """
        Initialize united camera view widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # ROI configuration (from CameraViewROI)
        self.roi = None
        self.entry_zone_pct = 0.15
        self.exit_zone_pct = 0.15

        # Tracking configuration (from CameraViewTracking)
        self.tracked_pieces = []

        logger.debug("CameraViewUnited initialized")

    # ========================================================================
    # PUBLIC CONFIGURATION METHODS
    # ========================================================================

    def set_roi(self, x: int, y: int, width: int, height: int):
        """
        Set the Region of Interest coordinates.

        Args:
            x: ROI top-left x coordinate
            y: ROI top-left y coordinate
            width: ROI width
            height: ROI height
        """
        self.roi = (x, y, width, height)
        logger.debug(f"ROI set to: ({x}, {y}, {width}, {height})")

    def set_zones(self, entry_pct: float, exit_pct: float):
        """
        Set entry and exit zone percentages.

        Args:
            entry_pct: Entry zone percentage (0.0 to 1.0)
            exit_pct: Exit zone percentage (0.0 to 1.0)
        """
        self.entry_zone_pct = entry_pct
        self.exit_zone_pct = exit_pct
        logger.debug(f"Zones set to: entry={entry_pct}, exit={exit_pct}")

    def set_tracked_pieces(self, pieces: List[Any]):
        """
        Update the list of tracked pieces to display.

        Args:
            pieces: List of piece objects with bbox and status attributes
        """
        self.tracked_pieces = pieces if pieces else []

    # ========================================================================
    # FRAME PROCESSING
    # ========================================================================

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply both ROI and tracking overlays to frame.

        Order of drawing:
        1. ROI zones (semi-transparent, drawn first)
        2. Tracked pieces (opaque, drawn on top)
        3. ROI border (drawn last for visibility)

        Args:
            frame: Input frame

        Returns:
            Frame with all overlays
        """
        # Apply ROI overlays first
        frame = self._apply_roi_overlays(frame)

        # Apply tracking overlays on top
        frame = self._apply_tracking_overlays(frame)

        return frame

    # ========================================================================
    # HELPER METHODS - ROI Overlays
    # ========================================================================

    def _apply_roi_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply ROI and zone overlays (reused from CameraViewROI logic).

        Args:
            frame: Input frame

        Returns:
            Frame with ROI overlays
        """
        if self.roi is None:
            return frame

        x, y, w, h = self.roi

        # Draw zones first (so borders are on top)
        self._draw_entry_zone(frame, x, y, w, h)
        self._draw_exit_zone(frame, x, y, w, h)

        # Draw ROI rectangle border
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                     (0, 255, 255), ViewStyles.LINE_WIDTH_ROI)

        return frame

    def _draw_entry_zone(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw entry zone overlay (same as CameraViewROI)."""
        entry_width = int(w * self.entry_zone_pct)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + entry_width, y + h),
                     (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cv2.rectangle(frame, (x, y), (x + entry_width, y + h),
                     (0, 255, 0), ViewStyles.LINE_WIDTH_ZONE)

        cv2.putText(frame, "ENTRY", (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _draw_exit_zone(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw exit zone overlay (same as CameraViewROI)."""
        exit_width = int(w * self.exit_zone_pct)
        exit_x = x + w - exit_width

        overlay = frame.copy()
        cv2.rectangle(overlay, (exit_x, y), (x + w, y + h),
                     (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cv2.rectangle(frame, (exit_x, y), (x + w, y + h),
                     (0, 0, 255), ViewStyles.LINE_WIDTH_ZONE)

        cv2.putText(frame, "EXIT", (exit_x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ========================================================================
    # HELPER METHODS - Tracking Overlays
    # ========================================================================

    def _apply_tracking_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply piece tracking overlays (reused from CameraViewTracking logic).

        Args:
            frame: Input frame

        Returns:
            Frame with tracking overlays
        """
        if not self.tracked_pieces:
            return frame

        for piece in self.tracked_pieces:
            self._draw_tracked_piece(frame, piece)

        return frame

    def _draw_tracked_piece(self, frame: np.ndarray, piece: Any):
        """Draw overlay for tracked piece (same as CameraViewTracking)."""
        if not hasattr(piece, 'bbox') or piece.bbox is None:
            return

        x, y, w, h = piece.bbox
        color = self._get_piece_color(piece)

        cv2.rectangle(frame, (x, y), (x + w, y + h),
                     color, ViewStyles.LINE_WIDTH_PIECE)

        if hasattr(piece, 'id'):
            label = f"ID: {piece.id}"
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        self._draw_piece_status(frame, piece, x, y, w, h, color)

    def _get_piece_color(self, piece: Any) -> Tuple[int, int, int]:
        """Determine piece color based on status (same as CameraViewTracking)."""
        if hasattr(piece, 'being_processed') and piece.being_processed:
            return (255, 255, 0)  # Cyan in BGR
        elif hasattr(piece, 'captured') and piece.captured:
            return (0, 255, 0)  # Green in BGR
        elif hasattr(piece, 'error') and piece.error:
            return (0, 0, 255)  # Red in BGR
        else:
            return (255, 255, 255)  # White in BGR

    def _draw_piece_status(self, frame: np.ndarray, piece: Any,
                          x: int, y: int, w: int, h: int, color: Tuple[int, int, int]):
        """Draw piece status indicators (same as CameraViewTracking)."""
        if hasattr(piece, 'in_entry_zone') and piece.in_entry_zone:
            cv2.putText(frame, "ENTRY", (x, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        elif hasattr(piece, 'in_exit_zone') and piece.in_exit_zone:
            cv2.putText(frame, "EXIT", (x, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


# ============================================================================
# SORTING CAMERA VIEW WIDGET (SELF-CONFIGURING)
# ============================================================================

class CameraViewSorting(CameraViewUnited):
    """
    Self-configuring camera view widget for sorting operations.

    This widget automatically loads ROI and zone configuration from the
    config manager, and provides color-coded piece tracking based on
    processing state. It integrates with the detector coordinator and
    orchestrator to determine piece processing states.

    Features:
    - Auto-loads ROI and zones from configuration (no manual setup required)
    - Color-coded bounding boxes showing processing stage
    - Real-time piece state tracking
    - Integrates seamlessly with sorting GUI

    Color Coding:
    - Gray (128, 128, 128): Detected, not captured yet
    - Orange (0, 165, 255): Captured, queued for processing
    - Green (0, 255, 0): Identified successfully
    - Magenta (255, 0, 255): Unknown/overflow piece

    Usage:
        # In sorting GUI - just create and use:
        view = CameraViewSorting(
            config_manager=config_manager,
            detector_coordinator=detector_coordinator,
            orchestrator=orchestrator
        )
        camera.register_consumer("sorting_gui", view.receive_frame, "async", 5)

        # ROI and zones are automatically loaded and configured!
        # No need to call set_roi() or set_zones()

    The widget automatically updates piece colors based on their state in
    the detection and processing pipeline, providing real-time visual feedback
    during sorting operations.
    """

    def __init__(self, config_manager, detector_coordinator, orchestrator, parent=None):
        """
        Initialize sorting camera view with automatic configuration.

        This constructor:
        1. Calls parent CameraViewUnited.__init__() to setup base functionality
        2. Stores references to system modules for piece state tracking
        3. Automatically loads ROI and zone configuration from config
        4. Initializes piece tracking dictionaries

        Args:
            config_manager: Configuration manager for loading ROI/zone settings
            detector_coordinator: For accessing tracked pieces and their states
            orchestrator: For accessing identified pieces dictionary
            parent: Optional parent widget
        """
        # Call parent constructor first
        super().__init__(parent)

        # Store module references for piece state tracking
        self.config_manager = config_manager
        self.detector_coordinator = detector_coordinator
        self.orchestrator = orchestrator

        # Initialize piece tracking dictionaries
        # These are updated periodically by sorting GUI via update_piece_tracking()
        self.tracked_pieces_dict = {}  # Maps piece_id -> TrackedPiece
        self.identified_pieces_dict = {}  # Maps piece_id -> IdentifiedPiece

        # Auto-configure ROI and zones from configuration
        self._load_configuration()

        logger.info("CameraViewSorting initialized with auto-configuration")

    def _load_configuration(self):
        """
        Automatically load ROI and zone configuration from config manager.

        This method is called during __init__ to configure the widget without
        requiring manual setup by the parent GUI. It loads:
        - ROI coordinates (x, y, width, height) from detector_roi config
        - Zone percentages (entry, exit) from detector config

        This eliminates the need for the sorting GUI to manually configure
        the widget after creation.
        """
        try:
            # Load ROI configuration
            roi_config = self.config_manager.get_module_config("detector_roi")
            self.set_roi(
                roi_config["x"],
                roi_config["y"],
                roi_config["w"],
                roi_config["h"]
            )
            logger.info(f"Auto-loaded ROI: ({roi_config['x']}, {roi_config['y']}, "
                        f"{roi_config['w']}, {roi_config['h']})")

            # Load zone configuration
            detector_config = self.config_manager.get_module_config("detector")
            zones = detector_config.get("zones", {})
            entry_pct = zones.get("entry_percentage", 15) / 100.0
            exit_pct = zones.get("exit_percentage", 15) / 100.0
            self.set_zones(entry_pct, exit_pct)
            logger.info(f"Auto-loaded zones: entry={entry_pct:.2f}, exit={exit_pct:.2f}")

        except Exception as e:
            logger.error(f"Error loading configuration for CameraViewSorting: {e}", exc_info=True)
            # Set default values if config loading fails
            logger.warning("Using default ROI and zone values")
            self.set_roi(100, 50, 1720, 980)
            self.set_zones(0.15, 0.15)

    def update_piece_tracking(self, tracked_pieces_dict: Dict, identified_pieces_dict: Dict):
        """
        Update the dictionaries used for piece state tracking and color-coding.

        This should be called periodically by the sorting GUI (typically in a
        timer callback) to keep the piece colors synchronized with the current
        processing state.

        Args:
            tracked_pieces_dict: {piece_id: TrackedPiece} from detector coordinator
            identified_pieces_dict: {piece_id: IdentifiedPiece} from orchestrator
        """
        self.tracked_pieces_dict = tracked_pieces_dict
        self.identified_pieces_dict = identified_pieces_dict

    def _get_piece_color(self, piece_id: int) -> tuple:
        """
        Determine bounding box color based on piece processing stage.

        This method overrides the parent's _get_piece_color() to provide
        sorting-specific color logic based on the piece's current state
        in the detection and processing pipeline.

        Color Logic:
        1. Gray: Piece is detected but not yet captured
        2. Orange: Piece is captured and queued for processing
        3. Green: Piece is successfully identified with valid bin assignment
        4. Magenta: Piece is identified as unknown (bin 0 = overflow)

        The method looks up the piece in both tracked_pieces_dict (for capture
        state) and identified_pieces_dict (for identification state) to
        determine the appropriate color.

        Args:
            piece_id: The piece ID to look up

        Returns:
            BGR color tuple for OpenCV drawing
        """
        # Check if piece is in tracked pieces dictionary
        if piece_id not in self.tracked_pieces_dict:
            logger.debug(f"Piece {piece_id}: Not in tracked_pieces_dict → GRAY")
            return (128, 128, 128)  # Gray - detected but not tracked yet

        tracked_piece = self.tracked_pieces_dict[piece_id]

        # Check if piece has been captured
        if not tracked_piece.captured:
            logger.debug(f"Piece {piece_id}: Not captured → GRAY")
            return (128, 128, 128)  # Gray - detected but not captured

        # Check if piece has been identified
        if piece_id in self.identified_pieces_dict:
            identified_piece = self.identified_pieces_dict[piece_id]

            # Check if it's unknown/overflow (bin 0)
            if hasattr(identified_piece, 'bin_number') and identified_piece.bin_number == 0:
                logger.info(f"Piece {piece_id}: Unknown/Overflow → MAGENTA")
                return (255, 0, 255)  # Magenta - unknown/overflow
            else:
                logger.info(f"Piece {piece_id}: Identified → GREEN")
                return (0, 255, 0)  # Green - identified successfully

        # Piece is captured but not yet identified
        logger.info(f"Piece {piece_id}: Captured, awaiting identification → ORANGE")
        return (0, 165, 255)  # Orange - captured, queued for processing

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame with ROI, zones, and color-coded piece tracking.

        This method overrides the parent's process_frame() to provide the
        complete sorting view with:
        1. ROI rectangle and zones (from parent)
        2. Color-coded piece bounding boxes (custom logic)
        3. Piece ID labels

        The color of each piece's bounding box reflects its current state
        in the processing pipeline, providing real-time visual feedback.

        Args:
            frame: Input frame from camera

        Returns:
            Frame with all overlays drawn
        """
        # Start with parent's ROI and zone overlays
        display_frame = super().process_frame(frame)

        # Draw tracked pieces with custom color-coding
        if self.tracked_pieces:
            for piece in self.tracked_pieces:
                piece_id = piece.get('id')
                bbox = piece.get('bbox')

                if bbox:
                    x, y, w, h = bbox

                    # Get color based on processing stage
                    color = self._get_piece_color(piece_id)

                    # Draw bounding box
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                    # Draw piece ID label with colored background
                    label = f"ID:{piece_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(display_frame, (x, y - label_size[1] - 5),
                                  (x + label_size[0], y), color, -1)
                    cv2.putText(display_frame, label, (x, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display_frame