"""
sorting_gui.py - Runtime GUI for LEGO Sorting Machine

FILE LOCATION: GUI/sorting_gui.py

This GUI displays the machine's operation in real-time during sorting sessions.
It shows live camera feed with tracked pieces, bin status, and recently processed pieces.

LAYOUT:
┌─────────────────────────────────────┬──────────┐
│                                     │          │
│                                     │ Recently │
│      Camera View (2/3 height)       │   Proc.  │
│      - Live feed with ROI           │  Piece   │
│      - Color-coded piece tracking   │          │
│      - Zone overlays                │  [img]   │
│                                     │  Brick   │
├─────────────────────────────────────┤  2x4     │
│  Bin Status Panel (1/3 height)      │  #3001   │
│  [1] [2] [3] [4] [5] [6] [7] [8]   │  → Bin2  │
│  2x4 Plt ... ... ... ... ... ...    │          │
│  ████░░░ ██░░ ... ... ... ... ...   │          │
│  [Reset][Reset]... ...              │          │
└─────────────────────────────────────┴──────────┘

CALLBACKS:
- bin_assignment_module calls on_bin_assigned(bin_num, category)
- processing_coordinator calls on_piece_identified(identified_piece)
- hardware_controller calls on_piece_sorted(bin_num)

FEATURES:
- Real-time camera view with ROI, zones, and tracked pieces
- Color-coded bounding boxes showing piece processing stage
- Bin status showing category assignment and capacity
- Recently processed piece display with identification details
- Bin reset buttons for manual capacity management
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QGroupBox, QGridLayout,
    QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor

# Import camera view widget
from GUI.widgets.camera_view import CameraViewUnited, ViewStyles

logger = logging.getLogger(__name__)


# ============================================================================
# STYLING CONSTANTS (matching configuration_gui.py)
# ============================================================================

class SortingGUIStyles:
    """Styling constants for sorting GUI - matches config_gui aesthetic."""

    # Color scheme (matching config_gui buttons)
    COLOR_PRIMARY = "#27AE60"  # Green - success/ready
    COLOR_PRIMARY_HOVER = "#229954"
    COLOR_WARNING = "#F39C12"  # Orange - warning
    COLOR_WARNING_HOVER = "#E67E22"
    COLOR_DANGER = "#E74C3C"  # Red - critical
    COLOR_DANGER_HOVER = "#C0392B"
    COLOR_INFO = "#3498DB"  # Blue - info
    COLOR_INFO_HOVER = "#2980B9"
    COLOR_NEUTRAL = "#95A5A6"  # Gray - neutral
    COLOR_NEUTRAL_HOVER = "#7F8C8D"

    # Piece tracking colors (BGR for OpenCV, matching camera_view.py)
    # These are used for bounding box drawing
    COLOR_DETECTED = (128, 128, 128)  # Gray - detected, not captured
    COLOR_CAPTURED = (0, 165, 255)  # Orange - captured, queued
    COLOR_IDENTIFIED = (0, 255, 0)  # Green - identified successfully
    COLOR_UNKNOWN = (255, 0, 255)  # Magenta - identified as unknown

    # Background colors
    BACKGROUND_DARK = "#2b2b2b"
    BACKGROUND_MEDIUM = "#3d3d3d"
    BACKGROUND_LIGHT = "#555555"

    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_MUTED = "#7F8C8D"

    # Font settings
    FONT_FAMILY = "Arial"
    FONT_SIZE_TITLE = 14
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_SMALL = 9

    @staticmethod
    def get_button_style(bg_color: str, hover_color: str, text_color: str = "#ffffff") -> str:
        """Get button stylesheet matching config_gui style."""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
                border-radius: 4px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {bg_color};
            }}
            QPushButton:disabled {{
                background-color: #555555;
                color: #888888;
            }}
        """

    @staticmethod
    def get_groupbox_style() -> str:
        """Get groupbox stylesheet matching config_gui style."""
        return """
            QGroupBox {
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """

    @staticmethod
    def get_label_style(size: int = 11, bold: bool = False, color: str = "#ffffff") -> str:
        """Get label stylesheet."""
        weight = "bold" if bold else "normal"
        return f"""
            QLabel {{
                color: {color};
                font-size: {size}px;
                font-weight: {weight};
            }}
        """


# ============================================================================
# BIN STATUS WIDGET
# ============================================================================

class BinStatusWidget(QWidget):
    """
    Widget displaying status for a single bin.

    Shows:
    - Bin number
    - Assigned category (or "Unassigned")
    - Capacity percentage with color-coded progress bar
    - Reset button to clear bin

    The capacity bar changes color based on fill level:
    - Green: < 75%
    - Yellow: 75-90%
    - Red: > 90%
    """

    # Signal emitted when reset button clicked
    reset_requested = pyqtSignal(int)  # Emits bin_number

    def __init__(self, bin_number: int, parent=None):
        """
        Initialize bin status widget.

        Args:
            bin_number: The bin number (0-based index, but displayed as 1-based)
        """
        super().__init__(parent)
        self.bin_number = bin_number
        self.current_capacity = 0
        self.assigned_category = None

        self._setup_ui()

    def _setup_ui(self):
        """Create the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Bin number and category label
        self.label = QLabel(f"Bin {self.bin_number + 1}: Unassigned")
        self.label.setStyleSheet(SortingGUIStyles.get_label_style(
            size=10, bold=True, color=SortingGUIStyles.TEXT_PRIMARY
        ))
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Capacity progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(20)
        self._update_progress_bar_style()
        layout.addWidget(self.progress_bar)

        # Reset button
        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.setStyleSheet(SortingGUIStyles.get_button_style(
            SortingGUIStyles.COLOR_WARNING,
            SortingGUIStyles.COLOR_WARNING_HOVER
        ))
        self.reset_button.setToolTip(f"Reset capacity for Bin {self.bin_number + 1}")
        self.reset_button.clicked.connect(lambda: self.reset_requested.emit(self.bin_number))
        self.reset_button.setEnabled(False)  # Disabled until bin has content
        layout.addWidget(self.reset_button)

    def update_assignment(self, category: str):
        """
        Update the bin's assigned category.

        Called by: SortingGUI.on_bin_assigned() callback

        Args:
            category: Category name assigned to this bin
        """
        self.assigned_category = category
        self.label.setText(f"Bin {self.bin_number + 1}: {category}")
        logger.debug(f"Bin {self.bin_number + 1} assigned to '{category}'")

    def update_capacity(self, percentage: int):
        """
        Update the bin's capacity percentage.

        Called by: SortingGUI.on_piece_sorted() callback

        Args:
            percentage: Fill percentage (0-100)
        """
        self.current_capacity = percentage
        self.progress_bar.setValue(percentage)
        self._update_progress_bar_style()

        # Enable reset button if bin has content
        self.reset_button.setEnabled(percentage > 0)

        logger.debug(f"Bin {self.bin_number + 1} capacity: {percentage}%")

    def _update_progress_bar_style(self):
        """Update progress bar color based on capacity level."""
        if self.current_capacity > 90:
            color = SortingGUIStyles.COLOR_DANGER
        elif self.current_capacity > 75:
            color = SortingGUIStyles.COLOR_WARNING
        else:
            color = SortingGUIStyles.COLOR_PRIMARY

        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)

    def reset_capacity(self):
        """Reset the bin capacity to 0. Called after bin is physically emptied."""
        self.update_capacity(0)
        logger.info(f"Bin {self.bin_number + 1} capacity reset to 0%")


# ============================================================================
# BIN STATUS PANEL
# ============================================================================

class BinStatusPanel(QGroupBox):
    """
    Panel displaying status for all bins in a grid.

    CALLBACK CONNECTION:
    This panel's bins are updated via:
    1. on_bin_assigned() -> calls update_bin_assignment()
    2. on_piece_sorted() -> calls increment_bin_capacity()

    The panel also manages reset button clicks, calling the
    bin_capacity_module.reset_bin() method.
    """

    def __init__(self, bin_capacity_module, num_bins: int = 8, parent=None):
        """
        Initialize bin status panel.

        Args:
            bin_capacity_module: Reference to hardware.bin_capacity_module
            num_bins: Total number of bins to display
        """
        super().__init__("Bin Status", parent)
        self.bin_capacity_module = bin_capacity_module
        self.num_bins = num_bins
        self.bin_widgets = []

        self.setStyleSheet(SortingGUIStyles.get_groupbox_style())
        self._setup_ui()

    def _setup_ui(self):
        """Create the grid of bin widgets."""
        layout = QGridLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 20, 10, 10)

        # Create bin widgets in a grid (2 rows x 4 columns for 8 bins)
        rows = 2
        cols = self.num_bins // rows

        for i in range(self.num_bins):
            bin_widget = BinStatusWidget(i, self)

            # Connect reset signal to handler
            bin_widget.reset_requested.connect(self.on_bin_reset_requested)

            # Add to grid
            row = i // cols
            col = i % cols
            layout.addWidget(bin_widget, row, col)

            self.bin_widgets.append(bin_widget)

    def update_bin_assignment(self, bin_number: int, category: str):
        """
        Update a bin's assigned category.

        CALLBACK: Called by SortingGUI.on_bin_assigned()
        TRIGGERED BY: bin_assignment_module when new category assigned

        Args:
            bin_number: Bin index (0-based)
            category: Category name assigned to bin
        """
        if 0 <= bin_number < self.num_bins:
            self.bin_widgets[bin_number].update_assignment(category)

    def increment_bin_capacity(self, bin_number: int):
        """
        Update a bin's capacity after a piece is sorted.

        CALLBACK: Called by SortingGUI.on_piece_sorted()
        TRIGGERED BY: hardware_controller when servo moves piece

        Args:
            bin_number: Bin index (0-based)
        """
        if 0 <= bin_number < self.num_bins:
            # Get current capacity from bin_capacity_module
            capacity_pct = self.bin_capacity_module.get_bin_capacity_percentage(bin_number)
            self.bin_widgets[bin_number].update_capacity(int(capacity_pct))

    def on_bin_reset_requested(self, bin_number: int):
        """
        Handle bin reset button click.

        Calls bin_capacity_module.reset_bin() and updates display.

        Args:
            bin_number: Bin index (0-based)
        """
        logger.info(f"Reset requested for Bin {bin_number + 1}")

        # Reset in the capacity module
        self.bin_capacity_module.reset_bin(bin_number)

        # Update display
        self.bin_widgets[bin_number].reset_capacity()


# ============================================================================
# RECENTLY PROCESSED PIECE PANEL
# ============================================================================

class RecentlyProcessedPanel(QGroupBox):
    """
    Panel displaying the most recently processed piece.

    Shows:
    - Cropped piece image
    - Part name
    - Element ID
    - Assigned bin number
    - Processing timestamp

    CALLBACK CONNECTION:
    Updated via SortingGUI.on_piece_identified() callback
    """

    def __init__(self, parent=None):
        """Initialize recently processed piece panel."""
        super().__init__("Recently Processed", parent)
        self.setStyleSheet(SortingGUIStyles.get_groupbox_style())
        self._setup_ui()

    def _setup_ui(self):
        """Create the UI components."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 20, 10, 10)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setMaximumSize(300, 300)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {SortingGUIStyles.BACKGROUND_MEDIUM};
                border: 2px solid {SortingGUIStyles.BACKGROUND_LIGHT};
                border-radius: 5px;
            }}
        """)
        self.image_label.setText("No pieces\nprocessed yet")
        layout.addWidget(self.image_label)

        # Piece information labels
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)

        self.name_label = QLabel("Part Name: -")
        self.name_label.setStyleSheet(SortingGUIStyles.get_label_style(
            size=12, bold=True, color=SortingGUIStyles.TEXT_PRIMARY
        ))
        self.name_label.setWordWrap(True)
        info_layout.addWidget(self.name_label)

        self.element_id_label = QLabel("Element ID: -")
        self.element_id_label.setStyleSheet(SortingGUIStyles.get_label_style(
            size=11, color=SortingGUIStyles.TEXT_SECONDARY
        ))
        info_layout.addWidget(self.element_id_label)

        self.bin_label = QLabel("→ Bin: -")
        self.bin_label.setStyleSheet(SortingGUIStyles.get_label_style(
            size=12, bold=True, color=SortingGUIStyles.COLOR_PRIMARY
        ))
        info_layout.addWidget(self.bin_label)

        self.timestamp_label = QLabel("Time: -")
        self.timestamp_label.setStyleSheet(SortingGUIStyles.get_label_style(
            size=9, color=SortingGUIStyles.TEXT_MUTED
        ))
        info_layout.addWidget(self.timestamp_label)

        layout.addLayout(info_layout)
        layout.addStretch()

    def display_piece(self, identified_piece):
        """
        Display a newly identified piece.

        CALLBACK: Called by SortingGUI.on_piece_identified()
        TRIGGERED BY: processing_coordinator when identification completes

        Args:
            identified_piece: IdentifiedPiece object with all identification data
        """
        try:
            # Display the processed image
            if hasattr(identified_piece, 'processed_image') and identified_piece.processed_image is not None:
                image = identified_piece.processed_image

                # Convert BGR to RGB for Qt display
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image

                # Convert to QPixmap
                h, w = image_rgb.shape[:2]
                bytes_per_line = 3 * w
                q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                # Scale to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)

            # Update text labels
            part_name = identified_piece.part_name if identified_piece.part_name else "Unknown"
            self.name_label.setText(f"Part Name: {part_name}")

            element_id = identified_piece.element_id if identified_piece.element_id else "?"
            self.element_id_label.setText(f"Element ID: {element_id}")

            bin_num = identified_piece.assigned_bin
            if bin_num == 0:
                self.bin_label.setText("→ Bin: 0 (Overflow)")
                self.bin_label.setStyleSheet(SortingGUIStyles.get_label_style(
                    size=12, bold=True, color=SortingGUIStyles.COLOR_DANGER
                ))
            else:
                self.bin_label.setText(f"→ Bin: {bin_num}")
                self.bin_label.setStyleSheet(SortingGUIStyles.get_label_style(
                    size=12, bold=True, color=SortingGUIStyles.COLOR_PRIMARY
                ))

            # Show timestamp
            import time
            timestamp_str = time.strftime("%H:%M:%S", time.localtime())
            self.timestamp_label.setText(f"Time: {timestamp_str}")

            logger.debug(f"Displayed piece: {part_name} (ID: {element_id}) → Bin {bin_num}")

        except Exception as e:
            logger.error(f"Error displaying piece: {e}", exc_info=True)


# ============================================================================
# CAMERA VIEW WITH TRACKING
# ============================================================================

class TrackingCameraView(CameraViewUnited):
    """
    Extended camera view with color-coded piece tracking.

    This extends CameraViewUnited to add color-coding based on piece
    processing stage (detected, captured, identified, unknown).

    REQUIRES:
    - Access to tracked_pieces dict from detector
    - Access to identified_pieces dict from processing

    The orchestrator must call update_piece_tracking() with both dicts.
    """

    def __init__(self, parent=None):
        """Initialize tracking camera view."""
        super().__init__(parent)

        # Storage for piece status lookups
        self.tracked_pieces_dict = {}
        self.identified_pieces_dict = {}

    def update_piece_tracking(self, tracked_pieces_dict: Dict, identified_pieces_dict: Dict):
        """
        Update the dictionaries used for color-coding pieces.

        Should be called by orchestrator whenever piece status changes.

        Args:
            tracked_pieces_dict: {piece_id: TrackedPiece} from detector
            identified_pieces_dict: {piece_id: IdentifiedPiece} from processing
        """
        self.tracked_pieces_dict = tracked_pieces_dict
        self.identified_pieces_dict = identified_pieces_dict

    def _get_piece_color(self, piece_id: int) -> tuple:
        """
        Determine bounding box color based on piece processing stage.

        Color logic:
        - Gray: Detected, not captured yet
        - Orange: Captured, queued for processing
        - Green: Identified successfully
        - Magenta: Identified as unknown/overflow

        Args:
            piece_id: The piece ID to look up

        Returns:
            BGR color tuple for OpenCV drawing
        """
        # Check if piece is tracked
        if piece_id not in self.tracked_pieces_dict:
            return SortingGUIStyles.COLOR_DETECTED

        tracked_piece = self.tracked_pieces_dict[piece_id]

        # Check if captured
        if not tracked_piece.captured:
            return SortingGUIStyles.COLOR_DETECTED

        # Check if identified
        if piece_id in self.identified_pieces_dict:
            identified_piece = self.identified_pieces_dict[piece_id]

            # Check if unknown/overflow
            if (hasattr(identified_piece, 'is_unknown') and identified_piece.is_unknown) or \
                    (hasattr(identified_piece, 'assigned_bin') and identified_piece.assigned_bin == 0):
                return SortingGUIStyles.COLOR_UNKNOWN
            else:
                return SortingGUIStyles.COLOR_IDENTIFIED

        # Captured but not yet identified
        return SortingGUIStyles.COLOR_CAPTURED

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame with ROI, zones, and color-coded piece tracking.

        Overrides parent method to add custom color logic.

        Args:
            frame: Input frame from camera

        Returns:
            Frame with all overlays drawn
        """
        # Start with parent's processing (ROI and zones)
        display_frame = super().process_frame(frame)

        # Draw tracked pieces with custom colors
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

                    # Draw piece ID label
                    label = f"ID:{piece_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(display_frame, (x, y - label_size[1] - 5),
                                  (x + label_size[0], y), color, -1)
                    cv2.putText(display_frame, label, (x, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display_frame


# ============================================================================
# MAIN SORTING GUI
# ============================================================================

class SortingGUI(QMainWindow):
    """
    Main GUI window for LEGO Sorting Machine runtime operation.

    INITIALIZATION:
    Called by LegoSorting008 orchestrator after all modules are initialized.

    LAYOUT:
    - Top 2/3: Camera view with tracking (left 3/4) + Recently processed (right 1/4)
    - Bottom 1/3: Bin status panel

    CALLBACKS REGISTERED:
    This GUI registers callbacks with:
    1. bin_assignment_module.register_assignment_callback(on_bin_assigned)
    2. processing_coordinator.register_identification_callback(on_piece_identified)
    3. hardware_controller.register_sort_callback(on_piece_sorted)

    UPDATES:
    - Camera view: Updates automatically via camera consumer pattern
    - Bin assignments: Via on_bin_assigned callback
    - Recently processed: Via on_piece_identified callback
    - Bin capacity: Via on_piece_sorted callback
    """

    def __init__(self, orchestrator):
        """
        Initialize sorting GUI.

        Args:
            orchestrator: Reference to main LegoSorting008 orchestrator
                         Provides access to all modules via properties
        """
        super().__init__()

        self.orchestrator = orchestrator

        # Component references (for convenience)
        self.camera = orchestrator.camera
        self.detector = orchestrator.detector_coordinator
        self.processing = orchestrator.processing_coordinator
        self.hardware = orchestrator.hardware_controller
        self.bin_assignment = orchestrator.bin_assignment_module
        self.bin_capacity = orchestrator.bin_capacity_module

        # UI Components (created in _setup_ui)
        self.camera_view = None
        self.bin_status_panel = None
        self.recent_piece_panel = None

        # State tracking
        self.is_running = False

        # Initialize UI
        self._setup_ui()

        # Register callbacks with modules
        self._register_callbacks()

        # Start camera feed
        self._start_camera_feed()

        # Setup update timer for piece tracking colors
        self._setup_update_timer()

        logger.info("Sorting GUI initialized")

    def _setup_ui(self):
        """Create the main UI layout."""
        self.setWindowTitle("LEGO Sorting Machine - Runtime")
        self.setMinimumSize(1400, 900)

        # Apply dark theme stylesheet
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {SortingGUIStyles.BACKGROUND_DARK};
            }}
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left side: Camera and bin status (3/4 width)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Camera view (2/3 height of left side)
        self.camera_view = TrackingCameraView()
        self.camera_view.setMinimumSize(800, 600)
        self.camera_view.set_fps_visible(True)
        left_layout.addWidget(self.camera_view, stretch=2)

        # Bin status panel (1/3 height of left side)
        self.bin_status_panel = BinStatusPanel(
            self.bin_capacity,
            num_bins=8
        )
        left_layout.addWidget(self.bin_status_panel, stretch=1)

        main_layout.addWidget(left_widget, stretch=3)

        # Right side: Recently processed panel (1/4 width)
        self.recent_piece_panel = RecentlyProcessedPanel()
        self.recent_piece_panel.setMinimumWidth(250)
        self.recent_piece_panel.setMaximumWidth(350)
        main_layout.addWidget(self.recent_piece_panel, stretch=1)

        logger.info("UI layout created")

    # ========================================================================
    # CALLBACK REGISTRATION
    # ========================================================================

    def _register_callbacks(self):
        """
        Register callbacks with all relevant modules.

        This is the key integration point - we tell each module
        "call my method when your event happens."
        """
        logger.info("Registering GUI callbacks with modules...")

        # 1. Register for bin assignment updates
        # WHEN: bin_assignment_module assigns a category to a bin
        # CALLS: self.on_bin_assigned(bin_number, category)
        if hasattr(self.bin_assignment, 'register_assignment_callback'):
            self.bin_assignment.register_assignment_callback(self.on_bin_assigned)
            logger.info("  ✓ Registered bin assignment callback")
        else:
            logger.warning("  ✗ bin_assignment_module missing register_assignment_callback()")

        # 2. Register for piece identification completion
        # WHEN: processing_coordinator finishes identifying a piece
        # CALLS: self.on_piece_identified(identified_piece)
        if hasattr(self.processing, 'register_identification_callback'):
            self.processing.register_identification_callback(self.on_piece_identified)
            logger.info("  ✓ Registered identification callback")
        else:
            logger.warning("  ✗ processing_coordinator missing register_identification_callback()")

        # 3. Register for servo sort events
        # WHEN: hardware_controller moves servo to sort a piece
        # CALLS: self.on_piece_sorted(bin_number)
        if hasattr(self.hardware, 'register_sort_callback'):
            self.hardware.register_sort_callback(self.on_piece_sorted)
            logger.info("  ✓ Registered sort callback")
        else:
            logger.warning("  ✗ hardware_controller missing register_sort_callback()")

        logger.info("Callback registration complete")

    # ========================================================================
    # CALLBACK METHODS (called BY modules, not by us)
    # ========================================================================

    def on_bin_assigned(self, bin_number: int, category: str):
        """
        Callback: Called when a bin is assigned to a category.

        TRIGGERED BY: bin_assignment_module._notify_assignment_callbacks()
        WHEN: New category is dynamically assigned OR pre-assignments loaded

        Args:
            bin_number: Bin index (0-based)
            category: Category name assigned to bin
        """
        logger.info(f"GUI callback: Bin {bin_number} assigned to '{category}'")

        # Update bin status panel
        self.bin_status_panel.update_bin_assignment(bin_number, category)

    def on_piece_identified(self, identified_piece):
        """
        Callback: Called when a piece is identified.

        TRIGGERED BY: processing_coordinator._notify_identification_complete()
        WHEN: Piece identification completes (success or unknown)

        Args:
            identified_piece: IdentifiedPiece object with all data
        """
        logger.info(f"GUI callback: Piece identified - {identified_piece.part_name}")

        # Update recently processed panel
        self.recent_piece_panel.display_piece(identified_piece)

    def on_piece_sorted(self, bin_number: int):
        """
        Callback: Called when a piece is sorted into a bin.

        TRIGGERED BY: hardware_controller._notify_sort_callbacks()
        WHEN: Servo moves to direct piece into bin

        Args:
            bin_number: Bin index (0-based) where piece was sorted
        """
        logger.info(f"GUI callback: Piece sorted into Bin {bin_number}")

        # Update bin capacity display
        self.bin_status_panel.increment_bin_capacity(bin_number)

    # ========================================================================
    # CAMERA FEED MANAGEMENT
    # ========================================================================

    def _start_camera_feed(self):
        """
        Start camera feed to the camera view widget.

        Registers the camera_view as a consumer of camera frames.
        """
        if self.camera:
            logger.info("Registering camera view as frame consumer")
            self.camera.register_consumer(
                name="sorting_gui",
                callback=self.camera_view.receive_frame,
                processing_type="async",
                priority=5
            )
            logger.info("Camera feed started")
        else:
            logger.error("No camera available for feed")

    def _setup_update_timer(self):
        """
        Setup timer to periodically update piece tracking colors.

        This timer calls update_piece_tracking() to refresh the
        color-coding based on current piece processing states.
        """
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_piece_colors)
        self.update_timer.start(100)  # Update every 100ms

    def _update_piece_colors(self):
        """
        Update piece tracking colors in camera view.

        Called periodically by timer. Fetches current tracked pieces
        and identified pieces, then updates camera view.
        """
        try:
            # Check if detector exists (won't exist in test mode)
            if self.detector is None:
                return  # Skip update if no detector

            # Get tracked pieces from detector
            tracked_pieces = self.detector.get_tracked_pieces()

            # Convert list to dict for lookup
            tracked_dict = {p.id: p for p in tracked_pieces}

            # Get identified pieces from ORCHESTRATOR instead of processing
            identified_dict = {}
            if self.orchestrator and hasattr(self.orchestrator, 'get_identified_pieces'):
                identified_dict = self.orchestrator.get_identified_pieces()

            # Update camera view with both dicts
            self.camera_view.update_piece_tracking(tracked_dict, identified_dict)

            # Set tracked pieces for drawing
            tracked_pieces_dicts = []
            for piece in tracked_pieces:
                tracked_pieces_dicts.append({
                    'id': piece.id,
                    'bbox': piece.bbox,
                    'center': piece.center,
                    'status': piece.status
                })

            self.camera_view.set_tracked_pieces(tracked_pieces_dicts)

        except Exception as e:
            logger.error(f"Error updating piece colors: {e}")

    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Sorting GUI closing")

        # Stop update timer
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()

        # Unregister camera consumer
        if self.camera:
            try:
                self.camera.unregister_consumer("sorting_gui")
            except Exception as e:
                logger.error(f"Error unregistering camera consumer: {e}")

        event.accept()


# ============================================================================
# STANDALONE TESTING
# ============================================================================

def main():
    """
    Standalone test of sorting GUI (for development).

    Creates a mock orchestrator with dummy modules for testing UI layout.
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create mock orchestrator with dummy modules
    class MockOrchestrator:
        def __init__(self):
            self.camera = None
            self.detector_coordinator = None
            self.processing_coordinator = None
            self.hardware_controller = None
            self.bin_assignment_module = None
            self.bin_capacity_module = MockBinCapacity()

    class MockBinCapacity:
        def get_bin_capacity_percentage(self, bin_num):
            return 0

        def reset_bin(self, bin_num):
            pass

    app = QApplication(sys.argv)

    orchestrator = MockOrchestrator()
    gui = SortingGUI(orchestrator)
    gui.show()

    logger.info("Sorting GUI test mode - UI only, no actual functionality")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()