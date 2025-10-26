"""
sorting_gui.py - Runtime GUI for LEGO Sorting Machine

FILE LOCATION: GUI/sorting_gui.py

This GUI displays the machine's operation in real-time during sorting sessions.
It shows live camera feed with tracked pieces, bin status, and recently processed pieces.

ARCHITECTURE CHANGES (Refactored):
- CameraViewSorting now auto-configures ROI and zones from config
- All widget-related frame processing logic moved to camera_view.py
- Sorting GUI simplified to focus on display and coordination

LAYOUT:
┌─────────────────────────────────┬──────────┐
│                                 │          │
│                                 │ Recently │
│      Camera View (2/3 height)   │   Proc.  │
│      - Live feed with ROI       │  Piece   │
│      - Color-coded piece tracking│         │
│      - Zone overlays            │  [img]   │
│                                 │  Brick   │
├─────────────────────────────────┤  2x4     │
│  Bin Status Panel (1/3 height)  │  #3001   │
│  [1] [2] [3] [4] [5] [6] [7] [8]│  → Bin2  │
│  2x4 Plt ... ... ... ... ... ... │         │
│  ████░░░ ██░░ ... ... ... ... ...│         │
│  [Reset][Reset]... ...           │          │
└─────────────────────────────────┴──────────┘

CALLBACKS:
- bin_assignment_module calls on_bin_assigned(bin_num, category)
- processing_coordinator calls on_piece_identified(identified_piece)
- hardware_controller calls on_piece_sorted(bin_num)

FEATURES:
- Real-time camera view with ROI, zones, and tracked pieces
- Auto-configured ROI and zones (loaded from config automatically)
- Color-coded bounding boxes showing piece processing stage
- Bin status showing category assignment and capacity
- Recently processed piece display with identification details
- Bin reset buttons for manual capacity management
"""

# ============================================================================
# IMPORTS
# ============================================================================

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

# Import widgets - using self-configuring CameraViewSorting and BinStatusWidget
from GUI.widgets.camera_view import CameraViewSorting, ViewStyles
from GUI.widgets.bin_status import BinStatusWidget
from GUI.widgets.gui_styles import SortingGUIStyles

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
            status = self.bin_capacity_module.get_bin_status(bin_number)
            count = status['count']
            max_capacity = status['max_capacity']
            percentage = int(status['percentage'] * 100)  # Convert 0.0-1.0 to 0-100

            self.bin_widgets[bin_number].update_capacity(count, max_capacity, percentage)

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
            piece_name = identified_piece.name if identified_piece.name else "Unknown"
            self.name_label.setText(f"Part Name: {piece_name}")

            element_id = identified_piece.element_id if identified_piece.element_id else "?"
            self.element_id_label.setText(f"Element ID: {element_id}")

            bin_num = identified_piece.bin_number
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

            logger.debug(f"Displayed piece: {piece_name} (ID: {element_id}) → Bin {bin_num}")

        except Exception as e:
            logger.error(f"Error displaying piece: {e}", exc_info=True)


# ============================================================================
# MAIN SORTING GUI
# ============================================================================

class SortingGUI(QMainWindow):
    """
    Main GUI window for LEGO Sorting Machine runtime operation.

    INITIALIZATION:
    Called by LegoSorting008 orchestrator after all modules are initialized.

    ARCHITECTURE IMPROVEMENT:
    The CameraViewSorting widget now auto-configures itself with ROI and zones
    from the config manager. This eliminates the need for manual configuration
    in this GUI class, keeping widget-related logic in the widget module.

    LAYOUT:
    - Top 2/3: Camera view with tracking (left 3/4) + Recently processed (right 1/4)
    - Bottom 1/3: Bin status panel

    CALLBACKS REGISTERED:
    This GUI registers callbacks with:
    1. bin_assignment_module.register_assignment_callback(on_bin_assigned)
    2. processing_coordinator.register_identification_callback(on_piece_identified)
    3. hardware_controller.register_sort_callback(on_piece_sorted)

    UPDATES:
    - Camera view: Auto-configured on creation, updates via camera consumer pattern
    - Bin assignments: Via on_bin_assigned callback
    - Recently processed: Via on_piece_identified callback
    - Bin capacity: Via on_piece_sorted callback
    - Piece colors: Via periodic timer updating piece tracking state

    THREAD SAFETY:
    Callbacks may be called from non-GUI threads. Internal signals are used
    to ensure all GUI updates happen in the main thread.
    """

    # Thread-safe signals for GUI updates (emitted from any thread, processed in GUI thread)
    _bin_assigned_signal = pyqtSignal(int, str)  # bin_number, category
    _piece_identified_signal = pyqtSignal(object)  # identified_piece
    _piece_sorted_signal = pyqtSignal(int)  # bin_number

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
        self.hardware = orchestrator.hardware_coordinator
        self.bin_assignment = orchestrator.bin_assignment_module
        self.bin_capacity = orchestrator.bin_capacity_manager

        # UI Components (created in _setup_ui)
        self.camera_view = None
        self.bin_status_panel = None
        self.recent_piece_panel = None

        # State tracking
        self.is_running = False

        # Initialize UI
        self._setup_ui()

        # Connect internal signals to GUI update slots (thread-safe)
        self._connect_signals()

        # Register callbacks with modules
        self._register_callbacks()

        # Start camera feed
        self._start_camera_feed()

        # Setup update timer for piece tracking colors
        self._setup_update_timer()

        logger.info("Sorting GUI initialized")

    # ========================================================================
    # UI SETUP
    # ========================================================================

    def _setup_ui(self):
        """
        Create the main UI layout.

        ARCHITECTURE NOTE:
        The camera view is now created using CameraViewSorting, which
        automatically loads its ROI and zone configuration from the
        config manager. No manual configuration is needed.
        """
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
        # ARCHITECTURE IMPROVEMENT: CameraViewSorting auto-configures ROI and zones
        # from config manager - no manual setup required!
        self.camera_view = CameraViewSorting(
            config_manager=self.orchestrator.config_manager,
            detector_coordinator=self.detector,
            orchestrator=self.orchestrator
        )
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

        logger.info("UI layout created with auto-configured camera view")

    # ========================================================================
    # SIGNAL CONNECTIONS (Thread Safety)
    # ========================================================================

    def _connect_signals(self):
        """
        Connect internal signals to GUI update slots.

        THREAD SAFETY:
        Qt's signal/slot mechanism is thread-safe. Signals emitted from
        background threads are automatically queued and processed in the
        GUI thread, preventing QPainter errors and race conditions.
        """
        self._bin_assigned_signal.connect(self._update_bin_assignment)
        self._piece_identified_signal.connect(self._update_piece_display)
        self._piece_sorted_signal.connect(self._update_bin_capacity)

        logger.info("Internal signals connected for thread-safe GUI updates")

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
    # CALLBACK METHODS (called BY modules from any thread)
    # ========================================================================
    #
    # NOTE: These callbacks may be called from non-GUI threads.
    # They emit signals which are safely queued to the GUI thread.
    #

    def on_bin_assigned(self, bin_number: int, category: str):
        """
        Callback: Called when a bin is assigned to a category.

        TRIGGERED BY: bin_assignment_module._notify_assignment_callbacks()
        WHEN: New category is dynamically assigned OR pre-assignments loaded
        THREAD SAFETY: May be called from any thread - emits signal for GUI thread

        Args:
            bin_number: Bin index (0-based)
            category: Category name assigned to bin
        """
        logger.debug(f"GUI callback: Bin {bin_number} assigned to '{category}' (emitting signal)")
        self._bin_assigned_signal.emit(bin_number, category)

    def on_piece_identified(self, identified_piece):
        """
        Callback: Called when a piece is identified.

        TRIGGERED BY: processing_coordinator._notify_identification_complete()
        WHEN: Piece identification completes (success or unknown)
        THREAD SAFETY: May be called from any thread - emits signal for GUI thread

        Args:
            identified_piece: IdentifiedPiece object with all data
        """
        logger.debug(f"GUI callback: Piece identified - {identified_piece.name} (emitting signal)")
        self._piece_identified_signal.emit(identified_piece)

    def on_piece_sorted(self, bin_number: int):
        """
        Callback: Called when a piece is sorted into a bin.

        TRIGGERED BY: hardware_controller._notify_sort_callbacks()
        WHEN: Servo moves to direct piece into bin
        THREAD SAFETY: May be called from any thread - emits signal for GUI thread

        Args:
            bin_number: Bin index (0-based) where piece was sorted
        """
        logger.debug(f"GUI callback: Piece sorted into Bin {bin_number} (emitting signal)")
        self._piece_sorted_signal.emit(bin_number)

    # ========================================================================
    # GUI UPDATE SLOTS (always run in GUI thread)
    # ========================================================================
    #
    # These methods perform actual GUI updates and are connected to signals.
    # Qt automatically ensures they run in the GUI thread.
    #

    def _update_bin_assignment(self, bin_number: int, category: str):
        """
        Slot: Update bin assignment display (thread-safe).

        CALLED BY: _bin_assigned_signal (always in GUI thread)

        Args:
            bin_number: Bin index (0-based)
            category: Category name assigned to bin
        """
        logger.info(f"GUI update: Bin {bin_number} → '{category}'")
        self.bin_status_panel.update_bin_assignment(bin_number, category)

    def _update_piece_display(self, identified_piece):
        """
        Slot: Update recently processed piece display (thread-safe).

        CALLED BY: _piece_identified_signal (always in GUI thread)

        Args:
            identified_piece: IdentifiedPiece object with all data
        """
        logger.info(f"GUI update: Display piece {identified_piece.name}")
        self.recent_piece_panel.display_piece(identified_piece)

    def _update_bin_capacity(self, bin_number: int):
        """
        Slot: Update bin capacity display (thread-safe).

        CALLED BY: _piece_sorted_signal (always in GUI thread)

        Args:
            bin_number: Bin index (0-based) where piece was sorted
        """
        logger.info(f"GUI update: Bin {bin_number} capacity")
        self.bin_status_panel.increment_bin_capacity(bin_number)

    # ========================================================================
    # CAMERA FEED MANAGEMENT
    # ========================================================================

    def _start_camera_feed(self):
        """
        Start camera feed to the camera view widget.

        Registers the camera_view as a consumer of camera frames.
        The CameraViewSorting widget handles all frame processing internally,
        including ROI/zone overlays and color-coded piece tracking.
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
        Setup timer to periodically update piece tracking state.

        This timer calls _update_piece_colors() which provides the
        CameraViewSorting widget with current piece tracking dictionaries.
        The widget uses these to determine proper color-coding for each piece.

        ARCHITECTURE NOTE:
        The widget handles all color logic internally. We just provide it
        with the current state via update_piece_tracking().
        """
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_piece_colors)
        self.update_timer.start(100)  # Update every 100ms

    def _update_piece_colors(self):
        """
        Update piece tracking state in camera view.

        Called periodically by timer. Fetches current tracked pieces
        and identified pieces, then updates camera view for color-coding.

        ARCHITECTURE IMPROVEMENT:
        The CameraViewSorting widget handles all color logic internally.
        We just provide it with the current state dictionaries, and it
        determines the appropriate color for each piece based on its
        processing stage (detected/captured/identified/unknown).

        This separation of concerns keeps widget logic in the widget module
        and GUI logic in the GUI module.
        """
        try:
            # Check if detector exists (won't exist in test mode)
            if self.detector is None:
                return  # Skip update if no detector

            # Get tracked pieces from detector
            tracked_pieces = self.detector.get_tracked_pieces()

            # Convert list to dict for lookup
            # The widget uses this dict to check piece capture status
            tracked_dict = {p.id: p for p in tracked_pieces}

            # Get identified pieces from orchestrator
            # The widget uses this dict to check piece identification status
            identified_dict = {}
            if self.orchestrator and hasattr(self.orchestrator, 'get_identified_pieces'):
                identified_dict = self.orchestrator.get_identified_pieces()

            # Update camera view's piece tracking state
            # The CameraViewSorting widget uses these dicts internally to
            # determine the appropriate color for each piece's bounding box
            self.camera_view.update_piece_tracking(tracked_dict, identified_dict)

            # Convert tracked pieces to dict format for display
            # The widget needs pieces in this format for drawing bounding boxes
            tracked_pieces_dicts = []
            for piece in tracked_pieces:
                tracked_pieces_dicts.append({
                    'id': piece.id,
                    'bbox': piece.bbox,
                    'center': piece.center
                })

            # Set tracked pieces for drawing
            # This tells the widget which pieces to draw on the frame
            self.camera_view.set_tracked_pieces(tracked_pieces_dicts)

        except Exception as e:  # ✅ CORRECT - aligned with try
            logger.error(f"Error updating piece tracking: {e}", exc_info=True)

    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================

    def closeEvent(self, event):
        """
        Handle window close event.

        Performs cleanup operations when the GUI window is closed:
        - Stops the piece tracking update timer
        - Unregisters as a camera frame consumer
        """
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
    This allows testing the GUI appearance and basic functionality without
    requiring the full sorting system to be running.

    NOTE: This test mode will not display ROI overlays since the mock
    config manager returns empty configuration.
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create mock orchestrator with dummy modules
    class MockConfigManager:
        """Mock config manager for testing."""

        def get_module_config(self, module_name):
            """Return dummy config."""
            if module_name == "detector_roi":
                return {"x": 100, "y": 50, "w": 1720, "h": 980}
            elif module_name == "detector":
                return {"zones": {"entry_percentage": 15, "exit_percentage": 15}}
            return {}

    class MockOrchestrator:
        """Mock orchestrator for testing."""

        def __init__(self):
            self.config_manager = MockConfigManager()
            self.camera = None
            self.detector_coordinator = None
            self.processing_coordinator = None
            self.hardware_coordinator = None
            self.bin_assignment_module = None
            self.bin_capacity_manager = MockBinCapacity()

        def get_identified_pieces(self):
            """Return empty dict for testing."""
            return {}

    class MockBinCapacity:
        """Mock bin capacity manager for testing."""

        def get_bin_capacity_percentage(self, bin_num):
            return 0

        def reset_bin(self, bin_num):
            pass

    app = QApplication(sys.argv)

    orchestrator = MockOrchestrator()
    gui = SortingGUI(orchestrator)
    gui.show()

    logger.info("Sorting GUI test mode - UI only, no actual functionality")
    logger.info("Note: Camera view will show 'No camera feed' message")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()