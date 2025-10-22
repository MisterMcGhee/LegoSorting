"""
bin_status.py - Bin status display widget for LEGO Sorting Machine

This module provides a reusable widget for displaying the status of a single bin,
including category assignment, capacity tracking, and reset functionality.

WIDGET: BinStatusWidget
- Displays bin number and assigned category
- Shows current count and capacity (e.g., "35/50")
- Visual progress bar with color coding (green/yellow/red)
- Reset button to clear bin capacity

USAGE:
    from GUI.widgets.bin_status import BinStatusWidget

    bin_widget = BinStatusWidget(bin_number=0)
    bin_widget.update_assignment("Brick 2x4")
    bin_widget.update_capacity(count=35, max_capacity=50, percentage=70)

    # Connect to reset signal
    bin_widget.reset_requested.connect(on_reset_handler)
"""

import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar
from PyQt5.QtCore import Qt, pyqtSignal

from GUI.widgets.gui_styles import SortingGUIStyles

logger = logging.getLogger(__name__)


class BinStatusWidget(QWidget):
    """
    Widget displaying status for a single bin.

    Shows:
    - Bin number
    - Assigned category (or "Unassigned")
    - Current count and max capacity (e.g., "35/50")
    - Capacity percentage with color-coded progress bar
    - Reset button to clear bin

    The capacity bar changes color based on fill level:
    - Green: < 75%
    - Yellow: 75-90%
    - Red: > 90%

    SIGNALS:
    - reset_requested(int): Emitted when reset button clicked (passes bin_number)
    """

    # Signal emitted when reset button clicked
    reset_requested = pyqtSignal(int)  # Emits bin_number

    def __init__(self, bin_number: int, parent=None):
        """
        Initialize bin status widget.

        Args:
            bin_number: The bin number (0-based index, but displayed as 1-based)
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.bin_number = bin_number
        self.current_capacity = 0
        self.current_count = 0
        self.max_capacity = 50  # Default, will be updated from config
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

        # Count label (shows "0/50" format)
        self.count_label = QLabel(f"0/{self.max_capacity}")
        self.count_label.setStyleSheet(SortingGUIStyles.get_label_style(
            size=11, bold=True, color=SortingGUIStyles.TEXT_SECONDARY
        ))
        self.count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.count_label)

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

        Args:
            category: Category name assigned to this bin
        """
        self.assigned_category = category
        self.label.setText(f"Bin {self.bin_number + 1}: {category}")
        logger.debug(f"Bin {self.bin_number + 1} assigned to '{category}'")

    def update_capacity(self, count: int, max_capacity: int, percentage: int):
        """
        Update the bin's capacity display.

        Args:
            count: Current number of pieces in bin
            max_capacity: Maximum capacity of the bin
            percentage: Fill percentage (0-100)
        """
        self.current_count = count
        self.max_capacity = max_capacity
        self.current_capacity = percentage

        # Update count label
        self.count_label.setText(f"{count}/{max_capacity}")

        # Update progress bar
        self.progress_bar.setValue(percentage)
        self._update_progress_bar_style()

        # Enable reset button if bin has content
        self.reset_button.setEnabled(percentage > 0)

        logger.debug(f"Bin {self.bin_number + 1} capacity: {count}/{max_capacity} ({percentage}%)")

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
        self.update_capacity(0, self.max_capacity, 0)
        logger.info(f"Bin {self.bin_number + 1} capacity reset to 0%")
