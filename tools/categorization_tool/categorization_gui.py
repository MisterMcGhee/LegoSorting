# tools/categorization_tool/categorization_gui.py
"""
categorization_gui.py - GUI components for piece categorization

This module contains all PyQt5 GUI components for the categorization tool.
It has NO business logic - all operations delegate to the controller.

RESPONSIBILITIES:
- Create and layout PyQt5 widgets
- Handle user interaction events
- Display data from controller
- Call controller methods for business operations

DOES NOT:
- Perform CSV operations directly
- Validate data (delegates to controller)
- Make business decisions
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QComboBox, QPushButton,
                             QProgressBar, QMessageBox, QFormLayout, QLineEdit)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont
import logging
from typing import Optional

# Import business logic
from tools.categorization_tool.categorization_logic import (
    CategorizationController,
    UnknownPiece
)

logger = logging.getLogger(__name__)


# ============================================================================
# STYLING CONSTANTS
# ============================================================================

class ToolStyles:
    """Styling constants matching existing GUI style."""

    # Color scheme
    COLOR_PRIMARY = "#27AE60"  # Green - save action
    COLOR_PRIMARY_HOVER = "#229954"
    COLOR_INFO = "#3498DB"  # Blue - navigation
    COLOR_INFO_HOVER = "#2980B9"
    COLOR_NEUTRAL = "#95A5A6"  # Gray - skip
    COLOR_NEUTRAL_HOVER = "#7F8C8D"

    @staticmethod
    def get_button_style(bg_color: str, hover_color: str) -> str:
        """Get button stylesheet."""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                padding: 8px 16px;
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
        """Get groupbox stylesheet."""
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

    # Dark theme
    DARK_THEME = """
        QMainWindow {
            background-color: #2b2b2b;
        }
        QLabel {
            color: #ffffff;
        }
        QComboBox, QLineEdit {
            background-color: #3d3d3d;
            color: white;
            border: 1px solid #666;
            padding: 6px 8px;
            border-radius: 3px;
            font-size: 11px;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ccc;
            margin-right: 5px;
        }
        QComboBox:hover {
            background-color: #4a4a4a;
            border: 1px solid #777;
        }
        QComboBox QAbstractItemView {
            background-color: #3d3d3d;
            color: white;
            selection-background-color: #4a7ba7;
            selection-color: white;
            border: 1px solid #666;
        }
        QLineEdit:hover {
            background-color: #4a4a4a;
            border: 1px solid #777;
        }
        QLineEdit:focus {
            background-color: #4a4a4a;
            border: 1px solid #3498DB;
        }
        QProgressBar {
            border: 2px solid #555;
            border-radius: 5px;
            text-align: center;
            color: white;
            background-color: #333;
        }
        QProgressBar::chunk {
            background-color: #27AE60;
        }
    """


# ============================================================================
# MAIN GUI WINDOW
# ============================================================================

class PieceCategorizationGUI(QMainWindow):
    """
    Main GUI window for piece categorization.

    This window presents pieces one at a time and allows the user
    to assign categories. All business logic is delegated to the controller.

    Signals:
        categorization_complete(int): Emitted when tool closes with count
    """

    # Signal emitted when tool closes
    categorization_complete = pyqtSignal(int)

    def __init__(self, controller: CategorizationController):
        """
        Initialize the GUI window.

        Args:
            controller: Business logic controller
        """
        super().__init__()

        logger.info("Initializing PieceCategorizationGUI")

        # Store controller
        self.controller = controller

        # State
        self.unknown_pieces = []
        self.current_index = 0
        self.total_categorized = 0
        self.new_categories_added = False

        # UI references (set in init_ui)
        self.progress_bar = None
        self.progress_label = None
        self.element_id_label = None
        self.name_label = None
        self.encounter_label = None
        self.dates_label = None
        self.primary_combo = None
        self.secondary_combo = None
        self.tertiary_combo = None
        self.primary_input = None
        self.secondary_input = None
        self.tertiary_input = None
        self.prev_btn = None
        self.skip_btn = None
        self.save_btn = None
        self.close_btn = None
        self.image_placeholder = None

        # Setup UI
        self.init_ui()

        # Load data
        if self.load_unknown_pieces():
            self.display_current_piece()
        else:
            self.show_empty_state()

        logger.info("GUI initialized")

    # ========================================================================
    # UI INITIALIZATION
    # ========================================================================

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Missing Piece Categorization Tool")
        self.setStyleSheet(ToolStyles.DARK_THEME)
        self.setGeometry(100, 100, 1000, 750)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Add sections
        main_layout.addWidget(self.create_progress_section())
        main_layout.addWidget(self.create_piece_info_section())
        main_layout.addWidget(self.create_category_section())
        main_layout.addStretch()
        main_layout.addWidget(self.create_navigation_section())

    def create_progress_section(self) -> QWidget:
        """Create progress bar section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)

        self.progress_label = QLabel("Loading...")
        self.progress_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: bold;")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v of %m pieces categorized")
        layout.addWidget(self.progress_bar)

        return widget

    def create_piece_info_section(self) -> QGroupBox:
        """Create piece information display section."""
        group = QGroupBox("Piece Information")
        group.setStyleSheet(ToolStyles.get_groupbox_style())
        layout = QHBoxLayout(group)
        layout.setSpacing(15)

        # Image placeholder
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self.image_placeholder = QLabel("🖼\n\nImage Preview\nComing in Phase 2")
        self.image_placeholder.setAlignment(Qt.AlignCenter)
        self.image_placeholder.setStyleSheet("""
            QLabel {
                background-color: #3d3d3d;
                border: 2px dashed #666;
                border-radius: 5px;
                color: #888;
                font-size: 14px;
                padding: 20px;
            }
        """)
        self.image_placeholder.setFixedSize(250, 250)
        image_layout.addWidget(self.image_placeholder)

        layout.addWidget(image_container)

        # Piece details
        details_container = QWidget()
        details_container.setMinimumWidth(500)  # Ensure enough width for long names
        details_layout = QFormLayout(details_container)
        details_layout.setSpacing(12)
        details_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        details_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Element ID (large and prominent)
        self.element_id_label = QLabel("---")
        self.element_id_label.setStyleSheet("color: #3498DB; font-size: 18px; font-weight: bold;")
        details_layout.addRow("Element ID:", self.element_id_label)

        # Piece name (allow for long names with wrapping)
        self.name_label = QLabel("---")
        self.name_label.setStyleSheet("""
            color: #ffffff; 
            font-size: 14px;
            padding: 5px;
            background-color: #3d3d3d;
            border-radius: 3px;
        """)
        self.name_label.setWordWrap(True)
        self.name_label.setMinimumHeight(40)  # Allow for multiple lines
        self.name_label.setMaximumHeight(80)  # But not too tall
        details_layout.addRow("Name:", self.name_label)

        # Encounter count
        self.encounter_label = QLabel("---")
        self.encounter_label.setStyleSheet("color: #F39C12; font-size: 12px; font-weight: bold;")
        details_layout.addRow("Encounters:", self.encounter_label)

        # Dates
        self.dates_label = QLabel("---")
        self.dates_label.setStyleSheet("color: #7F8C8D; font-size: 11px;")
        self.dates_label.setWordWrap(True)
        details_layout.addRow("Dates:", self.dates_label)

        layout.addWidget(details_container)
        layout.setStretch(1, 1)  # Give details more space to expand

        return group

    def create_category_section(self) -> QGroupBox:
        """Create category selection section."""
        group = QGroupBox("Category Selection")
        group.setStyleSheet(ToolStyles.get_groupbox_style())
        layout = QFormLayout(group)
        layout.setSpacing(10)
        layout.setLabelAlignment(Qt.AlignRight)

        # Helper to create combo + input row
        def create_category_row(label_text: str) -> tuple:
            combo = QComboBox()
            combo.setMinimumWidth(200)

            line_edit = QLineEdit()
            line_edit.setPlaceholderText("Or type new category...")
            line_edit.setMinimumWidth(200)

            container = QWidget()
            row_layout = QHBoxLayout(container)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)
            row_layout.addWidget(combo)
            row_layout.addWidget(QLabel("or"))
            row_layout.addWidget(line_edit)
            row_layout.addStretch()

            return combo, line_edit, container

        # Primary category
        self.primary_combo, self.primary_input, primary_row = create_category_row("Primary")
        self.primary_combo.currentTextChanged.connect(self.on_primary_changed)
        layout.addRow("Primary Category: *", primary_row)

        # Secondary category
        self.secondary_combo, self.secondary_input, secondary_row = create_category_row("Secondary")
        self.secondary_combo.currentTextChanged.connect(self.on_secondary_changed)
        layout.addRow("Secondary Category: *", secondary_row)

        # Tertiary category
        self.tertiary_combo, self.tertiary_input, tertiary_row = create_category_row("Tertiary")
        layout.addRow("Tertiary Category:", tertiary_row)

        # Info
        info = QLabel("* Required fields • Tertiary is optional")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        layout.addRow("", info)

        tip = QLabel("💡 Tip: Type in the text field to add a new category. "
                     "It will auto-reload for the next piece.")
        tip.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        tip.setWordWrap(True)
        layout.addRow("", tip)

        return group

    def create_navigation_section(self) -> QWidget:
        """Create navigation buttons section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        info = QLabel("Changes are saved immediately when you click 'Save & Next'")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.setToolTip("Go to previous piece")
        self.prev_btn.clicked.connect(self.go_to_previous)
        self.prev_btn.setStyleSheet(ToolStyles.get_button_style(
            ToolStyles.COLOR_INFO, ToolStyles.COLOR_INFO_HOVER
        ))
        button_layout.addWidget(self.prev_btn)

        self.skip_btn = QPushButton("⏭ Skip")
        self.skip_btn.setToolTip("Skip this piece for now (move to end of queue)")
        self.skip_btn.clicked.connect(self.skip_current)
        self.skip_btn.setStyleSheet(ToolStyles.get_button_style(
            ToolStyles.COLOR_NEUTRAL, ToolStyles.COLOR_NEUTRAL_HOVER
        ))
        button_layout.addWidget(self.skip_btn)

        button_layout.addStretch()

        self.save_btn = QPushButton("✓ Save & Next ▶")
        self.save_btn.setToolTip("Save categorization and move to next piece")
        self.save_btn.clicked.connect(self.save_and_next)
        self.save_btn.setStyleSheet(ToolStyles.get_button_style(
            ToolStyles.COLOR_PRIMARY, ToolStyles.COLOR_PRIMARY_HOVER
        ))
        self.save_btn.setMinimumWidth(150)
        button_layout.addWidget(self.save_btn)

        self.close_btn = QPushButton("✓ Close")
        self.close_btn.setToolTip("Close the tool")
        self.close_btn.clicked.connect(self.close_tool)
        self.close_btn.setStyleSheet(ToolStyles.get_button_style(
            ToolStyles.COLOR_NEUTRAL, ToolStyles.COLOR_NEUTRAL_HOVER
        ))
        self.close_btn.setVisible(False)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        return widget

    # ========================================================================
    # DATA LOADING AND DISPLAY
    # ========================================================================

    def load_unknown_pieces(self) -> bool:
        """
        Load unknown pieces from controller.

        Returns:
            True if pieces loaded, False otherwise
        """
        try:
            self.unknown_pieces = self.controller.load_unknown_pieces()

            if not self.unknown_pieces:
                return False

            self.progress_bar.setMaximum(len(self.unknown_pieces))
            self.progress_bar.setValue(0)
            self.update_progress_label()

            return True

        except FileNotFoundError as e:
            QMessageBox.information(
                self,
                "No Unknown Pieces",
                f"The unknown_pieces.csv file was not found.\n\n"
                f"This file is created when pieces are encountered that aren't "
                f"in the database."
            )
            return False
        except ValueError as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load unknown pieces:\n\n{e}"
            )
            return False

    def display_current_piece(self):
        """Display the piece at current_index."""
        if not self.unknown_pieces or self.current_index >= len(self.unknown_pieces):
            return

        piece = self.unknown_pieces[self.current_index]

        # Update labels
        self.element_id_label.setText(piece.element_id)
        self.name_label.setText(piece.name or "Unknown Name")
        self.encounter_label.setText(f"{piece.encounter_count} time(s)")
        self.dates_label.setText(f"First: {piece.first_seen}\nLast: {piece.last_seen}")

        # Reset categories
        self.reset_category_selections()
        self.populate_primary_dropdown()
        self.update_progress_label()

        # Reload categories if new ones were added
        if self.new_categories_added:
            self.controller.reload_category_hierarchy()
            self.new_categories_added = False

    def reset_category_selections(self):
        """Clear all category selections."""
        self.primary_combo.blockSignals(True)
        self.secondary_combo.blockSignals(True)
        self.tertiary_combo.blockSignals(True)

        self.primary_combo.clear()
        self.secondary_combo.clear()
        self.tertiary_combo.clear()

        self.primary_input.clear()
        self.secondary_input.clear()
        self.tertiary_input.clear()

        self.primary_combo.blockSignals(False)
        self.secondary_combo.blockSignals(False)
        self.tertiary_combo.blockSignals(False)

    def update_progress_label(self):
        """Update progress display."""
        total = len(self.unknown_pieces)
        remaining = total - self.total_categorized

        self.progress_label.setText(
            f"Progress: {self.total_categorized} of {total} pieces categorized "
            f"({remaining} remaining)"
        )
        self.progress_bar.setValue(self.total_categorized)

    def show_empty_state(self):
        """Show empty state when no pieces remain."""
        self.progress_label.setText("✓ All pieces categorized!")
        self.progress_bar.setValue(self.progress_bar.maximum())

        self.element_id_label.setText("---")
        self.name_label.setText("No pieces to categorize")
        self.encounter_label.setText("---")
        self.dates_label.setText("---")

        # Disable controls
        self.primary_combo.setEnabled(False)
        self.secondary_combo.setEnabled(False)
        self.tertiary_combo.setEnabled(False)
        self.primary_input.setEnabled(False)
        self.secondary_input.setEnabled(False)
        self.tertiary_input.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self.close_btn.setVisible(True)

        QMessageBox.information(
            self,
            "Complete",
            f"✓ All unknown pieces have been categorized!\n\n"
            f"Total categorized this session: {self.total_categorized}"
        )

    # ========================================================================
    # CATEGORY DROPDOWNS
    # ========================================================================

    def populate_primary_dropdown(self):
        """Populate primary category dropdown."""
        primaries = self.controller.get_primary_categories()

        self.primary_combo.blockSignals(True)
        self.primary_combo.clear()
        self.primary_combo.addItem("")
        self.primary_combo.addItems(primaries)
        self.primary_combo.blockSignals(False)

    def on_primary_changed(self):
        """Handle primary category change."""
        primary = self.primary_combo.currentText()

        self.secondary_combo.blockSignals(True)
        self.tertiary_combo.blockSignals(True)

        self.secondary_combo.clear()
        self.tertiary_combo.clear()

        self.secondary_combo.blockSignals(False)
        self.tertiary_combo.blockSignals(False)

        if primary:
            secondaries = self.controller.get_secondary_categories(primary)
            self.secondary_combo.blockSignals(True)
            self.secondary_combo.addItem("")
            self.secondary_combo.addItems(secondaries)
            self.secondary_combo.blockSignals(False)

    def on_secondary_changed(self):
        """Handle secondary category change."""
        primary = self.primary_combo.currentText()
        secondary = self.secondary_combo.currentText()

        self.tertiary_combo.blockSignals(True)
        self.tertiary_combo.clear()
        self.tertiary_combo.blockSignals(False)

        if primary and secondary:
            tertiaries = self.controller.get_tertiary_categories(primary, secondary)
            self.tertiary_combo.blockSignals(True)
            self.tertiary_combo.addItem("")
            self.tertiary_combo.addItems(tertiaries)
            self.tertiary_combo.blockSignals(False)

    # ========================================================================
    # NAVIGATION
    # ========================================================================

    def go_to_previous(self):
        """Go to previous piece."""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_piece()

    def skip_current(self):
        """Skip current piece (move to end)."""
        if not self.unknown_pieces:
            return

        piece = self.unknown_pieces.pop(self.current_index)
        self.unknown_pieces.append(piece)

        if self.current_index >= len(self.unknown_pieces):
            self.current_index = 0

        self.display_current_piece()

    def save_and_next(self):
        """Save categorization and move to next piece."""
        if not self.unknown_pieces:
            return

        piece = self.unknown_pieces[self.current_index]

        # Get categories
        primary = self.primary_input.text().strip() or self.primary_combo.currentText()
        secondary = self.secondary_input.text().strip() or self.secondary_combo.currentText()
        tertiary = self.tertiary_input.text().strip() or self.tertiary_combo.currentText()

        # Check for new categories
        if (self.primary_input.text().strip() or
                self.secondary_input.text().strip() or
                self.tertiary_input.text().strip()):
            self.new_categories_added = True

        # Check for duplicates
        allow_update = False
        if self.controller.database_manager.element_exists(piece.element_id):
            reply = QMessageBox.question(
                self,
                "Duplicate Element ID",
                f"Element ID {piece.element_id} already exists in the database.\n\n"
                "Do you want to update it with the new categorization?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            allow_update = True

        # Save via controller
        success, message = self.controller.save_categorization(
            piece.element_id,
            piece.name,
            primary,
            secondary,
            tertiary,
            allow_update
        )

        if not success:
            QMessageBox.warning(self, "Save Error", message)
            return

        # Remove from list and advance
        self.unknown_pieces.pop(self.current_index)
        self.total_categorized += 1

        if not self.unknown_pieces:
            self.show_empty_state()
        else:
            if self.current_index >= len(self.unknown_pieces):
                self.current_index = 0
            self.display_current_piece()

    # ========================================================================
    # WINDOW MANAGEMENT
    # ========================================================================

    def close_tool(self):
        """Close the tool."""
        self.categorization_complete.emit(self.total_categorized)
        self.close()

    def closeEvent(self, event):
        """Handle window close."""
        self.categorization_complete.emit(self.total_categorized)
        event.accept()