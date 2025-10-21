"""
processing_tab.py - Processing/sorting strategy configuration tab

FILE LOCATION: GUI/config_tabs/processing_tab.py

This tab provides sorting strategy and bin assignment configuration with
dynamic category dropdowns based on the selected strategy. It integrates with
the category hierarchy service to show only valid category options.

Features:
- Three sorting strategies (Primary, Secondary, Tertiary)
- Dynamic category dropdowns that update based on strategy and selections
- Pre-assignment table for mapping categories to specific bins
- Bin capacity and overflow configuration
- Queue size configuration for processing pipeline
- Confidence threshold settings
- Comprehensive validation of all settings

Sorting Strategies:
    - Primary: Sort all pieces by primary category (Basic, Technic, etc.)
    - Secondary: Sort pieces matching target primary by secondary category
    - Tertiary: Sort pieces matching target primary+secondary by tertiary category

Configuration Mapping:
    {
        "sorting": {
            "strategy": "primary",
            "target_primary_category": "",
            "target_secondary_category": "",
            "max_bins": 9,
            "overflow_bin": 0,
            "confidence_threshold": 0.7,
            "pre_assignments": {
                "Basic": 1,
                "Technic": 2
            },
            "max_pieces_per_bin": 50,
            "bin_warning_threshold": 0.8
        },
        "processing_queue": {
            "queue_size": 100
        }
    }
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QComboBox, QPushButton, QSpinBox,
                             QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                             QMessageBox, QRadioButton, QButtonGroup, QDoubleSpinBox,
                             QAbstractItemView, QDialog, QDialogButtonBox)
from PyQt5.QtCore import pyqtSignal, Qt
from typing import Dict, Any, Optional, List
import logging

# Import base class
from GUI.config_tabs.base_tab import BaseConfigTab
from enhanced_config_manager import ModuleConfig

# Initialize logger
logger = logging.getLogger(__name__)


class ProcessingConfigTab(BaseConfigTab):
    """
    Processing/sorting strategy configuration tab.

    This tab allows users to:
    - Select sorting strategy (Primary, Secondary, or Tertiary)
    - Configure target categories for secondary/tertiary strategies
    - Set maximum bins and overflow bin
    - Configure queue size for processing pipeline
    - Configure confidence thresholds
    - Create pre-assignments mapping categories to specific bins
    - Set bin capacity and warning thresholds

    The UI dynamically updates based on the selected strategy, showing only
    relevant controls and populating dropdowns with valid categories from
    the category hierarchy service.

    Signals:
        strategy_changed(str): Emitted when sorting strategy changes
        bins_changed(int): Emitted when max_bins changes
        target_categories_changed(str, str): Emitted when target categories change
    """

    # Custom signals
    strategy_changed = pyqtSignal(str)  # strategy name
    bins_changed = pyqtSignal(int)  # max_bins
    target_categories_changed = pyqtSignal(str, str)  # (primary, secondary)

    def __init__(self, config_manager, category_service=None, parent=None):
        """
        Initialize processing configuration tab.

        Args:
            config_manager: Configuration manager instance
            category_service: Category hierarchy service for dynamic dropdowns
            parent: Parent widget
        """
        # Store category service reference
        self.category_service = category_service

        # Log what we received
        if self.category_service is None:
            logger.warning("⚠️  ProcessingConfigTab initialized WITHOUT category service")
            logger.warning("    Target category dropdowns will not populate")
        else:
            logger.info("✓ ProcessingConfigTab initialized with category service")

        # Pre-assignment tracking
        self.pre_assignments = {}  # Dict[str, int] - category name -> bin number

        # Call parent init
        super().__init__(config_manager, parent)

        # Initialize UI and load config
        self.init_ui()
        self.load_config()

        # Populate initial category options
        self.update_available_categories()

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (Required by BaseConfigTab)
    # ========================================================================

    def get_module_name(self) -> str:
        """Return the configuration module name this tab manages."""
        return ModuleConfig.SORTING.value

    def init_ui(self):
        """Create the processing configuration user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Title
        title = QLabel("Processing & Sorting Configuration")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #2E86AB;
                color: white;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(title)

        # Main content in horizontal layout
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        # Left panel: Strategy, categories, and bins
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        strategy_group = self.create_strategy_group()
        left_layout.addWidget(strategy_group)

        category_group = self.create_category_group()
        left_layout.addWidget(category_group)

        bins_group = self.create_bins_group()
        left_layout.addWidget(bins_group)

        left_layout.addStretch()
        content_layout.addWidget(left_panel, stretch=1)

        # Right panel: Pre-assignments and thresholds
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        preassign_group = self.create_preassignment_group()
        right_layout.addWidget(preassign_group, stretch=1)

        threshold_group = self.create_threshold_group()
        right_layout.addWidget(threshold_group)

        content_layout.addWidget(right_panel, stretch=1)

        # Info panel at bottom
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel)

        self.logger.info("Processing tab UI initialized")

    def create_strategy_group(self) -> QGroupBox:
        """Create sorting strategy selection group."""
        group = QGroupBox("Sorting Strategy")
        layout = QVBoxLayout()

        # Strategy radio buttons
        self.strategy_group = QButtonGroup()

        self.primary_radio = QRadioButton("Primary")
        self.primary_radio.setToolTip("Sort all pieces by primary category (Basic, Technic, etc.)")
        self.strategy_group.addButton(self.primary_radio, 0)
        layout.addWidget(self.primary_radio)

        self.secondary_radio = QRadioButton("Secondary")
        self.secondary_radio.setToolTip("Sort pieces matching target primary by secondary category")
        self.strategy_group.addButton(self.secondary_radio, 1)
        layout.addWidget(self.secondary_radio)

        self.tertiary_radio = QRadioButton("Tertiary")
        self.tertiary_radio.setToolTip("Sort pieces matching target primary+secondary by tertiary category")
        self.strategy_group.addButton(self.tertiary_radio, 2)
        layout.addWidget(self.tertiary_radio)

        # Set default
        self.primary_radio.setChecked(True)

        # Connect signal
        self.strategy_group.buttonClicked.connect(self.on_strategy_changed)

        # Description label
        self.strategy_desc = QLabel()
        self.strategy_desc.setWordWrap(True)
        self.strategy_desc.setStyleSheet("color: #7F8C8D; font-style: italic; padding: 5px;")
        self.update_strategy_description()
        layout.addWidget(self.strategy_desc)

        group.setLayout(layout)
        return group

    def create_category_group(self) -> QGroupBox:
        """Create target category selection group."""
        group = QGroupBox("Target Categories")
        layout = QFormLayout()

        # Primary category dropdown (for Secondary & Tertiary strategies)
        self.target_primary_label = QLabel("Primary Category:")
        self.target_primary_combo = QComboBox()
        self.target_primary_combo.setToolTip("Select primary category to focus on")
        self.target_primary_combo.currentTextChanged.connect(self.on_primary_category_changed)
        layout.addRow(self.target_primary_label, self.target_primary_combo)

        # Secondary category dropdown (for Tertiary strategy only)
        self.target_secondary_label = QLabel("Secondary Category:")
        self.target_secondary_combo = QComboBox()
        self.target_secondary_combo.setToolTip("Select secondary category to focus on")
        self.target_secondary_combo.currentTextChanged.connect(self.on_secondary_category_changed)
        layout.addRow(self.target_secondary_label, self.target_secondary_combo)

        # Info label
        self.category_info = QLabel("Select strategy above to configure target categories")
        self.category_info.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        self.category_info.setWordWrap(True)
        layout.addRow("", self.category_info)

        # Initially hide category selection (shown for secondary/tertiary)
        self.target_primary_label.setVisible(False)
        self.target_primary_combo.setVisible(False)
        self.target_secondary_label.setVisible(False)
        self.target_secondary_combo.setVisible(False)

        group.setLayout(layout)
        return group

    def create_bins_group(self) -> QGroupBox:
        """Create bin and queue configuration group."""
        group = QGroupBox("Bin & Queue Configuration")
        layout = QFormLayout()

        # Maximum bins (total bins including overflow)
        self.max_bins_spin = QSpinBox()
        self.max_bins_spin.setRange(2, 13)  # Minimum 2 (1 overflow + 1 sorting)
        self.max_bins_spin.setValue(9)
        self.max_bins_spin.setToolTip(
            "Total number of physical bins (e.g., 9 means bins 0-8)\n"
            "Bin 0 is reserved for overflow, leaving N-1 bins for sorting"
        )
        self.max_bins_spin.valueChanged.connect(self.on_max_bins_changed)
        layout.addRow("Total Bins:", self.max_bins_spin)

        # Available bins info (clarifies the sorting bins)
        self.available_bins_label = QLabel("Sorting: Bins 1-8 (8 bins)")
        self.available_bins_label.setStyleSheet("color: #27AE60; font-weight: bold;")
        self.max_bins_spin.valueChanged.connect(self.update_available_bins_label)
        layout.addRow("", self.available_bins_label)

        # Overflow bin (read-only, always 0)
        self.overflow_bin_label = QLabel("Bin 0 (reserved)")
        self.overflow_bin_label.setStyleSheet("font-weight: bold; color: #E67E22;")
        layout.addRow("Overflow:", self.overflow_bin_label)

        # Separator for visual clarity
        separator1 = QLabel("─" * 40)
        separator1.setStyleSheet("color: #BDC3C7;")
        layout.addRow("", separator1)

        # Queue size configuration (processing pipeline)
        self.queue_size_spin = QSpinBox()
        self.queue_size_spin.setRange(10, 1000)
        self.queue_size_spin.setValue(100)
        self.queue_size_spin.setSuffix(" items")
        self.queue_size_spin.setToolTip(
            "Maximum number of pieces queued for processing.\n"
            "Larger values use more memory but handle bursts better.\n"
            "Recommended: 100-200 for normal operation"
        )
        self.queue_size_spin.valueChanged.connect(self.on_queue_size_changed)
        layout.addRow("Queue Size:", self.queue_size_spin)

        # Queue size warning
        self.queue_warning_label = QLabel("")
        self.queue_warning_label.setWordWrap(True)
        self.queue_warning_label.setStyleSheet("color: #E67E22; font-weight: bold;")
        layout.addRow("", self.queue_warning_label)

        # Separator for visual clarity
        separator2 = QLabel("─" * 40)
        separator2.setStyleSheet("color: #BDC3C7;")
        layout.addRow("", separator2)

        # Max pieces per bin
        self.max_pieces_spin = QSpinBox()
        self.max_pieces_spin.setRange(1, 200)
        self.max_pieces_spin.setValue(50)
        self.max_pieces_spin.setToolTip("Maximum pieces per bin before it's considered full")
        self.max_pieces_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Max Pieces/Bin:", self.max_pieces_spin)

        # Warning threshold
        self.warning_threshold_spin = QDoubleSpinBox()
        self.warning_threshold_spin.setRange(0.0, 1.0)
        self.warning_threshold_spin.setSingleStep(0.05)
        self.warning_threshold_spin.setValue(0.8)
        self.warning_threshold_spin.setDecimals(2)
        self.warning_threshold_spin.setToolTip("Warn when bin reaches this percentage of capacity")
        self.warning_threshold_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Warning Threshold:", self.warning_threshold_spin)

        # Info
        info = QLabel("Example: 9 total bins = Bin 0 (overflow) + Bins 1-8 (sorting)")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        info.setWordWrap(True)
        layout.addRow("", info)

        group.setLayout(layout)
        return group

    def create_threshold_group(self) -> QGroupBox:
        """Create confidence threshold group."""
        group = QGroupBox("Detection Thresholds")
        layout = QFormLayout()

        # Confidence threshold
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setToolTip("Minimum confidence for piece identification")
        self.confidence_spin.valueChanged.connect(self.mark_modified)
        layout.addRow("Confidence Threshold:", self.confidence_spin)

        # Info
        info = QLabel("Pieces below this confidence will go to overflow bin")
        info.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        info.setWordWrap(True)
        layout.addRow("", info)

        group.setLayout(layout)
        return group

    def create_preassignment_group(self) -> QGroupBox:
        """Create pre-assignment table group."""
        group = QGroupBox("Category Pre-Assignments")
        layout = QVBoxLayout()

        # Description
        desc = QLabel("Pre-assign specific categories to bins. These assignments take "
                      "priority over dynamic allocation.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #555; padding: 5px;")
        layout.addWidget(desc)

        # Table
        self.preassign_table = QTableWidget()
        self.preassign_table.setColumnCount(3)
        self.preassign_table.setHorizontalHeaderLabels(["Category", "Bin", "Actions"])
        self.preassign_table.horizontalHeader().setStretchLastSection(False)
        self.preassign_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.preassign_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.preassign_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.preassign_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preassign_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Set maximum height to reduce vertical space
        self.preassign_table.setMaximumHeight(200)

        layout.addWidget(self.preassign_table)

        # Add button row
        button_layout = QHBoxLayout()

        self.add_assignment_btn = QPushButton("➕ Add Assignment")
        self.add_assignment_btn.setToolTip("Add a new category-to-bin assignment")
        self.add_assignment_btn.clicked.connect(self.add_assignment)
        self.add_assignment_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        button_layout.addWidget(self.add_assignment_btn)

        button_layout.addStretch()

        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.setToolTip("Remove all pre-assignments")
        clear_btn.clicked.connect(self.clear_assignments)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        button_layout.addWidget(clear_btn)

        layout.addLayout(button_layout)

        # Current assignments summary
        self.assignment_summary = QLabel("0 pre-assignments")
        self.assignment_summary.setStyleSheet("color: #7F8C8D; font-style: italic; padding: 5px;")
        layout.addWidget(self.assignment_summary)

        group.setLayout(layout)
        return group

    def create_info_panel(self) -> QWidget:
        """Create information panel."""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        info = QLabel("💡 Tip: Pre-assignments ensure specific categories always go to designated bins")
        info.setStyleSheet("color: #7F8C8D; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)

        return panel

    # ========================================================================
    # STRATEGY AND CATEGORY MANAGEMENT
    # ========================================================================

    def on_strategy_changed(self):
        """Handle sorting strategy change."""
        strategy = self.get_current_strategy()

        self.logger.info(f"Strategy changed to: {strategy}")

        # Clear pre-assignments when strategy changes since they may no longer be valid
        if self.pre_assignments:
            self.logger.info("Clearing pre-assignments due to strategy change")
            self.pre_assignments.clear()
            self.preassign_table.setRowCount(0)
            self.update_assignment_summary()

        # Update UI visibility based on strategy
        is_secondary = strategy == "secondary"
        is_tertiary = strategy == "tertiary"

        # Show/hide target category controls
        self.target_primary_label.setVisible(is_secondary or is_tertiary)
        self.target_primary_combo.setVisible(is_secondary or is_tertiary)
        self.target_secondary_label.setVisible(is_tertiary)
        self.target_secondary_combo.setVisible(is_tertiary)

        # Update info text
        if strategy == "primary":
            self.category_info.setText("Primary strategy sorts by primary category - no target needed")
        elif strategy == "secondary":
            self.category_info.setText("Select which primary category to focus on")
        else:  # tertiary
            self.category_info.setText("Select primary and secondary categories to focus on")

        # Update description
        self.update_strategy_description()

        # Update available categories
        self.update_available_categories()

        # Mark as modified and emit signal
        self.mark_modified()
        self.strategy_changed.emit(strategy)

    def on_primary_category_changed(self):
        """Handle primary category selection change."""
        # Clear pre-assignments when target primary changes since categories will be different
        if self.pre_assignments:
            self.logger.info("Clearing pre-assignments due to target primary category change")
            self.pre_assignments.clear()
            self.preassign_table.setRowCount(0)
            self.update_assignment_summary()

        # Update secondary dropdown options
        self.update_secondary_categories()

        # Update available categories for pre-assignments
        self.update_available_categories()

        # Mark modified and emit signal
        self.mark_modified()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()
        self.target_categories_changed.emit(primary, secondary)

    def on_secondary_category_changed(self):
        """Handle secondary category selection change."""
        # Clear pre-assignments when target secondary changes since categories will be different
        if self.pre_assignments:
            self.logger.info("Clearing pre-assignments due to target secondary category change")
            self.pre_assignments.clear()
            self.preassign_table.setRowCount(0)
            self.update_assignment_summary()

        # Update available categories for pre-assignments
        self.update_available_categories()

        # Mark modified and emit signal
        self.mark_modified()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()
        self.target_categories_changed.emit(primary, secondary)

    def on_max_bins_changed(self):
        """Handle max bins change."""
        self.mark_modified()
        self.bins_changed.emit(self.max_bins_spin.value())
        self.update_available_bins_label()

    def on_queue_size_changed(self, value):
        """Handle queue size change and update warning."""
        if value < 50:
            self.queue_warning_label.setText(
                "⚠ Small queue size may cause dropped pieces during bursts"
            )
            self.queue_warning_label.setStyleSheet("color: #E67E22; font-weight: bold;")
        elif value > 500:
            self.queue_warning_label.setText(
                "⚠ Large queue size will use more memory"
            )
            self.queue_warning_label.setStyleSheet("color: #E67E22; font-weight: bold;")
        else:
            self.queue_warning_label.setText("")

        self.mark_modified()

    def update_available_bins_label(self):
        """Update the available bins label to show the range."""
        max_bins = self.max_bins_spin.value()
        sorting_bins = max_bins - 1  # Subtract 1 for overflow bin
        self.available_bins_label.setText(f"Sorting: Bins 1-{sorting_bins} ({sorting_bins} bins)")
        self.logger.debug(f"Available sorting bin range: 1-{sorting_bins}")

    def get_current_strategy(self) -> str:
        """Get currently selected strategy as lowercase string."""
        if self.primary_radio.isChecked():
            return "primary"
        elif self.secondary_radio.isChecked():
            return "secondary"
        else:
            return "tertiary"

    def update_strategy_description(self):
        """Update the strategy description text."""
        strategy = self.get_current_strategy()

        descriptions = {
            "primary": "Sorts all pieces by their primary category (Basic, Technic, Plates, etc.)",
            "secondary": "Focuses on one primary category and sorts by secondary categories within it",
            "tertiary": "Focuses on specific primary+secondary combination and sorts by tertiary categories"
        }

        self.strategy_desc.setText(descriptions[strategy])

    def update_available_categories(self):
        """
        Update available categories based on current strategy and selections.

        This is the key method that populates dropdowns and pre-assignment options
        using the category_hierarchy_service.
        """
        if not self.category_service:
            self.logger.warning("No category service available - cannot update categories")

            # Update info text to show why dropdowns are empty
            self.category_info.setText(
                "⚠️ Category service not available - dropdowns cannot populate"
            )
            self.category_info.setStyleSheet("color: red; font-style: italic; font-size: 10px;")

            self.available_categories = []
            return

        strategy = self.get_current_strategy()

        try:
            if strategy == "primary":
                # For primary strategy, populate primary dropdown (though not used)
                # and get primary categories for pre-assignments
                primaries = self.category_service.get_primary_categories()
                self.available_categories = primaries

            elif strategy == "secondary":
                # For secondary strategy, populate primary dropdown
                primaries = self.category_service.get_primary_categories()

                # Block signals to prevent recursive updates
                self.target_primary_combo.blockSignals(True)
                current_primary = self.target_primary_combo.currentText()
                self.target_primary_combo.clear()
                self.target_primary_combo.addItems(primaries)

                # Restore selection if valid
                if current_primary in primaries:
                    self.target_primary_combo.setCurrentText(current_primary)
                self.target_primary_combo.blockSignals(False)

                # Get secondary categories for selected primary
                selected_primary = self.target_primary_combo.currentText()
                if selected_primary:
                    secondaries = self.category_service.get_secondary_categories(selected_primary)
                    self.available_categories = secondaries
                else:
                    self.available_categories = []

            else:  # tertiary
                # Populate primary dropdown
                primaries = self.category_service.get_primary_categories()

                self.target_primary_combo.blockSignals(True)
                current_primary = self.target_primary_combo.currentText()
                self.target_primary_combo.clear()
                self.target_primary_combo.addItems(primaries)
                if current_primary in primaries:
                    self.target_primary_combo.setCurrentText(current_primary)
                self.target_primary_combo.blockSignals(False)

                # Update secondary categories based on selected primary
                self.update_secondary_categories()

                # Get tertiary categories for selected primary+secondary
                selected_primary = self.target_primary_combo.currentText()
                selected_secondary = self.target_secondary_combo.currentText()

                if selected_primary and selected_secondary:
                    tertiaries = self.category_service.get_tertiary_categories(
                        selected_primary, selected_secondary
                    )
                    self.available_categories = tertiaries
                else:
                    self.available_categories = []

            self.logger.debug(f"Updated available categories: {len(self.available_categories)} options")

        except Exception as e:
            self.logger.error(f"Error updating categories: {e}", exc_info=True)
            self.available_categories = []

    def update_secondary_categories(self):
        """Update secondary category dropdown based on selected primary."""
        if not self.category_service:
            return

        selected_primary = self.target_primary_combo.currentText()

        if selected_primary:
            secondaries = self.category_service.get_secondary_categories(selected_primary)

            # Block signals to prevent recursive updates
            self.target_secondary_combo.blockSignals(True)
            current_secondary = self.target_secondary_combo.currentText()
            self.target_secondary_combo.clear()
            self.target_secondary_combo.addItems(secondaries)

            # Restore selection if valid
            if current_secondary in secondaries:
                self.target_secondary_combo.setCurrentText(current_secondary)
            self.target_secondary_combo.blockSignals(False)

    # ========================================================================
    # PRE-ASSIGNMENT MANAGEMENT
    # ========================================================================

    def add_assignment(self):
        """Add a new pre-assignment via dialog."""
        if not self.available_categories:
            QMessageBox.warning(
                self,
                "No Categories Available",
                "No categories available for pre-assignment.\n\n"
                "Make sure you've selected a strategy and target categories."
            )
            return

        # Create simple dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Pre-Assignment")
        layout = QVBoxLayout(dialog)

        form = QFormLayout()

        # Category selection
        category_combo = QComboBox()
        category_combo.addItems(self.available_categories)
        form.addRow("Category:", category_combo)

        # Bin selection (1 to max_bins-1 inclusive, since bin 0 is overflow)
        max_bins = self.max_bins_spin.value()
        max_sorting_bin = max_bins - 1
        bin_spin = QSpinBox()
        bin_spin.setRange(1, max_sorting_bin)
        bin_spin.setValue(1)
        bin_spin.setToolTip(f"Select bin number (1-{max_sorting_bin})\nBin 0 is reserved for overflow")
        form.addRow("Bin Number:", bin_spin)

        # Info label
        info_label = QLabel(f"Available sorting bins: 1-{max_sorting_bin}\n(Bin 0 is overflow)")
        info_label.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 10px;")
        info_label.setWordWrap(True)
        form.addRow("", info_label)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            category = category_combo.currentText()
            bin_num = bin_spin.value()

            # Validate
            if category in self.pre_assignments:
                QMessageBox.warning(
                    self,
                    "Duplicate Assignment",
                    f"Category '{category}' is already assigned to bin {self.pre_assignments[category]}"
                )
                return

            # Check if bin already used
            if bin_num in self.pre_assignments.values():
                existing = [cat for cat, bn in self.pre_assignments.items() if bn == bin_num]
                QMessageBox.warning(
                    self,
                    "Bin Already Used",
                    f"Bin {bin_num} is already assigned to '{existing[0]}'"
                )
                return

            # Add assignment
            self.pre_assignments[category] = bin_num
            self.add_assignment_to_table(category, bin_num)
            self.update_assignment_summary()
            self.mark_modified()

            self.logger.info(f"Added pre-assignment: {category} → Bin {bin_num}")

    def add_assignment_to_table(self, category: str, bin_num: int):
        """Add an assignment to the table widget."""
        row = self.preassign_table.rowCount()
        self.preassign_table.insertRow(row)

        # Category column
        category_item = QTableWidgetItem(category)
        category_item.setFlags(category_item.flags() & ~Qt.ItemIsEditable)
        self.preassign_table.setItem(row, 0, category_item)

        # Bin column
        bin_item = QTableWidgetItem(str(bin_num))
        bin_item.setTextAlignment(Qt.AlignCenter)
        bin_item.setFlags(bin_item.flags() & ~Qt.ItemIsEditable)
        self.preassign_table.setItem(row, 1, bin_item)

        # Remove button
        remove_btn = QPushButton("❌ Remove")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        remove_btn.clicked.connect(lambda: self.remove_assignment(category))
        self.preassign_table.setCellWidget(row, 2, remove_btn)

    def remove_assignment(self, category: str):
        """Remove a pre-assignment."""
        if category not in self.pre_assignments:
            return

        bin_num = self.pre_assignments[category]
        del self.pre_assignments[category]

        # Remove from table
        for row in range(self.preassign_table.rowCount()):
            if self.preassign_table.item(row, 0).text() == category:
                self.preassign_table.removeRow(row)
                break

        self.update_assignment_summary()
        self.mark_modified()

        self.logger.info(f"Removed pre-assignment: {category} (was Bin {bin_num})")

    def clear_assignments(self):
        """Clear all pre-assignments."""
        if not self.pre_assignments:
            return

        reply = QMessageBox.question(
            self,
            "Clear All Assignments",
            "Remove all pre-assignments?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.pre_assignments.clear()
            self.preassign_table.setRowCount(0)
            self.update_assignment_summary()
            self.mark_modified()
            self.logger.info("Cleared all pre-assignments")

    def update_assignment_summary(self):
        """Update the assignment summary label."""
        count = len(self.pre_assignments)
        used_bins = len(set(self.pre_assignments.values()))

        if count == 0:
            self.assignment_summary.setText("0 pre-assignments")
        else:
            self.assignment_summary.setText(
                f"{count} pre-assignment{'' if count == 1 else 's'} using {used_bins} bin{'' if used_bins == 1 else 's'}"
            )

    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================

    def load_config(self) -> bool:
        """Load configuration from config_manager and populate UI."""
        try:
            # Get sorting config
            config = self.get_module_config(self.get_module_name())

            # Get processing_queue config for queue_size
            queue_config = self.config_manager.get_module_config(ModuleConfig.PROCESSING_QUEUE.value)

            if not config:
                self.logger.warning("No configuration found, using defaults")
                self.clear_modified()
                return True

            # Block signals during loading
            self.strategy_group.blockSignals(True)
            self.target_primary_combo.blockSignals(True)
            self.target_secondary_combo.blockSignals(True)
            self.max_bins_spin.blockSignals(True)
            self.max_pieces_spin.blockSignals(True)
            self.warning_threshold_spin.blockSignals(True)
            self.confidence_spin.blockSignals(True)
            self.queue_size_spin.blockSignals(True)

            # Load strategy
            strategy = config.get("strategy", "primary")
            if strategy == "primary":
                self.primary_radio.setChecked(True)
            elif strategy == "secondary":
                self.secondary_radio.setChecked(True)
            else:
                self.tertiary_radio.setChecked(True)

            # Load target categories
            target_primary = config.get("target_primary_category", "")
            target_secondary = config.get("target_secondary_category", "")

            # Load bin configuration
            self.max_bins_spin.setValue(config.get("max_bins", 9))
            self.max_pieces_spin.setValue(config.get("max_pieces_per_bin", 50))
            self.warning_threshold_spin.setValue(config.get("bin_warning_threshold", 0.8))

            # Load queue size from processing_queue module
            if queue_config:
                self.queue_size_spin.setValue(queue_config.get("queue_size", 100))
            else:
                self.queue_size_spin.setValue(100)

            # Load thresholds
            self.confidence_spin.setValue(config.get("confidence_threshold", 0.7))

            # Restore signals
            self.strategy_group.blockSignals(False)
            self.target_primary_combo.blockSignals(False)
            self.target_secondary_combo.blockSignals(False)
            self.max_bins_spin.blockSignals(False)
            self.max_pieces_spin.blockSignals(False)
            self.warning_threshold_spin.blockSignals(False)
            self.confidence_spin.blockSignals(False)
            self.queue_size_spin.blockSignals(False)

            # Update UI based on strategy
            self.on_strategy_changed()

            # Set target categories after dropdowns are populated
            if target_primary and self.target_primary_combo.findText(target_primary) >= 0:
                self.target_primary_combo.setCurrentText(target_primary)

            if target_secondary and self.target_secondary_combo.findText(target_secondary) >= 0:
                self.target_secondary_combo.setCurrentText(target_secondary)

            # Load pre-assignments
            self.pre_assignments = config.get("pre_assignments", {}).copy()
            self.preassign_table.setRowCount(0)
            for category, bin_num in self.pre_assignments.items():
                self.add_assignment_to_table(category, bin_num)
            self.update_assignment_summary()

            # Clear modified flag
            self.clear_modified()

            self.logger.info("Configuration loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            return False

    def save_config(self) -> bool:
        """Save current UI values to config_manager."""
        if not self.validate():
            return False

        try:
            # Get sorting configuration
            config = self.get_config()

            # Save sorting configuration
            self.config_manager.update_module_config(
                self.get_module_name(),
                config
            )

            # Save queue_size to processing_queue module
            self.config_manager.update_module_config(
                ModuleConfig.PROCESSING_QUEUE.value,
                {"queue_size": self.queue_size_spin.value()}
            )

            self.clear_modified()
            self.logger.info("Configuration saved successfully")

            # Emit signal
            self.config_changed.emit(self.get_module_name(), config)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}", exc_info=True)
            return False

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration as a dictionary."""
        strategy = self.get_current_strategy()

        config = {
            "strategy": strategy,
            "target_primary_category": self.target_primary_combo.currentText() if strategy in ["secondary",
                                                                                               "tertiary"] else "",
            "target_secondary_category": self.target_secondary_combo.currentText() if strategy == "tertiary" else "",
            "max_bins": self.max_bins_spin.value(),
            "overflow_bin": 0,  # Always 0
            "confidence_threshold": self.confidence_spin.value(),
            "pre_assignments": self.pre_assignments.copy(),
            "max_pieces_per_bin": self.max_pieces_spin.value(),
            "bin_warning_threshold": self.warning_threshold_spin.value()
        }

        return config

    def validate(self) -> bool:
        """Validate current configuration."""
        strategy = self.get_current_strategy()

        # Validate secondary strategy requirements
        if strategy == "secondary":
            if not self.target_primary_combo.currentText():
                self.show_validation_error(
                    "Secondary strategy requires a target primary category"
                )
                return False

        # Validate tertiary strategy requirements
        if strategy == "tertiary":
            if not self.target_primary_combo.currentText():
                self.show_validation_error(
                    "Tertiary strategy requires a target primary category"
                )
                return False

            if not self.target_secondary_combo.currentText():
                self.show_validation_error(
                    "Tertiary strategy requires a target secondary category"
                )
                return False

        # Validate pre-assignments don't have duplicate bins
        used_bins = set()
        for category, bin_num in self.pre_assignments.items():
            if bin_num in used_bins:
                self.show_validation_error(
                    f"Bin {bin_num} is assigned to multiple categories"
                )
                return False
            used_bins.add(bin_num)

        # Validate pre-assignment bin numbers are in range
        max_bins = self.max_bins_spin.value()
        max_sorting_bin = max_bins - 1  # Bin 0 is overflow
        for category, bin_num in self.pre_assignments.items():
            if not (1 <= bin_num <= max_sorting_bin):
                self.show_validation_error(
                    f"Pre-assignment for '{category}' uses bin {bin_num}, "
                    f"which is outside range 1-{max_sorting_bin}"
                )
                return False

        return True

    def reset_to_defaults(self):
        """Reset all values to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Reset all processing settings to default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Reset strategy
            self.primary_radio.setChecked(True)
            self.on_strategy_changed()

            # Reset bins
            self.max_bins_spin.setValue(9)
            self.max_pieces_spin.setValue(50)
            self.warning_threshold_spin.setValue(0.8)

            # Reset queue size
            self.queue_size_spin.setValue(100)

            # Reset thresholds
            self.confidence_spin.setValue(0.7)

            # Clear pre-assignments
            self.pre_assignments.clear()
            self.preassign_table.setRowCount(0)
            self.update_assignment_summary()

            self.mark_modified()
            self.logger.info("Settings reset to defaults")