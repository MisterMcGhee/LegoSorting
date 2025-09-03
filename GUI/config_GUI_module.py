"""
config_gui_module.py - Configuration GUI for Lego Sorting System
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import json
import logging
import time  # ADD if not present
import numpy as np  # ADD if not present
from typing import Dict, Any, Optional
from enhanced_config_manager import create_config_manager  # or your config manager
from GUI.gui_common import BaseGUIWindow, VideoWidget, ConfirmationDialog, validate_config
from camera_module import create_camera


logger = logging.getLogger(__name__)

class ConfigurationGUI(BaseGUIWindow):
    """Main configuration window for system setup"""

    # Signals
    configuration_complete = pyqtSignal(dict)  # Emitted when config is confirmed

    def __init__(self, config_manager=None):
        super().__init__(config_manager, "Lego Sorting System - Configuration")

        # Camera reference
        self.camera = None
        self.preview_active = False

        # FPS tracking
        self.last_preview_time = time.time()
        self.preview_frame_count = 0

        # Window setup (existing code continues...)
        self.setGeometry(100, 100, 1200, 800)
        self.center_window()

        # Initialize UI
        self.init_ui()
        self.load_current_config()

    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Add tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_camera_tab()
        self.create_detector_tab()
        self.create_sorting_tab()
        self.create_api_tab()
        self.create_arduino_tab()
        self.create_system_tab()

        # Bottom control panel
        self.create_control_panel()
        main_layout.addWidget(self.control_panel)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to configure")

    def create_camera_tab(self):
        """Create camera configuration tab"""
        camera_tab = QWidget()
        layout = QVBoxLayout()

        # Camera selection group
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout()

        # Device selection
        self.camera_device = QComboBox()
        self.camera_device.addItems(["0", "1", "2", "3"])
        camera_layout.addRow("Camera Device:", self.camera_device)

        # Resolution
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1920x1080", "1280x720", "640x480"])
        camera_layout.addRow("Resolution:", self.resolution_combo)

        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        camera_layout.addRow("Target FPS:", self.fps_spin)

        # Exposure
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-10, 10)
        self.exposure_slider.setValue(0)
        camera_layout.addRow("Exposure:", self.exposure_slider)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Preview area
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()

        self.preview_widget = VideoWidget()
        preview_layout.addWidget(self.preview_widget)

        self.preview_fps_label = QLabel("FPS: 0.0")
        preview_layout.addWidget(self.preview_fps_label)

        preview_buttons = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Preview")
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.stop_preview_btn.setEnabled(False)

        self.start_preview_btn.clicked.connect(self.start_camera_preview)
        self.stop_preview_btn.clicked.connect(self.stop_camera_preview)

        preview_buttons.addWidget(self.start_preview_btn)
        preview_buttons.addWidget(self.stop_preview_btn)
        preview_layout.addLayout(preview_buttons)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        camera_tab.setLayout(layout)
        self.tab_widget.addTab(camera_tab, "Camera")

    def create_detector_tab(self):
        """Create detector configuration tab"""
        detector_tab = QWidget()
        layout = QVBoxLayout()

        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout()

        # Min/Max area
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 10000)
        self.min_area_spin.setValue(500)
        detection_layout.addRow("Min Area:", self.min_area_spin)

        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1000, 100000)
        self.max_area_spin.setValue(50000)
        detection_layout.addRow("Max Area:", self.max_area_spin)

        # Threshold
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(30)
        detection_layout.addRow("Threshold:", self.threshold_slider)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # ROI settings
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QVBoxLayout()

        self.roi_info_label = QLabel("ROI not configured")
        roi_layout.addWidget(self.roi_info_label)

        self.configure_roi_btn = QPushButton("Configure ROI")
        self.configure_roi_btn.clicked.connect(self.configure_roi)
        roi_layout.addWidget(self.configure_roi_btn)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        layout.addStretch()
        detector_tab.setLayout(layout)
        self.tab_widget.addTab(detector_tab, "Detector")

    def create_sorting_tab(self):
        """Create sorting configuration tab aligned with sorting module requirements"""
        sorting_tab = QWidget()
        main_layout = QVBoxLayout()

        # Store references for dynamic updates
        self.category_dropdowns = {}
        self.pre_assignments = {}

        # === Sorting Strategy Group ===
        strategy_group = QGroupBox("Sorting Strategy")
        strategy_layout = QFormLayout()

        # Strategy selection dropdown
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Primary", "Secondary", "Tertiary"])
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_layout.addRow("Strategy Level:", self.strategy_combo)

        # Target primary category (shown for Secondary/Tertiary)
        self.target_primary_label = QLabel("Target Primary Category:")
        self.target_primary_combo = QComboBox()
        self.target_primary_combo.currentTextChanged.connect(self.on_primary_category_changed)
        strategy_layout.addRow(self.target_primary_label, self.target_primary_combo)

        # Target secondary category (shown for Tertiary only)
        self.target_secondary_label = QLabel("Target Secondary Category:")
        self.target_secondary_combo = QComboBox()
        self.target_secondary_combo.currentTextChanged.connect(self.on_secondary_category_changed)
        strategy_layout.addRow(self.target_secondary_label, self.target_secondary_combo)

        # Initially hide target category selectors
        self.target_primary_label.hide()
        self.target_primary_combo.hide()
        self.target_secondary_label.hide()
        self.target_secondary_combo.hide()

        strategy_group.setLayout(strategy_layout)
        main_layout.addWidget(strategy_group)

        # === Bin Configuration Group ===
        bin_config_group = QGroupBox("Bin Configuration")
        bin_layout = QFormLayout()

        # Maximum bins setting
        self.max_bins_spin = QSpinBox()
        self.max_bins_spin.setRange(1, 20)
        self.max_bins_spin.setValue(9)
        self.max_bins_spin.valueChanged.connect(self.on_max_bins_changed)
        bin_layout.addRow("Maximum Bins:", self.max_bins_spin)

        # Overflow bin (fixed at 0)
        overflow_info = QLabel("0 (fixed)")
        overflow_info.setStyleSheet("color: gray;")
        bin_layout.addRow("Overflow Bin:", overflow_info)

        # Confidence threshold
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.0, 1.0)
        self.confidence_threshold.setSingleStep(0.05)
        self.confidence_threshold.setValue(0.7)
        self.confidence_threshold.setDecimals(2)
        bin_layout.addRow("Confidence Threshold:", self.confidence_threshold)

        # Bin capacity settings
        capacity_label = QLabel("Capacity Management")
        capacity_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        bin_layout.addRow(capacity_label, QLabel())

        self.max_pieces_per_bin = QSpinBox()
        self.max_pieces_per_bin.setRange(1, 1000)
        self.max_pieces_per_bin.setValue(50)
        bin_layout.addRow("Max Pieces per Bin:", self.max_pieces_per_bin)

        self.warning_threshold = QDoubleSpinBox()
        self.warning_threshold.setRange(0.1, 1.0)
        self.warning_threshold.setSingleStep(0.1)
        self.warning_threshold.setValue(0.8)
        self.warning_threshold.setDecimals(2)
        self.warning_threshold.setSuffix("%")
        bin_layout.addRow("Warning Threshold:", self.warning_threshold)

        bin_config_group.setLayout(bin_layout)
        main_layout.addWidget(bin_config_group)

        # === Pre-Assignments Group ===
        assignments_group = QGroupBox("Category Pre-Assignments")
        assignments_layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Assign specific categories to bins (optional)")
        instructions.setStyleSheet("color: gray; font-style: italic;")
        assignments_layout.addWidget(instructions)

        # Pre-assignment table
        self.assignment_table = QTableWidget(0, 3)
        self.assignment_table.setHorizontalHeaderLabels(["Bin", "Category", "Action"])
        self.assignment_table.horizontalHeader().setStretchLastSection(True)
        self.assignment_table.setMaximumHeight(200)
        assignments_layout.addWidget(self.assignment_table)

        # Assignment control buttons
        assignment_buttons = QHBoxLayout()

        self.add_assignment_btn = QPushButton("Add Assignment")
        self.add_assignment_btn.clicked.connect(self.add_bin_assignment)
        assignment_buttons.addWidget(self.add_assignment_btn)

        self.clear_assignments_btn = QPushButton("Clear All")
        self.clear_assignments_btn.clicked.connect(self.clear_all_assignments)
        assignment_buttons.addWidget(self.clear_assignments_btn)

        assignment_buttons.addStretch()
        assignments_layout.addLayout(assignment_buttons)

        assignments_group.setLayout(assignments_layout)
        main_layout.addWidget(assignments_group)

        # === Available Categories Preview ===
        preview_group = QGroupBox("Categories to be Sorted")
        preview_layout = QVBoxLayout()

        # Info label
        self.category_info_label = QLabel("Categories available for current strategy:")
        preview_layout.addWidget(self.category_info_label)

        # Category list
        self.available_categories_list = QListWidget()
        self.available_categories_list.setMaximumHeight(150)
        self.available_categories_list.setSelectionMode(QListWidget.NoSelection)
        preview_layout.addWidget(self.available_categories_list)

        # Statistics
        self.category_stats_label = QLabel("0 categories available")
        self.category_stats_label.setStyleSheet("color: gray;")
        preview_layout.addWidget(self.category_stats_label)

        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        main_layout.addStretch()
        sorting_tab.setLayout(main_layout)
        self.tab_widget.addTab(sorting_tab, "Sorting")

        # Load categories from CSV if config manager available
        if self.config_manager:
            self.load_category_hierarchy()
            self.update_available_categories()

    def on_strategy_changed(self, strategy_text):
        """Handle sorting strategy selection change with smart dropdown population"""
        print(f"Strategy changed to: {strategy_text}")  # Debug line
        if not self.config_manager:
            print("WARNING: config_manager is None!")
            return

        strategy = strategy_text.lower()
        print(f"Strategy (lowercase): {strategy}")  # Debug line

        # Get the category hierarchy for populating dropdowns
        hierarchy = self.config_manager.get_category_hierarchy()

        # Handle visibility and populate dropdowns based on strategy
        if strategy == "primary":
            # Hide all target category selectors for primary strategy
            self.target_primary_label.hide()
            self.target_primary_combo.hide()
            self.target_secondary_label.hide()
            self.target_secondary_combo.hide()

        elif strategy == "secondary":
            # Show only primary selector for secondary strategy
            self.target_primary_label.show()
            self.target_primary_combo.show()
            self.target_secondary_label.hide()
            self.target_secondary_combo.hide()

            # Populate primary dropdown with all available primary categories
            primary_categories = hierarchy.get("primary", [])
            current_selection = self.target_primary_combo.currentText()

            self.target_primary_combo.clear()
            self.target_primary_combo.addItem("")  # Add empty option
            self.target_primary_combo.addItems(sorted(primary_categories))

            # Restore previous selection if it still exists
            if current_selection in primary_categories:
                self.target_primary_combo.setCurrentText(current_selection)

        elif strategy == "tertiary":
            # Show both selectors for tertiary strategy
            self.target_primary_label.show()
            self.target_primary_combo.show()
            self.target_secondary_label.show()
            self.target_secondary_combo.show()

            # Populate primary dropdown with all available primary categories
            primary_categories = hierarchy.get("primary", [])
            current_primary = self.target_primary_combo.currentText()

            self.target_primary_combo.clear()
            self.target_primary_combo.addItem("")  # Add empty option
            self.target_primary_combo.addItems(sorted(primary_categories))

            # If we had a previous selection, restore it and populate secondary
            if current_primary in primary_categories:
                self.target_primary_combo.setCurrentText(current_primary)

                # Populate secondary dropdown based on selected primary
                secondary_categories = hierarchy.get("primary_to_secondary", {}).get(current_primary, [])
                current_secondary = self.target_secondary_combo.currentText()

                self.target_secondary_combo.clear()
                self.target_secondary_combo.addItem("")  # Add empty option
                self.target_secondary_combo.addItems(sorted(secondary_categories))

                # Restore secondary selection if it still exists
                if current_secondary in secondary_categories:
                    self.target_secondary_combo.setCurrentText(current_secondary)
            else:
                # Clear secondary dropdown if no primary selected
                self.target_secondary_combo.clear()
                self.target_secondary_combo.addItem("")

        # Update available categories list based on new strategy
        self.update_available_categories()

        # Clear pre-assignments as they may no longer be valid
        self.validate_assignments()

        # Update status to show what categories are available
        self.update_dropdown_status()

        self.strategy_combo.parentWidget().update()
        self.strategy_combo.parentWidget().adjustSize()

    def on_primary_category_changed(self, primary_category):
        """Handle primary category selection change with smart secondary dropdown update"""
        if not primary_category or not self.config_manager:
            # Clear secondary dropdown if no primary selected
            if self.strategy_combo.currentText().lower() == "tertiary":
                self.target_secondary_combo.clear()
                self.target_secondary_combo.addItem("")
            self.update_available_categories()
            return

        strategy = self.strategy_combo.currentText().lower()

        # Update secondary category dropdown if in tertiary mode
        if strategy == "tertiary":
            # Get secondary categories for selected primary
            hierarchy = self.config_manager.get_category_hierarchy()
            secondary_categories = hierarchy.get("primary_to_secondary", {}).get(primary_category, [])

            # Remember current selection if any
            current_secondary = self.target_secondary_combo.currentText()

            # Populate secondary dropdown
            self.target_secondary_combo.clear()
            self.target_secondary_combo.addItem("")  # Empty option

            if secondary_categories:
                self.target_secondary_combo.addItems(sorted(secondary_categories))

                # Restore selection if it's still valid
                if current_secondary in secondary_categories:
                    self.target_secondary_combo.setCurrentText(current_secondary)

                # Enable the secondary dropdown
                self.target_secondary_combo.setEnabled(True)

                # Update tooltip with helpful information
                tooltip = f"Select a secondary category within '{primary_category}'\n"
                tooltip += f"Available options: {len(secondary_categories)} categories"
                self.target_secondary_combo.setToolTip(tooltip)
            else:
                # No secondary categories available
                self.target_secondary_combo.addItem("(No secondary categories available)")
                self.target_secondary_combo.setEnabled(False)

        # Update available categories based on selection
        self.update_available_categories()
        self.validate_assignments()

        # Update status
        self.update_dropdown_status()

    def on_secondary_category_changed(self, secondary_category):
        """Handle secondary category selection change"""
        self.update_available_categories()
        self.validate_assignments()

        # Update status to reflect the selection
        self.update_dropdown_status()

    def on_max_bins_changed(self, value):
        """Handle max bins value change"""
        self.validate_assignments()

    def update_dropdown_status(self):
        """Update status bar with current dropdown selection information"""
        strategy = self.strategy_combo.currentText().lower()

        if strategy == "primary":
            # Get count of primary categories available
            categories = self.config_manager.get_categories_for_strategy(strategy)
            self.status_bar.showMessage(f"Primary sorting: {len(categories)} categories available")

        elif strategy == "secondary":
            primary = self.target_primary_combo.currentText()
            if primary:
                categories = self.config_manager.get_categories_for_strategy(strategy, primary)
                self.status_bar.showMessage(
                    f"Secondary sorting in '{primary}': {len(categories)} subcategories available"
                )
            else:
                self.status_bar.showMessage("Secondary sorting: Please select a primary category")

        elif strategy == "tertiary":
            primary = self.target_primary_combo.currentText()
            secondary = self.target_secondary_combo.currentText()

            if primary and secondary:
                categories = self.config_manager.get_categories_for_strategy(
                    strategy, primary, secondary
                )
                self.status_bar.showMessage(
                    f"Tertiary sorting in '{primary}/{secondary}': {len(categories)} categories available"
                )
            elif primary:
                self.status_bar.showMessage(
                    f"Tertiary sorting in '{primary}': Please select a secondary category"
                )
            else:
                self.status_bar.showMessage("Tertiary sorting: Please select primary and secondary categories")

    def load_category_hierarchy(self):
        """Load category hierarchy from CSV via config manager and populate initial dropdowns"""
        if not self.config_manager:
            return

        try:
            # Parse categories from CSV
            hierarchy = self.config_manager.get_category_hierarchy()

            # Get all primary categories
            primary_categories = hierarchy.get("primary", [])

            # Store hierarchy for later use
            self.category_hierarchy = hierarchy

            # Initially populate primary dropdown (will be shown/hidden based on strategy)
            self.target_primary_combo.clear()
            self.target_primary_combo.addItem("")  # Empty option
            self.target_primary_combo.addItems(sorted(primary_categories))

            # Add tooltips to help users
            self.target_primary_combo.setToolTip(
                f"Select from {len(primary_categories)} available primary categories"
            )
            self.target_secondary_combo.setToolTip(
                "Secondary categories will appear after selecting a primary category"
            )

            logger.info(f"Loaded category hierarchy with {len(primary_categories)} primary categories")

        except Exception as e:
            logger.error(f"Failed to load category hierarchy: {e}")
            QMessageBox.warning(self, "Warning",
                                f"Could not load categories from CSV: {str(e)}")

    def update_available_categories(self):
        """Update the list of available categories based on current strategy"""
        if not self.config_manager:
            return

        self.available_categories_list.clear()

        strategy = self.strategy_combo.currentText().lower()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()

        # Get categories for current strategy
        categories = self.config_manager.get_categories_for_strategy(
            strategy, primary, secondary
        )

        # Update list widget
        for category in sorted(categories):
            item = QListWidgetItem(category)

            # Check if this category is pre-assigned
            if category in self.pre_assignments:
                bin_num = self.pre_assignments[category]
                item.setText(f"{category} → Bin {bin_num}")
                item.setForeground(QColor(0, 128, 0))  # Green for assigned

            self.available_categories_list.addItem(item)

        # Update statistics
        count = len(categories)
        assigned_count = sum(1 for cat in categories if cat in self.pre_assignments)

        stats_text = f"{count} categories available"
        if assigned_count > 0:
            stats_text += f" ({assigned_count} pre-assigned)"
        self.category_stats_label.setText(stats_text)

        # Update info label based on strategy
        if strategy == "primary":
            self.category_info_label.setText("Sorting by primary categories:")
        elif strategy == "secondary":
            if primary:
                self.category_info_label.setText(f"Sorting secondary categories within '{primary}':")
            else:
                self.category_info_label.setText("Select a primary category first")
        elif strategy == "tertiary":
            if primary and secondary:
                self.category_info_label.setText(f"Sorting tertiary categories within '{primary}/{secondary}':")
            else:
                self.category_info_label.setText("Select primary and secondary categories first")

    def validate_dropdown_selections(self):
        """Validate that dropdown selections are valid for the current strategy"""
        strategy = self.strategy_combo.currentText().lower()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()

        errors = []

        if strategy == "secondary":
            if not primary:
                errors.append("Secondary sorting requires selecting a primary category")

            # Check if the selected primary has any secondary categories
            elif self.config_manager:
                categories = self.config_manager.get_categories_for_strategy(strategy, primary)
                if not categories:
                    errors.append(f"Primary category '{primary}' has no secondary categories")

        elif strategy == "tertiary":
            if not primary:
                errors.append("Tertiary sorting requires selecting a primary category")
            elif not secondary:
                errors.append("Tertiary sorting requires selecting a secondary category")

            # Check if the combination has any tertiary categories
            elif self.config_manager:
                categories = self.config_manager.get_categories_for_strategy(
                    strategy, primary, secondary
                )
                if not categories:
                    errors.append(
                        f"Combination '{primary}/{secondary}' has no tertiary categories"
                    )

        return errors

    def refresh_dropdowns(self):
        """Force refresh all dropdowns from the CSV data"""
        # Reload category hierarchy
        self.load_category_hierarchy()

        # Trigger strategy change to repopulate dropdowns
        current_strategy = self.strategy_combo.currentText()
        self.on_strategy_changed(current_strategy)

        self.status_bar.showMessage("Dropdowns refreshed from CSV data", 3000)
    def add_bin_assignment(self):
        """Add a new bin assignment"""
        if not self.config_manager:
            return

        # Get available categories for current strategy
        strategy = self.strategy_combo.currentText().lower()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()

        categories = self.config_manager.get_categories_for_strategy(
            strategy, primary, secondary
        )

        # Filter out already assigned categories
        available = [cat for cat in categories if cat not in self.pre_assignments]

        if not available:
            QMessageBox.information(self, "No Categories",
                                    "All available categories are already assigned")
            return

        # Create assignment dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Bin Assignment")
        dialog.setModal(True)

        layout = QFormLayout()

        # Category selection
        category_combo = QComboBox()
        category_combo.addItems(sorted(available))
        layout.addRow("Category:", category_combo)

        # Bin selection
        bin_spin = QSpinBox()
        bin_spin.setRange(1, self.max_bins_spin.value())

        # Find next available bin
        used_bins = set(self.pre_assignments.values())
        for i in range(1, self.max_bins_spin.value() + 1):
            if i not in used_bins:
                bin_spin.setValue(i)
                break

        layout.addRow("Bin Number:", bin_spin)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            category = category_combo.currentText()
            bin_num = bin_spin.value()

            # Check if bin is already assigned
            for cat, assigned_bin in self.pre_assignments.items():
                if assigned_bin == bin_num:
                    reply = QMessageBox.question(self, "Bin Already Assigned",
                                                 f"Bin {bin_num} is already assigned to '{cat}'.\n"
                                                 f"Replace with '{category}'?",
                                                 QMessageBox.Yes | QMessageBox.No)

                    if reply == QMessageBox.Yes:
                        # Remove old assignment
                        del self.pre_assignments[cat]
                        self.remove_assignment_from_table(cat)
                    else:
                        return

            # Add assignment
            self.pre_assignments[category] = bin_num
            self.add_assignment_to_table(category, bin_num)
            self.update_available_categories()

    def add_assignment_to_table(self, category, bin_num):
        """Add an assignment to the table"""
        row = self.assignment_table.rowCount()
        self.assignment_table.insertRow(row)

        # Bin number
        bin_item = QTableWidgetItem(str(bin_num))
        bin_item.setTextAlignment(Qt.AlignCenter)
        self.assignment_table.setItem(row, 0, bin_item)

        # Category name
        category_item = QTableWidgetItem(category)
        self.assignment_table.setItem(row, 1, category_item)

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self.remove_assignment(category))
        self.assignment_table.setCellWidget(row, 2, remove_btn)

    def remove_assignment(self, category):
        """Remove a bin assignment"""
        if category in self.pre_assignments:
            del self.pre_assignments[category]
            self.remove_assignment_from_table(category)
            self.update_available_categories()

    def remove_assignment_from_table(self, category):
        """Remove an assignment from the table"""
        for row in range(self.assignment_table.rowCount()):
            item = self.assignment_table.item(row, 1)
            if item and item.text() == category:
                self.assignment_table.removeRow(row)
                break

    def clear_all_assignments(self):
        """Clear all bin assignments"""
        reply = QMessageBox.question(self, "Clear Assignments",
                                     "Remove all bin assignments?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.pre_assignments.clear()
            self.assignment_table.setRowCount(0)
            self.update_available_categories()

    def validate_assignments(self):
        """Validate that current assignments are still valid for the strategy"""
        if not self.pre_assignments:
            return

        strategy = self.strategy_combo.currentText().lower()
        primary = self.target_primary_combo.currentText()
        secondary = self.target_secondary_combo.currentText()

        # Get currently valid categories
        valid_categories = self.config_manager.get_categories_for_strategy(
            strategy, primary, secondary
        )

        # Check each assignment
        invalid = []
        for category in list(self.pre_assignments.keys()):
            if category not in valid_categories:
                invalid.append(category)

        # Remove invalid assignments
        if invalid:
            for category in invalid:
                del self.pre_assignments[category]
                self.remove_assignment_from_table(category)

            QMessageBox.information(self, "Assignments Updated",
                                    f"Removed {len(invalid)} assignment(s) that are no longer valid "
                                    f"for the selected strategy.")

            self.update_available_categories()

    def get_sorting_config(self):
        """Get the current sorting configuration as a dictionary"""
        strategy = self.strategy_combo.currentText().lower()

        config = {
            "strategy": strategy,
            "target_primary_category": self.target_primary_combo.currentText() if strategy in ["secondary",
                                                                                               "tertiary"] else "",
            "target_secondary_category": self.target_secondary_combo.currentText() if strategy == "tertiary" else "",
            "max_bins": self.max_bins_spin.value(),
            "overflow_bin": 0,
            "confidence_threshold": self.confidence_threshold.value(),
            "pre_assignments": dict(self.pre_assignments),
            "max_pieces_per_bin": self.max_pieces_per_bin.value(),
            "bin_warning_threshold": self.warning_threshold.value()
        }

        return config

    def load_sorting_config(self, config):
        """Load sorting configuration from a dictionary"""
        if not config:
            return

        # Set strategy
        strategy = config.get("strategy", "primary")
        strategy_map = {"primary": "Primary", "secondary": "Secondary", "tertiary": "Tertiary"}
        self.strategy_combo.setCurrentText(strategy_map.get(strategy, "Primary"))

        # Set target categories
        if strategy in ["secondary", "tertiary"]:
            primary = config.get("target_primary_category", "")
            self.target_primary_combo.setCurrentText(primary)

        if strategy == "tertiary":
            secondary = config.get("target_secondary_category", "")
            self.target_secondary_combo.setCurrentText(secondary)

        # Set bin configuration
        self.max_bins_spin.setValue(config.get("max_bins", 9))
        self.confidence_threshold.setValue(config.get("confidence_threshold", 0.7))
        self.max_pieces_per_bin.setValue(config.get("max_pieces_per_bin", 50))
        self.warning_threshold.setValue(config.get("bin_warning_threshold", 0.8))

        # Load pre-assignments
        self.pre_assignments = config.get("pre_assignments", {}).copy()
        self.assignment_table.setRowCount(0)
        for category, bin_num in self.pre_assignments.items():
            self.add_assignment_to_table(category, bin_num)

        # Update display
        self.update_available_categories()
    def create_api_tab(self):
        """Create API configuration tab"""
        api_tab = QWidget()
        layout = QVBoxLayout()

        # API settings
        api_group = QGroupBox("API Settings")
        api_layout = QFormLayout()

        # API Key
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        api_layout.addRow("API Key:", self.api_key_edit)

        # Mode
        self.api_mode_combo = QComboBox()
        self.api_mode_combo.addItems(["Production", "Development", "Mock"])
        api_layout.addRow("Mode:", self.api_mode_combo)

        # Timeout
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 60)
        self.timeout_spin.setValue(10)
        api_layout.addRow("Timeout (s):", self.timeout_spin)

        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # Test connection
        test_group = QGroupBox("Connection Test")
        test_layout = QVBoxLayout()

        self.test_api_btn = QPushButton("Test Connection")
        self.test_api_btn.clicked.connect(self.test_api_connection)
        test_layout.addWidget(self.test_api_btn)

        self.api_status_label = QLabel("Not tested")
        test_layout.addWidget(self.api_status_label)

        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

        layout.addStretch()
        api_tab.setLayout(layout)
        self.tab_widget.addTab(api_tab, "API")

    def create_arduino_tab(self):
        """Create Arduino configuration tab"""
        arduino_tab = QWidget()
        layout = QVBoxLayout()

        # Connection settings
        connection_group = QGroupBox("Arduino Connection")
        connection_layout = QFormLayout()

        # Port selection
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.refresh_ports_btn = QPushButton("Refresh")
        self.refresh_ports_btn.clicked.connect(self.refresh_serial_ports)

        port_layout = QHBoxLayout()
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.refresh_ports_btn)
        connection_layout.addRow("Serial Port:", port_layout)

        # Baud rate
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "115200"])
        connection_layout.addRow("Baud Rate:", self.baud_combo)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        # Servo settings
        servo_group = QGroupBox("Servo Configuration")
        servo_layout = QVBoxLayout()

        # Calibration button
        self.calibrate_servo_btn = QPushButton("Calibrate Servo")
        self.calibrate_servo_btn.clicked.connect(self.calibrate_servo)
        servo_layout.addWidget(self.calibrate_servo_btn)

        # Bin positions table
        self.servo_table = QTableWidget(10, 2)
        self.servo_table.setHorizontalHeaderLabels(["Bin", "Position"])

        for i in range(10):
            self.servo_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.servo_table.setItem(i, 1, QTableWidgetItem(str(90)))

        servo_layout.addWidget(self.servo_table)

        servo_group.setLayout(servo_layout)
        layout.addWidget(servo_group)

        arduino_tab.setLayout(layout)
        self.tab_widget.addTab(arduino_tab, "Arduino")

    def create_system_tab(self):
        """Create system configuration tab"""
        system_tab = QWidget()
        layout = QVBoxLayout()

        # Performance settings
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QFormLayout()

        # Threading
        self.threading_check = QCheckBox("Enable Multithreading")
        self.threading_check.setChecked(True)
        perf_layout.addRow("Threading:", self.threading_check)

        # Queue size
        self.queue_size_spin = QSpinBox()
        self.queue_size_spin.setRange(10, 1000)
        self.queue_size_spin.setValue(100)
        perf_layout.addRow("Queue Size:", self.queue_size_spin)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Logging settings
        log_group = QGroupBox("Logging Settings")
        log_layout = QFormLayout()

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        log_layout.addRow("Log Level:", self.log_level_combo)

        self.save_images_check = QCheckBox("Save Detected Pieces")
        self.save_images_check.setChecked(True)
        log_layout.addRow("Save Images:", self.save_images_check)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        system_tab.setLayout(layout)
        self.tab_widget.addTab(system_tab, "System")

    def create_control_panel(self):
        """Create bottom control panel"""
        self.control_panel = QWidget()
        layout = QHBoxLayout()

        # Profile controls
        self.load_profile_btn = QPushButton("Load Profile")
        self.save_profile_btn = QPushButton("Save Profile")

        layout.addWidget(self.load_profile_btn)
        layout.addWidget(self.save_profile_btn)

        layout.addStretch()

        # Action buttons
        self.validate_btn = QPushButton("Validate Config")
        self.validate_btn.clicked.connect(self.validate_configuration)

        self.start_sorting_btn = QPushButton("Start Sorting")
        self.start_sorting_btn.clicked.connect(self.confirm_configuration)
        self.start_sorting_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                font-size: 14px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        layout.addWidget(self.validate_btn)
        layout.addWidget(self.start_sorting_btn)

        self.control_panel.setLayout(layout)

    # ========== Configuration Methods ==========

    def load_current_config(self):
        """Load current configuration into UI"""
        if not self.config_manager:
            return

        # Load camera settings
        camera_config = self.config_manager.get_module_config("camera")
        if camera_config:
            self.camera_device.setCurrentText(str(camera_config.get("device_id", 0)))
            res = camera_config.get("resolution", [1920, 1080])
            self.resolution_combo.setCurrentText(f"{res[0]}x{res[1]}")
            self.fps_spin.setValue(camera_config.get("fps", 30))

        # Load detector settings
        detector_config = self.config_manager.get_module_config("detector")
        if detector_config:
            self.min_area_spin.setValue(detector_config.get("min_area", 500))
            self.max_area_spin.setValue(detector_config.get("max_area", 50000))
            self.threshold_slider.setValue(detector_config.get("threshold", 30))

        # Load other settings...
        # (Continue for all configuration items)

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration from UI"""
        config = {}

        # Camera configuration
        resolution = self.resolution_combo.currentText().split('x')
        config['camera'] = {
            'device_id': int(self.camera_device.currentText()),
            'resolution': [int(resolution[0]), int(resolution[1])],
            'fps': self.fps_spin.value(),
            'exposure': self.exposure_slider.value()
        }

        # Detector configuration
        config['detector'] = {
            'min_area': self.min_area_spin.value(),
            'max_area': self.max_area_spin.value(),
            'threshold': self.threshold_slider.value()
        }

        # Sorting configuration
        config['sorting'] = {
            'strategy': self.strategy_combo.currentText(),
            'overflow_bin': self.overflow_bin.value()
        }

        # API configuration
        config['api'] = {
            'api_key': self.api_key_edit.text(),
            'mode': self.api_mode_combo.currentText(),
            'timeout': self.timeout_spin.value()
        }

        # System configuration
        config['system'] = {
            'threading_enabled': self.threading_check.isChecked(),
            'queue_size': self.queue_size_spin.value(),
            'log_level': self.log_level_combo.currentText(),
            'save_images': self.save_images_check.isChecked()
        }

        return config

    def validate_configuration(self):
        """Validate the current configuration"""
        is_valid, error_msg = validate_config(self.config_manager)

        if is_valid:
            QMessageBox.information(self, "Validation Success",
                                    "Configuration is valid!")
            self.status_bar.showMessage("Configuration validated successfully")
        else:
            QMessageBox.warning(self, "Validation Failed",
                                f"Configuration error: {error_msg}")
            self.status_bar.showMessage(f"Validation failed: {error_msg}")

    def confirm_configuration(self):
        """Confirm configuration and start sorting"""
        # Validate first
        is_valid, error_msg = validate_config(self.config_manager)

        if not is_valid:
            QMessageBox.warning(self, "Invalid Configuration",
                                f"Cannot start sorting: {error_msg}")
            return

        # Confirm with user
        if ConfirmationDialog.confirm(self, "Start Sorting",
                                      "Start sorting with current configuration?"):
            # Save configuration
            config = self.get_configuration()
            for module, settings in config.items():
                self.config_manager.update_module_config(module, settings)
            self.config_manager.save_config()

            # Emit signal to start sorting
            self.configuration_complete.emit(config)

    # ========== Camera Preview Methods ==========

    def start_camera_preview(self):
        """Start live camera preview using consumer system"""
        try:
            # Get or create camera instance
            if not self.camera:
                self.camera = create_camera(config_manager=self.config_manager)

                # Initialize if needed
                if not self.camera.is_initialized:
                    if not self.camera.initialize():
                        QMessageBox.warning(self, "Camera Error",
                                            "Failed to initialize camera")
                        return

            # Register this GUI as a frame consumer
            success = self.camera.register_consumer(
                name="config_preview",
                callback=self._preview_frame_callback,
                processing_type="async",  # Non-blocking for GUI
                priority=50  # Medium priority for preview
            )

            if not success:
                logger.warning("Preview consumer already registered")

            # Start camera capture if not already running
            if not self.camera.is_capturing:
                self.camera.start_capture()

            # Update UI state
            self.preview_active = True
            self.start_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(True)
            self.status_bar.showMessage("Camera preview started")

            logger.info("Live camera preview started")

        except Exception as e:
            logger.error(f"Failed to start camera preview: {e}")
            QMessageBox.critical(self, "Preview Error",
                                 f"Failed to start preview: {str(e)}")

    def stop_camera_preview(self):
        """Stop camera preview and unregister consumer"""
        try:
            if self.camera:
                # Unregister this GUI as a consumer
                self.camera.unregister_consumer("config_preview")

                # Note: We don't stop capture here as other consumers might need it
                # The camera module handles this efficiently

            # Update UI state
            self.preview_active = False
            self.preview_widget.setText("No Video Signal")
            self.preview_fps_label.setText("FPS: 0.0")
            self.start_preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(False)
            self.status_bar.showMessage("Camera preview stopped")

            logger.info("Camera preview stopped")

        except Exception as e:
            logger.error(f"Error stopping preview: {e}")

    def _preview_frame_callback(self, frame: np.ndarray):
        """
        Callback function for camera frames.
        This is called by the camera module for each new frame.
        """
        if not self.preview_active:
            return

        try:
            # Use Qt's thread-safe mechanism to update GUI
            # Since this callback might be from a different thread
            QMetaObject.invokeMethod(
                self,
                "_update_preview_display",
                Qt.QueuedConnection,
                Q_ARG(object, frame)
            )

            # Track FPS
            self.preview_frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_preview_time

            if elapsed > 1.0:  # Update FPS every second
                fps = self.preview_frame_count / elapsed
                # Update FPS label in GUI thread
                QMetaObject.invokeMethod(
                    self.preview_fps_label,
                    "setText",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"FPS: {fps:.1f}")
                )
                self.last_preview_time = current_time
                self.preview_frame_count = 0

        except Exception as e:
            logger.error(f"Error in preview callback: {e}")

    @pyqtSlot(object)
    def _update_preview_display(self, frame):
        """
        Thread-safe method to update the preview widget.
        This runs in the GUI thread.
        """
        if self.preview_active and frame is not None:
            self.preview_widget.update_frame(frame)

    def closeEvent(self, event):
        """Handle window close event"""
        # Make sure to stop preview and clean up
        if self.preview_active:
            self.stop_camera_preview()

        # Call parent close event
        super().closeEvent(event)

    # ========== Other UI Methods ==========

    def configure_roi(self):
        """Open ROI configuration dialog"""
        # This would open a separate window for ROI configuration
        QMessageBox.information(self, "ROI Configuration",
                                "ROI configuration would open here")

    def test_api_connection(self):
        """Test API connection"""
        self.api_status_label.setText("Testing...")
        # Actual API test would go here
        QTimer.singleShot(1000, lambda: self.api_status_label.setText("Connection OK"))

    def calibrate_servo(self):
        """Start servo calibration"""
        QMessageBox.information(self, "Servo Calibration",
                                "Servo calibration would start here")

    def refresh_serial_ports(self):
        """Refresh available serial ports"""
        # Would enumerate actual serial ports
        self.port_combo.clear()
        self.port_combo.addItems(["COM3", "COM4", "/dev/ttyUSB0"])
