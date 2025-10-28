"""
gui_styles.py - Shared styling constants for LEGO Sorting GUI

This module provides consistent styling across all GUI widgets.
Matches the aesthetic of configuration_gui.py.

COLOR SCHEME:
- Primary (Green): Success, ready states, normal operation
- Warning (Orange): Warning states, approaching limits
- Danger (Red): Critical states, errors, full capacity
- Info (Blue): Informational elements
- Neutral (Gray): Disabled or inactive elements

USAGE:
    from GUI.widgets.gui_styles import SortingGUIStyles

    button.setStyleSheet(SortingGUIStyles.get_button_style(
        SortingGUIStyles.COLOR_PRIMARY,
        SortingGUIStyles.COLOR_PRIMARY_HOVER
    ))
"""


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
