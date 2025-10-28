"""
Widgets package for the Lego Sorting System GUI.

This package contains reusable custom widgets for camera views,
bin status displays, configuration controls, and visualization components.
"""

from .camera_view import (
    BaseCameraViewWidget,
    CameraViewRaw,
    CameraViewROI,
    CameraViewTracking,
    CameraViewUnited,
    CameraViewSorting,
    ViewStyles
)
from .bin_status import BinStatusWidget
from .gui_styles import SortingGUIStyles

__all__ = [
    'BaseCameraViewWidget',
    'CameraViewRaw',
    'CameraViewROI',
    'CameraViewTracking',
    'CameraViewUnited',
    'CameraViewSorting',
    'ViewStyles',
    'BinStatusWidget',
    'SortingGUIStyles'
]