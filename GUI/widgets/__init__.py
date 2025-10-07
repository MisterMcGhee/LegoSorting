"""
Widgets package for the Lego Sorting System GUI.

This package contains reusable custom widgets for camera views,
configuration controls, and visualization components.
"""

from .camera_view import (
    BaseCameraViewWidget,
    CameraViewRaw,
    CameraViewROI,
    CameraViewTracking,
    CameraViewUnited,
    ViewStyles
)

__all__ = [
    'BaseCameraViewWidget',
    'CameraViewRaw',
    'CameraViewROI',
    'CameraViewTracking',
    'CameraViewUnited',
    'ViewStyles'
]