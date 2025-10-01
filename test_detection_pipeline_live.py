"""
test_detection_pipeline_live.py - Improved test harness for the computer vision pipeline

This test environment integrates the camera module with the complete CV pipeline
to provide real-time testing with live camera feed, visualization, and logging.

Features:
- Real-time visualization with ROI and zone overlays
- Tracked piece bounding boxes with zone status indicators
- Background subtraction mask view (toggle with 'd')
- Recently captured images preview panel
- Console logging of tracking events and captures
- Performance statistics display
"""

import cv2
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Import camera module
from camera_module import create_camera

# Import CV pipeline components
from detector.detector_coordinator import create_detector_coordinator
from detector.zone_manager import create_zone_manager
from detector.capture_controller import create_capture_controller
from enhanced_config_manager import create_config_manager, ModuleConfig
from detector.data_models import RegionOfInterest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST ENVIRONMENT CONFIGURATION
# ============================================================================

class TestConfiguration:
    """Configuration for the test environment."""

    # Display settings
    WINDOW_NAME = "CV Pipeline Test Environment"
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    # Visualization colors (BGR format for OpenCV)
    COLOR_ROI = (255, 255, 0)  # Cyan
    COLOR_ENTRY_ZONE = (0, 255, 255)  # Yellow
    COLOR_VALID_ZONE = (0, 255, 0)  # Green
    COLOR_EXIT_ZONE = (0, 0, 255)  # Red

    # Piece status colors
    COLOR_DETECTED = (128, 128, 128)  # Gray
    COLOR_STABLE = (255, 165, 0)  # Blue
    COLOR_PROCESSING = (0, 165, 255)  # Orange
    COLOR_IDENTIFIED = (0, 255, 0)  # Green

    # Zone indicator colors (for dots on pieces)
    ZONE_INDICATOR_ENTRY = (0, 255, 255)  # Yellow
    ZONE_INDICATOR_VALID = (0, 255, 0)  # Green
    ZONE_INDICATOR_EXIT = (0, 0, 255)  # Red
    ZONE_INDICATOR_NONE = (128, 128, 128)  # Gray

    # Text settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    # Capture preview settings
    CAPTURE_PREVIEW_COLS = 4
    CAPTURE_PREVIEW_ROWS = 2
    CAPTURE_PREVIEW_MAX = 8  # Total captures to show

    # Debug settings
    SHOW_BACKGROUND_MASK = False  # Toggle with 'd' key


# ============================================================================
# VISUALIZATION OVERLAY MANAGER
# ============================================================================

class VisualizationOverlay:
    """Handles all visualization overlays for the test environment."""

    def __init__(self, config: TestConfiguration):
        self.config = config

    def draw_roi_overlay(self, frame: np.ndarray, roi_info: Dict[str, Any]) -> np.ndarray:
        """Draw ROI boundary on frame."""
        if not roi_info.get("configured", False):
            return frame

        x, y, w, h = roi_info["roi"]

        # Draw ROI rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      self.config.COLOR_ROI, 2)

        # Add ROI label
        cv2.putText(frame, "ROI", (x + 5, y + 25),
                    self.config.FONT, self.config.FONT_SCALE,
                    self.config.COLOR_ROI, self.config.FONT_THICKNESS)

        return frame

    def draw_zone_overlays(self, frame: np.ndarray, zone_info: Dict[str, Any]) -> np.ndarray:
        """Draw zone boundaries on frame."""
        if not zone_info.get("configured", False):
            return frame

        roi_x, roi_y, roi_w, roi_h = zone_info["roi"]
        zones = zone_info["zones"]

        # Draw entry zone (left side)
        entry = zones["entry"]
        entry_x = roi_x + entry["start_x"]
        entry_w = entry["width"]
        cv2.rectangle(frame, (entry_x, roi_y),
                      (entry_x + entry_w, roi_y + roi_h),
                      self.config.COLOR_ENTRY_ZONE, 1)
        cv2.putText(frame, "ENTRY", (entry_x + 5, roi_y + 20),
                    self.config.FONT, 0.5, self.config.COLOR_ENTRY_ZONE, 1)

        # Draw valid zone (middle)
        valid = zones["valid"]
        valid_x = roi_x + valid["start_x"]
        valid_w = valid["width"]
        cv2.rectangle(frame, (valid_x, roi_y),
                      (valid_x + valid_w, roi_y + roi_h),
                      self.config.COLOR_VALID_ZONE, 1)
        cv2.putText(frame, "VALID", (valid_x + valid_w // 2 - 30, roi_y + 20),
                    self.config.FONT, 0.5, self.config.COLOR_VALID_ZONE, 1)

        # Draw exit zone (right side)
        exit_zone = zones["exit"]
        exit_x = roi_x + exit_zone["start_x"]
        exit_w = exit_zone["width"]
        cv2.rectangle(frame, (exit_x, roi_y),
                      (exit_x + exit_w, roi_y + roi_h),
                      self.config.COLOR_EXIT_ZONE, 1)
        cv2.putText(frame, "EXIT", (exit_x + 5, roi_y + 20),
                    self.config.FONT, 0.5, self.config.COLOR_EXIT_ZONE, 1)

        return frame

    def draw_tracked_pieces(self, frame: np.ndarray,
                            tracked_pieces: list) -> np.ndarray:
        """Draw bounding boxes, labels, and zone indicators for tracked pieces."""
        try:
            for piece in tracked_pieces:
                piece_id = piece["id"]
                x, y, w, h = piece["bbox"]
                status = piece["status"]

                # Select color based on status
                color_map = {
                    "detected": self.config.COLOR_DETECTED,
                    "stable": self.config.COLOR_STABLE,
                    "processing": self.config.COLOR_PROCESSING,
                    "identified": self.config.COLOR_IDENTIFIED
                }
                color = color_map.get(status, self.config.COLOR_DETECTED)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw piece ID
                label = f"ID:{piece_id}"
                label_size = cv2.getTextSize(label, self.config.FONT, 0.5, 1)[0]
                cv2.rectangle(frame, (x, y - label_size[1] - 5),
                              (x + label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y - 3),
                            self.config.FONT, 0.5, (255, 255, 255), 1)

                # Draw center point
                center = piece["center"]
                cv2.circle(frame, (int(center[0]), int(center[1])),
                           3, color, -1)

                # NEW: Draw zone status indicator
                self._draw_zone_indicator(frame, piece, x, y, w, h)

                # NEW: Draw capture eligibility diagnostics
                self._draw_capture_diagnostics(frame, piece, x, y, w, h)

        except Exception as e:
            logger.error(f"Error drawing tracked pieces: {e}", exc_info=True)
            logger.error(f"Piece data: {piece if 'piece' in locals() else 'N/A'}")

        return frame

    def _draw_zone_indicator(self, frame: np.ndarray, piece: dict,
                             x: int, y: int, w: int, h: int):
        """Draw colored circle indicator showing which zone the piece is in."""
        # Position indicator in top-right corner of bounding box
        indicator_x = x + w - 15
        indicator_y = y + 15

        # Determine zone color
        if piece.get("in_exit_zone", False):
            zone_color = self.config.ZONE_INDICATOR_EXIT
            zone_label = "EXIT"
        elif piece.get("in_valid_zone", False):
            zone_color = self.config.ZONE_INDICATOR_VALID
            zone_label = "VALID"
        elif piece.get("in_entry_zone", False):
            zone_color = self.config.ZONE_INDICATOR_ENTRY
            zone_label = "ENTRY"
        else:
            zone_color = self.config.ZONE_INDICATOR_NONE
            zone_label = "NONE"

        # Draw filled circle with black border
        cv2.circle(frame, (indicator_x, indicator_y), 8, (0, 0, 0), -1)
        cv2.circle(frame, (indicator_x, indicator_y), 6, zone_color, -1)

        # Optionally add small text label next to circle
        label_x = indicator_x + 12
        cv2.putText(frame, zone_label, (label_x, indicator_y + 4),
                    self.config.FONT, 0.3, zone_color, 1)

    def _draw_capture_diagnostics(self, frame: np.ndarray, piece: dict,
                                  x: int, y: int, w: int, h: int):
        """Draw diagnostic information showing why a piece isn't being captured."""
        # Position diagnostics below the bounding box
        diag_y = y + h + 15
        diag_x = x

        # Check capture eligibility criteria
        fully_in_frame = piece.get("fully_in_frame", False)
        in_valid_zone = piece.get("in_valid_zone", False)
        update_count = piece.get("update_count", 0)
        captured = piece.get("captured", False)

        # Build status indicators
        indicators = []

        # Fully in frame check
        if fully_in_frame:
            indicators.append(("FIF", (0, 255, 0)))  # Green = good
        else:
            indicators.append(("FIF", (0, 0, 255)))  # Red = bad

        # Valid zone check
        if in_valid_zone:
            indicators.append(("VZ", (0, 255, 0)))
        else:
            indicators.append(("VZ", (0, 0, 255)))

        # Update count check (assume min_stability_updates = 5)
        if update_count >= 5:
            indicators.append((f"U{update_count}", (0, 255, 0)))
        else:
            indicators.append((f"U{update_count}", (0, 165, 255)))  # Orange = pending

        # Captured check
        if not captured:
            indicators.append(("RDY", (0, 255, 0)))
        else:
            indicators.append(("CAP", (128, 128, 128)))  # Gray = already captured

        # Draw indicators horizontally
        offset_x = 0
        for label, color in indicators:
            # Draw background
            label_size = cv2.getTextSize(label, self.config.FONT, 0.4, 1)[0]
            cv2.rectangle(frame,
                          (diag_x + offset_x, diag_y - 12),
                          (diag_x + offset_x + label_size[0] + 4, diag_y + 2),
                          (0, 0, 0), -1)
            # Draw text
            cv2.putText(frame, label, (diag_x + offset_x + 2, diag_y - 2),
                        self.config.FONT, 0.4, color, 1)
            offset_x += label_size[0] + 8

    def draw_statistics_overlay(self, frame: np.ndarray,
                                stats: Dict[str, Any]) -> np.ndarray:
        """Draw performance statistics and diagnostic legend on frame."""
        # Create semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Display statistics
        y_offset = 30
        text_color = (255, 255, 255)

        stats_text = [
            f"FPS: {stats.get('fps', 0):.1f}",
            f"Total Pieces: {stats.get('total_pieces', 0)}",
            f"Detections: {stats.get('detections_this_frame', 0)}",
            f"In Valid Zone: {stats.get('status_breakdown', {}).get('stable', 0)}",
            f"Processing: {stats.get('status_breakdown', {}).get('processing', 0)}"
        ]

        for text in stats_text:
            cv2.putText(frame, text, (20, y_offset),
                        self.config.FONT, 0.5, text_color, 1)
            y_offset += 25

        # Add diagnostic legend
        y_offset += 10
        cv2.putText(frame, "--- Capture Diagnostics ---", (20, y_offset),
                    self.config.FONT, 0.5, (200, 200, 200), 1)
        y_offset += 20

        legend_items = [
            ("FIF = Fully In Frame", (0, 255, 0)),
            ("VZ = Valid Zone", (0, 255, 0)),
            ("U# = Update Count (need 5+)", (0, 165, 255)),
            ("RDY/CAP = Ready/Captured", (0, 255, 0))
        ]

        for text, color in legend_items:
            cv2.putText(frame, text, (20, y_offset),
                        self.config.FONT, 0.35, color, 1)
            y_offset += 18

        return frame


# ============================================================================
# CAPTURE PREVIEW MANAGER
# ============================================================================

class CapturePreviewManager:
    """Manages the display of recently captured piece images."""

    def __init__(self, config: TestConfiguration):
        self.config = config
        self.capture_history = []  # List of (piece_id, image_path, timestamp)

    def add_capture(self, piece_id: int, image_path: str):
        """Add a newly captured image to the history."""
        self.capture_history.append({
            'piece_id': piece_id,
            'image_path': image_path,
            'timestamp': time.time()
        })

        # Keep only the most recent captures
        if len(self.capture_history) > self.config.CAPTURE_PREVIEW_MAX:
            self.capture_history.pop(0)

    def create_preview_panel(self, panel_width: int, panel_height: int) -> np.ndarray:
        """
        Create a panel showing recently captured images in a grid.

        Args:
            panel_width: Width of the preview panel
            panel_height: Height of the preview panel

        Returns:
            Preview panel image
        """
        # Create black background panel
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

        if not self.capture_history:
            # Show "No captures yet" message
            text = "No Captures Yet"
            text_size = cv2.getTextSize(text, self.config.FONT, 0.8, 2)[0]
            text_x = (panel_width - text_size[0]) // 2
            text_y = panel_height // 2
            cv2.putText(panel, text, (text_x, text_y),
                        self.config.FONT, 0.8, (128, 128, 128), 2)
            return panel

        # Calculate cell dimensions
        cols = self.config.CAPTURE_PREVIEW_COLS
        rows = self.config.CAPTURE_PREVIEW_ROWS
        cell_width = panel_width // cols
        cell_height = panel_height // rows

        # Draw grid with captured images
        for idx, capture in enumerate(self.capture_history):
            if idx >= self.config.CAPTURE_PREVIEW_MAX:
                break

            row = idx // cols
            col = idx % cols

            # Calculate cell position
            cell_x = col * cell_width
            cell_y = row * cell_height

            # Try to load the captured image
            img_path = Path(capture['image_path'])
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Resize to fit cell (with padding)
                    padding = 10
                    target_w = cell_width - padding * 2
                    target_h = cell_height - padding * 2

                    # Maintain aspect ratio
                    h, w = img.shape[:2]
                    scale = min(target_w / w, target_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)

                    img_resized = cv2.resize(img, (new_w, new_h))

                    # Center in cell
                    offset_x = cell_x + (cell_width - new_w) // 2
                    offset_y = cell_y + (cell_height - new_h) // 2

                    # Place image in panel
                    panel[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = img_resized

                    # Add piece ID label
                    label = f"ID: {capture['piece_id']}"
                    cv2.putText(panel, label,
                                (cell_x + 5, cell_y + 20),
                                self.config.FONT, 0.5, (255, 255, 255), 1)

            # Draw cell border
            cv2.rectangle(panel, (cell_x, cell_y),
                          (cell_x + cell_width, cell_y + cell_height),
                          (50, 50, 50), 1)

        # Add title
        title = "Recently Captured Pieces"
        cv2.putText(panel, title, (10, 25),
                    self.config.FONT, 0.7, (255, 255, 255), 2)

        return panel


# ============================================================================
# MAIN TEST ENVIRONMENT CLASS
# ============================================================================

class CVPipelineTestEnvironment:
    """Main test environment that integrates camera and CV pipeline."""

    def __init__(self):
        """Initialize the test environment."""
        logger.info("Initializing CV Pipeline Test Environment")

        self.config = TestConfiguration()
        self.visualizer = VisualizationOverlay(self.config)
        self.capture_preview_manager = CapturePreviewManager(self.config)

        # Initialize components
        self.config_manager = None
        self.camera = None
        self.detector_coordinator = None
        self.zone_manager = None
        self.capture_controller = None

        # State tracking
        self.latest_frame = None
        self.latest_results = None
        self.frame_count = 0
        self.last_log_time = time.time()

        # Debug mode toggle
        self.show_debug_view = self.config.SHOW_BACKGROUND_MASK

    def initialize_components(self):
        """Initialize all CV pipeline components."""
        logger.info("Initializing CV pipeline components")

        # Create configuration manager
        self.config_manager = create_config_manager("config.json")

        # Validate configuration
        validation_report = self.config_manager.get_validation_report()
        if not validation_report["valid"]:
            logger.warning("Configuration validation issues detected:")
            for module, report in validation_report["modules"].items():
                if not report["valid"]:
                    logger.warning(f"  {module}: {report['errors']}")

        # Initialize camera
        logger.info("Initializing camera module")
        self.camera = create_camera("webcam", self.config_manager)

        # Initialize CV pipeline components
        logger.info("Creating detector coordinator")
        self.detector_coordinator = create_detector_coordinator(self.config_manager)

        # Get the zone_manager directly from detector_coordinator to ensure consistency
        logger.info("Getting zone manager from detector coordinator")
        self.zone_manager = self.detector_coordinator.zone_manager

        logger.info("Creating capture controller")
        # Capture controller shares the same zone_manager instance as the pipeline
        self.capture_controller = create_capture_controller(
            self.config_manager, self.zone_manager
        )

        # Get first frame to set up ROI
        logger.info("Waiting for first frame to configure ROI")
        time.sleep(0.5)  # Let camera warm up
        sample_frame = self.camera.get_frame()

        if sample_frame is not None:
            # Get ROI from config
            roi_config = self.config_manager.get_module_config(ModuleConfig.DETECTOR_ROI.value)
            roi_coords = (
                roi_config["x"],
                roi_config["y"],
                roi_config["w"],
                roi_config["h"]
            )

            logger.info(f"Setting ROI from config: {roi_coords}")
            self.detector_coordinator.set_roi_from_sample_frame(
                sample_frame, roi_coords
            )

            # Configure zone manager with ROI
            roi = RegionOfInterest(*roi_coords)
            self.zone_manager.set_roi(roi)

            logger.info("ROI and zones configured successfully")
        else:
            logger.error("Failed to get sample frame for ROI configuration")
            raise RuntimeError("Could not initialize ROI")

        logger.info("All components initialized successfully")

    def process_frame_callback(self, frame: np.ndarray):
        """
        Callback function for camera frame consumer.

        Args:
            frame: Captured camera frame
        """
        try:
            self.latest_frame = frame.copy()
            self.frame_count += 1

            # Process frame through CV pipeline
            results = self.detector_coordinator.process_frame_for_consumer(frame)

            if results.get("error"):
                logger.error(f"Pipeline error: {results['error']}")
                return

            # Get tracked pieces
            tracked_pieces_objects = self.detector_coordinator.get_tracked_pieces()

            # Get tracked pieces with zone flags for capture controller
            # Note: get_tracked_pieces() returns TrackedPiece objects with complete info
            tracked_pieces_objects = self.detector_coordinator.get_tracked_pieces()

            # Check for captures
            capture_packages = self.capture_controller.check_and_process_captures(
                frame, tracked_pieces_objects)

            # Log captures and add to preview
            if capture_packages:
                for package in capture_packages:
                    self.log_capture_event(package)

                    # Construct image path matching capture_controller's format
                    # capture_controller saves as: piece_{id}_{timestamp}.jpg
                    capture_dir = self.config_manager.get_module_config(
                        ModuleConfig.DETECTOR.value).get("capture_directory", "captured_pieces")
                    timestamp_str = str(int(package.capture_timestamp))
                    filename = f"piece_{package.piece_id}_{timestamp_str}.jpg"
                    img_path = f"{capture_dir}/{filename}"

                    logger.info(f"Adding captured image to preview: {img_path}")

                    self.capture_preview_manager.add_capture(package.piece_id, str(img_path))

            # Store results
            self.latest_results = results

            # Debug: Log tracked pieces info
            tracked_pieces_from_results = results.get("tracked_pieces", [])
            if self.frame_count % 30 == 0:  # Log every 30 frames
                logger.debug(f"Frame {self.frame_count}: {len(tracked_pieces_from_results)} pieces in results")
                if tracked_pieces_from_results:
                    logger.debug(f"First piece keys: {list(tracked_pieces_from_results[0].keys())}")

            # Log tracking information periodically
            current_time = time.time()
            if current_time - self.last_log_time >= 2.0:
                self.log_tracking_status(results)
                self.last_log_time = current_time

        except Exception as e:
            logger.error(f"Error in frame processing: {e}", exc_info=True)

    def log_capture_event(self, capture_package):
        """Log when a piece is captured."""
        logger.info("=" * 60)
        logger.info(f"CAPTURE EVENT - Piece ID: {capture_package.piece_id}")
        logger.info(f"  Position: {capture_package.capture_position}")
        logger.info(f"  Bbox: {capture_package.original_bbox}")
        logger.info(f"  Image size: {capture_package.processed_image.shape}")
        logger.info(f"  Timestamp: {capture_package.capture_timestamp:.2f}")
        logger.info("=" * 60)

    def log_tracking_status(self, results: Dict[str, Any]):
        """Log current tracking status with capture diagnostics."""
        stats = results.get("statistics", {})
        pieces = results.get("tracked_pieces", [])

        logger.info("-" * 60)
        logger.info(f"TRACKING STATUS (Frame {self.frame_count})")
        logger.info(f"  FPS: {stats.get('fps', 0):.1f}")
        logger.info(f"  Total Pieces: {len(pieces)}")
        logger.info(f"  Detections This Frame: {stats.get('detections_this_frame', 0)}")

        if pieces:
            logger.info(f"  Active Pieces:")
            for piece in pieces:
                center = piece['center']
                # Determine zone for logging
                zone = "NONE"
                if piece.get('in_exit_zone'):
                    zone = "EXIT"
                elif piece.get('in_valid_zone'):
                    zone = "VALID"
                elif piece.get('in_entry_zone'):
                    zone = "ENTRY"

                # Check capture eligibility
                fully_in_frame = piece.get('fully_in_frame', False)
                update_count = piece.get('update_count', 0)
                captured = piece.get('captured', False)

                # Build capture status string
                capture_status = []
                if not fully_in_frame:
                    capture_status.append("NOT_FULLY_IN_FRAME")
                if not piece.get('in_valid_zone'):
                    capture_status.append("NOT_IN_VALID_ZONE")
                if update_count < 5:
                    capture_status.append(f"INSUFFICIENT_UPDATES({update_count}<5)")
                if captured:
                    capture_status.append("ALREADY_CAPTURED")

                if not capture_status:
                    capture_status.append("CAPTURE_ELIGIBLE")

                logger.info(f"    ID {piece['id']}: "
                            f"pos=({center[0]:.0f}, {center[1]:.0f}), "
                            f"zone={zone}, "
                            f"status={piece['status']}, "
                            f"capture={' | '.join(capture_status)}")
        else:
            logger.info("  No pieces currently tracked")

        logger.info("-" * 60)

    def create_visualization_frame(self) -> Optional[np.ndarray]:
        """
        Create the main visualization frame with all overlays.

        Returns:
            Visualization frame or None if no frame available
        """
        if self.latest_frame is None or self.latest_results is None:
            return None

        # Start with copy of original frame
        viz_frame = self.latest_frame.copy()

        # Draw ROI overlay
        roi_info = self.latest_results.get("roi_info", {})
        viz_frame = self.visualizer.draw_roi_overlay(viz_frame, roi_info)

        # Draw zone overlays
        zone_info = self.zone_manager.get_zone_boundaries_info()
        viz_frame = self.visualizer.draw_zone_overlays(viz_frame, zone_info)

        # Draw tracked pieces with zone indicators
        tracked_pieces = self.latest_results.get("tracked_pieces", [])
        viz_frame = self.visualizer.draw_tracked_pieces(viz_frame, tracked_pieces)

        # Draw statistics
        stats = self.latest_results.get("statistics", {})
        viz_frame = self.visualizer.draw_statistics_overlay(viz_frame, stats)

        return viz_frame

    def create_debug_mask_view(self) -> Optional[np.ndarray]:
        """
        Create background subtraction mask view for debugging.

        Returns:
            Background mask as BGR image or None if unavailable
        """
        if not hasattr(self.detector_coordinator.vision_processor, 'get_debug_mask'):
            return None

        mask = self.detector_coordinator.vision_processor.get_debug_mask()
        if mask is None:
            return None

        # Convert grayscale mask to BGR for display
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def create_composite_view(self) -> Optional[np.ndarray]:
        """
        Create composite view with main visualization and capture preview.

        If debug mode is enabled, shows a 2x2 grid with:
          - Top-left: Original frame
          - Top-right: Background mask
          - Bottom-left: Tracked visualization
          - Bottom-right: Capture preview

        Otherwise shows side-by-side:
          - Left: Tracked visualization
          - Right: Capture preview

        Returns:
            Composite view or None if no frame available
        """
        viz_frame = self.create_visualization_frame()
        if viz_frame is None:
            return None

        frame_h, frame_w = viz_frame.shape[:2]

        if self.show_debug_view:
            # Create 2x2 grid layout
            grid_h = frame_h * 2
            grid_w = frame_w * 2
            composite = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            # Top-left: Original frame
            if self.latest_frame is not None:
                composite[0:frame_h, 0:frame_w] = self.latest_frame
                cv2.putText(composite, "ORIGINAL", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Top-right: Background mask
            debug_mask = self.create_debug_mask_view()
            if debug_mask is not None:
                composite[0:frame_h, frame_w:frame_w * 2] = debug_mask
                cv2.putText(composite, "BACKGROUND MASK", (frame_w + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Show "unavailable" message
                text = "Background Mask Unavailable"
                cv2.putText(composite, text, (frame_w + 10, frame_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            # Bottom-left: Tracked visualization
            composite[frame_h:frame_h * 2, 0:frame_w] = viz_frame
            cv2.putText(composite, "TRACKED", (10, frame_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Bottom-right: Capture preview
            preview_panel = self.capture_preview_manager.create_preview_panel(
                frame_w, frame_h)
            composite[frame_h:frame_h * 2, frame_w:frame_w * 2] = preview_panel

        else:
            # Side-by-side layout: visualization + capture preview
            preview_w = frame_w // 2
            preview_panel = self.capture_preview_manager.create_preview_panel(
                preview_w, frame_h)

            composite = np.zeros((frame_h, frame_w + preview_w, 3), dtype=np.uint8)
            composite[0:frame_h, 0:frame_w] = viz_frame
            composite[0:frame_h, frame_w:frame_w + preview_w] = preview_panel

        return composite

    def run(self):
        """Main test environment loop."""
        try:
            # Initialize all components
            self.initialize_components()

            # Register frame consumer with camera
            logger.info("Registering CV pipeline as frame consumer")
            self.camera.register_consumer(
                name="cv_pipeline",
                callback=self.process_frame_callback,
                processing_type="sync",
                priority=10
            )

            # Start camera capture
            logger.info("Starting camera capture")
            self.camera.start_capture()

            # Create display window
            cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)

            logger.info("=" * 60)
            logger.info("TEST ENVIRONMENT RUNNING")
            logger.info("Press 'q' to quit")
            logger.info("Press 'd' to toggle debug view (background mask)")
            logger.info("Press 's' to save current frame")
            logger.info("Press 'r' to reset detection system")
            logger.info("")
            logger.info("CAPTURE DIAGNOSTICS:")
            logger.info("  Look below each piece for capture status indicators:")
            logger.info("  FIF = Fully In Frame (green=yes, red=no)")
            logger.info("  VZ = Valid Zone (green=yes, red=no)")
            logger.info("  U# = Update Count (green=5+, orange=<5)")
            logger.info("  RDY/CAP = Ready or Already Captured")
            logger.info("")
            capture_dir = self.config_manager.get_module_config(
                ModuleConfig.DETECTOR.value).get("capture_directory", "captured_pieces")
            logger.info(f"Captured images will be saved to: {capture_dir}/")
            logger.info("=" * 60)

            # Main display loop
            while True:
                # Create composite view
                composite_frame = self.create_composite_view()

                if composite_frame is not None:
                    cv2.imshow(self.config.WINDOW_NAME, composite_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('d'):
                    # Toggle debug view
                    self.show_debug_view = not self.show_debug_view
                    logger.info(f"Debug view: {'ON' if self.show_debug_view else 'OFF'}")
                elif key == ord('s'):
                    # Save current frame
                    if composite_frame is not None:
                        timestamp = int(time.time())
                        filename = f"test_frame_{timestamp}.jpg"
                        cv2.imwrite(filename, composite_frame)
                        logger.info(f"Saved frame to {filename}")
                elif key == ord('r'):
                    # Reset detection system
                    logger.info("Resetting detection system")
                    self.detector_coordinator.reset_detection_system()
                    logger.info("Detection system reset complete")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in test environment: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up test environment")

        if self.camera:
            self.camera.stop_capture()
            self.camera.release()

        cv2.destroyAllWindows()

        # Print final statistics
        logger.info("=" * 60)
        logger.info("TEST ENVIRONMENT FINAL STATISTICS")
        logger.info(f"  Total Frames Processed: {self.frame_count}")
        logger.info(f"  Total Captures: {len(self.capture_preview_manager.capture_history)}")

        if self.capture_controller:
            capture_stats = self.capture_controller.get_capture_statistics()
            logger.info(f"  Capture Controller Statistics:")
            for key, value in capture_stats.items():
                logger.info(f"    {key}: {value}")

        logger.info("=" * 60)
        logger.info("Test environment shutdown complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the test environment."""
    print("\n" + "=" * 60)
    print("CV PIPELINE TEST ENVIRONMENT")
    print("=" * 60)
    print("\nThis test environment will:")
    print("  - Capture live camera feed")
    print("  - Process frames through the CV pipeline")
    print("  - Display tracked pieces with zone indicators")
    print("  - Show recently captured piece images")
    print("  - Optionally show background subtraction mask (press 'd')")
    print("  - Log tracking and capture events")
    print("\n" + "=" * 60 + "\n")

    # Create and run test environment
    test_env = CVPipelineTestEnvironment()
    test_env.run()


if __name__ == "__main__":
    main()