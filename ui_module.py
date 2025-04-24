"""
ui_module.py - User interface module for the Lego sorting application

This module provides visualization of the Lego sorting system, including
the main video feed with detection visualization, metrics dashboard,
recently processed piece display, and sorting information.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Union


class UIManager:
    """Manages the user interface elements for the Lego sorting application."""

    def __init__(self, config_manager=None):
        """Initialize UI manager with optional configuration.

        Args:
            config_manager: Optional configuration manager instance
        """
        # Default configuration
        self.config = {
            "panels": {
                "metrics_dashboard": True,
                "processed_piece": True,
                "sorting_information": True,
                "help_overlay": True
            },
            "opacity": 0.7,
            "colors": {
                "panel_bg": (30, 30, 30),
                "text": (255, 255, 255),
                "roi": (0, 255, 0),
                "entry_zone": (255, 0, 0),
                "exit_zone": (0, 0, 255),
                "active_piece": (0, 0, 255),
                "captured_piece": (0, 255, 0),
                "processing_piece": (255, 0, 255),
                "highlight": (0, 255, 255),
                "error": (0, 0, 255)
            }
        }

        # Update with provided config manager
        if config_manager:
            ui_config = config_manager.get_section("ui")
            if ui_config:
                # Update panels configuration
                if "panels" in ui_config:
                    for key, value in ui_config["panels"].items():
                        if key in self.config["panels"]:
                            self.config["panels"][key] = value

                # Update other configuration values
                for key, value in ui_config.items():
                    if key != "panels" and key in self.config:
                        self.config[key] = value

        # Panel layout information
        self.layout = {
            "metrics": (10, 10, 300, 150),
            "processed": (None, 10, 300, 200),  # x will be calculated based on frame width
            "sorting": (None, None, None, 200)  # x, y, width will be calculated
        }

        # Toggle control keys
        self.toggle_keys = {
            ord('m'): "metrics_dashboard",
            ord('p'): "processed_piece",
            ord('s'): "sorting_information",
            ord('h'): "help_overlay"
        }

    def calculate_layout(self, frame_width: int, frame_height: int) -> None:
        """Calculate panel positions based on frame size.

        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
        """
        # Update positions that depend on frame size
        self.layout["processed"] = (frame_width - 310, 10, 300, 200)
        self.layout["sorting"] = (160, frame_height - 190, frame_width - 320, 180)

    def create_display_frame(self, frame: np.ndarray,
                          detector_data: Optional[Dict[str, Any]] = None,
                          system_data: Optional[Dict[str, Any]] = None,
                          piece_data: Optional[Dict[str, Any]] = None,
                          sorting_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Create the complete UI display frame.

        Args:
            frame: Original video frame
            detector_data: Data from detector module (ROI, zones, tracked pieces)
            system_data: System performance metrics
            piece_data: Recently processed piece information
            sorting_data: Sorting strategy and bin assignments

        Returns:
            Display frame with all enabled UI elements
        """
        # Create a copy of the frame
        display = frame.copy()

        # Calculate layout based on frame size
        h, w = display.shape[:2]
        self.calculate_layout(w, h)

        # Always render the main video feed with basic detection visualization
        if detector_data:
            display = self._render_main_feed(display, detector_data)

        # Only render optional panels if enabled in config
        if self.config["panels"]["metrics_dashboard"] and system_data:
            display = self._render_metrics_dashboard(display, system_data)

        if self.config["panels"]["processed_piece"] and piece_data:
            display = self._render_processed_piece(display, piece_data)

        if self.config["panels"]["sorting_information"] and sorting_data:
            display = self._render_sorting_info(display, sorting_data)

        # Add help overlay if enabled
        if self.config["panels"]["help_overlay"]:
            display = self._render_help_overlay(display)

        return display

    def add_sorting_info(self, frame: np.ndarray, sorting_data: Dict[str, Any]) -> np.ndarray:
        """Add the sorting information panel to the frame.

        Args:
            frame: Video frame
            sorting_data: Sorting strategy and bin assignments

        Returns:
            Frame with sorting information added
        """
        if self.config["panels"]["sorting_information"] and sorting_data:
            h, w = frame.shape[:2]
            self.calculate_layout(w, h)
            return self._render_sorting_info(frame, sorting_data)
        return frame

    def _create_panel(self, frame: np.ndarray, x: int, y: int,
                    width: int, height: int,
                    title: Optional[str] = None) -> np.ndarray:
        """Create a semi-transparent panel.

        Args:
            frame: Video frame
            x, y: Top-left corner coordinates
            width, height: Panel dimensions
            title: Optional panel title

        Returns:
            Frame with panel added
        """
        # Create a copy of the input frame
        result = frame.copy()

        # Create semi-transparent overlay
        overlay = result.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height),
                      self.config["colors"]["panel_bg"], -1)

        # Apply transparency
        cv2.addWeighted(overlay, self.config["opacity"], result,
                        1 - self.config["opacity"], 0, result)

        # Add title if provided
        if title:
            cv2.putText(result, str(title), (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, self.config["colors"]["text"], 2)
            # Add underline
            cv2.line(result, (x + 10, y + 30), (x + len(str(title)) * 14, y + 30),
                     self.config["colors"]["text"], 1)

        return result

    def _render_main_feed(self, frame: np.ndarray,
                        detector_data: Dict[str, Any]) -> np.ndarray:
        """Render the main video feed with detection visualization.

        Args:
            frame: Video frame
            detector_data: Data from detector module

        Returns:
            Frame with detection visualization
        """
        display = frame.copy()

        # Draw ROI rectangle if available
        if "roi" in detector_data:
            x, y, w, h = detector_data["roi"]
            cv2.rectangle(display, (x, y), (x + w, y + h),
                          self.config["colors"]["roi"], 2)

        # Draw entry/exit zones if available
        if "entry_zone" in detector_data and "exit_zone" in detector_data:
            roi_y = detector_data["roi"][1]
            roi_h = detector_data["roi"][3]

            # Entry zone
            entry_x = detector_data["entry_zone"][1]
            cv2.line(display, (entry_x, roi_y), (entry_x, roi_y + roi_h),
                     self.config["colors"]["entry_zone"], 2)

            # Exit zone
            exit_x = detector_data["exit_zone"][0]
            cv2.line(display, (exit_x, roi_y), (exit_x, roi_y + roi_h),
                     self.config["colors"]["exit_zone"], 2)

        # Draw tracked pieces
        if "tracked_pieces" in detector_data:
            for piece in detector_data["tracked_pieces"]:
                # Get bounding box coordinates with ROI offset
                roi_x, roi_y = detector_data["roi"][0], detector_data["roi"][1]
                px, py, pw, ph = piece["bbox"]
                px += roi_x  # Add ROI x offset
                py += roi_y  # Add ROI y offset

                # Different colors based on status
                if piece.get("being_processed", False):
                    color = self.config["colors"]["processing_piece"]
                elif piece.get("captured", False):
                    color = self.config["colors"]["captured_piece"]
                else:
                    color = self.config["colors"]["active_piece"]

                # Draw bounding box
                cv2.rectangle(display, (px, py), (px + pw, py + ph), color, 2)

                # Add ID and update count
                piece_id = piece.get('id', 0)
                update_count = piece.get('update_count', 0)
                label = f"ID:{piece_id} U:{update_count}"
                cv2.putText(display, label, (px, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return display

    def _render_metrics_dashboard(self, frame: np.ndarray,
                                system_data: Dict[str, Any]) -> np.ndarray:
        """Render the metrics dashboard panel.

        Args:
            frame: Video frame
            system_data: System performance metrics

        Returns:
            Frame with metrics dashboard added
        """
        x, y, w, h = self.layout["metrics"]

        # Create panel
        display = self._create_panel(frame, x, y, w, h, "Metrics Dashboard")

        # Add metrics text
        if "fps" in system_data:
            fps_text = f"FPS: {system_data['fps']:.1f}"
            cv2.putText(display, fps_text, (x + 20, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config["colors"]["text"], 1)

        if "queue_size" in system_data:
            queue_text = f"Queue: {system_data['queue_size']}"
            cv2.putText(display, queue_text, (x + 20, y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config["colors"]["text"], 1)

        if "processed_count" in system_data:
            processed_text = f"Processed: {system_data['processed_count']}"
            cv2.putText(display, processed_text, (x + 20, y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config["colors"]["text"], 1)

        if "uptime" in system_data:
            uptime = float(system_data["uptime"])
            uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime))
            uptime_text = f"Uptime: {uptime_str}"
            cv2.putText(display, uptime_text, (x + 20, y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config["colors"]["text"], 1)

        return display

    def _render_processed_piece(self, frame: np.ndarray,
                              piece_data: Dict[str, Any]) -> np.ndarray:
        """Render the recently processed piece panel.

        Args:
            frame: Video frame
            piece_data: Recently processed piece information

        Returns:
            Frame with processed piece panel added
        """
        x, y, w, h = self.layout["processed"]

        # Create panel
        display = self._create_panel(frame, x, y, w, h, "Recently Processed")

        # Display piece image if available
        if "image" in piece_data and piece_data["image"] is not None:
            img = piece_data["image"]
            # Resize image to fit in panel
            img_h, img_w = img.shape[:2]
            max_img_w = w - 40
            max_img_h = 100
            scale = min(max_img_w / img_w, max_img_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)

            resized_img = cv2.resize(img, (new_w, new_h))

            # Calculate position to center the image
            img_x = x + (w - new_w) // 2
            img_y = y + 40

            # Place image on panel
            display[img_y:img_y + new_h, img_x:img_x + new_w] = resized_img

            # Draw border around image
            cv2.rectangle(display, (img_x, img_y), (img_x + new_w, img_y + new_h),
                          (255, 255, 255), 1)

        # Add piece information
        text_y = y + 150

        # Add error information if piece has error status
        if piece_data.get("status") == "error":
            error_msg = piece_data.get("error", "Unknown error")
            cv2.putText(display, "Error:", (x + 20, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config["colors"]["error"], 1)
            text_y += 20

            # Truncate long error messages
            if len(str(error_msg)) > 30:
                error_msg = str(error_msg)[:27] + "..."

            cv2.putText(display, str(error_msg), (x + 20, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config["colors"]["error"], 1)
            text_y += 20
        else:
            # Regular info for successful pieces
            if "element_id" in piece_data:
                element_text = f"ID: {piece_data['element_id']}"
                cv2.putText(display, element_text, (x + 20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config["colors"]["text"], 1)
                text_y += 20

            if "name" in piece_data:
                name_text = f"Name: {piece_data['name']}"
                cv2.putText(display, name_text, (x + 20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config["colors"]["text"], 1)
                text_y += 20

            if "bin_number" in piece_data:
                bin_text = f"Bin: {piece_data['bin_number']}"
                cv2.putText(display, bin_text, (x + 20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config["colors"]["text"], 1)

        return display

    def _render_sorting_info(self, frame: np.ndarray,
                           sorting_data: Dict[str, Any]) -> np.ndarray:
        """Render the sorting information panel.

        Args:
            frame: Video frame
            sorting_data: Sorting strategy and bin assignments

        Returns:
            Frame with sorting information panel added
        """
        x, y, w, h = self.layout["sorting"]

        # Create panel
        display = self._create_panel(frame, x, y, w, h, "Sorting Information")

        # Add sorting strategy
        if "strategy" in sorting_data:
            strategy_text = f"Strategy: {sorting_data['strategy']}"
            cv2.putText(display, strategy_text, (x + 20, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config["colors"]["text"], 1)

        # Add bin assignments
        if "bin_assignments" in sorting_data:
            # Calculate layout for bin assignments
            bins_per_row = 3
            bin_w = (w - 40) // bins_per_row
            bin_h = 40
            bin_x_start = x + 20
            bin_y_start = y + 90

            for i, (bin_num, category) in enumerate(sorting_data["bin_assignments"].items()):
                row = i // bins_per_row
                col = i % bins_per_row

                bin_x = bin_x_start + col * bin_w
                bin_y = bin_y_start + row * bin_h

                # Draw bin rectangle
                cv2.rectangle(display, (bin_x, bin_y), (bin_x + bin_w - 10, bin_y + bin_h - 5),
                              (100, 100, 100), 1)

                # Add bin label
                bin_text = f"Bin {bin_num}: {category}"
                cv2.putText(display, bin_text, (bin_x + 5, bin_y + bin_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config["colors"]["text"], 1)

        return display

    def _render_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render the help overlay panel.

        Args:
            frame: Video frame

        Returns:
            Frame with help overlay added
        """
        h, w = frame.shape[:2]
        x = 10
        y = h - 160
        width = 250
        height = 150

        # Create panel
        display = self._create_panel(frame, x, y, width, height, "Controls")

        # Add key bindings
        text_y = y + 60

        key_bindings = [
            ("M", "Toggle Metrics Dashboard"),
            ("P", "Toggle Processed Piece"),
            ("S", "Toggle Sorting Information"),
            ("H", "Toggle Help Overlay"),
            ("ESC", "Exit Program")
        ]

        for key, description in key_bindings:
            binding_text = f"{key}: {description}"
            cv2.putText(display, binding_text, (x + 20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config["colors"]["text"], 1)
            text_y += 20

        return display

    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input for UI controls.

        Args:
            key: Key code from cv2.waitKey()

        Returns:
            True if program should continue, False if should exit
        """
        # Check for ESC key
        if key == 27:  # ESC
            return False

        # Check for panel toggle keys
        if key in self.toggle_keys:
            panel_name = self.toggle_keys[key]
            self.config["panels"][panel_name] = not self.config["panels"][panel_name]
            print(f"Toggled {panel_name}: {self.config['panels'][panel_name]}")

        return True


# Factory function
def create_ui_manager(config_manager=None):
    """Create a UI manager instance.

    Args:
        config_manager: Optional configuration manager

    Returns:
        UIManager instance
    """
    return UIManager(config_manager)


# Test case
if __name__ == "__main__":
    """Test the UI module with mock data."""
    # Create UI manager
    ui_manager = create_ui_manager()

    # Create a test frame (black canvas)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Create mock data
    detector_data = {
        "roi": (200, 100, 880, 400),
        "entry_zone": (200, 280),
        "exit_zone": (1000, 1080),
        "tracked_pieces": [
            {
                "id": 1,
                "bbox": (250, 150, 80, 60),
                "update_count": 7,
                "captured": False,
                "being_processed": False
            },
            {
                "id": 2,
                "bbox": (400, 200, 70, 50),
                "update_count": 12,
                "captured": True,
                "being_processed": False
            },
            {
                "id": 3,
                "bbox": (600, 300, 90, 70),
                "update_count": 5,
                "captured": False,
                "being_processed": True
            }
        ]
    }

    system_data = {
        "fps": 29.7,
        "queue_size": 2,
        "processed_count": 42,
        "uptime": 305  # seconds
    }

    piece_data = {
        "image": np.ones((120, 160, 3), dtype=np.uint8) * 150,  # Gray test image
        "element_id": "3001",
        "name": "Brick 2x4",
        "bin_number": 2,
        "primary_category": "Basic",
        "secondary_category": "Brick",
        "status": "success"
    }

    # Error piece example
    error_piece_data = {
        "element_id": "Unknown",
        "name": "Unknown Piece",
        "error": "Low confidence score: 0.45 < 0.7",
        "status": "error"
    }

    # Draw a simple brick shape on the test image
    brick_img = piece_data["image"]
    cv2.rectangle(brick_img, (20, 40), (140, 80), (0, 0, 200), -1)

    sorting_data = {
        "strategy": "Sorting by primary category",
        "bin_assignments": {
            "1": "Basic",
            "2": "Technic",
            "3": "Angle",
            "4": "Plate"
        }
    }

    # Main loop
    running = True
    show_error = False
    while running:
        # Toggle between normal and error pieces every 2 seconds
        current_time = time.time()
        if int(current_time) % 4 < 2:
            current_piece_data = piece_data
        else:
            current_piece_data = error_piece_data

        # Create UI
        display = ui_manager.create_display_frame(
            frame, detector_data, system_data, current_piece_data, sorting_data
        )

        # Display the result
        cv2.imshow("Lego Sorting UI Test", display)

        # Wait for keyboard input
        key = cv2.waitKey(100) & 0xFF
        running = ui_manager.handle_keyboard_input(key)

    cv2.destroyAllWindows()
    print("UI test completed")