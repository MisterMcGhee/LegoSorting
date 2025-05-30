"""
ui_module.py - User interface module for the Lego sorting application

This module provides visualization of the Lego sorting system, including
the main video feed with detection visualization, metrics dashboard,
recently processed piece display, and sorting information.
"""

import cv2
import numpy as np
import time
import os
from typing import Dict, Any, List, Tuple, Optional, Union


class UIManager:
    """Manages the user interface elements for the Lego sorting application."""

    def __init__(self, config_manager=None, piece_history=None):
        """Initialize UI manager with optional configuration.

        Args:
            config_manager: Optional configuration manager instance
            piece_history: Optional piece history manager
        """
        # Store piece history reference
        self.piece_history = piece_history

        # Add tracking for last displayed piece
        self.last_displayed_piece_id = None

        # Store the last displayed piece data and image
        self.current_piece_data = None
        self.current_piece_image = None

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
        """Create the complete UI display frame."""
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

        # Check for new piece from piece_history if available
        if self.config["panels"]["processed_piece"]:
            new_piece_available = False

            if self.piece_history:
                # Check if there's a new piece available
                latest_piece = self.piece_history.get_latest_piece()

                if latest_piece and (
                    self.last_displayed_piece_id is None or
                    latest_piece.get('piece_id') != self.last_displayed_piece_id
                ):
                    new_piece_available = True
                    self.last_displayed_piece_id = latest_piece.get('piece_id')
                    self.current_piece_data = latest_piece

                    # Load the image if available
                    if 'file_path' in latest_piece and os.path.exists(latest_piece['file_path']):
                        self.current_piece_image = cv2.imread(latest_piece['file_path'])
                    else:
                        self.current_piece_image = None

            # Use direct piece_data if provided and no piece_history is available
            elif piece_data:
                new_piece_available = True
                self.current_piece_data = piece_data
                self.current_piece_image = piece_data.get('image')

            # Always render the current piece data if available
            if self.current_piece_data:
                # Make a copy of the current piece data to add the image for rendering
                piece_to_render = self.current_piece_data.copy() if self.current_piece_data else None

                if piece_to_render and self.current_piece_image is not None:
                    piece_to_render['image'] = self.current_piece_image

                if piece_to_render:
                    display = self._render_processed_piece(display, piece_to_render)

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
            True if the program should continue, False if it should exit
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
def create_ui_manager(config_manager=None, piece_history=None):
    """Create a UI manager instance.

    Args:
        config_manager: Optional configuration manager
        piece_history: Optional piece history manager

    Returns:
        UIManager instance
    """
    return UIManager(config_manager, piece_history)