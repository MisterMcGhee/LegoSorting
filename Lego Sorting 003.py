import cv2
import csv
import os
import datetime
import requests
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from time import time


class CameraManager:
    def __init__(self, directory="LegoPictures"):
        """Initialize camera and directory for storing images."""
        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.count = len([name for name in os.listdir(self.directory)
                          if name.startswith("Lego") and name.endswith(".jpg")]) + 1

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to initialize camera")

    def capture_image(self):
        """Captures and saves an image. Returns filename, current count, and any error."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, "Failed to capture image"

        filename = os.path.join(self.directory, f"Lego{self.count:03}.jpg")
        try:
            cv2.imwrite(filename, frame)
            current_count = self.count  # Store current count before incrementing
            self.count += 1
            return filename, current_count, None
        except Exception as e:
            return None, None, f"Failed to save image: {str(e)}"

    def get_preview_frame(self):
        """Gets a frame for preview/detection without saving."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def cleanup(self):
        """Releases camera resources."""
        self.cap.release()


class PieceIdentifier:
    def __init__(self, csv_path='Lego_Categories.csv'):
        """Initialize piece identifier with category data."""
        self.lego_dict = {}
        self.primary_to_bin = {
            'Basic': 0, 'Wall': 1, 'SNOT': 2, 'Minifig': 3,
            'Clip': 4, 'Hinge': 5, 'Angle': 6, 'Vehicle': 7, 'Curved': 8
        }
        self._load_categories(csv_path)

    def _load_categories(self, filepath):
        """Load piece categories from CSV file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    self.lego_dict[row['element_id']] = {
                        'name': row['name'],
                        'primary_category': row['primary_category'],
                        'secondary_category': row['secondary_category']
                    }
        except FileNotFoundError:
            print(f"Error: Could not find file at {filepath}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    def identify_piece(self, image_path, sort_type='primary', target_category=None):
        """Identify piece and determine sorting bin."""
        api_result = self._send_to_brickognize_api(image_path)
        if "error" in api_result:
            return None, api_result["error"]

        result = self._process_identification(api_result, sort_type, target_category)
        return result, None

    def _send_to_brickognize_api(self, file_name):
        """Send image to Brickognize API."""
        url = "https://api.brickognize.com/predict/"
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']

        if not os.path.isfile(file_name):
            return {"error": f"File not found: {file_name}"}
        if os.path.getsize(file_name) == 0:
            return {"error": "File is empty"}
        if not any(file_name.lower().endswith(ext) for ext in valid_extensions):
            return {"error": f"Invalid file type. Allowed: {', '.join(valid_extensions)}"}

        try:
            with open(file_name, "rb") as image_file:
                files = {"query_image": (file_name, image_file, "image/jpeg")}
                response = requests.post(url, files=files)

            if response.status_code != 200:
                return {"error": f"API request failed: {response.status_code}"}

            data = response.json()
            if "items" in data and data["items"]:
                item = data["items"][0]
                return {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "score": item.get("score")
                }

            return {"error": "No items found in response"}

        except Exception as e:
            return {"error": f"Request error: {str(e)}"}

    def _process_identification(self, piece_identity, sort_type, target_category):
        """Process API result and determine sorting bin."""
        result = {
            "id": piece_identity.get("id"),
            "bin_number": 9,  # Default to overflow bin
            "confidence": piece_identity.get("score", 0)
        }

        if result["confidence"] < 0.7:
            result["error"] = "Low confidence score"
            return result

        piece_id = result["id"]
        if not piece_id or piece_id not in self.lego_dict:
            self._log_missing_piece(result)
            result["error"] = "Piece not found in dictionary"
            return result

        piece_info = self.lego_dict[piece_id]
        result.update({
            "element_id": piece_id,
            "name": piece_info['name'],
            "primary_category": piece_info['primary_category'],
            "secondary_category": piece_info['secondary_category']
        })

        # Determine bin number based on sort type
        if sort_type == 'primary':
            result["bin_number"] = self.primary_to_bin.get(
                piece_info['primary_category'], 9)
        else:
            result["bin_number"] = self._get_secondary_bin(
                piece_info, target_category)

        return result

    def _get_secondary_bin(self, piece_info, target_category):
        """Determine bin number for secondary sorting."""
        if piece_info['primary_category'] != target_category:
            return 9

        secondary_categories = self._get_secondary_categories(target_category)
        try:
            return secondary_categories.index(piece_info['secondary_category'])
        except ValueError:
            return 9

    def _get_secondary_categories(self, primary_category):
        """Get list of secondary categories for a primary category."""
        categories = {
            'Basic': ['Brick', 'Plate', 'Tile'],
            'Wall': ['Decorative', 'Groove_Rail', 'Panel', 'Window', 'Door', 'Stairs', 'Fence'],
            'SNOT': ['Bracket', 'Brick', 'Jumper'],
            'Minifig': ['Clothing', 'Body'],
            'Clip': ['Bar', 'Clip', 'Flag', 'Handle', 'Door', 'Flexible'],
            'Hinge': ['Click_brick', 'Click_plate', 'Click-other', 'Hinge', 'Turntable'],
            'Angle': ['Wedge-brick', 'Wedge-plate', 'Wedge-tile', 'Wedge-nose'],
            'Vehicle': ['Windscreen', 'Mudguard'],
            'Curved': ['plate', 'brick', 'tile', 'cylinder', 'Cone', 'Arch_bow']
        }
        return categories.get(primary_category, [])

    def _log_missing_piece(self, piece_data):
        """Log unknown pieces for review."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("Missing_Pieces.txt", 'a') as f:
            f.write(f"\nTimestamp: {timestamp}\n")
            f.write(f"Piece ID: {piece_data.get('id', 'Unknown')}\n")
            f.write(f"Name: {piece_data.get('name', 'Unknown')}\n")
            f.write(f"Confidence: {piece_data.get('confidence', 0)}\n")
            f.write("-" * 50)


@dataclass
class BeltColorProfile:
    """Data class to store belt color calibration results"""
    mean_color: np.ndarray
    std_dev: np.ndarray
    min_color: np.ndarray
    max_color: np.ndarray
    histograms: List
    roi: Tuple[slice, slice]  # Stores the ROI coordinates for future reference


class EnhancedConveyorCalibrator:
    def __init__(self, grid_size=(5, 5), sample_size=10):
        self.grid_size = grid_size
        self.sample_size = sample_size
        self.profile = None
        self._selection_points = []

    def calibrate(self, cap) -> BeltColorProfile:
        """Main calibration routine"""
        # Get user ROI selection
        roi_frame = self._capture_clean_frame(cap)
        roi = self._get_user_roi(roi_frame)

        # Sample grid within ROI
        samples = self._collect_grid_samples(roi_frame, roi)

        # Create color profile
        self.profile = self._create_color_profile(samples, roi)
        return self.profile

    def _capture_clean_frame(self, cap) -> np.ndarray:
        """Capture a clear frame for calibration"""
        for _ in range(5):  # Capture several frames to ensure clear image
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture calibration frame")
        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self._selection_points) < 2:
                self._selection_points.append((x, y))

    def _get_user_roi(self, frame) -> Tuple[slice, slice]:
        """Get user-selected ROI through mouse clicks"""
        window_name = "Select Belt Region - Click Upper Left and Lower Right Corners"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        instruction_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(instruction_frame,
                    "Click Upper Left and Lower Right corners of belt region",
                    (10, 30), font, 0.7, (0, 255, 0), 2)

        while len(self._selection_points) < 2:
            display = instruction_frame.copy()

            # Draw first point if it exists
            if len(self._selection_points) == 1:
                cv2.circle(display, self._selection_points[0], 5, (0, 255, 0), -1)

            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                cv2.destroyWindow(window_name)
                raise RuntimeError("ROI selection cancelled")

        # Get final points and clean up
        (x1, y1), (x2, y2) = self._selection_points
        cv2.destroyWindow(window_name)

        # Create slice objects for ROI
        row_slice = slice(min(y1, y2), max(y1, y2))
        col_slice = slice(min(x1, x2), max(x1, x2))

        return (row_slice, col_slice)

    def _collect_grid_samples(self, frame: np.ndarray, roi: Tuple[slice, slice]) -> List[np.ndarray]:
        """Collect color samples in a grid pattern within the ROI"""
        samples = []
        roi_height = roi[0].stop - roi[0].start
        roi_width = roi[1].stop - roi[1].start

        # Calculate grid spacing
        h_step = roi_height // (self.grid_size[0] + 1)
        w_step = roi_width // (self.grid_size[1] + 1)

        # Create debugging visualization
        debug_frame = frame.copy()

        # Sample points in grid pattern
        for i in range(1, self.grid_size[0] + 1):
            for j in range(1, self.grid_size[1] + 1):
                # Calculate sample point center
                y = roi[0].start + (i * h_step)
                x = roi[1].start + (j * w_step)

                # Get sample region
                sample_region = frame[
                                y - self.sample_size // 2:y + self.sample_size // 2,
                                x - self.sample_size // 2:x + self.sample_size // 2
                                ]

                # Store sample
                samples.append(np.mean(sample_region, axis=(0, 1)))

                # Visualize sampling points
                cv2.rectangle(debug_frame,
                              (x - self.sample_size // 2, y - self.sample_size // 2),
                              (x + self.sample_size // 2, y + self.sample_size // 2),
                              (0, 255, 0), 1)

        # Show sampling visualization
        cv2.imshow("Sampling Grid", debug_frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyWindow("Sampling Grid")

        return samples

    def _create_color_profile(self, samples: List[np.ndarray], roi: Tuple[slice, slice]) -> BeltColorProfile:
        """Create a color profile from collected samples"""
        samples_array = np.array(samples)

        # Compute histograms for each channel
        histograms = []
        for channel in range(3):  # BGR channels
            hist = np.histogram(samples_array[:, channel], bins=32, range=(0, 256))
            histograms.append(hist)

        return BeltColorProfile(
            mean_color=np.mean(samples_array, axis=0),
            std_dev=np.std(samples_array, axis=0),
            min_color=np.min(samples_array, axis=0),
            max_color=np.max(samples_array, axis=0),
            histograms=histograms,
            roi=roi
        )

    def is_belt_color(self, pixel: np.ndarray) -> bool:
        """Check if a pixel matches the belt color profile"""
        if self.profile is None:
            raise RuntimeError("Calibration profile not created")

        # Check if pixel is within acceptable range
        within_range = all(
            self.profile.min_color[i] <= pixel[i] <= self.profile.max_color[i]
            for i in range(3)
        )

        # Check if pixel is within standard deviation
        within_std = all(
            abs(pixel[i] - self.profile.mean_color[i]) <= 2 * self.profile.std_dev[i]
            for i in range(3)
        )

        return within_range and within_std


class ConveyorLegoDetector:
    def __init__(self, min_piece_area=1000, color_threshold=30):
        """Initialize detector with enhanced calibration."""
        self.min_piece_area = min_piece_area
        self.color_threshold = color_threshold
        self.last_detection_time = time()
        self.cooldown = 0.5  # Seconds between detections
        self.calibrator = EnhancedConveyorCalibrator()
        self.color_profile = None
        self.last_mask = None
        self.last_contours = None

    def calibrate(self, cap):
        """Calibrate using enhanced calibration system."""
        print("Calibrating belt color... Please select the belt region.")
        try:
            self.color_profile = self.calibrator.calibrate(cap)
            print(f"Calibrated belt color (BGR): {self.color_profile.mean_color}")
        except Exception as e:
            print(f"Calibration failed: {str(e)}")
            raise

    def detect_piece(self, roi_frame):
        """Detects if a Lego piece is in the ROI frame."""
        if self.color_profile is None:
            print("Error: Belt not calibrated")
            return False

        try:
            current_time = time()
            if current_time - self.last_detection_time < self.cooldown:
                return False

            # Create mask for non-belt objects
            height, width = roi_frame.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)

            # Check each pixel in ROI
            for y in range(height):
                for x in range(width):
                    pixel = roi_frame[y, x]
                    if not self.calibrator.is_belt_color(pixel):
                        mask[y, x] = 255

            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            self.last_mask = mask

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.last_contours = contours

            # Check for pieces in middle third
            middle_third = (height // 3, 2 * height // 3)

            for contour in contours:
                if cv2.contourArea(contour) > self.min_piece_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cy = int(M["m01"] / M["m00"])
                        if middle_third[0] < cy < middle_third[1]:
                            self.last_detection_time = current_time
                            return True
            return False

        except Exception as e:
            print(f"Error in detect_piece: {str(e)}")
            return False
    def draw_debug(self, frame):
        """Draws debug visualization on the frame."""
        try:
            debug_frame = frame.copy()

            if self.color_profile is None:
                return debug_frame

            roi = self.color_profile.roi
            height, width = frame.shape[:2]

            # Draw ROI
            cv2.rectangle(debug_frame,
                          (roi[1].start, roi[0].start),
                          (roi[1].stop, roi[0].stop),
                          (0, 255, 0), 2)  # Green

            # Draw middle third of ROI
            roi_height = roi[0].stop - roi[0].start
            middle_top = roi[0].start + roi_height // 3
            middle_bottom = roi[0].start + 2 * roi_height // 3
            cv2.rectangle(debug_frame,
                          (roi[1].start, middle_top),
                          (roi[1].stop, middle_bottom),
                          (255, 0, 0), 2)  # Blue

            # Draw detected contours if available
            if self.last_contours is not None:
                for contour in self.last_contours:
                    # Shift contour to match ROI position
                    shifted_contour = contour + [roi[1].start, roi[0].start]
                    cv2.drawContours(debug_frame, [shifted_contour], -1, (0, 0, 255), 2)

            # Add text showing detector status
            status_text = f"Threshold: {self.color_threshold}"
            cv2.putText(debug_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            color_text = f"Belt BGR: {self.color_profile.mean_color}"
            cv2.putText(debug_frame, color_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return debug_frame

        except Exception as e:
            print(f"Error in draw_debug: {str(e)}")
            return frame


class SortingManager:
    def __init__(self):
        self.camera = CameraManager()
        self.identifier = PieceIdentifier()
        self.detector = ConveyorLegoDetector()

    def run(self):
        """Main sorting loop."""
        try:
            print("\nInitializing sorting system...")

            # Show preview frame for belt region selection
            print("Opening camera preview for belt region selection.")
            print("Position the camera to view the belt clearly.")
            print("Then select the region of the belt to monitor by clicking:")
            print("1. Upper-left corner of belt region")
            print("2. Lower-right corner of belt region")
            print("\nPress ESC to exit")

            # Initialize calibration
            self.detector.calibrate(self.camera.cap)

            print("\nCalibration complete!")
            print("\nSorting system ready. Press ESC to exit.")

            while True:
                frame = self.camera.get_preview_frame()
                if frame is None:
                    print("Failed to get preview frame")
                    break

                # Only process the ROI for detection
                roi = self.detector.color_profile.roi
                roi_frame = frame[roi[0], roi[1]]

                if self.detector.detect_piece(roi_frame):
                    # Capture and process piece
                    image_path, image_count, error = self.camera.capture_image()
                    if error:
                        print(f"Capture error: {error}")
                        continue

                    result, error = self.identifier.identify_piece(image_path)
                    if error:
                        print(f"Identification error: {error}")
                        continue

                    print(f"\nPiece #{image_count:03} identified:")
                    print(f"Image: Lego{image_count:03}.jpg")
                    print(f"Element ID: {result.get('element_id', 'Unknown')}")
                    print(f"Name: {result.get('name', 'Unknown')}")
                    print(f"Primary Category: {result.get('primary_category', 'Unknown')}")
                    print(f"Secondary Category: {result.get('secondary_category', 'Unknown')}")
                    print(f"Bin Number: {result.get('bin_number', 9)}")

                # Show debug view
                debug_frame = self.detector.draw_debug(frame)
                cv2.imshow("Capture", debug_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.camera.cleanup()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
# Main program
if __name__ == "__main__":
    sorter = SortingManager()
    sorter.run()
