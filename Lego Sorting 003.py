import cv2
import csv
import os
import datetime
import requests
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from time import time


class CameraManager:
    def __init__(self, directory="LegoPictures"):
        """Initialize camera and directory for storing images."""
        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Initialize count by checking existing files
        self.count = 1  # Start at 1 by default
        if os.path.exists(self.directory):
            existing_files = [f for f in os.listdir(self.directory)
                              if f.startswith("Lego") and f.endswith(".jpg")]
            if existing_files:
                # Extract numbers from filenames and find the highest
                numbers = [int(f[4:7]) for f in existing_files]  # Lego001.jpg -> 1
                self.count = max(numbers) + 1

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
class TrackedPiece:
    """Represents a piece being tracked through the ROI"""
    id: int  # Unique identifier for this piece
    contour: np.ndarray  # Current contour
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    entry_time: float  # When the piece entered tracking
    captured: bool = False  # Whether this piece has been photographed
    image_number: Optional[int] = None  # The number used when saving the image


class TrackedLegoDetector:
    def __init__(self, min_piece_area=300, max_piece_area=20000):
        self.min_piece_area = min_piece_area
        self.max_piece_area = max_piece_area
        self.last_detection_time = time()
        self.tracked_pieces: List[TrackedPiece] = []
        self.next_piece_id = 1
        self.buffer_percent = 0.01
        self.crop_padding = 20
        self.detection_history = []
        self.history_length = 3

        # Initialize ROI-related attributes
        self.roi = None
        self.entry_zone = None
        self.exit_zone = None
        self.valid_zone = None

        # Create debug windows
        cv2.namedWindow('Debug View', cv2.WINDOW_NORMAL)

    def calibrate(self, frame):
        """Get user-selected ROI and calculate buffer zones"""
        # Get ROI selection from user
        roi = cv2.selectROI("Select Belt Region", frame, False)
        cv2.destroyWindow("Select Belt Region")

        self.roi = roi
        roi_width = roi[2]

        # Calculate buffer zones
        buffer_width = int(roi_width * self.buffer_percent)
        self.entry_zone = (roi[0], roi[0] + buffer_width)
        self.exit_zone = (roi[0] + roi[2] - buffer_width, roi[0] + roi[2])
        self.valid_zone = (self.entry_zone[1], self.exit_zone[0])

        print(f"ROI: {roi}")
        print(f"Entry buffer: {self.entry_zone}")
        print(f"Valid zone: {self.valid_zone}")
        print(f"Exit buffer: {self.exit_zone}")

    def _check_initialized(self):
        """Check if the detector has been properly initialized"""
        if self.roi is None:
            raise RuntimeError("TrackedLegoDetector must be calibrated before use. Call calibrate() first.")

    def _preprocess_frame(self, frame):
        """Preprocess frame optimized for magenta belt"""
        self._check_initialized()

        x, y, w, h = self.roi
        roi = frame[y:y + h, x:x + w]

        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create mask for magenta belt (to remove it)
        # Magenta is around 150 in HSV
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])
        magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)

        # Invert mask to get pieces
        piece_mask = cv2.bitwise_not(magenta_mask)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _find_new_contours(self, mask):
        """Find contours with filtering for Lego pieces"""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_piece_area or area > self.max_piece_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio
            aspect_ratio = float(w) / h

            # Less strict aspect ratio for Lego pieces
            if aspect_ratio > 4 or aspect_ratio < 0.25:
                continue

            # Verify position relative to ROI
            if self.roi[0] <= x < self.roi[0] + self.roi[2]:
                valid_contours.append((contour, (x, y, w, h)))

        return valid_contours

    def _match_and_update_tracks(self, new_contours):
        """Match new contours to existing tracks and update"""
        current_time = time()

        # Remove old tracks
        self.tracked_pieces = [piece for piece in self.tracked_pieces
                               if current_time - piece.entry_time < 5.0
                               and piece.bounding_box[0] < self.exit_zone[0]]

        # Match new contours to existing tracks or create new ones
        for contour, bbox in new_contours:
            x, y, w, h = bbox
            matched = False

            # Try to match with existing tracks
            for piece in self.tracked_pieces:
                old_x, old_y, old_w, old_h = piece.bounding_box

                # Calculate center points
                new_center = (x + w / 2, y + h / 2)
                old_center = (old_x + old_w / 2, old_y + old_h / 2)

                # Check if centers are close enough
                distance = np.sqrt((new_center[0] - old_center[0]) ** 2 +
                                   (new_center[1] - old_center[1]) ** 2)

                if distance < max(w, h) * 0.7:  # Increased movement threshold
                    piece.contour = contour
                    piece.bounding_box = bbox
                    matched = True
                    break

            if not matched and x <= self.entry_zone[1]:
                # Create new track
                new_piece = TrackedPiece(
                    id=self.next_piece_id,
                    contour=contour,
                    bounding_box=bbox,
                    entry_time=current_time
                )
                self.tracked_pieces.append(new_piece)
                self.next_piece_id += 1

    def _should_capture(self, piece: TrackedPiece) -> bool:
        """Determine if piece should be captured"""
        if piece.captured:
            return False

        x, y, w, h = piece.bounding_box
        piece_center_x = x + w / 2

        # Check if piece is in the valid capture zone
        in_capture_zone = (self.valid_zone[0] <= piece_center_x <= self.valid_zone[1])

        # Add to detection history
        self.detection_history.append(in_capture_zone)
        if len(self.detection_history) > self.history_length:
            self.detection_history.pop(0)

        # Only capture if piece has been consistently detected
        return (in_capture_zone and
                len(self.detection_history) == self.history_length and
                all(self.detection_history))

    def _crop_piece_image(self, frame, piece: TrackedPiece) -> np.ndarray:
        """Crop the frame to just the piece with padding"""
        x, y, w, h = piece.bounding_box
        roi_x, roi_y = self.roi[0], self.roi[1]

        # Add padding and ensure within frame bounds
        pad = self.crop_padding
        x1 = max(0, x + roi_x - pad)
        y1 = max(0, y + roi_y - pad)
        x2 = min(frame.shape[1], x + roi_x + w + pad)
        y2 = min(frame.shape[0], y + roi_y + h + pad)

        return frame[y1:y2, x1:x2]

    def process_frame(self, frame, current_count) -> Tuple[List[TrackedPiece], Optional[np.ndarray]]:
        """Process a frame and return any pieces to capture."""
        self._check_initialized()

        # Create debug frame
        debug_frame = frame.copy()

        # Preprocess frame and find contours
        mask = self._preprocess_frame(frame)
        new_contours = self._find_new_contours(mask)

        # Update tracking
        self._match_and_update_tracks(new_contours)

        # Draw debug visualization
        x, y, w, h = self.roi
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show mask in debug window
        debug_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Debug View', debug_mask)

        # Check for pieces to capture
        for piece in self.tracked_pieces:
            if self._should_capture(piece) and not piece.captured:
                cropped_image = self._crop_piece_image(frame, piece)
                piece.captured = True
                piece.image_number = current_count
                return self.tracked_pieces, cropped_image

        return self.tracked_pieces, None

    def draw_debug(self, frame):
        """Draw debug visualization"""
        debug_frame = frame.copy()

        # Draw ROI
        x, y, w, h = self.roi
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw pieces
        for piece in self.tracked_pieces:
            x, y, w, h = piece.bounding_box
            roi_x, roi_y = self.roi[0], self.roi[1]

            # Adjust coordinates to full frame
            x += roi_x
            y += roi_y

            # Draw bounding box
            color = (0, 255, 0) if piece.captured else (0, 0, 255)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)

            # Add label
            label = f"#{piece.image_number}" if piece.captured else "Tracking"
            label_color = (0, 255, 0) if piece.captured else (0, 0, 255)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(debug_frame, (x, y - label_size[1] - 10),
                          (x + label_size[0], y), (0, 0, 0), -1)

            # Draw label text
            cv2.putText(debug_frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        return debug_frame
class SortingManager:
    def __init__(self):
        """Initialize sorting manager and ensure camera count is set."""
        self.camera = CameraManager()
        self.identifier = PieceIdentifier()
        self.detector = TrackedLegoDetector()
        self.sort_type = None
        self.target_category = None
        print(f"Starting with image count: {self.camera.count}")  # Debug print

    def get_sorting_preference(self):
        """Get user preference for sorting method."""
        while True:
            print("\nSelect sorting method:")
            print("1. Sort by Primary Categories")
            print("2. Sort by Secondary Categories")

            choice = input("Enter choice (1 or 2): ").strip()

            if choice == '1':
                self.sort_type = 'primary'
                break
            elif choice == '2':
                self.sort_type = 'secondary'
                # For secondary sorting, need to select primary category
                print("\nSelect primary category to sort by secondary categories:")
                for idx, category in enumerate(self.identifier.primary_to_bin.keys(), 1):
                    print(f"{idx}. {category}")

                while True:
                    try:
                        cat_choice = int(input("Enter category number: "))
                        if 1 <= cat_choice <= len(self.identifier.primary_to_bin):
                            self.target_category = list(self.identifier.primary_to_bin.keys())[cat_choice - 1]
                            break
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def run(self):
        """Main sorting loop."""
        try:
            print("\nInitializing sorting system...")

            # Get sorting preferences first
            self.get_sorting_preference()

            print("\nOpening camera preview for belt region selection.")
            print("Position the camera to view the belt clearly.")
            print("Select the region to monitor by dragging a rectangle:")
            print("Press SPACE or ENTER to confirm selection")
            print("Press ESC to exit")

            # Get initial frame for ROI selection
            frame = self.camera.get_preview_frame()
            if frame is None:
                raise RuntimeError("Failed to get preview frame")

            # Initialize detector with ROI
            self.detector.calibrate(frame)

            print("\nSetup complete!")
            print("\nSorting system ready. Press ESC to exit.")

            while True:
                frame = self.camera.get_preview_frame()
                if frame is None:
                    print("Failed to get preview frame")
                    break

                # Process frame and get any pieces to capture
                tracked_pieces, piece_image = self.detector.process_frame(
                    frame=frame,
                    current_count=self.camera.count
                )
                # If we have a piece to capture
                if piece_image is not None:
                    # Save the cropped image
                    image_path = os.path.join(self.camera.directory, f"Lego{self.camera.count:03}.jpg")
                    cv2.imwrite(image_path, piece_image)

                    # Process with API using selected sort type
                    result, error = self.identifier.identify_piece(
                        image_path,
                        sort_type=self.sort_type,
                        target_category=self.target_category
                    )
                    if error:
                        print(f"Identification error: {error}")
                    else:
                        print(f"\nPiece #{self.camera.count:03} identified:")
                        print(f"Image: Lego{self.camera.count:03}.jpg")
                        print(f"Element ID: {result.get('element_id', 'Unknown')}")
                        print(f"Name: {result.get('name', 'Unknown')}")
                        print(f"Primary Category: {result.get('primary_category', 'Unknown')}")
                        print(f"Secondary Category: {result.get('secondary_category', 'Unknown')}")
                        print(f"Bin Number: {result.get('bin_number', 9)}")

                    self.camera.count += 1

                # Show debug view
                debug_frame = self.detector.draw_debug(frame)
                cv2.imshow("Capture", debug_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cleanup()


# Main program
if __name__ == "__main__":
    sorter = SortingManager()
    sorter.run()
