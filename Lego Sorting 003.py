import cv2
import csv
import os
import datetime
import requests
import numpy as np
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
        """Captures and saves an image."""
        ret, frame = self.cap.read()
        if not ret:
            return None, "Failed to capture image"

        filename = os.path.join(self.directory, f"Lego{self.count:03}.jpg")
        try:
            cv2.imwrite(filename, frame)
            self.count += 1
            return filename, None
        except Exception as e:
            return None, f"Failed to save image: {str(e)}"

    def get_preview_frame(self):
        """Gets a frame for preview/detection without saving."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def cleanup(self):
        """Releases camera resources."""
        self.cap.release()


class ConveyorLegoDetector:
    def __init__(self, detection_zone_y=(200, 280), min_piece_area=1000, color_threshold=30):
        """Initialize detector for conveyor belt setup."""
        self.detection_zone = detection_zone_y
        self.min_piece_area = min_piece_area
        self.color_threshold = color_threshold
        self.last_detection_time = time()
        self.cooldown = 0.5  # Seconds between detections
        self.belt_color = None

    def calibrate(self, cap):
        """Calibrate belt color using camera feed."""
        print("Calibrating belt color... Please ensure no pieces are on the belt.")
        samples = []
        for _ in range(30):
            ret, frame = cap.read()
            if ret:
                center_y = frame.shape[0] // 2
                center_x = frame.shape[1] // 2
                sample_region = frame[center_y - 10:center_y + 10, center_x - 10:center_x + 10]
                samples.append(np.mean(sample_region, axis=(0, 1)))
            cv2.waitKey(33)

        self.belt_color = np.mean(samples, axis=0).astype(int)
        print(f"Calibrated belt color (BGR): {self.belt_color}")

    def detect_piece(self, frame):
        """Detects if a Lego piece is in the detection zone."""
        if self.belt_color is None:
            return False

        current_time = time()
        if current_time - self.last_detection_time < self.cooldown:
            return False

        # Create mask for non-belt objects
        detection_area = frame[self.detection_zone[0]:self.detection_zone[1], :]
        color_diff = cv2.absdiff(detection_area, self.belt_color.reshape(1, 1, 3))
        color_diff_sum = np.sum(color_diff, axis=2)
        mask = (color_diff_sum > self.color_threshold).astype(np.uint8) * 255

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check for pieces in middle third
        zone_height = self.detection_zone[1] - self.detection_zone[0]
        middle_third = (zone_height // 3, 2 * zone_height // 3)

        for contour in contours:
            if cv2.contourArea(contour) > self.min_piece_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cy = int(M["m01"] / M["m00"])
                    if middle_third[0] < cy < middle_third[1]:
                        self.last_detection_time = current_time
                        return True
        return False

    def draw_debug(self, frame):
        """Draws debug visualization on the frame."""
        debug_frame = frame.copy()
        zone_height = self.detection_zone[1] - self.detection_zone[0]

        # Draw full detection zone
        cv2.rectangle(debug_frame,
                      (0, self.detection_zone[0]),
                      (frame.shape[1], self.detection_zone[1]),
                      (0, 255, 0), 2)

        # Draw middle third
        middle_top = self.detection_zone[0] + zone_height // 3
        middle_bottom = self.detection_zone[0] + 2 * zone_height // 3
        cv2.rectangle(debug_frame,
                      (0, middle_top),
                      (frame.shape[1], middle_bottom),
                      (255, 0, 0), 2)

        return debug_frame


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


class SortingManager:
    def __init__(self):
        self.camera = CameraManager()
        self.identifier = PieceIdentifier()
        self.detector = ConveyorLegoDetector()  # Your piece detection class

    def run(self):
        """Main sorting loop."""
        try:
            print("\nSorting system ready. Press ESC to exit.")

            while True:
                frame = self.camera.get_preview_frame()
                if frame is None:
                    print("Failed to get preview frame")
                    break

                if self.detector.detect_piece(frame):
                    # Capture and process piece
                    image_path, error = self.camera.capture_image()
                    if error:
                        print(f"Capture error: {error}")
                        continue

                    result, error = self.identifier.identify_piece(image_path)
                    if error:
                        print(f"Identification error: {error}")
                        continue

                    print("\nPiece identified:")
                    print(f"Element ID: {result['element_id']}")
                    print(f"Name: {result['name']}")
                    print(f"Primary Category: {result['primary_category']}")
                    print(f"Secondary Category: {result['secondary_category']}")

                # Show debug view
                debug_frame = self.detector.draw_debug(frame)
                cv2.imshow("Capture", debug_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        self.camera.cleanup()
        cv2.destroyAllWindows()


# Main program
if __name__ == "__main__":
    sorter = SortingManager()
    sorter.run()