import requests
import csv
import cv2
import os


class LegoRoughCategories:
    """
    Manages the categorization system for Lego pieces with 7 primary categories
    and 1 catch-all bin (8 total bins numbered 0-7).
    """

    def __init__(self):
        # Define the mapping of rough categories to their subcategories
        self.category_mapping = {
            'BASIC_ELEMENTS': [
                'Brick',
                'Plate',
                'Tile',
                'Baseplate',
                'Cylinder',
                'Brick, Modified',
                'Plate, Round',
            ],
            'TECHNIC_COMPONENTS': [
                'Technic',
                'Technic, Connector',
                'Technic, Gear',
                'Technic, Panel',
                'Technic, Liftarm',
                'Technic, Link',
                'Technic, Plate',
                'Technic, Disk',
                'Technic, Axle',
                'Technic, Pin',
                'Technic, Brick',
            ],
            'ANGLED_ELEMENTS': [
                'Slope',
                'Slope, Curved',
                'Slope, Inverted',
                'Wedge',
                'Wedge, Plate',
                'Roof',
                'Arch',
                'Cone',
                'Dish',
            ],
            'STRUCTURAL_COMPONENTS': [
                'Support',
                'Panel',
                'Bracket',
                'Bar',
                'Ladder',
                'Stairs',
                'Fence',
                'Turntable',
            ],
            'VEHICLE_PARTS': [
                'Vehicle',
                'Vehicle, Base',
                'Propeller',
                'Boat',
                'Crane',
            ],
            'WINDOWS_AND_DOORS': [
                'Window',
                'Door',
                'Windscreen',
            ],
            'SPECIALTY_ITEMS': [
                'Container',
                'Plant',
                'Rock',
                'Hinge',
            ]
        }

        # Create reverse lookup dictionary for efficient category finding
        self.reverse_lookup = {}
        for rough_cat, subcats in self.category_mapping.items():
            for subcat in subcats:
                self.reverse_lookup[subcat] = rough_cat

    def get_all_categories(self):
        """Returns list of all rough categories"""
        return list(self.category_mapping.keys())

    def get_subcategories(self, rough_category):
        """Returns list of subcategories for a given rough category"""
        return self.category_mapping.get(rough_category, [])

    def get_bin_number(self, rough_category):
        """
        Maps rough category to bin number (0-7).
        Bin 7 is reserved for errors and unrecognized categories.
        """
        bin_mapping = {
            'BASIC_ELEMENTS': 0,
            'TECHNIC_COMPONENTS': 1,
            'ANGLED_ELEMENTS': 2,
            'STRUCTURAL_COMPONENTS': 3,
            'VEHICLE_PARTS': 4,
            'WINDOWS_AND_DOORS': 5,
            'SPECIALTY_ITEMS': 6
        }
        return bin_mapping.get(rough_category, 7)  # Return 7 (error bin) for unknown categories


def create_lego_dictionary(filepath='Bricklink ID Name Category.csv'):
    """
    Creates a dictionary mapping Lego piece IDs to their categories and names.
    """
    lego_dict = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                lego_dict[row['ID']] = (row['Category'], row['Name'])
    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    return lego_dict


def send_to_brickognize_api(file_name):
    """
    Sends an image to the Brickognize API for piece identification.
    """
    url = "https://api.brickognize.com/predict/"
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    # Validate file
    file_name = os.path.abspath(file_name)
    if not os.path.isfile(file_name):
        return {"error": f"File not found: {file_name}"}
    if os.path.getsize(file_name) == 0:
        return {"error": "File is empty"}
    if not any(file_name.lower().endswith(ext) for ext in valid_extensions):
        return {"error": f"Invalid file type. Allowed: {', '.join(valid_extensions)}"}

    # Send request to API
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


def get_sorting_mode():
    """
    Prompts user to select sorting mode and category if needed.
    """
    while True:
        mode = input("\nSelect sorting mode:\n1: Rough Sort (8 categories)\n2: Detailed Sort\nEnter 1 or 2: ").strip()

        if mode not in ['1', '2']:
            print("Invalid input. Please enter 1 or 2.")
            continue

        if mode == '1':
            return 'rough', None
        else:
            # Get rough category for detailed sort
            sorter = LegoRoughCategories()
            categories = sorter.get_all_categories()

            print("\nAvailable categories for detailed sort:")
            for i, category in enumerate(categories, 1):
                print(f"{i}: {category}")

            while True:
                try:
                    cat_num = int(input("\nEnter category number: "))
                    if 1 <= cat_num <= len(categories):
                        return 'detailed', categories[cat_num - 1]
                    print("Invalid category number.")
                except ValueError:
                    print("Please enter a valid number.")


def attribute_piece(piece_identity, confidence_threshold=0.7, sort_mode=None, target_category=None):
    """
    Determines the appropriate bin for a piece based on its identity and sorting mode.
    """
    # Initialize result with error bin (8) as default
    result = {
        "id": piece_identity.get("id"),
        "name": piece_identity.get("name"),
        "bin_number": 8,  # Default to error/catchall bin
        "confidence": piece_identity.get("score", 0)
    }

    # Check confidence threshold
    if result["confidence"] < confidence_threshold:
        result["error"] = "Low confidence score"
        return result

    # Get piece category from dictionary
    piece_id = result["id"]
    if not piece_id or piece_id not in lego_dict:
        result["error"] = "Piece not found in dictionary"
        return result

    # Get category information
    category, name = lego_dict[piece_id]
    sorter = LegoRoughCategories()
    rough_category = sorter.reverse_lookup.get(category, 'SPECIALTY_ITEMS')

    # Handle different sorting modes
    if sort_mode == 'rough':
        # Map rough category to bin number (0-7)
        bin_mapping = {
            'BASIC_ELEMENTS': 0,
            'TECHNIC_COMPONENTS': 1,
            'ANGLED_ELEMENTS': 2,
            'STRUCTURAL_COMPONENTS': 3,
            'VEHICLE_PARTS': 4,
            'MODIFIED_ELEMENTS': 5,
            'WINDOWS_AND_DOORS': 6,
            'SPECIALTY_ITEMS': 7
        }
        result["bin_number"] = bin_mapping.get(rough_category, 8)

    elif sort_mode == 'detailed':
        # For detailed sort, check if piece belongs to target category
        if rough_category == target_category:
            # If it matches target category, assign to appropriate detail bin (0-7)
            subcategories = sorter.get_subcategories(target_category)
            try:
                result["bin_number"] = subcategories.index(category) % 8
            except ValueError:
                result["bin_number"] = 8  # Catchall if subcategory not found
        else:
            result["bin_number"] = 8  # Different category than target

    result["category"] = category
    result["rough_category"] = rough_category
    result["name"] = name

    return result


def image_capture(directory="LegoPictures", sort_mode=None, target_category=None):
    """
    Handles the camera interface and image capture process.
    """
    # Create directory if it doesn't exist
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get next image number
    count = len([name for name in os.listdir(directory)
                 if name.startswith("Lego") and name.endswith(".jpg")]) + 1

    def capture(event):
        """Callback function for mouse click event"""
        nonlocal count
        if event == cv2.EVENT_LBUTTONDOWN:
            ret, frame = cap.read()
            if ret:
                filename = os.path.join(directory, f"Lego{count:03}.jpg")
                cv2.imwrite(filename, frame)
                print(f"\nImage saved to {filename}")

                # Process image and get piece information
                result = send_to_brickognize_api(filename)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    result = attribute_piece(result, sort_mode=sort_mode,
                                             target_category=target_category)
                    print("Piece identification:", result)
                    print(f"Sort to bin: {result['bin_number']}")
                count += 1

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Capture")
    cv2.setMouseCallback("Capture", capture)

    # Main capture loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main program entry point"""
    # Initialize
    global lego_dict
    lego_dict = create_lego_dictionary()

    # Get sorting mode and category
    sort_mode, target_category = get_sorting_mode()

    print(f"\nInitializing {sort_mode} sort", end="")
    if target_category:
        print(f" for {target_category}")
    print("\nPress ESC to exit, click in window to capture image")

    # Start image capture
    image_capture(sort_mode=sort_mode, target_category=target_category)


if __name__ == "__main__":
    main()
