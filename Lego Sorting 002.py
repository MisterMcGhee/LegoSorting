import requests
import csv
import cv2
import os
import datetime

# Define the main category mappings as a constant dictionary
CATEGORY_MAPPING = {
    'BASIC_ELEMENTS': [
        'Brick',
        'Plate',
        'Tile',
        'Tile, Round',
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
        'Baseplate',
    ]
}

# Define bin mapping as a constant dictionary
BIN_MAPPING = {
    'BASIC_ELEMENTS': 0,
    'TECHNIC_COMPONENTS': 1,
    'ANGLED_ELEMENTS': 2,
    'STRUCTURAL_COMPONENTS': 3,
    'VEHICLE_PARTS': 4,
    'WINDOWS_AND_DOORS': 5,
    'SPECIALTY_ITEMS': 6
}

# Create reverse lookup dictionary for finding main categories
REVERSE_LOOKUP = {}
for main_cat, subcats in CATEGORY_MAPPING.items():
    for subcat in subcats:
        REVERSE_LOOKUP[subcat] = main_cat


def get_all_categories():
    """
    Returns list of all main categories.

    Returns:
        list: All main category names
    """
    return list(CATEGORY_MAPPING.keys())


def get_subcategories(main_category):
    """
    Returns list of subcategories for a given main category.

    Args:
        main_category (str): Name of the main category

    Returns:
        list: Subcategories for the given main category, or empty list if not found
    """
    return CATEGORY_MAPPING.get(main_category, [])


def get_bin_number(main_category):
    """
    Maps main category to bin number (0-7).
    Bin 7 is reserved for errors and unrecognized categories.

    Args:
        main_category (str): Name of the main category

    Returns:
        int: Bin number (0-7)
    """
    return BIN_MAPPING.get(main_category, 7)


def get_main_category(subcategory):
    """
    Finds the main category for a given subcategory.

    Args:
        subcategory (str): Name of the subcategory

    Returns:
        str: Main category name, or 'SPECIALTY_ITEMS' if not found
    """
    return REVERSE_LOOKUP.get(subcategory, 'SPECIALTY_ITEMS')


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
            categories = get_all_categories()

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


def log_missing_piece(piece_data):
    """
    Logs information about pieces not found in the dictionary.

    Args:
        piece_data (dict): Dictionary containing piece information
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "Missing_from_dictionary.txt"

    try:
        with open(log_file, 'a') as f:
            f.write(f"\nTimestamp: {timestamp}\n")
            f.write(f"Piece ID: {piece_data.get('id', 'Unknown')}\n")
            f.write(f"Name: {piece_data.get('name', 'Unknown')}\n")
            f.write(f"Confidence Score: {piece_data.get('confidence', 0)}\n")
            f.write("-" * 50)
    except Exception as e:
        print(f"Error logging missing piece: {str(e)}")


def attribute_piece(piece_identity, confidence_threshold=0.7, sort_mode=None, target_category=None):
    """
    Determines the appropriate bin for a piece based on its identity and sorting mode.
    Logs pieces not found in dictionary and assigns them to bin 7.
    """
    # Initialize result with provided piece information
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
        # Log the missing piece
        log_missing_piece(result)

        # Update result for missing piece
        result["bin_number"] = 7  # Assign to bin 7 for missing pieces
        result["error"] = "Piece not found in dictionary - logged for review"
        result["category"] = "Unknown"
        result["rough_category"] = "Unknown"
        return result

    # Rest of the function remains the same for known pieces
    category, name = lego_dict[piece_id]
    main_category = get_main_category(category)

    if sort_mode == 'rough':
        result["bin_number"] = get_bin_number(main_category)
    elif sort_mode == 'detailed':
        if main_category == target_category:
            subcategories = get_subcategories(target_category)
            try:
                result["bin_number"] = subcategories.index(category) % 8
            except ValueError:
                result["bin_number"] = 8
        else:
            result["bin_number"] = 8

    result["category"] = category
    result["rough_category"] = main_category
    result["name"] = name

    return result


def initialize_image_capture(directory="LegoPictures"):
    """
    Initialize camera and required directories for image capture.

    Args:
        directory (str): Directory path for storing captured images

    Returns:
        tuple: (VideoCapture object, directory path, initial image count)
    """
    # Create directory if it doesn't exist
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get next image number
    count = len([name for name in os.listdir(directory)
                 if name.startswith("Lego") and name.endswith(".jpg")]) + 1

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to initialize camera")

    return cap, directory, count


def capture_image(cap, directory, count):
    """
    Capture and save an image from the camera.

    Args:
        cap: OpenCV VideoCapture object
        directory (str): Directory to save the image
        count (int): Current image number

    Returns:
        tuple: (filename, success boolean, error message if any)
    """
    ret, frame = cap.read()
    if not ret:
        return None, False, "Failed to capture image"

    filename = os.path.join(directory, f"Lego{count:03}.jpg")
    try:
        cv2.imwrite(filename, frame)
        return filename, True, None
    except Exception as e:
        return None, False, f"Failed to save image: {str(e)}"


def process_image(filename, sort_mode=None, target_category=None):
    """
    Process captured image through Brickognize API and determine sorting.

    Args:
        filename (str): Path to the captured image
        sort_mode (str): Sorting mode ('rough' or 'detailed')
        target_category (str): Target category for detailed sorting

    Returns:
        dict: Processing results including bin number and piece information
    """
    # Get piece identification from API
    api_result = send_to_brickognize_api(filename)
    if "error" in api_result:
        return {"error": api_result["error"]}

    # Attribute piece to correct bin
    result = attribute_piece(api_result, sort_mode=sort_mode,
                             target_category=target_category)
    return result


def run_image_capture(sort_mode=None, target_category=None):
    """
    Main function to run the image capture and processing system.

    Args:
        sort_mode (str): Sorting mode ('rough' or 'detailed')
        target_category (str): Target category for detailed sorting
    """
    try:
        # Initialize capture system
        cap, directory, count = initialize_image_capture()
        cv2.namedWindow("Capture")

        def capture_callback(event, x, y, flags, param):
            """Callback function for mouse click event"""
            nonlocal count
            if event == cv2.EVENT_LBUTTONDOWN:
                # Capture and save image
                filename, success, error = capture_image(cap, directory, count)
                if not success:
                    print(f"Error: {error}")
                    return

                print(f"\nImage saved to {filename}")

                # Process image
                result = process_image(filename, sort_mode, target_category)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print("Piece identification:", result)
                    print(f"Sort to bin: {result['bin_number']}")

                count += 1

        cv2.setMouseCallback("Capture", capture_callback)

        # Main capture loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture preview frame")
                break

            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()


def main():
    """Main program entry point"""
    global lego_dict
    lego_dict = create_lego_dictionary()

    # Get sorting mode and category
    sort_mode, target_category = get_sorting_mode()

    print(f"\nInitializing {sort_mode} sort", end="")
    if target_category:
        print(f" for {target_category}")
    print("\nPress ESC to exit, click in window to capture image")

    # Start image capture with new function
    run_image_capture(sort_mode=sort_mode, target_category=target_category)


if __name__ == "__main__":
    main()
