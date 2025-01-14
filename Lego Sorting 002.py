import cv2
import csv
import os
import datetime
import requests

# Primary category to bin mapping
PRIMARY_CATEGORIES = {
    'Basic': 0,
    'Wall': 1,
    'SNOT': 2,
    'Minifig': 3,
    'Clip': 4,
    'Hinge': 5,
    'Angle': 6,
    'Vehicle': 7,
    'Curved': 8
}


def create_lego_dictionary(filepath='Lego_Categories.csv'):
    """
    Creates a dictionary mapping Lego piece IDs to their categories and names.

    Returns:
        dict: Dictionary with Element_ID as key and piece information as value
    """
    lego_dict = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                lego_dict[row['element_id']] = {
                    'name': row['name'],
                    'primary_category': row['primary_category'],
                    'secondary_category': row['secondary_category']
                }
    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    return lego_dict


def get_secondary_categories(primary_category):
    """
    Returns list of secondary categories for a given primary category.
    """
    secondary_categories = {
        'Basic': ['Brick', 'Plate', 'Tile'],
        'Wall': ['Decorative', 'Groove_Rail', 'Panel', 'Window', 'Door', 'Stairs', 'Fence'],
        'SNOT': ['Bracket', 'Brick', 'Jumper'],
        'Minifig': ['Clothing', 'Body'],
        'Clip': ['Bar', 'Clip', 'Flag', 'Handle', 'Door', 'Flexible'],
        'Hinge': ['Click_brick', 'Click_plate', 'Click-other', 'Hinge', 'Turntable',
                  'Socket_ball', 'Socket-click', 'Socket-towball'],
        'Angle': ['Wedge-brick', 'Wedge-plate', 'Wedge-tile', 'Wedge-nose',
                  'Slope-10-18-30', 'Slope-33', 'Slope-45', 'Slope-55-65-75'],
        'Vehicle': ['Windscreen', 'Mudguard'],
        'Curved': ['plate', 'doubleplate', 'brick', 'tile', 'clyinder', 'Cone',
                   'Arch_bow', 'Dish_done', 'Curved', 'Wedge', 'Ball', 'Heart_star', 'Other']
    }
    return secondary_categories.get(primary_category, [])


def get_bin_number(piece_info, sort_type, target_category=None):
    """
    Determines bin number based on piece categories and sort type.

    Args:
        piece_info (dict): Piece information including categories
        sort_type (str): 'primary' or 'secondary'
        target_category (str): Target primary category for secondary sorting

    Returns:
        int: Bin number (0-8, or 9 for unknown/error)
    """
    if sort_type == 'primary':
        return PRIMARY_CATEGORIES.get(piece_info['primary_category'], 9)

    elif sort_type == 'secondary' and target_category:
        if piece_info['primary_category'] != target_category:
            return 9

        valid_secondaries = get_secondary_categories(target_category)
        if piece_info['secondary_category'] in valid_secondaries:
            return valid_secondaries.index(piece_info['secondary_category']) % 9

    return 9


def send_to_brickognize_api(file_name):
    """
    Sends image to Brickognize API for piece identification.
    """
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


def log_missing_piece(piece_data):
    """
    Logs information about pieces not found in the dictionary.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "Missing_Pieces.txt"

    try:
        with open(log_file, 'a') as f:
            f.write(f"\nTimestamp: {timestamp}\n")
            f.write(f"Piece ID: {piece_data.get('id', 'Unknown')}\n")
            f.write(f"Name: {piece_data.get('name', 'Unknown')}\n")
            f.write(f"Confidence Score: {piece_data.get('confidence', 0)}\n")
            f.write("-" * 50)
    except Exception as e:
        print(f"Error logging missing piece: {str(e)}")


def initialize_camera(directory="LegoPictures"):
    """
    Initialize camera and required directories.
    """
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    count = len([name for name in os.listdir(directory)
                 if name.startswith("Lego") and name.endswith(".jpg")]) + 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to initialize camera")

    return cap, directory, count


def capture_and_process_image(cap, directory, count, sort_type, target_category=None):
    """
    Capture image and process it through the sorting system.
    """
    ret, frame = cap.read()
    if not ret:
        return None, "Failed to capture image"

    filename = os.path.join(directory, f"Lego{count:03}.jpg")
    try:
        cv2.imwrite(filename, frame)
    except Exception as e:
        return None, f"Failed to save image: {str(e)}"

    # Get piece identification from API
    api_result = send_to_brickognize_api(filename)
    if "error" in api_result:
        return None, api_result["error"]

    # Process piece information
    result = attribute_piece(api_result, sort_type, target_category)
    return result, None


def attribute_piece(piece_identity, sort_type, target_category=None):
    """
    Determines the appropriate bin for a piece.
    """
    result = {
        "id": piece_identity.get("id"),
        "bin_number": 9,
        "confidence": piece_identity.get("score", 0)
    }

    # Check confidence threshold
    if result["confidence"] < 0.7:  # Confidence threshold
        result["error"] = "Low confidence score"
        return result

    # Get piece information from dictionary
    piece_id = result["id"]
    if not piece_id or piece_id not in lego_dict:
        log_missing_piece(result)
        result["error"] = "Piece not found in dictionary - logged for review"
        return result

    # Get piece information and determine bin
    piece_info = lego_dict[piece_id]
    result.update({
        "element_id": piece_id,
        "name": piece_info['name'],  # Add name from dictionary
        "primary_category": piece_info['primary_category'],
        "secondary_category": piece_info['secondary_category']
    })

    result["bin_number"] = get_bin_number(piece_info, sort_type, target_category)
    return result


def get_sort_configuration():
    """
    Get sorting configuration from user.
    """
    while True:
        print("\nSelect sorting type:")
        print("1: Primary Sort (Sort by main categories)")
        print("2: Secondary Sort (Sort specific category by subcategories)")

        mode = input("Enter 1 or 2: ").strip()

        if mode == '1':
            return 'primary', None

        elif mode == '2':
            print("\nAvailable primary categories for secondary sorting:")
            categories = list(PRIMARY_CATEGORIES.keys())
            for i, category in enumerate(categories, 1):
                print(f"{i}: {category}")
                secondaries = get_secondary_categories(category)
                print(f"   Secondary categories: {', '.join(secondaries)}")

            while True:
                try:
                    cat_num = int(input("\nEnter primary category number: "))
                    if 1 <= cat_num <= len(categories):
                        return 'secondary', categories[cat_num - 1]
                    print("Invalid category number.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print("Invalid input. Please enter 1 or 2.")


def main():
    """
    Main program entry point.
    """
    global lego_dict
    lego_dict = create_lego_dictionary()

    # Get sort configuration
    sort_type, target_category = get_sort_configuration()

    print(f"\nInitializing {sort_type} sort", end="")
    if target_category:
        print(f" for {target_category}")
    print("\nPress ESC to exit, click in window to capture image")

    try:
        # Initialize camera
        cap, directory, count = initialize_camera()
        cv2.namedWindow("Capture")

        def capture_callback(event, x, y, flags, param):
            nonlocal count
            if event == cv2.EVENT_LBUTTONDOWN:
                result, error = capture_and_process_image(
                    cap, directory, count, sort_type, target_category)

                if error:
                    print(f"Error: {error}")
                else:
                    print("\nPiece identified:")
                    print(f"Element ID: {result['element_id']}")
                    print(f"Name: {result['name']}")
                    print(f"Primary Category: {result['primary_category']}")
                    print(f"Secondary Category: {result['secondary_category']}")
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
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
