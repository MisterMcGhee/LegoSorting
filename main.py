import requests
import os
import cv2
from collections import OrderedDict

# Sorting maps
bulk_sort_mapping = {
    "brick": "Bricks",
    "plate": "Plates",
    "tile": "Tiles",
    "slope": "Slopes",
    "technic": "Technic",
    "curved": "Curves",
    "panel": "Panels"
}

brick_sort_mapping = OrderedDict([
    ("brick, modified", "Modified Bricks"),
    ("brick, round", "Round Bricks"),
    ("brick 1 x 1", "1x1 Bricks"),
    ("brick 1 x 2", "1x2 Bricks"),
    ("brick 1 x ", "1x3 Bricks or larger"),
    ("brick 2 x 2", "2x2 Bricks"),
    ("brick 2 x 4", "2x4 Bricks"),
    ("brick 2 x ", "2x3 or 2x6 or larger")
])

plate_sort_mapping = OrderedDict([
    ("plate, modified", "Modified Plates"),
    ("plate, round", "Round Plates"),
    ("plate 1 x 1", "1x1 Plates"),
    ("plate 1 x 2", "1x2 Plates"),
    ("plate 1 x ", "1x3 Plates or larger"),
    ("plate 2 x 2", "2x2 Plates"),
    ("plate 2 x 4", "2x4 Plates"),
    ("plate 2 x ", "2x3 or 2x6 or larger"),
])

tile_sort_mapping = OrderedDict([
    ("tile, modified", "Modified Tiles"),
    ("tile, round", "Round Tiles"),
    ("tile, corner", "Corner Tiles"),
    ("tile, printed", "Printed Tiles"),
    ("tile 1 x 1", "1x1 Tiles"),
    ("tile 1 x 2", "1x2 Tiles"),
    ("tile 1 x ", "1x3 Tiles or larger"),
    ("tile 2 x 2", "2x2 Tiles"),
    ("tile 2 x 4", "2x4 Tiles"),
    ("tile 2 x ", "2x3 or 2x6 or larger"),
])

slope_sort_mapping = OrderedDict([
    ("slope, curved 1 x", "1x Curved Slopes"),
    ("slope, curved 2 x", "2x Curved Slopes"),
    ("slope, curved", "1x Curved Slopes"),
    ("slope, inverted 33", "33 degree Inverted Slopes"),
    ("slope, inverted 45", "45 degree Inverted Slopes"),
    ("slope, inverted", "60,65,75 degree degree Inverted Slopes"),
    ("slope", "Slope")
])

wedge_sort_mapping = OrderedDict([
    ("wedge 2 x", "1x Wedges"),
    ("wedge 3 x", "2x Wedges"),
    ("wedge 4 x", "4x Wedges"),
    ("wedge, plate 2 x", "2x Wedge Plates"),
    ("wedge, plate 3 x", "3x Wedge Plates"),
    ("wedge, plate 4 x ", "4x Wedge Plates"),
    ("wedge, plate", "Large Wedges")
])

technic_sort_mapping = {
    "technic, brick": "Technic Bricks",
    "technic, axle": "Technic Axles and Pins",
    "technic, pin": "Technic Axles and Pins",
    "technic, connector": "Technic Connectors",
    "technic, gear": "Technic Gears",
    "technic, liftarm": "Technic Liftarms",
    "technic, panel": "Technic Panels"
}

panel_sort_mapping = {
    "Panel 1 x": "1x Panels",
    "Panel 2 x ": "2x Panels",
    "Panel": "Larger Panels",
}

# Dictionary of all sorting maps
sorting_maps = {
    "bulk": bulk_sort_mapping,
    "brick": brick_sort_mapping,
    "plate": plate_sort_mapping,
    "tile": tile_sort_mapping,
    "slope": slope_sort_mapping,
    "wedge": wedge_sort_mapping,
    "technic": technic_sort_mapping,
    "panel": panel_sort_mapping
}


def upload_image_to_api(file_name):
    """
    Uploads an image to the Brickognize API and returns the API response.

    Args:
    file_name (str): The name of the image file to upload.

    Returns:
    dict: The API response or an error message.
    """
    url = "https://api.brickognize.com/predict/"
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    # Error handling
    if not os.path.isfile(file_name):
        return {"error": f"File not found or not a file: {file_name}"}

    if os.path.getsize(file_name) == 0:
        return {"error": f"File is empty: {file_name}"}

    if not any(file_name.lower().endswith(ext) for ext in valid_extensions):
        return {"error": f"Invalid file extension. Allowed: {', '.join(valid_extensions)}"}

    # Open the picture, load the file to payload, get response
    try:
        with open(file_name, "rb") as image_file:
            files = {"query_image": (file_name, image_file, "image/jpeg")}
            response = requests.post(url, files=files)

        # Test connection is valid
        if response.status_code != 200:
            return {"error": f"API request failed with status code: {response.status_code}"}

        return response.json()

    except requests.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def process_brickognize_data(api_response):
    """
    Processes the Brickognize API response and extracts relevant information.

    Args:
    api_response (dict): The response from the Brickognize API.

    Returns:
    dict: Processed data including id, name, category, and score.
    """
    if "items" not in api_response or not api_response["items"]:
        return {"error": "No predictions found in the API response"}

    top_prediction = api_response["items"][0]
    return {
        "id": top_prediction.get("id"),
        "name": top_prediction.get("name", "").lower(),
        "category": top_prediction.get("category", "").lower(),
        "score": top_prediction.get("score")
    }


def determine_sort(processed_data, sort_map="bulk"):
    """
    Determines the sorting category based on the processed Brickognize data and specified sorting map.

    Args:
    processed_data (dict): The processed data from process_brickognize_data.
    sort_map (str): The name of the sorting map to use (default is "bulk").

    Returns:
    str: The sorting category.
    """
    if sort_map == "bulk":
        categories = [cat.strip() for cat in processed_data.get("category", "").split(',')]

        # Select the appropriate sorting map
        current_map = sorting_maps["bulk"]

        # Try to match each category in order
        for cat in categories:
            for key, value in current_map.items():
                if key in cat:
                    return value

        # If no match is found, return "Other"
        return "Other"
    else:
        piece_name = processed_data.get("name", "").lower()

        # Select the appropriate sorting map
        current_map = sorting_maps.get(sort_map, bulk_sort_mapping)

        # Iterate through the map to find the first matching prefix
        for key, value in current_map.items():
            if piece_name.startswith(key):
                return value

        # If no match found, return "Other"
        return f"Other {sort_map.capitalize()}s"


def capture_and_process_image(save_folder='LegoPictures', sort_map="bulk"):
    """
    Captures images from the webcam, processes them, and uploads to the Brickognize API.

    Args:
    save_folder (str): The folder where images will be saved.
    sort_map (str): The name of the sorting map to use (default is "bulk").

    Returns:
    dict: A dictionary containing processed data for all captured Lego pieces.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    cap = cv2.VideoCapture(0)
    photo_counter = 0
    all_brick_data = {}

    def triggered_event(event, x, y, flags, param):
        nonlocal photo_counter
        if event == cv2.EVENT_LBUTTONDOWN:
            ret, frame = cap.read()
            if ret:
                processed_frame = image_processing(frame)

                photo_name = f'Lego{photo_counter}.jpg'
                photo_path = os.path.join(save_folder, photo_name)
                cv2.imwrite(photo_path, processed_frame)
                print(f'Photo saved: {photo_path}')

                api_response = upload_image_to_api(photo_path)
                processed_data = process_brickognize_data(api_response)

                if "error" not in processed_data:
                    sort_category = determine_sort(processed_data, sort_map)
                    processed_data["sort_category"] = sort_category

                brick_key = f"Lego{photo_counter}"
                all_brick_data[brick_key] = processed_data

                print(f"\nProcessed {brick_key}:")
                print(friendly_dictionary(processed_data))

                photo_counter += 1

    cv2.namedWindow('Webcam')
    cv2.setMouseCallback('Webcam', triggered_event)

    print(f"Using sorting map: {sort_map}")
    print("Click inside the window to capture and process a Lego piece. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return all_brick_data


def image_processing(image):
    """
    Process the captured image. Currently, a placeholder for future implementation.
    """
    return image


def friendly_dictionary(dictionary):
    """Convert a dictionary into a string with each key-value pair on a new line."""
    return '\n'.join(f'{key}: {value}' for key, value in dictionary.items())


def session_summary(all_brick_data, sort_map):
    """
    Creates a text file summarizing the session data for all processed Lego pieces.

    Args:
    all_brick_data (dict): A dictionary containing processed data for all captured Lego pieces.
    sort_map (str): The name of the sorting map used in this session.

    Returns:
    str: The path to the created summary file.
    """
    filename = "Brick_data.txt"

    with open(filename, 'w') as f:
        f.write(f"Lego Sorting Session Summary\n")
        f.write(f"Sorting Map Used: {sort_map}\n\n")

        for brick_key, brick_data in all_brick_data.items():
            f.write(f"{brick_key}:\n")
            for key, value in brick_data.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    return filename


def main():
    sort_map = input("Enter sorting map to use (bulk, brick, plate, tile, technic, slope, curved, panel): ").lower()
    if sort_map not in sorting_maps:
        print("Invalid sorting map. Using bulk sort.")
        sort_map = "bulk"

    all_brick_data = capture_and_process_image(sort_map=sort_map)

    summary_file = session_summary(all_brick_data, sort_map)
    print(f"\nSession summary has been saved to: {summary_file}")

if __name__ == "__main__":
    main()

