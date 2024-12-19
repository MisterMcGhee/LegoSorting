import requests
import csv
import cv2
import os


def create_lego_dictionary(filepath='Bricklink ID Name Category.csv'):
    """
    Creates a dictionary mapping Lego piece IDs to their categories and names.

    Args:
        filepath (str): Path to the CSV file containing Lego piece information

    Returns:
        dict: Dictionary with piece IDs as keys and tuples of (category, name) as values
    """
    lego_dict = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Store both category and name for each piece ID
                lego_dict[row['ID']] = (row['Category'], row['Name'])

    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    return lego_dict

def send_to_brickognize_api(file_name):
    """
    Uploads an image to the Brickognize API and returns parsed piece identity data.

    Args:
    file_name (str): The name of the image file to upload.

    Returns:
    dict: A dictionary containing the piece ID, name, and score.
    """
    url = "https://api.brickognize.com/predict/"
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    # Resolve file path
    file_name = os.path.abspath(file_name)

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

        data = response.json()
        if "items" in data and data["items"]:
            item = data["items"][0]
            piece_identity = {
                "id": item.get("id"),
                "name": item.get("name"),
                "score": item.get("score")
            }
            return piece_identity

        return {"error": "No items found in response."}

    except requests.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def attribute_piece(piece_identity, confidence_threshold=0.7):
    """
    Assigns a piece to its category based on the Brickognize API results.

    Args:
    piece_identity (dict): The dictionary containing 'id', 'name', and 'score'.
    confidence_threshold (float): The minimum confidence score to proceed.

    Returns:
    dict: Updated piece_identity dictionary including the category.
    """
    if piece_identity.get("score", 0) < confidence_threshold:
        piece_identity["category"] = "Error: Confidence score too low."
        return piece_identity

    piece_id = piece_identity.get("id")
    if not piece_id:
        piece_identity["category"] = "Error: Piece ID is missing."
        return piece_identity

    if piece_id in lego_dict:
        category, name = lego_dict[piece_id]
        piece_identity["category"] = category
        piece_identity["name"] = name
        return piece_identity

    piece_identity["category"] = "Error: Piece not found in dictionary."
    return piece_identity

def image_capture(directory="LegoPictures"):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    count = len([name for name in os.listdir(directory) if name.startswith("Lego") and name.endswith(".jpg")]) + 1

    def capture(event, x, y, flags, param):
        nonlocal count
        if event == cv2.EVENT_LBUTTONDOWN:
            ret, frame = cap.read()
            if ret:
                filename = os.path.join(directory, f"Lego{count:03}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Image saved to {filename}")

                # Send to Brickognize API and process result
                result = send_to_brickognize_api(filename)
                if "error" in result:
                    print(result["error"])
                else:
                    result = attribute_piece(result)
                    print("Piece identification:", result)
                count += 1

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Capture")
    cv2.setMouseCallback("Capture", capture)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
lego_dict = create_lego_dictionary('Bricklink ID Name Category.csv')
image_capture()
