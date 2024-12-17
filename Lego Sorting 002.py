import requests
import json
import cv2
import os

lego_piece_dict = {
    "Basic": {
        "BASIC-brick": {
            "1x1": [],
            "1x2": [],
            "1x": [],
            "Tall": []
        },
        "BASIC-plate": {
            "1x1": [],
            "1x2": [],
            "1x": [],
            "2x": [],
            "3x": [],
            "4x": [],
            "6x": []
        },
        "BASIC-tile": []
    },
    "Wall": {
        "WALL-panel": [],
        "WALL-window_door": [],
        "WALL-fence": [],
        "WALL-decorative": [],
        "WALL-groove_rail": [],
        "WALL-stairs": []
    },
    "SNOT": {
        "SNOT-brick": [],
        "SNOT-jumper": [],
        "SNOT-bracket": []
    },
    "Clip": {
        "CLIP-bar": [],
        "CLIP-handle": [],
        "CLIP-flag": [],
        "CLIP-clip": [],
        "CLIP-door": [],
        "CLIP-flexible": [],
        "CLIP-other": []
    },
    "Hinge": {
        "HINGE-click_brick": [],
        "HINGE-hinge": [],
        "HINGE-turntable": [],
        "HINGE-click_plate": [],
        "HINGE-click_other": []
    },
    "Socket": {
        "SOCKET-towball": [],
        "SOCKET-ball": [],
        "SOCKET-click": []
    },
    "Angle": {
        "ANGLE-slope_55_65_75": [],
        "ANGLE-slope_45": [],
        "ANGLE-slope_33": [],
        "ANGLE-slope_10_18_30": [],
        "ANGLE-slope_inverted": [],
        "ANGLE-windscreen": [],
        "ANGLE-wedge_plate": [],
        "ANGLE-wedge_tile": [],
        "ANGLE-wedge_brick": [],
        "ANGLE-wedge_nose": [],
        "ANGLE-wedge_inverted": []
    },
    "Curved": {
        "CURVED-plate": [],
        "CURVED-brick": [],
        "CURVED-tile": [],
        "CURVED-cylinder": [],
        "CURVED-doubleplate": [],
        "CURVED-curved": [],
        "CURVED-arch_bow": [],
        "CURVED-dish_dome": [],
        "CURVED-cone": [],
        "CURVED-ball": [],
        "CURVED-heart_star": [],
        "CURVED-mudguard": [],
        "CURVED-wedge": [],
        "CURVED-other": []
    },
    "Vehicle": {
        "VEHICLE-wheel": [],
        "VEHICLE-pin": [],
        "VEHICLE-bracket": [],
        "VEHICLE-body": [],
        "VEHICLE-train": []
    },
    "Minifig": {
        "MINIFIG-CATEGORY-animals": [],
        "MINIFIG-minifig": [],
        "MINIFIG-nanofig": [],
        "MINIFIG-CATEGORY-accessories": [],
        "MINIFIG-CATEGORY-clothing_hair": [],
        "MINIFIG-minidoll": [],
        "MINIFIG-CATEGORY-weapons": [],
        "MINIFIG-hair-accessory": [],
        "MINIFIG-food_drink": [],
        "MINIFIG-footwear": [],
        "MINIFIG-tools_other": [],
        "MINIFIG-clothing": [],
        "MINIFIG-container_seat": [],
        "MINIFIG-weapon": [],
        "MINIFIG-transportation": []
    },
    "Nature": {
        "NATURE-elemental": [],
        "NATURE-flower": [],
        "NATURE-plant": [],
        "NATURE-produce": [],
        "NATURE-barb_horn_tail": [],
        "NATURE-tooth": []
    },
    "Technic": {
        "TECHNIC-brick": [],
        "TECHNIC-plate": [],
        "TECHNIC-liftarm": [],
        "TECHNIC-brick-round": [],
        "TECHNIC-frame": [],
        "TECHNIC-panel": [],
        "TECHNIC-axle": [],
        "TECHNIC-pin": [],
        "TECHNIC-connector": [],
        "TECHNIC-hub": [],
        "TECHNIC-gears": [],
        "TECHNIC-rack": [],
        "TECHNIC-other": [],
        "TECHNIC-mechanical": [],
        "TECHNIC-link": [],
        "TECHNIC-steering": [],
        "TECHNIC-engine": [],
        "TECHNIC-ball_socket": [],
        "TECHNIC-blade_propeller": []
    },
    "Electronics": {
        "ELECTRONICS-hub": [],
        "ELECTRONICS-motors": [],
        "ELECTRONICS-sensors_accessories": [],
        "ELECTRONICS-CATEGORIES": []
    },
    "Other": {
        "OTHER-railing_ladder": [],
        "OTHER-shooter": [],
        "OTHER-structural": [],
        "OTHER-miscellaneous": [],
        "OTHER-chain_string": []
    },
    "Retired": {
        "RETIRED-BASIC": [],
        "RETIRED-WALL": [],
        "RETIRED-SNOT": [],
        "RETIRED-CLIP": [],
        "RETIRED-SOCKET-ball": [],
        "RETIRED-SOCKET-click": [],
        "RETIRED-ANGLE": [],
        "RETIRED-HINGE": [],
        "RETIRED-TECHNIC": [],
        "RETIRED-CURVE": [],
        "RETIRED-MINIFIGURE": [],
        "RETIRED-NATURE": [],
        "RETIRED-VEHICLE": [],
        "RETIRED-ELECTRONICS": []
    },
    "Duplo": {
        "DUPLO-brick": [],
        "DUPLO-plate": [],
        "DUPLO-angle": [],
        "DUPLO-other": [],
        "DUPLO-curved": [],
        "QUATRO": []
    }
}

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
    str: The category the piece belongs to, or an error message if it cannot be classified.
    """
    if piece_identity.get("score", 0) < confidence_threshold:
        return "Error: Confidence score too low."

    piece_id = piece_identity.get("id")
    if not piece_id:
        return "Error: Piece ID is missing."

    for category, subcategories in lego_piece_dict.items():
        for subcategory, pieces in subcategories.items():
            if piece_id in pieces:
                return f"{category} -> {subcategory}"

    return "Error: Piece not found in dictionary."

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
                    category = attribute_piece(result)
                    print("Piece identification:", result)
                    print("Assigned Category:", category)
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
image_capture()
