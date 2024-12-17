import cv2
import time


def detect_lego_and_capture(camera_index=0, threshold=25, min_contour_area=500):
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)

    # Create background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=90, detectShadows=False)

    # Initialize variables
    last_check_time = time.time()
    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = back_sub.apply(frame)

        # Check if it's time to process (once per second)
        current_time = time.time()
        if current_time - last_check_time >= 1:
            last_check_time = current_time

            # Apply threshold to get binary image
            _, binary = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if any contour is large enough (to avoid small noise)
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:
                    # LEGO piece detected, capture image
                    image_name = f"lego_piece_{frame_count}.jpg"
                    cv2.imwrite(image_name, frame)
                    print(f"LEGO piece detected! Image saved as {image_name}")
                    frame_count += 1
                    break  # Only capture one image per second

        # Display the resulting frame (optional, for debugging)
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Usage example:
detect_lego_and_capture(camera_index=0, threshold=25, min_contour_area=500)