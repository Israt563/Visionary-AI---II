import cv2
import numpy as np

# Initialize video capture with DirectShow for reliable Windows support
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam. Make sure no other application (like Zoom) is using it.")
    exit()

print("Webcam started successfully! Place your hand in front of the camera. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to capture image from camera.")
        break

    # Flip the frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)

    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Broadened HSV range for skin color to handle indoor room lighting better
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    # Create a binary mask to isolate skin tones
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Clean up background noise using morphological operations (erode and dilate)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Apply the mask to the frame to show only the detected skin areas
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours (shapes) in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, track the largest one (the hand/face)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)  # Find the largest contour
        
        # Ignore small background noise blobs (area threshold of 1000 pixels)
        if cv2.contourArea(max_contour) > 1000:
            # Draw a bounding box around the detected hand
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Skin Tracked', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calculate and draw the center point of the hand
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

    # Display both the original video feed and the filtered mask output
    cv2.imshow('Original Frame (Mirror View)', frame)
    cv2.imshow('Filtered Mask (What the Camera Detects)', result)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and release system resources
cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()