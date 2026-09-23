import cv2
import numpy as np
import time
import pyautogui

# Configurations
SCROLL_SPEED = 300
SCROLL_DELAY = 1

# Initialize camera using MSMF backend for stable Windows streaming
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

last_scroll = 0
p_time = time.time()
print("Gesture Scroll Control Active! Press 'q' to exit.")

# Background subtractor to isolate moving objects (like your hand)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

while True:
    success, img = cap.read()
    if not success or img is None:
        time.sleep(0.05)
        continue

    # Mirror flip for a natural view
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    
    # Apply background subtraction to isolate your hand motion
    fg_mask = bg_subtractor.apply(img)
    
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    gesture = "none"
    handedness = "Right"

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        
        # Filter out small movements
        if area > 4000:
            x, y, w_box, h_box = cv2.boundingRect(max_contour)
            
            # Draw tracking rectangle and center point over your hand
            cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            center_x = x + (w_box // 2)
            center_y = y + (h_box // 2)
            cv2.circle(img, (center_x, center_y), 8, (255, 0, 0), cv2.FILLED)

            # Determine gesture based on bounding box proportions (Open Palm vs Fist)
            box_area = w_box * h_box
            solidity = area / float(box_area) if box_area > 0 else 0
            
            if solidity > 0.60:
                gesture = "scroll_down" # Compact shape / Fist
            else:
                gesture = "scroll_up"   # Spread shape / Open Palm

            # Trigger system scroll with a cooldown delay
            if (time.time() - last_scroll) > SCROLL_DELAY:
                if gesture == "scroll_up":
                    pyautogui.scroll(SCROLL_SPEED)
                elif gesture == "scroll_down":
                    pyautogui.scroll(-SCROLL_SPEED)
                last_scroll = time.time()

    # Calculate FPS
    current_time = time.time()
    fps = int(1 / (current_time - p_time)) if (current_time - p_time) > 0 else 0
    p_time = current_time

    # Display HUD status overlay
    cv2.putText(img, f"FPS: {fps} | Hand: {handedness} | Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show active window
    cv2.imshow("Gesture Control", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()