import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils

# Custom drawing styles to match the screenshot (White connections, Red landmark dots)[cite: 1]
landmark_spec = draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)    # Red dots (BGR format)
connection_spec = draw.DrawingSpec(color=(255, 255, 255), thickness=2)          # White lines

TH, IX = mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP

# Setup Audio (Pycaw)
try:
    dev = AudioUtilities.GetDefaultOutputDevice() if hasattr(AudioUtilities, "GetDefaultOutputDevice") else AudioUtilities.GetSpeakers()
    volctl = dev.EndpointVolume.QueryInterface(IAudioEndpointVolume)
    minv, maxv = volctl.GetVolumeRange()[:2]
except Exception as e:
    print(f"Pycaw error: {e}")
    exit()

# Setup Webcam with DirectShow for robust Windows support
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

WIN = "Hand Gesture Control"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
print("Gesture control started! Left hand = Volume, Right hand = Brightness. Press 'q' to quit.")

while True:
    ok, img = cap.read()
    if not ok:
        print("Error: Failed to grab frame.")
        break
        
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    
    # Process frame with MediaPipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb_img)

    if res.multi_hand_landmarks and res.multi_handedness:
        for i, hand in enumerate(res.multi_hand_landmarks):
            label = res.multi_handedness[i].classification[0].label
            
            # Draw landmarks with custom white lines and red dots[cite: 1]
            draw.draw_landmarks(
                img, 
                hand, 
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )
            
            lm = hand.landmark
            tp = (int(lm[TH].x * w), int(lm[TH].y * h))
            ip = (int(lm[IX].x * w), int(lm[IX].y * h))
            
            # Blue circles on thumb and index tips, connected by a green line[cite: 1]
            cv2.circle(img, tp, 8, (255, 0, 0), cv2.FILLED)  # Blue tip
            cv2.circle(img, ip, 8, (255, 0, 0), cv2.FILLED)  # Blue tip
            cv2.line(img, tp, ip, (0, 255, 0), 3)           # Green connecting line
            
            dist = float(np.hypot(ip[0] - tp[0], ip[1] - tp[1]))

            if label == "Left":  # Real right hand due to mirror flip -> Volume
                v = np.interp(dist, [30, 300], [minv, maxv])
                try:
                    volctl.SetMasterVolumeLevel(v, None)
                except Exception as e:
                    print(f"Volume error: {e}")
                    
                bar = int(np.interp(dist, [30, 300], [400, 150]))
                pct = int(np.interp(dist, [30, 300], [0, 100]))
                cv2.rectangle(img, (w - 85, 150), (w - 50, 400), (0, 255, 0), 2)
                cv2.rectangle(img, (w - 85, bar), (w - 50, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f"{pct}%", (w - 115, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            elif label == "Right":  # Real left hand due to mirror flip -> Brightness
                b = int(np.interp(dist, [30, 300], [0, 100]))
                try:
                    sbc.set_brightness(b)
                except Exception as e:
                    print(f"Brightness error: {e}")
                    
                bar = int(np.interp(dist, [30, 300], [400, 150]))
                x1, x2 = w - 85, w - 50
                cv2.rectangle(img, (x1, 150), (x2, 400), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, bar), (x2, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f"{b}%", (w - 115, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow(WIN, img)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord("q")):
        break
        
    try:
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break

cap.release()
cv2.destroyAllWindows()