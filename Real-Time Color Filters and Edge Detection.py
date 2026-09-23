import cv2
import numpy as np

def apply_filter(image, ftype):
    """Apply a filter to the image based on the filter type."""
    img = image.copy()
    if ftype == "red_tint":
        img[:, :, 1] = 0  # Green channel to 0
        img[:, :, 0] = 0  # Blue channel to 0
    elif ftype == "green_tint":
        img[:, :, 0] = 0  # Blue channel to 0
        img[:, :, 2] = 0  # Red channel to 0
    elif ftype == "blue_tint":
        img[:, :, 1] = 0  # Green channel to 0
        img[:, :, 2] = 0  # Red channel to 0
    elif ftype == "sobel":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(sx)
        abs_y = cv2.convertScaleAbs(sy)
        sob = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        img = cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR)
    elif ftype == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        can = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(can, cv2.COLOR_GRAY2BGR)
    elif ftype == "cartoon":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(image, 9, 300, 300)
        img = cv2.bitwise_and(color, color, mask=edges)
    return img

def main():
    # Added cv2.CAP_DSHOW for reliable Windows webcam initialization
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera. Make sure no other app is using it.")
        return
    
    ftype = "original"
    print("\n--- Real-Time Video Filter Script ---")
    print("Keys: r=Red, g=Green, b=Blue, s=Sobel, c=Canny, t=Cartoon, q=Quit")
    print("---------------------------------------")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break
            
        out = apply_filter(frame, ftype)
        cv2.imshow("Real-Time Filter (Press 'q' to quit)", out)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            ftype = "red_tint"
            print("Filter: Red Tint")
        elif key == ord('g'):
            ftype = "green_tint"
            print("Filter: Green Tint")
        elif key == ord('b'):
            ftype = "blue_tint"
            print("Filter: Blue Tint")
        elif key == ord('s'):
            ftype = "sobel"
            print("Filter: Sobel Edges")
        elif key == ord('c'):
            ftype = "canny"
            print("Filter: Canny Edges")
        elif key == ord('t'):
            ftype = "cartoon"
            print("Filter: Cartoon Effect")
        elif key == ord('q'):
            print("Exiting...")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()