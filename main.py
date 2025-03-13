import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    color_ranges = {
        "Red": (np.array([0, 120, 70]), np.array([10, 255, 255])),
        "Green": (np.array([40, 40, 40]), np.array([80, 255, 255])),
        "Blue": (np.array([90, 50, 50]), np.array([130, 255, 255])),
        "Yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
        "Orange": (np.array([10, 100, 100]), np.array([25, 255, 255])),
        "Purple": (np.array([130, 50, 50]), np.array([160, 255, 255])),
        "Pink": (np.array([160, 100, 100]), np.array([180, 255, 255])),
        "Brown": (np.array([10, 100, 20]), np.array([20, 255, 200])),
        "Black": (np.array([0, 0, 0]), np.array([180, 255, 30])),
        "White": (np.array([0, 0, 200]), np.array([180, 20, 255])),
        "Gray": (np.array([0, 0, 40]), np.array([180, 20, 200])),
        "Cyan": (np.array([80, 100, 100]), np.array([100, 255, 255])),
        "Magenta": (np.array([140, 100, 100]), np.array([160, 255, 255])),
        "Teal": (np.array([85, 100, 100]), np.array([95, 255, 255])),
        "Maroon": (np.array([0, 100, 20]), np.array([10, 255, 100])),
        "Navy": (np.array([100, 100, 20]), np.array([130, 255, 100])),
        "Olive": (np.array([30, 100, 50]), np.array([40, 255, 150])),
        "Gold": (np.array([20, 150, 150]), np.array([30, 255, 255])),
        "Silver": (np.array([0, 0, 180]), np.array([180, 25, 255])),
        "Beige": (np.array([10, 30, 150]), np.array([20, 70, 255])),
        "Coral": (np.array([0, 50, 200]), np.array([10, 150, 255])),
        "Lavender": (np.array([120, 50, 200]), np.array([140, 150, 255])),
        "Peach": (np.array([10, 100, 150]), np.array([20, 200, 255])),
        "Turquoise": (np.array([85, 150, 150]), np.array([95, 255, 255])),
        "Violet": (np.array([130, 50, 200]), np.array([160, 150, 255])),
        "Indigo": (np.array([110, 50, 100]), np.array([130, 255, 200])),
        "Mint": (np.array([70, 150, 150]), np.array([80, 255, 255])),
        "Charcoal": (np.array([0, 0, 20]), np.array([180, 10, 80])),
        "Amber": (np.array([20, 100, 200]), np.array([30, 255, 255])),
        "Cream": (np.array([10, 20, 200]), np.array([20, 50, 255]))
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        largest_contour = None
        largest_area = 0
        detected_color = None

        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_frame, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500 and area > largest_area:
                    largest_area = area
                    largest_contour = contour
                    detected_color = color

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{detected_color} - X: {x} Y: {y}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Object Detection and Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Now the script focuses on the largest object in front of the camera and only tracks that! Let me know if you want me to fine-tune it further!

