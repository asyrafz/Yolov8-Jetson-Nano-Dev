import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load YOLOv8 model (assuming 'yolov8n.pt' for YOLOv8 nano model)
model = YOLO('yolov8n.pt')

def detect_traffic_lights(image):
    results = model(image)
    # Extract bounding boxes from the results
    bboxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2, conf = map(int, box.xyxy[0])
            bboxes.append((x1, y1, x2, y2, conf))
    return bboxes

def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=30
    )
    return circles

def detect_color(circle, image):
    x, y, r = circle
    mask = np.zeros_like(image)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
    mean_val = cv2.mean(image, mask=mask[:, :, 0])
    b, g, r = mean_val[:3]
    if r > 100 and g < 100 and b < 100:
        return "Red"
    elif r > 100 and g > 100 and b < 100:
        return "Yellow"
    elif g > 100 and r < 100 and b < 100:
        return "Green"
    else:
        return "Unknown"

def process_frame(frame):
    traffic_lights = detect_traffic_lights(frame)
    
    for (x1, y1, x2, y2, conf) in traffic_lights:
        cropped_image = frame[y1:y2, x1:x2]
        circles = detect_circles(cropped_image)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                color = detect_color((x, y, r), cropped_image)
                cv2.circle(cropped_image, (x, y), r, (0, 255, 0), 4)
                cv2.putText(cropped_image, color, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Traffic Light", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the current frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Traffic Light Detection', processed_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
