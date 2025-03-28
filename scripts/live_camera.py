import cv2
import torch
from ultralytics import YOLO

# Load the finetuned YOLOv8 model
model_path = '../models/ball_detection/run8.pt'
model = YOLO(model_path)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference with confidence threshold
    results = model(frame, conf=0.5)  # Set confidence to 0.5

    # Render results on the frame
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow('Webcam Object Detection', annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()