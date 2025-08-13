import random
import time
import cv2
import numpy as np
from playsound import playsound
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Read class names from coco.txt
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().strip().split("\n")

# Generate random colors for each class
detection_colors = []
for _ in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Use video file or webcam (0)
cap = cv2.VideoCapture("13.mp4")  # or cap = cv2.VideoCapture(0) for webcam
if not cap.isOpened():
    print("Cannot open video source")
    exit()

last_alert_time = 0
alert_cooldown = 2  # seconds cooldown between alerts

plt.ion()  # interactive mode on for matplotlib

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot read frame")
        break

    # Run prediction on current frame
    results = model.predict(source=[frame], conf=0.45, save=False)
    DP = results[0].numpy()

    alert_triggered = False

    if len(DP) != 0:
        for i in range(len(results[0])):
            boxes = results[0].boxes
            box = boxes[i]
            clsID = int(box.cls.numpy()[0])
            conf = float(box.conf.numpy()[0])
            bb = box.xyxy.numpy()[0]

            # Draw bounding box on frame
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                3,
            )

            # Put class name and confidence
            cv2.putText(
                frame,
                f"{class_list[clsID]} {conf:.2f}",
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # If detected class is "person", trigger alert
            if class_list[clsID] == "person":
                alert_triggered = True

    # Play alert sound with cooldown
    if alert_triggered and (time.time() - last_alert_time) > alert_cooldown:
        playsound("alert.mp3")
        last_alert_time = time.time()

    # Convert BGR frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame_rgb)
    plt.title("Object Detection")
    plt.axis('off')
    plt.pause(0.001)  # short pause to update plot
    plt.clf()  # clear for next frame

cap.release()
plt.close()
