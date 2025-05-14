from IPython import display
display.clear_output()

import cv2
import os
import time
import pyautogui
import numpy as np
import mss
from ultralytics import YOLO

# Load model
model = YOLO("C:/Users/joaoc/runs/detect/train14/weights/best.pt")

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Parameters
TURN_DISTANCE = 0.15    # Percentage of screen width
MIN_TURN_DURATION = 0.1
MAX_TURN_DURATION = 0.5
NITRO_KEY = 'n'
TURN_ANGLE_ZONE = 0.4   # For drift-based turn assistance

# Helper function
def capture_screen():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (640, 480))  # Standardize input size
        return img, img.shape[1], img.shape[0]

def calculate_turn_duration(distance, img_width):
    normalized_dist = distance / img_width
    return MAX_TURN_DURATION * (1 - normalized_dist) + MIN_TURN_DURATION

# Initialize loop
step = 0

print("[INFO] Starting real-time inference. Press 'q' on image window to quit.")

while True:
    img, img_width, img_height = capture_screen()
    results = model(img, conf=0.35)  # Try lower conf threshold
    own_car_x_center = None
    boundary_edges = []
    class_names_detected = []

    # Draw boxes and collect data
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        class_names_detected.append(class_name)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if class_name == "Own_Car":
            own_car_x_center = (x1 + x2) / 2
        elif class_name == "boundary":
            boundary_edges.append((x1, x2))

    turning = False

    # --- Turning Logic ---
    if own_car_x_center and boundary_edges:
        closest_dist = float('inf')
        turn_key = None

        for (x1, x2) in boundary_edges:
            boundary_center = (x1 + x2) / 2
            dist = abs(own_car_x_center - boundary_center)

            if dist < closest_dist:
                closest_dist = dist
                turn_key = 'left' if boundary_center > own_car_x_center else 'right'

        if closest_dist < TURN_DISTANCE * img_width:
            turn_duration = calculate_turn_duration(closest_dist, img_width)
            pyautogui.keyDown(turn_key)
            time.sleep(turn_duration)
            pyautogui.keyUp(turn_key)
            turning = True

    # --- Steering Assist ---
    if own_car_x_center:
        if own_car_x_center < img_width * (0.5 - TURN_ANGLE_ZONE):
            pyautogui.keyDown('right')
            time.sleep(0.05)
            pyautogui.keyUp('right')
        elif own_car_x_center > img_width * (0.5 + TURN_ANGLE_ZONE):
            pyautogui.keyDown('left')
            time.sleep(0.05)
            pyautogui.keyUp('left')

    # --- Nitro Use ---
    if not turning and "Nitro" in class_names_detected:
        pyautogui.press(NITRO_KEY)

    # --- Forward Movement ---
    if step % 5 == 0:
        pyautogui.keyDown('up')
        time.sleep(0.05)
        pyautogui.keyUp('up')

    step += 1
    time.sleep(0.05)  # Loop control
