from IPython import display
display.clear_output()

import cv2
import time
import pyautogui
import numpy as np
import mss
from ultralytics import YOLO

# Load model
model = YOLO("C:/Users/joaoc/runs/detect/train16/weights/best.pt")

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Parameters
TURN_DISTANCE = 0.10
MIN_TURN_DURATION = 0.08
MAX_TURN_DURATION = 0.65
NITRO_KEY = 'n'
OBSTACLE_AVOID_DISTANCE = 0.2
ROAD_CENTER_WEIGHT = 0.7
FORWARD_INTERVAL = 0.8
FORWARD_HOLD_TIME = 0.7

# Helper function
def capture_screen():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (640, 480))
        return img, img.shape[1], img.shape[0]

def calculate_turn_duration(distance, img_width):
    normalized_dist = min(1.0, distance / img_width)
    return MIN_TURN_DURATION + (MAX_TURN_DURATION - MIN_TURN_DURATION) * normalized_dist

print("[INFO] Starting real-time inference. Press Ctrl+C to stop.")

last_forward_time = time.time()

while True:
    img, img_width, img_height = capture_screen()
    results = model(img, conf=0.35)

    own_car_x_center = None
    road_centers = []
    obstacles = []
    nitro_detected = False

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_name = model.names[int(box.cls[0])]
        box_center = (x1 + x2) / 2

        if class_name == "player_kart":
            own_car_x_center = box_center
        elif class_name == "road":
            road_centers.append(box_center)
        elif class_name in ["banana", "pumpkin"]:
            obstacles.append((box_center, class_name))
        elif class_name == "nitro":
            nitro_detected = True

    if not own_car_x_center:
        continue

    target_x = None
    if road_centers:
        road_center = np.mean(road_centers)
        target_x = road_center

        for obstacle_center, _ in obstacles:
            obstacle_dist = abs(obstacle_center - own_car_x_center) / img_width
            if obstacle_dist < OBSTACLE_AVOID_DISTANCE:
                avoid_dir = 1 if road_center > obstacle_center else -1
                target_x = (ROAD_CENTER_WEIGHT * road_center +
                            (1 - ROAD_CENTER_WEIGHT) * (obstacle_center + avoid_dir * img_width * 0.3))

    # Steering
    if target_x:
        dist = abs(own_car_x_center - target_x)
        if dist > TURN_DISTANCE * img_width:
            turn_key = 'left' if target_x < own_car_x_center else 'right'
            duration = calculate_turn_duration(dist, img_width)
            pyautogui.keyDown(turn_key)
            time.sleep(duration)
            pyautogui.keyUp(turn_key)

    # Nitro
    if nitro_detected and (not target_x or abs(own_car_x_center - target_x) < TURN_DISTANCE * img_width * 2):
        pyautogui.press(NITRO_KEY)

    # Forward
    current_time = time.time()
    if current_time - last_forward_time >= FORWARD_INTERVAL:
        pyautogui.keyDown('up')
        time.sleep(FORWARD_HOLD_TIME)
        pyautogui.keyUp('up')
        last_forward_time = current_time

    time.sleep(0.05)