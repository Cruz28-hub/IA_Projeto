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

# Helper: screen capture
def capture_screen():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (640, 480))
        return img, img.shape[1], img.shape[0]

# Helper: calculate steering duration
def calculate_turn_duration(distance, img_width):
    normalized_dist = min(1.0, distance / img_width)
    return MIN_TURN_DURATION + (MAX_TURN_DURATION - MIN_TURN_DURATION) * normalized_dist

# Key press tracking
pressed_keys = set()

def press_key(key):
    if key not in pressed_keys:
        pyautogui.keyDown(key)
        pressed_keys.add(key)

def release_key(key):
    if key in pressed_keys:
        pyautogui.keyUp(key)
        pressed_keys.remove(key)

print("[INFO] Starting real-time inference. Press Ctrl+C to stop.")

# Key timing
last_forward_time = 0
forward_end_time = 0
turn_end_time = 0
current_turn_key = None

while True:
    now = time.time()
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

    # Compute desired target X
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

    # --- Steering logic ---
    if target_x:
        dist = abs(own_car_x_center - target_x)
        if dist > TURN_DISTANCE * img_width and current_turn_key is None:
            current_turn_key = 'left' if target_x < own_car_x_center else 'right'
            duration = calculate_turn_duration(dist, img_width)
            turn_end_time = now + duration
            press_key(current_turn_key)

    # End turn if time passed
    if current_turn_key and now > turn_end_time:
        release_key(current_turn_key)
        current_turn_key = None

    # --- Nitro logic ---
    if nitro_detected and (not target_x or abs(own_car_x_center - target_x) < TURN_DISTANCE * img_width * 2):
        pyautogui.press(NITRO_KEY)

    # --- Forward motion logic ---
    if now - last_forward_time >= FORWARD_INTERVAL:
        press_key('up')
        forward_end_time = now + FORWARD_HOLD_TIME
        last_forward_time = now

    # End forward key if time passed
    if 'up' in pressed_keys and now > forward_end_time:
        release_key('up')

    time.sleep(0.01)
