from IPython import display
display.clear_output()

import cv2
import time
import pyautogui
import numpy as np
import mss
from ultralytics import YOLO

# Load model
model = YOLO("C:/Users/luiss/Documents/Estagio/IA_Projeto/runs/version_9/treino_versao_9/weights/best.pt")

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Parameters ajustados
TURN_DISTANCE = 0.25  # Aumentado para começar a virar mais cedo
MIN_TURN_DURATION = 0.2
MAX_TURN_DURATION = 0.8
NITRO_KEY = 'n'
TURN_ANGLE_ZONE = 0.3
ACCELERATION_DURATION = 1  # Tempo que mantém a aceleração pressionada

# Helper functions
def capture_screen():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (640, 480))
        return img, img.shape[1], img.shape[0]

def calculate_turn_duration(distance, img_width):
    normalized_dist = distance / img_width
    return MAX_TURN_DURATION * (1 - normalized_dist) + MIN_TURN_DURATION

# Initialize
step = 0
last_turn_time = time.time()
pyautogui.keyDown('up')  # Mantém acelerador pressionado continuamente

print("[INFO] Starting real-time inference. Press 'q' to quit.")

while True:
    img, img_width, img_height = capture_screen()
    results = model(img, conf=0.35)
    own_car_x_center = None
    track_edges = []
    class_names_detected = []

    # Process detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        class_names_detected.append(class_name)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if class_name == "playerKart":
            own_car_x_center = (x1 + x2) / 2
        elif class_name == "track":
            track_edges.append((x1, x2))

    # --- Turning Logic ---
    if own_car_x_center and track_edges:
        left_boundary = min([x1 for (x1, x2) in track_edges])
        right_boundary = max([x2 for (x1, x2) in track_edges])
        track_width = right_boundary - left_boundary
        
        dist_to_left = own_car_x_center - left_boundary
        dist_to_right = right_boundary - own_car_x_center
        
        # Virar mais agressivamente quando próximo das bordas
        if dist_to_left < TURN_DISTANCE * track_width:
            turn_duration = calculate_turn_duration(dist_to_left, track_width)
            pyautogui.keyUp('left')  # Garante que não está pressionado
            pyautogui.keyDown('right')
            time.sleep(turn_duration)
            pyautogui.keyUp('right')
            last_turn_time = time.time()
            
        elif dist_to_right < TURN_DISTANCE * track_width:
            turn_duration = calculate_turn_duration(dist_to_right, track_width)
            pyautogui.keyUp('right')  # Garante que não está pressionado
            pyautogui.keyDown('left')
            time.sleep(turn_duration)
            pyautogui.keyUp('left')
            last_turn_time = time.time()

    # --- Center Positioning Assist ---
    if own_car_x_center and track_edges:
        track_center = (left_boundary + right_boundary) / 2
        deviation = own_car_x_center - track_center
        
        # Correção mais suave mas contínua
        if abs(deviation) > track_width * 0.15:
            if deviation > 0 and time.time() - last_turn_time > 0.2:
                pyautogui.keyDown('left')
                time.sleep(0.1)
                pyautogui.keyUp('left')
                last_turn_time = time.time()
            elif deviation < 0 and time.time() - last_turn_time > 0.2:
                pyautogui.keyDown('right')
                time.sleep(0.1)
                pyautogui.keyUp('right')
                last_turn_time = time.time()

    # --- Nitro Use ---
    if "Nitro" in class_names_detected:
        pyautogui.press(NITRO_KEY)

    # Display
    cv2.imshow('Screen Capture', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pyautogui.keyUp('up')  # Libera o acelerador ao sair
        break

    time.sleep(0.02)  # Loop mais rápido
    step += 1

cv2.destroyAllWindows()