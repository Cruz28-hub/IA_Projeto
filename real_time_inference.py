from IPython import display
display.clear_output()

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import roboflow

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

import mss
import numpy as np
import time
import pyautogui

# Carregar o modelo
model = YOLO("runs/detect/train2/weights/best.pt")

screen_width, screen_height = pyautogui.size()

# Mapear os IDs para os nomes das classes
class_names = {
    0: "Banana",
    1: "Car",
    2: "Nitro",
    3: "Pumpkin",
    4: "left_boundary",
    5: "right_boundary"
}

# função que captura o ecrã e devolve a imagem
def capture_screen():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])  # Capturar do monitor principal
        img = np.array(screenshot)  # Converter em imagem
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Converter BGRA para RGB
        img_height, img_width, _ = img.shape

        timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
        return img, timestamp, img_width, img_height

while True:
    img, timestamp, img_width, img_height = capture_screen()

    results = model(img)

    # Extract detections (bounding boxes)
    boxes = results[0].boxes
    detections = boxes.xyxy  # Bounding boxes (x1, y1, x2, y2)
    class_ids = boxes.cls.cpu().numpy().astype(int)  # Class IDs

    if len(detections) > 0:
        print(f"Detetou {len(detections)} objetos.")

        annotated_frame = results[0].plot()
        result_path = os.path.join('detections', f"result_{timestamp}.jpg")
        cv2.imwrite(result_path, annotated_frame)

        for i, (box, class_id) in enumerate(zip(detections.tolist(), class_ids)):
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            scaled_x = int((center_x / img_width) * screen_width)
            scaled_y = int((center_y / img_height) * screen_height)

            class_name = class_names.get(class_id, "Unknown")
            print(f"Objeto {i+1}: {class_name} em ({scaled_x}, {scaled_y})")

            # Ação genérica de exemplo
            pyautogui.moveTo(scaled_x, scaled_y, duration=0.3)
            pyautogui.click()
            time.sleep(0.25)

    time.sleep(3)
