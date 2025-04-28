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
import cv2
import numpy as np
import time
import pyautogui

# Carregar o modelo
model = YOLO("runs/detect/train2/weights/best.pt")

screen_width, screen_height = pyautogui.size()

# função que captura o ecrã e devolve a imagem
def capture_screen():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])  # Capturar do monitor principal
        img = np.array(screenshot)  # Converter em imagem
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # Converter BGRA para RGB
        img_height, img_width, _ = img.shape

        # Opcionalmente, guardar a imagem capturada
        # timestamp = time.strftime("%Y%m%d-%H%M%S-%f")  # Include microseconds
        # img_path = os.path.join('captured_images', f"capture_{timestamp}.jpg")
        # cv2.imwrite(img_path, img)

        return img, timestamp, img_width, img_height

while True:
    img, timestamp, img_width, img_height = capture_screen()
 
    #results = model.predict(source=img, save=True, save_txt=True, conf=0.1) 
    results = model(img)

    # Extract detections (bounding boxes)
    detections = results[0].boxes.xyxy  # Bounding boxes (x1, y1, x2, y2)

    if len(detections) > 0:
        print(f"Detetou {len(detections)} objetos.")

        # Opcionalmente, guardar imagem com as deteções
        annotated_frame = results[0].plot()  # Draw bounding boxes on the image
        result_path = os.path.join('detections', f"result_{timestamp}.jpg")
        cv2.imwrite(result_path, annotated_frame)

        for i, (x1, y1, x2, y2) in enumerate(detections.tolist()):
            # Calculate the center of the object
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # converter coordenadas
            scaled_x = int((center_x / img_width) * screen_width)
            scaled_y = int((center_y / img_height) * screen_height)

            # mover o rato
            pyautogui.moveTo(scaled_x, scaled_y, duration=0.3)
            pyautogui.click()
            print(f"Moved to object {i+1} at ({scaled_x}, {scaled_y})")

            # Pausa breve, entre objetos
            time.sleep(0.25)  # Adjust delay as needed

    time.sleep(3)