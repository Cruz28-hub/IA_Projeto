import time
import os
import cv2
import numpy as np
import mss
import pyautogui
from ultralytics import YOLO

# Mapeamento de teclas
CONTROLS = {
    'left': 'left',
    'right': 'right',
    'accelerate': 'up',
    'brake': 'down',
    'power_up': 'space',
    'nitro': 'n',
}

# Carregar o modelo treinado
model = YOLO("runs/detect/train2/weights/best.pt")

# Obter dimensões da tela
screen_width, screen_height = pyautogui.size()

# Função para capturar a tela
def capture_screen():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])  # Captura do monitor principal
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_height, img_width, _ = img.shape
        return img, img_width, img_height

# Função para controlar o kart
def control_kart(action, duration=0.1):
    pyautogui.keyDown(CONTROLS[action])
    time.sleep(duration)
    pyautogui.keyUp(CONTROLS[action])

# Início do loop principal
print("🎮 Iniciando controle automático do SuperTuxKart...")
pyautogui.keyDown(CONTROLS['accelerate'])  # Aceleração constante

try:
    while True:
        img, img_width, img_height = capture_screen()
        results = model(img)

        # Verificar se há detecções
        if results and results[0].boxes:
            for detection in results[0].boxes:
                cls_id = int(detection.cls)
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = detection.xyxy[0]
                x_center = (x1 + x2) / 2

                if cls_name == 'enemy_kart':
                    # Desviar do kart inimigo
                    if x_center < img_width / 2:
                        control_kart('left', 0.2)
                    else:
                        control_kart('right', 0.2)

                elif cls_name == 'power_up':
                    # Alinhar o kart com o power-up
                    if x_center < img_width / 2 - 50:
                        control_kart('left', 0.1)
                    elif x_center > img_width / 2 + 50:
                        control_kart('right', 0.1)

                elif cls_name == 'bonus':
                    # Alinhar o kart com o bónus
                    if x_center < img_width / 2 - 50:
                        control_kart('left', 0.1)
                    elif x_center > img_width / 2 + 50:
                        control_kart('right', 0.1)
                    else:
                        control_kart('power_up')

                elif cls_name == 'nitro':
                    # Alinhar o kart com o nitro
                    if x_center < img_width / 2 - 50:
                        control_kart('left', 0.1)
                    elif x_center > img_width / 2 + 50:
                        control_kart('right', 0.1)
                    else:
                        control_kart('nitro')

        time.sleep(0.05)  # Reduzir o delay para melhor responsividade

except KeyboardInterrupt:
    print("🚨 Interrompido pelo usuário.")
    pyautogui.keyUp(CONTROLS['accelerate'])  # Liberar a tecla de aceleração
print("🏁 Controle automático encerrado.")