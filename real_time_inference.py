from IPython import display
display.clear_output()

import cv2
import time
import pyautogui
import numpy as np
import mss
from ultralytics import YOLO

# 1. Carregar modelo YOLO treinado
model = YOLO("C:/Users/luiss/Documents/Estagio/IA_Projeto/runs/version_8_50Epochs/treino_versao_8_50Epochs/weights/best.pt")
print("Classes do modelo carregado:")
print(model.names)

# 2. Parâmetros ajustáveis
TURN_DISTANCE = 0.25
MIN_TURN_DURATION = 0.1
MAX_TURN_DURATION = 0.6
NITRO_KEY = 'n'
CONFIDENCE_THRESHOLD = 0.45

# 3. Função para capturar tela
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (640, 480))
        return img, img.shape[1], img.shape[0]

# 4. Função para calcular tempo de viragem
def calculate_turn(distance, max_width):
    ratio = min(distance / max_width, 1.0)
    return MIN_TURN_DURATION + (MAX_TURN_DURATION - MIN_TURN_DURATION) * ratio

# 5. Delay para o usuário abrir o jogo
print("[SISTEMA] Iniciando em 3 segundos...")
time.sleep(3)

last_turn_time = time.time()
print("[SISTEMA] Controle automático iniciado. Pressione 'Q' para sair.")

while True:
    frame, width, height = capture_screen()
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    kart_pos = None
    left_bound = float('inf')
    right_bound = -float('inf')
    nitro_detected = False

    print(">> Objetos detectados no frame:")
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_name = model.names[int(box.cls[0])]
        print(f" - {class_name}")

        if class_name == "player_kart":
            kart_pos = (x1 + x2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        elif class_name == "track":
            left_bound = min(left_bound, x1)
            right_bound = max(right_bound, x2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        elif class_name == "nitro":
            nitro_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Verificação segura
    if kart_pos is not None and left_bound < right_bound:
        pyautogui.keyDown('up')  # Só acelera se tudo estiver detectado
        track_width = right_bound - left_bound
        dist_left = (kart_pos - left_bound) / track_width
        dist_right = (right_bound - kart_pos) / track_width

        if nitro_detected:
            pyautogui.press(NITRO_KEY)
            print(">> NITRO ativado!")
            cv2.putText(frame, "NITRO ATIVADO!", (width//2-100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        turn_direction = None
        if dist_left < TURN_DISTANCE:
            turn_direction = 'right'
            turn_duration = calculate_turn(kart_pos - left_bound, track_width)
        elif dist_right < TURN_DISTANCE:
            turn_direction = 'left'
            turn_duration = calculate_turn(right_bound - kart_pos, track_width)

        if turn_direction and (time.time() - last_turn_time > 0.2):
            print(f">> Virando para {turn_direction.upper()} por {turn_duration:.2f}s")
            pyautogui.keyDown(turn_direction)
            time.sleep(turn_duration)
            pyautogui.keyUp(turn_direction)
            last_turn_time = time.time()

            cv2.putText(frame, f"VIRANDO {turn_direction.upper()}!", 
                        (width//2-80, height-50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
    else:
        pyautogui.keyUp('up')  # Não acelera se não estiver tudo certo

    # Debug visual
    cv2.putText(frame, f"Kart Position: {kart_pos if kart_pos is not None else 'N/D'}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Track Bounds: L={left_bound} | R={right_bound}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Controle Automático', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        pyautogui.keyUp('up')
        break

cv2.destroyAllWindows()
print("[SISTEMA] Controle desativado.")
