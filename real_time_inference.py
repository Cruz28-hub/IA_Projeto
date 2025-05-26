from datetime import datetime
import cv2
import os
import time
import numpy as np
import pyautogui
import mss
from ultralytics import YOLO
import random
import ultralytics

ultralytics.checks()


class KartController:
    def __init__(self):
        self.model = YOLO("runs/detect/train/weights/best.pt")
        os.makedirs('detections', exist_ok=True)
        self.last_turn_time = time.time()
        self.last_turn_direction = None
        self.consecutive_straight_frames = 0
        self.last_successful_detection = time.time()
        self.last_positions = []
        self.last_reset_time = time.time() - 30
        self.screen_width, self.screen_height = pyautogui.size()

        self.track_center_history = []
        self.steering_smoothing = 1.0
        self.last_steering_direction = "straight"
        self.dead_zone = 0.05

        self.accelerate = True
        self.last_accel_toggle = time.time()
        self.accel_on_time = 0.01
        self.accel_off_time = 0.15

    def capture_screen(self):
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[0])
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            return img, timestamp, img.shape[1], img.shape[0]

    def smooth_steering_input(self, direction, intensity):
        return intensity if direction == self.last_steering_direction else intensity * self.steering_smoothing

    def steer_car(self, direction, intensity=1.0):
        if direction == "straight":
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
            self.consecutive_straight_frames += 1
            self.last_steering_direction = "straight"
            return

        intensity = self.smooth_steering_input(direction, intensity)
        if intensity < self.dead_zone:
            return

        self.consecutive_straight_frames = 0
        self.last_turn_time = time.time()
        self.last_turn_direction = direction
        self.last_steering_direction = direction

        opposite = "right" if direction == "left" else "left"
        pyautogui.keyUp(opposite)

        if intensity >= 0.8:
            print(f"VIRAGEM FORTE {direction.upper()}! (intensidade: {intensity:.2f})")
            pyautogui.keyDown(direction)
            time.sleep(0.04)
        elif intensity >= 0.4:
            print(f"Viragem média {direction} (intensidade: {intensity:.2f})")
            pyautogui.keyDown(direction)
            time.sleep(0.025)
            pyautogui.keyUp(direction)
            time.sleep(0.005)
            pyautogui.keyDown(direction)
            time.sleep(0.01)
            pyautogui.keyUp(direction)
        else:
            print(f"Ajuste suave {direction} (intensidade: {intensity:.2f})")
            pyautogui.keyDown(direction)
            time.sleep(0.01)
            pyautogui.keyUp(direction)

    def reset_car(self):
        print("RESET COM BACKSPACE")
        pyautogui.press("backspace")
        self.last_reset_time = time.time()

    def check_if_player_missing(self):
        if time.time() - self.last_reset_time > 8:
            if self.last_positions:
                last_time = self.last_positions[-1][1]
                if time.time() - last_time > 5:
                    print(f"PERSONAGEM NÃO DETECTADO POR {time.time() - last_time:.1f}s! Possível crash.")
                    self.reset_car()

    def check_if_game_frozen(self):
        if time.time() - self.last_reset_time > 10 and len(self.last_positions) >= 25:
            ref_pos = self.last_positions[0][0]
            if all(abs(pos[0] - ref_pos[0]) <= 2 and abs(pos[1] - ref_pos[1]) <= 2 for pos, _ in self.last_positions[1:]):
                print("JOGO CONGELADO OU COMPLETAMENTE PARADO! Forçando reset...")
                self.reset_car()

    def get_smoothed_track_center(self, current_center):
        self.track_center_history.append(current_center)
        if len(self.track_center_history) > 5:
            self.track_center_history.pop(0)

        weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        weighted_sum = sum(center * weights[i] for i, center in enumerate(self.track_center_history))
        return weighted_sum / sum(weights[:len(self.track_center_history)])

    def process_track_detection(self, track_center_x, track_width, img_width):
        screen_center = img_width // 2
        smoothed_center = self.get_smoothed_track_center(track_center_x)
        offset = smoothed_center - screen_center
        dead_zone = max(track_width * 0.08, 15)

        print(f"Track center: {smoothed_center:.1f}, Offset: {offset:.1f}, Dead zone: {dead_zone:.1f}")

        if abs(offset) < dead_zone:
            print("CENTRO - Mantendo direção reta")
            self.steer_car("straight")
            return

        normalized_offset = min(1.0, abs(offset) / (img_width * 0.25))

        if normalized_offset < 0.3:
            intensity = 0.15 + (normalized_offset * 0.4)
        elif normalized_offset < 0.6:
            intensity = 0.35 + (normalized_offset * 0.4)
        else:
            intensity = 0.65 + (normalized_offset * 0.35)

        intensity *= 0.8

        direction = "left" if offset > 0 else "right"
        print(f"{direction.upper()} - Offset: {abs(offset):.1f}px, Intensidade: {intensity:.2f}")
        self.steer_car(direction, intensity)

    def start(self):
        try:
            print("Iniciando controle de kart com visão computacional...")
            pyautogui.keyDown('up')

            while True:
                img, timestamp, img_width, img_height = self.capture_screen()
                results = self.model(img, conf=0.3)
                boxes = results[0].boxes

                current_time = time.time()

                if self.accelerate:
                    if current_time - self.last_accel_toggle > self.accel_on_time:
                        pyautogui.keyUp('up')
                        self.accelerate = False
                        self.last_accel_toggle = current_time
                else:
                    if current_time - self.last_accel_toggle > self.accel_off_time:
                        pyautogui.keyDown('up')
                        self.accelerate = True
                        self.last_accel_toggle = current_time

                user_center = None
                track_boxes = []

                if boxes is not None and len(boxes) > 0:
                    annotated = results[0].plot()
                    cv2.imshow("Detections", annotated)
                    cv2.imwrite(os.path.join("detections", f"detection_{timestamp}.jpg"), annotated)

                    for box in boxes:
                        cls = int(box.cls[0])
                        name = self.model.names[cls]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2

                        if name == "User":
                            user_center = (center_x, (y1 + y2) // 2)
                            self.last_positions.append((user_center, current_time))
                        elif name == "Track-c6Id" and conf > 0.45:
                            track_boxes.append({'center_x': center_x, 'width': x2 - x1, 'conf': conf})

                    if track_boxes:
                        ref_x = user_center[0] if user_center else img_width // 2
                        track_boxes.sort(key=lambda t: abs(t['center_x'] - ref_x))
                        best_track = track_boxes[0]
                        print(f"Track detectado - Centro: {best_track['center_x']}, Largura: {best_track['width']}, Conf: {best_track['conf']:.2f}")
                        self.process_track_detection(best_track['center_x'], best_track['width'], img_width)
                        self.last_successful_detection = current_time
                    else:
                        print("NÃO FOI DETECTADA UMA PISTA! Resetando...")
                        if time.time() - self.last_reset_time > 5:
                            self.reset_car()

                self.check_if_player_missing()
                self.check_if_game_frozen()

                if cv2.waitKey(1) == 27:
                    break

        except KeyboardInterrupt:
            print("Interrompido pelo utilizador.")
        finally:
            pyautogui.keyUp('up')
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
            cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = KartController()
    controller.start()
