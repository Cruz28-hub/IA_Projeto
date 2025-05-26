import mss
import cv2
import numpy as np
import pygetwindow as gw
import time
import os

GAME_WINDOW_TITLE = "SuperTuxKart"
OUTPUT_FOLDER = "captured_frames"


def get_game_bbox():
    try:
        # Get first matching window without index error risk
        win = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
        if win and not win[0].isMinimized:  # Removed invalid 'visible' check
            left, top, right, bottom = win[0].left, win[0].top, win[0].right, win[0].bottom
            return {"top": top, "left": left, "width": right - left, "height": bottom - top}
    except IndexError:
        return None


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

frame_count = 0
with mss.mss() as sct:
    print("Waiting for SuperTuxKart...")
    while True:
        bbox = get_game_bbox()
        if bbox:
            break
        time.sleep(1)

    print("Game window found. Capturing... Press 'q' to quit.")

    while True:
        bbox = get_game_bbox()
        if not bbox:
            print("Game window not found or minimized. Exiting.")
            break

        try:  # Added error handling for frame capture
            img = np.array(sct.grab(bbox))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"Error capturing frame: {e}")
            break

        filename = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, img)
        frame_count += 1

        cv2.imshow("Game Capture", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames to '{OUTPUT_FOLDER}'")
