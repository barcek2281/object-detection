"""
Simplified Face Authenticity Detection UI
- Detects faces using MediaPipe
- Classifies as 'real' or 'fake' via classify_frame()
- Displays camera feed with bounding box around detected face
- Shows probability, FPS, and simple interface

Run:
    pip install opencv-python mediapipe numpy
    python face_auth_ui.py
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from model import classify_frame

# ============================================================================
# CONFIGURATION
# ============================================================================
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
CAMERA_WINDOW = "Face Authenticity Detection"
FPS_SMOOTHING = 0.9

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


def main():
    print("Starting simplified Face Authenticity Detector...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WINDOW, WINDOW_WIDTH, WINDOW_HEIGHT)

    last_time = time.time()
    ema_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        # FPS calculation
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        ema_fps = FPS_SMOOTHING * ema_fps + (1 - FPS_SMOOTHING) * fps

        # Face detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        # Classification (real/fake)
        label, prob = classify_frame(frame)
        prob_text = f"{label.upper()} ({prob*100:.1f}%)"

        # Draw results
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, prob_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, prob_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Draw FPS and instructions
        cv2.putText(frame, f"FPS: {int(ema_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, WINDOW_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # Show camera window
        cv2.imshow(CAMERA_WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()
