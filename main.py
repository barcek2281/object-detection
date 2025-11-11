"""
Tongue Detection Meme Display
A MediaPipe + OpenCV application that detects when your tongue is out
and displays different meme images accordingly.

See TUTORIAL.md for detailed explanations
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from model import classify_frame

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Window settings - approximately half monitor size (1920x1080 / 2)
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

CAMERA_WINDOW = "Camera Input"
MEME_WINDOW = "Meme Output"


# Initialize MediaPipe Face Mesh
# This creates a face detection model that tracks 468 facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,  # Confidence threshold for initial detection
    min_tracking_confidence=0.5,   # Confidence threshold for tracking
    max_num_faces=1                # We only need to track one face
)


def main():    
    # ========================================================================
    # STEP 1: Load and prepare meme images
    # ========================================================================
    
    print("Tongue Detection Meme Display")
    print("=" * 60)
    
    
    # Load images using OpenCV (images are loaded in BGR format)
    apple_img = cv2.imread('img/lebron.png')
    appletongue_img = cv2.imread('img/cat.png')
    robocop_img = cv2.imread('img/robocop.png')
    kake_img = cv2.imread('img/kake.jfif')
    
    
    
    
    # Verify images loaded successfully
    # if apple_img is None or appletongue_img is None:
    #     print("\n[ERROR] Could not load meme images.")
    #     exit(1)

    
    print("[OK] Meme images loaded successfully!")
    
    # Resize images to fit the output window
    # This ensures consistent display regardless of original image size
    apple_img = cv2.resize(apple_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    appletongue_img = cv2.resize(appletongue_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    robocop_img = cv2.resize(robocop_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    kake_img = cv2.resize(kake_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    
    
    d = {
        "face": apple_img, 
        "robots": robocop_img,
        "animal": apple_img,
        "items": apple_img,
        "fake_faces": appletongue_img,
        "anime_faces": apple_img 
        }
    
    # ========================================================================
    # STEP 2: Initialize webcam
    # ========================================================================
    
    # Open the default webcam (index 0)
    # If you have multiple cameras, try changing 0 to 1, 2, etc.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n[ERROR] Could not open webcam.")
        exit(1)
    
    # Set webcam resolution (may not match exactly depending on hardware)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    
    print("[OK] Webcam initialized successfully!")
    
    # ========================================================================
    # STEP 3: Create display windows
    # ========================================================================
    
    # Create two windows: one for camera input, one for meme output
    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MEME_WINDOW, cv2.WINDOW_NORMAL)
    
    # Set window sizes
    cv2.resizeWindow(CAMERA_WINDOW, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow(MEME_WINDOW, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    print("\n" + "=" * 60)
    print("[OK] Application started successfully!")
    print("=" * 60)
    print("\n[CAMERA] Windows opened")
    print("[TONGUE] Stick your tongue out to change the meme!")
    print("[QUIT] Press 'q' to quit\n")
    
    # Default state - start with normal apple image
    current_meme = apple_img.copy()
    
    # ========================================================================
    # STEP 4: Main detection loop
    # ========================================================================
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Check if frame was captured successfully
        if not ret:
            print("\n[ERROR] Could not read frame from webcam.")
            break
        
        # Flip frame horizontally for mirror effect (makes it easier to use)
        # Without this, moving left would make the image move right
        frame = cv2.flip(frame, 1)
        
        # Ensure frame matches our target window size
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Process the frame with MediaPipe Face Mesh
        
        object_detected, object_detected_pro = classify_frame(frame)
        
        cv2.putText(frame, object_detected, (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, str(round(object_detected_pro * 100, 2)) + "%", (10, 80), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        
        # Show camera feed with detection status
        cv2.imshow(CAMERA_WINDOW, frame)
        
        # Show current meme image
        cv2.imshow(MEME_WINDOW, d[object_detected].copy())
        
        # Wait 1ms for key press, check if 'q' was pressed
        # The & 0xFF is needed for compatibility with some systems
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[QUIT] Quitting application...")
            break
        
    # ========================================================================
    # STEP 5: Cleanup and exit
    # ========================================================================
    
    # Release webcam
    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    # Close MediaPipe Face Mesh
    face_mesh.close()
    
    print("[OK] Application closed successfully.")
    print("Thanks for using Tongue Detection Meme Display!\n")

if __name__ == "__main__":
    main()

