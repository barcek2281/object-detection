import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("simple_cnn.h5", compile=False)
model.summary()

IMG_SIZE = (128, 128)
LABELS = ["face", "robots", "animal", "items", "fake_faces", "anime_faces"] 

def classify_frame(frame) -> tuple[str, int]:
    # Resize and preprocess frame for MobileNetV2
    img = cv2.resize(frame, IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    # Predict
    preds = model.predict(img, verbose=0)
    class_idx = np.argmax(preds[0])
    class_name = LABELS[class_idx]
    confidence = preds[0][class_idx]
    return class_name, confidence
