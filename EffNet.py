
import os
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor, as_completed

VIDEO_DIR = "F:/documents/Download/deepfake-detection-challenge/train_sample_videos"
PROCESSED_DIR = "processed_faces"
MODEL_DIR = "saved_models"

# VIDEO_DIR = "/app/videos"
# PROCESSED_DIR = "/app/processed_faces"
# MODEL_DIR = "/app/saved_models"

MODEL_NAME = "ViViT_DeepFake"
METADATA_PATH = os.path.join(VIDEO_DIR, "metadata.json")
IMG_SIZE = (224, 224)

NUM_FRAMES = 5
TUBELET_TIME = 2
PATCH_SIZE = 16
NUM_PATCHES_PER_FRAME = (IMG_SIZE[0] // PATCH_SIZE) * (IMG_SIZE[1] // PATCH_SIZE)
NUM_TUBELETS = (NUM_FRAMES // TUBELET_TIME) * NUM_PATCHES_PER_FRAME

BATCH_SIZE = 8
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15
FINE_TUNE_LAYERS = 50
AUTOTUNE = tf.data.AUTOTUNE
DEBUG_MODE = True
DEBUG_FRAC = 0.1
MAX_WORKERS = 8


# checking

NUM_FRAMES = 16
TUBELET_TIME = 2
PATCH_SIZE = 16
EMBED_DIM = 256
TRANSFORMER_LAYERS = 4
BATCH_SIZE = 8
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15
FINE_TUNE_LAYERS = 30

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

with open(METADATA_PATH) as f:
    metadata = json.load(f)
df = pd.DataFrame([{'filename': k, 'label': 1 if v.get('label') == 'FAKE' else 0} for k, v in metadata.items()])

n_samples = min(len(df[df['label']==0]), len(df[df['label']==1]))
df_balanced = pd.concat([df[df['label']==0].sample(n_samples, random_state=42),
                         df[df['label']==1].sample(n_samples, random_state=42)]).sample(frac=1, random_state=42).reset_index(drop=True)

if DEBUG_MODE:
    df_balanced = df_balanced.sample(frac=DEBUG_FRAC, random_state=42).reset_index(drop=True)

train_df = df_balanced.sample(frac=0.8, random_state=42).reset_index(drop=True)
val_df = df_balanced.drop(train_df.index).reset_index(drop=True)

detector = MTCNN()

def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector.detect_faces(rgb)
    if dets:
        x, y, w, h = dets[0]['box']
        x, y = max(0, x), max(0, y)
        face = cv2.resize(rgb[y:y+h, x:x+w], IMG_SIZE)
    else:
        face = np.zeros(IMG_SIZE + (3,), dtype=np.uint8)
    # лёгкая аугментация
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1)
        face = cv2.warpAffine(face, M, IMG_SIZE)
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        face = np.clip(face * factor, 0, 1)
    return face.astype(np.float32)/255.0

def extract_faces(video_path):
    filename = os.path.basename(video_path)
    npy_path = os.path.join(PROCESSED_DIR, os.path.splitext(filename)[0] + ".npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        if arr.shape == (NUM_FRAMES, *IMG_SIZE, 3):
            return arr
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total-1, 0), NUM_FRAMES, dtype=int)
    faces = []
    last_face = np.zeros((*IMG_SIZE,3), dtype=np.float32)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            faces.append(last_face)
            continue
        face = process_frame(frame)
        if np.sum(face) > 0:
            last_face = face
        faces.append(last_face)
    cap.release()
    faces = np.array(faces, dtype=np.float32)
    np.save(npy_path, faces)
    return faces

def prepare_dataset_parallel(df, max_workers=MAX_WORKERS):
    X = [None] * len(df)
    y = [None] * len(df)

    def process_video(idx, row):
        video_path = os.path.join(VIDEO_DIR, row['filename'])
        faces = extract_faces(video_path)
        return idx, faces, row['label']

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_video, idx, row) for idx, row in df.iterrows()]
        for future in as_completed(futures):
            idx, faces, label = future.result()
            X[idx] = faces
            y[idx] = label

    X = np.array(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    return X, y

print("[INFO] Подготовка тренировочных данных (многопоточно)...")
X_train, y_train = prepare_dataset_parallel(train_df, max_workers=MAX_WORKERS)

print("[INFO] Подготовка валидационных данных (многопоточно)...")
X_val, y_val = prepare_dataset_parallel(val_df, max_workers=MAX_WORKERS)

def create_dataset(X, y, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

train_ds = create_dataset(X_train, y_train)
val_ds = create_dataset(X_val, y_val, shuffle=False)

print(f"[INFO] train_ds: {len(X_train)} samples, val_ds: {len(X_val)} samples")

class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_tubelets, **kwargs):
        super().__init__(**kwargs)
        self.num_tubelets = num_tubelets
        
    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(1, self.num_tubelets, EMBED_DIM), 
            initializer='random_normal', 
            trainable=True,
            dtype=tf.float32
)

        
    def call(self, x): 
        return x + self.pos_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_tubelets": self.num_tubelets,
        })
        return config

def create_vivit():
    inputs = Input(shape=(NUM_FRAMES, *IMG_SIZE, 3))
    x = layers.Conv3D(
        EMBED_DIM,
        (TUBELET_TIME, PATCH_SIZE, PATCH_SIZE),
        strides=(TUBELET_TIME, PATCH_SIZE, PATCH_SIZE),
        padding='valid'
    )(inputs)
    x = layers.Reshape((NUM_TUBELETS, EMBED_DIM))(x)
    x = AddPositionEmbedding(num_tubelets=NUM_TUBELETS)(x)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = LayerNormalization()(x)
        attn = MultiHeadAttention(num_heads=8, key_dim=EMBED_DIM//8)(x1, x1)
        x = layers.Add()([x, attn])
        x2 = LayerNormalization()(x)
        ff = Dense(EMBED_DIM*3, activation='gelu')(x2)
        ff = Dropout(0.1)(ff)
        ff = Dense(EMBED_DIM)(ff)
        x = layers.Add()([x, ff])

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax')(x)
    return Model(inputs, outputs)

model_keras_path = os.path.join(MODEL_DIR, MODEL_NAME + ".keras")

custom_objects = {'AddPositionEmbedding': AddPositionEmbedding}

print("[INFO] Создание новой модели...")
model = create_vivit()

model.summary()

print("[INFO] Stage 1: Заморозка слоев...")
for layer in model.layers[:-6]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Начало обучения Stage 1...")
history1 = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS_STAGE1,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(model_keras_path, save_best_only=True, monitor='val_accuracy', verbose=1)
    ]
)

print("Stage 1 завершён!")

print("[INFO] Stage 2: Fine-tuning...")
for layer in model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in model.layers[-FINE_TUNE_LAYERS:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("[INFO] Начало обучения Stage 2...")
history2 = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS_STAGE2,
    callbacks=[
        EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(model_keras_path, save_best_only=True, monitor='val_accuracy')
    ]
)

print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")

model.save(model_keras_path)
print(f"Модель сохранена: {model_keras_path}")