import os
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Conv3D, Reshape, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("tensorflow:", tf.__version__)

VIDEO_DIR = "F:/documents/Download/deepfake-detection-challenge/train_sample_videos"
PROCESSED_DIR = "processed_faces"
MODEL_DIR = "saved_models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

METADATA_PATH = os.path.join(VIDEO_DIR, "metadata.json")
IMG_SIZE = (224, 224)
NUM_FRAMES = 16
TUBELET_TIME = 2
PATCH_SIZE = 16
EMBED_DIM = 64
TRANSFORMER_LAYERS = 2
BATCH_SIZE = 4
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 20
FINE_TUNE_LAYERS = 20
AUTOTUNE = tf.data.AUTOTUNE
MAX_WORKERS = 4
IMG_SIZE = (160, 160)
num_patches_per_frame = (IMG_SIZE[0] // PATCH_SIZE) * (IMG_SIZE[1] // PATCH_SIZE)
NUM_TUBELETS = (NUM_FRAMES // TUBELET_TIME) * num_patches_per_frame
print("NUM_TUBELETS =", NUM_TUBELETS)

DEBUG_MODE = False
DEBUG_FRAC = 0.1

SKIP_PREPROCESSING = True

num_patches_per_frame = (IMG_SIZE[0] // PATCH_SIZE) * (IMG_SIZE[1] // PATCH_SIZE)
NUM_TUBELETS = (NUM_FRAMES // TUBELET_TIME) * num_patches_per_frame
print("NUM_TUBELETS =", NUM_TUBELETS)
print(f"DEBUG MODE: {DEBUG_MODE}, DEBUG_FRAC: {DEBUG_FRAC}")

with open(METADATA_PATH) as f:
    metadata = json.load(f)
df = pd.DataFrame([{'filename': k, 'label': 1 if v.get('label') == 'FAKE' else 0} for k, v in metadata.items()])

n_samples = min(len(df[df['label']==0]), len(df[df['label']==1]))
df_balanced = pd.concat([
    df[df['label']==0].sample(n_samples, random_state=42),
    df[df['label']==1].sample(n_samples, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

if DEBUG_MODE:
    df_balanced = df_balanced.sample(frac=DEBUG_FRAC, random_state=42).reset_index(drop=True)
    print(f"DEBUG: Using {len(df_balanced)} samples ({DEBUG_FRAC*100}% of data)")

print(f"Total data: {len(df_balanced)}")
print(f"Real: {len(df_balanced[df_balanced['label']==0])}")
print(f"Fake: {len(df_balanced[df_balanced['label']==1])}")

train_df = df_balanced.sample(frac=0.8, random_state=42).reset_index(drop=True)
val_df = df_balanced.drop(train_df.index).reset_index(drop=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}")

def validate_dataset(X, y, df):
    print("\nDATA VALIDATION")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X value range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"X mean: {X.mean():.3f}")
    
    unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    return X, y

def load_cached_data_only():
    all_npy_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.npy')]
    
    X = []
    filenames = []
    valid_labels = []
    
    for npy_file in all_npy_files:
        try:
            video_name_without_ext = npy_file.replace('.npy', '')
            possible_names = [
                video_name_without_ext + '.mp4',
                video_name_without_ext + '.avi', 
                video_name_without_ext + '.mov',
                video_name_without_ext, 
                npy_file  
            ]
            
            matched_name = None
            for name in possible_names:
                if name in metadata:
                    matched_name = name
                    break
            
            if matched_name is None:
                print(f"⚠️ File {npy_file} not found in metadata, skipping")
                continue
                
            faces = np.load(os.path.join(PROCESSED_DIR, npy_file))
            X.append(faces)
            filenames.append(matched_name)
            
            label = 1 if metadata[matched_name].get('label') == 'FAKE' else 0
            valid_labels.append(label)
            
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
    
    X = np.array(X)
    y_cached = tf.keras.utils.to_categorical(valid_labels, 2)
    
    print(f" Loaded {len(X)} files from cache")
    print(f" Matches in metadata: {len(filenames)}")
    return X, filenames, y_cached

X_cached, cached_filenames, y_cached = load_cached_data_only()

print("[INFO] Loading data from cache...")
X_all = X_cached
y_all = y_cached

X_all, y_all = validate_dataset(X_all, y_all, df_balanced)

train_indices = []
val_indices = []

for i, filename in enumerate(cached_filenames):
    if filename in train_df['filename'].values:
        train_indices.append(i)
    elif filename in val_df['filename'].values:
        val_indices.append(i)

print(f"Found train samples: {len(train_indices)}")
print(f"Found val samples: {len(val_indices)}")

if len(train_indices) == 0 or len(val_indices) == 0:
    print("Could not match with original split, creating new one...")
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(X_all)), 
        test_size=0.2, 
        random_state=42,
        stratify=y_all
    )

X_train = X_all[train_indices]
y_train = y_all[train_indices]
X_val = X_all[val_indices] 
y_val = y_all[val_indices]

def preprocess_frames(frames):
    resized = np.array([cv2.resize(f, (160,160)) for f in frames])
    return resized

X_train = np.array([preprocess_frames(f) for f in X_train])
X_val   = np.array([preprocess_frames(f) for f in X_val])

print(f"Data split: Train {len(X_train)} samples, Val {len(X_val)} samples")
print("[INFO] Data successfully loaded from cache!")

def create_dataset(X, y, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

train_ds = create_dataset(X_train, y_train)
val_ds = create_dataset(X_val, y_val, shuffle=False)

print(f"Created datasets: Train {len(list(train_ds))} batches, Val {len(list(val_ds))} batches")

class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_tubelets, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_tubelets = num_tubelets
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(1, self.num_tubelets, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, x):
        return x + self.pos_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_tubelets": self.num_tubelets,
            "embed_dim": self.embed_dim
        })
        return config

def create_vivit_model():
    inputs = Input(shape=(NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3))

    x = Conv3D(
        EMBED_DIM,
        kernel_size=(TUBELET_TIME, PATCH_SIZE, PATCH_SIZE),
        strides=(TUBELET_TIME, PATCH_SIZE, PATCH_SIZE),
        padding='valid'
    )(inputs)

    x = Reshape((NUM_TUBELETS, EMBED_DIM))(x)

    x = AddPositionEmbedding(num_tubelets=NUM_TUBELETS, embed_dim=EMBED_DIM)(x)
    x = Dropout(0.1)(x)

    for i in range(TRANSFORMER_LAYERS):
        x1 = LayerNormalization()(x)
        attn_output = MultiHeadAttention(
            num_heads=8, 
            key_dim=EMBED_DIM // 8,
            dropout=0.1
        )(x1, x1)
        x = Add()([x, attn_output])
        
        x2 = LayerNormalization()(x)
        ff = Dense(EMBED_DIM * 2, activation='gelu')(x2)
        ff = Dropout(0.1)(ff)
        ff = Dense(EMBED_DIM)(ff)
        ff = Dropout(0.1)(ff)
        x = Add()([x, ff])

    x = GlobalAveragePooling1D()(x)
    x = LayerNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs, name="ViViT_DeepFake")
    return model

print("[INFO] Creating model...")
model = create_vivit_model()

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

if DEBUG_MODE:
    EPOCHS_STAGE1 = 3
    EPOCHS_STAGE2 = 3
    print(f"DEBUG: Reduced epochs - Stage1: {EPOCHS_STAGE1}, Stage2: {EPOCHS_STAGE2}")

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=3 if DEBUG_MODE else 10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "vivit_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("[INFO] Stage 1: Training classifier...")

for layer in model.layers[:-8]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Starting Stage 1 training...")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks,
    verbose=1
)

print("[INFO] Stage 2: Fine-tuning...")

for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Starting Stage 2 training...")
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks,
    verbose=1
)

print("[INFO] Saving model...")
model.save(os.path.join(MODEL_DIR, "vivit_final.keras"))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'] + history2.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'] + history2.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
plt.show()

print("[INFO] Evaluating model...")
train_loss, train_acc = model.evaluate(train_ds, verbose=0)
val_loss, val_acc = model.evaluate(val_ds, verbose=0)

print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Training Loss: {train_loss:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")

print("[INFO] Training completed!")

def create_baseline_model(input_shape=(NUM_FRAMES, *IMG_SIZE, 3)):
    inputs = Input(shape=input_shape)
    x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(inputs)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    return Model(inputs, outputs, name="Baseline_CNN")

baseline = create_baseline_model()
baseline.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
baseline.fit(train_ds, validation_data=val_ds, epochs=20)
baseline.save(os.path.join(MODEL_DIR, "baseline_final.keras"))