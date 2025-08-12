# sanity_check_lstm.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- File Paths and Parameters ---
FEATURES_PATH = '../../features/BETA_AllSubjects_EEGNet_Block2_Features.npz'
BATCH_SIZE = 64
CLASSES = 40
EPOCHS = 100

# --- 1. DATA LOADING ---
print(f"--- Loading features from {FEATURES_PATH}... ---")
data = np.load(FEATURES_PATH)
features = data['features']
labels = data['labels']
print("Features and labels loaded.")

# --- 2. TAKE A SINGLE BATCH ---
X_sanity = features[:BATCH_SIZE]
y_sanity = labels[:BATCH_SIZE]
print(f"Sanity check data shape: {X_sanity.shape}")

# --- 3. CREATE A MINIMAL LSTM MODEL ---
print("\n--- Building a simple LSTM model for the ultimate sanity check ---")
lstm_model = keras.Sequential([
    keras.layers.Input(shape=(X_sanity.shape[1], X_sanity.shape[2])), # (250, 96)
    # A single LSTM layer to see if any temporal pattern can be learned
    keras.layers.LSTM(64),
    # A dense layer to classify the output
    keras.layers.Dense(CLASSES)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
lstm_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
lstm_model.summary()

# --- 4. TRAIN THE LSTM MODEL ---
print("\n--- Training LSTM on a single batch ---")
history = lstm_model.fit(
    X_sanity, y_sanity,
    epochs=EPOCHS,
    validation_data=(X_sanity, y_sanity),
    verbose=2
)

# --- 5. CHECK THE RESULT ---
print("\n--- Sanity Check Complete ---")
final_accuracy = history.history['val_accuracy'][-1]
print(f"Final validation accuracy of LSTM on the single batch: {final_accuracy:.4f}")
if final_accuracy > 0.95:
    print("SUCCESS: The features are learnable! The problem is in the Transformer implementation.")
else:
    print("FAILURE: The features themselves may be the problem. Check the feature extraction process.")