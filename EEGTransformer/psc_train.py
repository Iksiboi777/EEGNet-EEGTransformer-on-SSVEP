# train_transformer_large_model_stable_lr.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
from tensorflow import keras
# Ensure you are using the corrected EEGTransformer_Class.py
from EEGTransformer_Class import EEGTransformer 

###############################################################################
# # 1. SETUP PARAMETERS
###############################################################################

# --- Data Parameters ---
CLASSES = 40

# --- Subject Partitioning (60/5/5 Split) ---
TRAIN_SUBJECT_IDS = list(range(1, 61))
VALIDATION_SUBJECT_IDS = list(range(61, 66))
TEST_SUBJECT_IDS = list(range(66, 71))

# --- Transformer Model Hyperparameters (High Capacity) ---
# Restoring the powerful configuration that showed learning potential.
NUM_HEADS = 8
KEY_DIM = 512 
FFN_INTERMEDIATE_DIM = 2048
DROPOUT_RATE = 0.3 # Keeping high dropout for regularization

# --- Training Parameters ---
EPOCHS = 250 # Increased epochs for slow, stable learning
BATCH_SIZE = 64
# Using a single, small, fixed learning rate for maximum stability.
LEARNING_RATE = 0.0001 

# --- File Paths ---
FEATURES_PATH = '../../features/BETA_AllSubjects_EEGNet_Block2_Features.npz'
MODELS_OUTPUT_DIR = '../../models_transformer'
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

###############################################################################
# # 2. DATA LOADING AND SPLITTING
###############################################################################

print(f"--- Loading Block 2 features from {FEATURES_PATH}... ---")
data = np.load(FEATURES_PATH)
features = data['features']
labels = data['labels']
subject_indices = data['subject_indices']
print(f"Full dataset shape: {features.shape}")

def get_data_for_subjects(subject_ids):
    """Filters the dataset to include only data from the specified subjects."""
    indices = np.isin(subject_indices, subject_ids)
    return features[indices], labels[indices]

X_train, y_train = get_data_for_subjects(TRAIN_SUBJECT_IDS)
X_val, y_val = get_data_for_subjects(VALIDATION_SUBJECT_IDS)
X_test, y_test = get_data_for_subjects(TEST_SUBJECT_IDS)

print(f"Training set: {X_train.shape[0]} samples.")
print(f"Validation set: {X_val.shape[0]} samples.")
print(f"Test set: {X_test.shape[0]} samples.")

###############################################################################
# # 3. MODEL TRAINING
###############################################################################

print("\n############################################################")
print("   Training High-Capacity Transformer with Stable Fixed LR   ")
print("############################################################\n")

# --- 1. Create Model ---
model = EEGTransformer(
    output_dim=CLASSES,
    num_heads=NUM_HEADS,
    key_dim=KEY_DIM,
    ffn_intermediate_dim=FFN_INTERMEDIATE_DIM,
    dropout_rate=DROPOUT_RATE
)
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 2. Set up Callbacks ---
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODELS_OUTPUT_DIR, 'EEGTransformer_Large_Stable_Best.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=60, # Increased patience for slow convergence
    restore_best_weights=True
)

# --- 3. Train the model ---
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint, early_stopping],
    verbose=2
)

# --- 4. Final Evaluation on the Unseen Test Set ---
print("\n--- Training complete. Evaluating final model on the unseen test set... ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n############################################################")
print("           FINAL MODEL PERFORMANCE          ")
print("############################################################\n")
print(f"Final Test Accuracy on subjects {TEST_SUBJECT_IDS}: {accuracy:.4f}")
print("\n############################################################")