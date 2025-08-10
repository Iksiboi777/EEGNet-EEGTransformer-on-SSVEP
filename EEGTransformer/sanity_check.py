import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import LeaveOneGroupOut
from EEGTransformer_Class import EEGTransformer

###############################################################################
# # 1. SETUP PARAMETERS
###############################################################################

# --- Data Parameters ---
CLASSES = 40
NUM_SUBJECTS = 70
TRIALS_PER_SUBJECT = 160 # 4 blocks * 40 conditions

# --- Transformer Model Hyperparameters ---
# CORRECTED: Using a much simpler model to see if it can learn at all.
# A large, complex model can be more prone to training instability.
NUM_HEADS = 8
KEY_DIM = 512  # Total attention dim = 8 * 12 = 96, matching feature dim
FFN_INTERMEDIATE_DIM = 2048
DROPOUT_RATE = 0.25

# --- Training Parameters ---
EPOCHS = 150 
BATCH_SIZE = 64
# Using a small, stable learning rate.
LEARNING_RATE = 0.0005
WARMUP_EPOCHS = 10  # Warmup steps for learning rate scheduler

# --- File Paths ---
FEATURES_PATH = '../../features/BETA_EEGNet_Ensemble_Features.npz'

###############################################################################
# # 2. DATA LOADING
###############################################################################

print(f"--- Loading pre-computed features from {FEATURES_PATH}... ---")
try:
    data = np.load(FEATURES_PATH)
    features = data['features']
    labels = data['labels']
    subject_indices = data['subject_indices']
    print("Features, labels, and subject indices loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Feature file not found at {FEATURES_PATH}.")
    exit()

###############################################################################
# # 3. SANITY CHECK: OVERFIT ON A SINGLE BATCH
# The goal is to prove the model can learn *anything* at all.
# This version uses a simpler model architecture to improve stability.
###############################################################################

print("\n############################################################")
print("        STARTING SANITY CHECK: OVERFIT ON ONE BATCH       ")
print("############################################################\n")

# Take only the first batch of data
X_sanity = features[:BATCH_SIZE]
y_sanity = labels[:BATCH_SIZE]

print(f"Sanity check data shape: {X_sanity.shape}")
print(f"Sanity check labels shape: {y_sanity.shape}")

# Create and compile the simpler model
model = EEGTransformer(
    output_dim=CLASSES,
    num_heads=NUM_HEADS,
    key_dim=KEY_DIM,
    ffn_intermediate_dim=FFN_INTERMEDIATE_DIM,
    dropout_rate=DROPOUT_RATE
)

# Using gradient clipping as a safety measure
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# To test our hypothesis about standardization, we need to modify the model class.
# Please go to the EEGTransformer_Class.py file and comment out the
# initial standardization block in the call() method like this:
#
# def call(self, inputs, training=False):
#     """
#     Defines the forward pass of the EEGTransformer model.
#     """
#     # --- 1. Input Standardization (Commented out for this test) ---
#     # mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
#     # std = tf.math.reduce_std(inputs, axis=1, keepdims=True)
#     # x = (inputs - mean) / (std + 1e-6)
#     x = inputs # Use the raw features directly

#     # ... rest of the call method
#
# After commenting out those lines, run this training script again.

# Train the model on the single batch and validate on the same batch
history = model.fit(
    X_sanity, y_sanity,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_sanity, y_sanity), # Validate on the training data
    verbose=2
)

print("\n--- Sanity Check Complete ---")
final_accuracy = history.history['val_accuracy'][-1]
print(f"Final validation accuracy on the single batch: {final_accuracy:.4f}")
if final_accuracy > 0.95:
    print("SUCCESS: The model is capable of learning and overfitting.")
else:
    print("FAILURE: The model is not learning. There may be a fundamental issue.")

