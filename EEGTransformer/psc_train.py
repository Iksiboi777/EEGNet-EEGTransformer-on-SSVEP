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
# Using the larger, more powerful configuration you found to be effective.
NUM_HEADS = 8
KEY_DIM = 512 
FFN_INTERMEDIATE_DIM = 2048
DROPOUT_RATE = 0.25

# --- Training Parameters ---
EPOCHS = 150 
BATCH_SIZE = 64
# The PEAK learning rate for the scheduler.
PEAK_LEARNING_RATE = 0.0005
WARMUP_EPOCHS = 15

# --- File Paths ---
FEATURES_PATH = '../../features/BETA_EEGNet_Ensemble_Features.npz'
MODELS_OUTPUT_DIR = '../../models_transformer_stage'
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

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
# # 3. LEARNING RATE SCHEDULER WITH WARM-UP
###############################################################################

def lr_warmup_cosine_decay(global_step, warmup_steps, total_steps, peak_lr):
    """
    A learning rate scheduler that implements a linear warm-up followed by
    a cosine decay.
    """
    global_step = tf.cast(global_step, dtype=tf.float32)
    warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
    total_steps = tf.cast(total_steps, dtype=tf.float32)
    
    if global_step < warmup_steps:
        # Linear warm-up
        lr = peak_lr * (global_step / warmup_steps)
    else:
        # Cosine decay
        progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * progress))
        lr = peak_lr * cosine_decay
        
    return tf.keras.backend.get_value(lr)

###############################################################################
# # 4. CROSS-VALIDATION TRAINING LOOP
###############################################################################

logo = LeaveOneGroupOut()
all_accuracies = []

print("\n############################################################")
print("   Starting EEGTransformer Cross-Subject Training   ")
print("############################################################\n")

for fold_num, (train_val_idx, test_idx) in enumerate(logo.split(features, labels, subject_indices)):
    test_subject_id = fold_num + 1
    print(f"--- Fold {test_subject_id}/{NUM_SUBJECTS}: Testing on Subject {test_subject_id} ---")
    
    # --- 1. Create the final Test Set ---
    X_test, y_test = features[test_idx], labels[test_idx]
    
    # --- 2. Create a robust Train/Validation Split (68 subjects / 1 subject) ---
    # Take the last subject in the training group to be the validation subject.
    val_idx = train_val_idx[-TRIALS_PER_SUBJECT:]
    train_idx = train_val_idx[:-TRIALS_PER_SUBJECT]
    
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]

    print(f"Training on {len(X_train)} samples ({NUM_SUBJECTS-2} subjects)...")
    print(f"Validating on {len(X_val)} samples (1 subject)...")
    print(f"Testing on {len(X_test)} samples (1 subject)...")

    # --- 3. Model Creation and Compilation ---
    model = EEGTransformer(
        output_dim=CLASSES,
        num_heads=NUM_HEADS,
        key_dim=KEY_DIM,
        ffn_intermediate_dim=FFN_INTERMEDIATE_DIM,
        dropout_rate=DROPOUT_RATE
    )

    optimizer = keras.optimizers.Adam(clipnorm=1.0) # LR will be handled by the scheduler
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- 4. Setup Callbacks with the LR Scheduler ---
    total_steps = (len(X_train) // BATCH_SIZE) * EPOCHS
    warmup_steps = (len(X_train) // BATCH_SIZE) * WARMUP_EPOCHS

    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(
        lambda epoch: lr_warmup_cosine_decay(
            global_step=epoch * (len(X_train) // BATCH_SIZE),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=PEAK_LEARNING_RATE
        )
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_OUTPUT_DIR, f'EEGTransformer_Test_S{test_subject_id}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=35, # Give it more patience with the new scheduler
        restore_best_weights=True
    )

    # --- 5. Model Training ---
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint, early_stopping, lr_scheduler_callback],
        verbose=2
    )

    # --- 6. Final Evaluation ---
    print(f"Evaluating final model on test subject {test_subject_id}...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    all_accuracies.append(accuracy)
    print(f"--- Accuracy for Subject {test_subject_id}: {accuracy:.4f} ---")

# --- Final Results ---
print("\n############################################################")
print("          Cross-Validation Training Complete          ")
print("############################################################\n")
mean_accuracy = np.mean(all_accuracies)
print(f"Mean Cross-Validation Accuracy across all {NUM_SUBJECTS} subjects: {mean_accuracy:.4f}")
