import numpy as np
import os
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow.keras as keras
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
import tensorflow as tf
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow informational messages

# Import the TensorFlow version of the EEGTransformer class.
# This assumes the class is saved in a file named 'EEGTransformer_tf.py'
# in the same directory.
from EEGTransformer_Class import EEGTransformer, PositionalEncoding

# Set a seed for reproducibility
np.random.seed(42)

###############################################################################
# # 1. GPU AND ENVIRONMENT SETUP
###############################################################################
print("--- Checking for GPU availability ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured.")
  except RuntimeError as e:
    print(e)
else:
    print("No GPU found. The model will run on the CPU.")
print("------------------------------------")

###############################################################################
# # 2. SETUP PARAMETERS
###############################################################################

# --- Data and Feature Parameters ---
FEATURES_PATH = '../../features/BETA_EEGNet_features.npz'
NUM_SUBJECTS = 70
CLASSES = 40

# --- Transformer Model Hyperparameters ---
# These should be tuned based on the feature dimensions.
NUM_HEADS = 8
# The key_dim must be divisible by the number of heads.
# The feature dimension from the extractor is 96.
KEY_DIM = 512 
FFN_INTERMEDIATE_DIM = 2048
DROPOUT_RATE = 0.25

# --- Training Parameters ---
EPOCHS = 100 
BATCH_SIZE = 64 # Can often be larger when training on smaller feature sets

###############################################################################
# # 3. DATA LOADING
# This section loads the pre-computed features and labels.
###############################################################################

print(f"\n--- Loading pre-computed features from {FEATURES_PATH}... ---")
try:
    data = np.load(FEATURES_PATH)
    X_features = data['features']
    y_labels = data['labels']
    print("Features and labels loaded successfully.")
    print(f"Features shape: {X_features.shape}")
    print(f"Labels shape: {y_labels.shape}")
except FileNotFoundError:
    print(f"ERROR: Feature file not found at '{FEATURES_PATH}'.")
    print("Please run the 'create_feature_extractor.py' script first.")
    exit()

# Create the subject indices array for cross-validation
# We know each subject has 160 trials
trials_per_subject = 160
subject_indices = np.repeat(np.arange(1, NUM_SUBJECTS + 1), trials_per_subject)

# One-hot encode the labels
y_categorical = np_utils.to_categorical(y_labels, num_classes=CLASSES)

# Get the input shape for the Transformer from the features
# Shape is (batch, sequence_length, feature_dimension)
# In our case, sequence_length is timepoints, feature_dimension is channels/filters
sequence_length = X_features.shape[1]
feature_dimension = X_features.shape[2]


###############################################################################
# # 4. CROSS-SUBJECT TRAINING OF THE TRANSFORMER
###############################################################################

logo = LeaveOneGroupOut()
all_subject_accuracies: List[float] = []
os.makedirs('../../models_transformer_stage', exist_ok=True)

print("\n############################################################")
print("  Starting EEGTransformer Training on Extracted Features  ")
print("############################################################")

for fold_num, (train_idx, test_idx) in enumerate(logo.split(X_features, y_categorical, subject_indices)):
    test_subject_id = fold_num + 1
    print(f"\n--- Fold {test_subject_id}/{NUM_SUBJECTS}: Testing on Subject {test_subject_id} ---")

    X_train, X_test = X_features[train_idx], X_features[test_idx]
    y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]

    val_split = int(0.7 * len(X_train))
    X_train_part, X_val_part = X_train[:val_split], X_train[val_split:]
    y_train_part, y_val_part = y_train[:val_split], y_train[val_split:]

    # Instantiate the EEGTransformer model
    # Note: num_channels is now the feature_dimension from the extractor,
    # and num_timepoints is the new, shorter sequence_length.
    model: Model = EEGTransformer(
        # num_channels=feature_dimension,
        # num_timepoints=sequence_length,
        output_dim=CLASSES,
        num_heads=NUM_HEADS,
        key_dim=KEY_DIM,
        ffn_intermediate_dim=FFN_INTERMEDIATE_DIM,
        dropout_rate=DROPOUT_RATE
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=False)

    checkpoint_path: str = f'../../models_transformer_stage/EEGTransformer_Test_S{test_subject_id}.h5'
    
    checkpoint: ModelCheckpoint = ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max'
    )
    
    early_stop: EarlyStopping = EarlyStopping(
        monitor='val_accuracy', patience=25, mode='max', restore_best_weights=True
    )
    
    reduce_lr: ReduceLROnPlateau = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=10, verbose=1,
        mode='max', min_lr=0.00001
    )

    print(f"Training on {len(X_train_part)} samples, validating on {len(X_val_part)} samples...")
    history = model.fit(X_train_part, y_train_part, batch_size=BATCH_SIZE, 
                            epochs=EPOCHS, verbose=1, 
                            validation_data=(X_val_part, y_val_part),
                            callbacks=[checkpoint, early_stop, reduce_lr])        
    
    print("Training stopped. Model has been restored to its best weights.")
    
    print(f"Evaluating final model on test subject {test_subject_id}...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    all_subject_accuracies.append(test_acc)

    print(f"--- Accuracy for Subject {test_subject_id}: {test_acc:.4f} ---")


###############################################################################
# # 5. FINAL RESULTS
###############################################################################

overall_mean_accuracy: float = np.mean(all_subject_accuracies)
print("\n=========================================================")
print("        FINAL TRANSFORMER STAGE CROSS-SUBJECT RESULTS        ")
print("=========================================================")
print(f"Overall Mean Accuracy Across All {NUM_SUBJECTS} Folds: {overall_mean_accuracy:.4f}")
print("=========================================================")


