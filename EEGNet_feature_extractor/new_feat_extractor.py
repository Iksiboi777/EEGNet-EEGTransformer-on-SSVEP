# train_feature_extractor.py
import numpy as np
import os
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import the specialized EEGNet_SSVEP model.
from EEGModels import EEGNet_SSVEP

# Import typing for explicit type hints
from typing import Tuple, List, Dict, Any

# Set a seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
# # 2. CONFIGURATION AND PARAMETERS
###############################################################################

# --- File Paths ---
# Path to the directory containing the BETA dataset .mat files
DATA_PATH = '../../Data'
# Directory where the final, trained feature extractor model will be saved
OUTPUT_DIR = '../new_feature_extractor'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Data Parameters ---
NUM_SUBJECTS = 70
SAMPLING_RATE = 250  # Hz
MAX_TRIAL_SECS = 4.0

# --- EEGNet_SSVEP Model Parameters (justified by Waytowich et al., 2018) ---
CHANS = 64
CLASSES = 40
SAMPLES = int(MAX_TRIAL_SECS * SAMPLING_RATE)
KERN_LENGTH = 250
F1 = 96  # Number of temporal filters
D = 1    # Number of spatial filters per temporal filter
F2 = 96  # Number of pointwise filters

# --- Training Parameters ---
EPOCHS = 100 # Increased epochs slightly, as EarlyStopping will find the best one
BATCH_SIZE = 32

# --- NEW: Subject Partitioning ---
# Here we define which subjects to use for training the single feature extractor
# and which to use for validating it. The rest are implicitly the test set.
# This split ensures no data leakage.

# Subjects 1-60 will be used to train the feature extractor
TRAIN_SUBJECT_IDS = list(range(1, 61))
# Subjects 61-65 will be used to validate the training process
VALIDATION_SUBJECT_IDS = list(range(61, 66))
# Subjects 66-70 are the final, unseen test set for the entire MTGNet pipeline
TEST_SUBJECT_IDS = list(range(66, 71))


###############################################################################
# # 3. DATA LOADING FUNCTION FOR SPECIFIC PARTITIONS
###############################################################################

def load_partitioned_data(subject_ids: List[int], data_path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and prepares the BETA dataset for a specific list of subject IDs.

    Args:
        subject_ids (List[int]): A list of subject numbers to load data for.
        data_path (str): Path to the directory containing the BETA dataset.

    Returns:
        A tuple of (X, y) for the specified data partition.
    """
    print(f"\n--- Loading data for subjects: {subject_ids[0]} to {subject_ids[-1]} ---")

    all_X = []
    all_y = []

    for subject_id in subject_ids:
        print(f"Loading S{subject_id}...", end='\r')
        file_path: str = os.path.join(data_path, f'S{subject_id}.mat')
        mat_data: Dict[str, Any] = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        eeg_data: np.ndarray = mat_data['data'].EEG
        # Transpose to (blocks, conditions, channels, timepoints)
        eeg_data: np.ndarray = np.transpose(eeg_data, (2, 3, 0, 1))

        num_blocks: int = eeg_data.shape[0]
        num_conditions: int = eeg_data.shape[1]

        # Reshape to (trials, channels, timepoints)
        X: np.ndarray = eeg_data.reshape(-1, CHANS, eeg_data.shape[-1])

        # Pad trials shorter than MAX_TRIAL_SECS (4s)
        if X.shape[2] < SAMPLES:
            padded_X: np.ndarray = np.zeros((X.shape[0], CHANS, SAMPLES))
            padded_X[:, :, :X.shape[2]] = X
            X = padded_X

        # Generate labels
        y: np.ndarray = np.array([i for i in range(num_conditions)] * num_blocks)

        all_X.append(X)
        all_y.append(y)

    # Concatenate lists of arrays into single large numpy arrays
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)

    print(f"\nFinished loading partition. Final shape: {X_final.shape}")

    return X_final, y_final


###############################################################################
# # 4. MAIN SCRIPT: TRAINING THE GENERALIZED FEATURE EXTRACTOR
###############################################################################

if __name__ == "__main__":

    print("\n######################################################################")
    print("## Training a Single, Generalized EEGNet Feature Extractor ##")
    print("######################################################################")

    # --- 1. Load Training and Validation Data ---
    X_train, y_train = load_partitioned_data(TRAIN_SUBJECT_IDS)
    X_val, y_val = load_partitioned_data(VALIDATION_SUBJECT_IDS)

    # --- 2. Reshape and Categorize Data for the Model ---
    X_train = X_train.reshape(X_train.shape[0], CHANS, SAMPLES, 1)
    y_train = np_utils.to_categorical(y_train, num_classes=CLASSES)

    X_val = X_val.reshape(X_val.shape[0], CHANS, SAMPLES, 1)
    y_val = np_utils.to_categorical(y_val, num_classes=CLASSES)
    
    # --- 3. Instantiate and Compile the EEGNet_SSVEP Model ---
    model: Model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                                dropoutRate=0.5, kernLength=KERN_LENGTH,
                                F1=F1, D=D, F2=F2)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- 4. Define Callbacks for Robust Training ---
    # Define the path for the final model file
    model_path: str = os.path.join(OUTPUT_DIR, 'BETA_EEGNet_Feature_Extractor.h5')
    
    # ModelCheckpoint saves the best model based on validation accuracy
    checkpoint: ModelCheckpoint = ModelCheckpoint(
        filepath=model_path, monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max'
    )
    
    # EarlyStopping halts training if validation accuracy plateaus
    early_stop: EarlyStopping = EarlyStopping(
        monitor='val_accuracy', patience=25, verbose=1, mode='max',
        restore_best_weights=True
    )
    
    # ReduceLROnPlateau lowers the learning rate on a plateau
    reduce_lr: ReduceLROnPlateau = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=10, verbose=1,
        mode='max', min_lr=0.00001
    )

    # --- 5. Train the Model ---
    print("\n--- Starting Training ---")
    print(f"Training on data from {len(TRAIN_SUBJECT_IDS)} subjects.")
    print(f"Validating on data from {len(VALIDATION_SUBJECT_IDS)} subjects.")
    
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint, early_stop, reduce_lr])

    # --- 6. Conclusion ---
    print("\n=========================================================")
    print("      FEATURE EXTRACTOR TRAINING COMPLETE      ")
    print("=========================================================")
    print(f"The best model has been saved to: {model_path}")
    print("\nYou can now use this single model to extract a consistent set of features")
    print("for all subjects to use as input for your Transformer and GNN models.")
    print("=========================================================")