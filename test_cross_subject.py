import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
# The user's script used these imports, which are fine. 
# tensorflow.keras is the modern standard, but this works.
import tensorflow.keras as keras
from keras import utils as np_utils
# Import ModelCheckpoint, EarlyStopping, and the new ReduceLROnPlateau callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
import tensorflow as tf

# Import typing for explicit type hints
from typing import Tuple, List, Dict, Any

# Import the specialized EEGNet_SSVEP model.
from EEGModels import EEGNet_SSVEP

# Set a seed for reproducibility
np.random.seed(42)

###############################################################################
# # 1. GPU AND ENVIRONMENT SETUP
# This section checks for GPU availability and sets memory growth to prevent
# TensorFlow from allocating all GPU memory at once.
###############################################################################
print("--- Checking for GPU availability ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured.")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
else:
    print("No GPU found. The model will run on the CPU.")
print("------------------------------------")


###############################################################################
# # 2. SETUP PARAMETERS
###############################################################################

# Set the path to the directory containing the BETA dataset
DATA_PATH = '../Data' 

# --- Data Parameters ---
NUM_SUBJECTS = 70
SAMPLING_RATE = 250  # Hz
MAX_TRIAL_SECS = 4.0

# --- EEGNet_SSVEP Model Parameters ---
CHANS = 64
CLASSES = 40
SAMPLES = int(MAX_TRIAL_SECS * SAMPLING_RATE)
KERN_LENGTH = 250

# --- PARAMETERS FOR CROSS-SUBJECT CLASSIFICATION ---
# Reverting to the larger, more powerful model parameters from the Waytowich paper,
# as the model will be trained on a much larger dataset (69 subjects).
F1 = 96 # Number of temporal filters
D = 1   # Number of spatial filters per temporal filter
F2 = 96 # Number of pointwise filters

# --- Training Parameters ---
EPOCHS = 50 
BATCH_SIZE = 32

###############################################################################
# # 3. DATA LOADING AND PREPARATION FUNCTION
# This function is now designed to load data for ALL subjects at once.
###############################################################################

def load_all_beta_data(data_path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and prepares the BETA dataset for all subjects.
    
    Args:
        data_path (str): Path to the directory containing the BETA dataset.
        
    Returns:
        A tuple of (X_all, y_all, subject_indices) for cross-subject validation.
    """
    print("\n--- Loading data for ALL subjects... ---")
    
    all_X = []
    all_y = []
    all_subject_indices = []

    for subject_id in range(1, NUM_SUBJECTS + 1):
        print(f"Loading S{subject_id}...", end='\r')
        file_path: str = os.path.join(data_path, f'S{subject_id}.mat')
        mat_data: Dict[str, Any] = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        eeg_data: np.ndarray = mat_data['data'].EEG
        eeg_data: np.ndarray = np.transpose(eeg_data, (2, 3, 0, 1))

        num_blocks: int = eeg_data.shape[0]
        num_conditions: int = eeg_data.shape[1]
        
        X: np.ndarray = eeg_data.reshape(-1, CHANS, eeg_data.shape[-1])

        if X.shape[2] < SAMPLES:
            padded_X: np.ndarray = np.zeros((X.shape[0], CHANS, SAMPLES))
            padded_X[:, :, :X.shape[2]] = X
            X = padded_X

        y: np.ndarray = np.array([i for i in range(num_conditions)] * num_blocks)    
        
        all_X.append(X)
        all_y.append(y)
        # Create an array that marks every trial with the current subject's ID
        all_subject_indices.append(np.full(X.shape[0], subject_id))
    
    # Concatenate lists of arrays into single large numpy arrays
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subject_indices = np.concatenate(all_subject_indices, axis=0)

    print("\nFinished loading all subject data.")
    print(f"Total data shape: {X_all.shape}, Total labels shape: {y_all.shape}")

    return X_all, y_all, subject_indices
 

###############################################################################
# # 4. CROSS-SUBJECT TRAINING FUNCTION
###############################################################################

# Load all data into memory
X_all, y_all, subject_indices = load_all_beta_data()

# Reshape data and labels for the model
X_all = X_all.reshape(X_all.shape[0], CHANS, SAMPLES, 1)
y_all_categorical = np_utils.to_categorical(y_all, num_classes=CLASSES)

# Use scikit-learn's LeaveOneGroupOut for Leave-One-Subject-Out cross-validation
logo = LeaveOneGroupOut()

all_subject_accuracies: List[float] = []
os.makedirs('models_cross_subject', exist_ok=True)

print("\n############################################################")
print("Starting EEGNet_SSVEP CROSS-SUBJECT Training and Evaluation")
print("############################################################")

# The 'groups' parameter for logo.split is our subject_indices array
for fold_num, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all_categorical, subject_indices)):
    test_subject_id = fold_num + 1
    print(f"\n--- Fold {test_subject_id}/{NUM_SUBJECTS}: Testing on Subject {test_subject_id} ---")

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all_categorical[train_idx], y_all_categorical[test_idx]

    # Use a small subset of the training data as a validation set for callbacks
    # This is crucial for EarlyStopping and ModelCheckpoint to work correctly
    val_split = int(0.9 * len(X_train))
    X_train_part, X_val_part = X_train[:val_split], X_train[val_split:]
    y_train_part, y_val_part = y_train[:val_split], y_train[val_split:]

    model: Model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                         dropoutRate=0.5, kernLength=KERN_LENGTH,
                         F1=F1, D=D, F2=F2)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path: str = f'../models_cross_subject/EEGNet_SSVEP_Test_S{test_subject_id}.h5'
    
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
    
    # Evaluate the final model on the held-out test subject
    print(f"Evaluating final model on test subject {test_subject_id}...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    all_subject_accuracies.append(test_acc)

    print(f"--- Accuracy for Subject {test_subject_id}: {test_acc:.4f} ---")

###############################################################################
# # 5. FINAL RESULTS
###############################################################################

overall_mean_accuracy: float = np.mean(all_subject_accuracies)
print("\n=========================================================")
print("              FINAL CROSS-SUBJECT RESULTS              ")
print("=========================================================")
print(f"Overall Mean Classification Accuracy Across All {NUM_SUBJECTS} Folds: {overall_mean_accuracy:.4f}")
print("=========================================================")
