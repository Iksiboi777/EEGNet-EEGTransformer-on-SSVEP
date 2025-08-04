import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow.keras as keras
from keras import utils as np_utils
from keras.models import Model
import tensorflow as tf
from typing import Tuple, List, Dict, Any

# Import the specialized EEGNet_SSVEP model so we can recreate the architecture
# before loading the saved weights.
from EEGModels import EEGNet_SSVEP

# Set a seed for reproducibility
np.random.seed(42)

###############################################################################
# # 1. SETUP PARAMETERS (Must match the original training script)
###############################################################################

DATA_PATH: str = '../Data' 
NUM_SUBJECTS: int = 70
SAMPLING_RATE: int = 250
MAX_TRIAL_SECS: float = 4.0
CHANS: int = 64
CLASSES: int = 40
SAMPLES: int = int(MAX_TRIAL_SECS * SAMPLING_RATE)
KERN_LENGTH: int = 250
F1: int = 96
D: int = 1
F2: int = 96

###############################################################################
# # 2. DATA LOADING FUNCTION (Copied from the original script)
# We need this function to load the correct test data for each fold.
###############################################################################

def load_subject_data(subject_id: int, data_path: str = DATA_PATH) -> np.ndarray:
    """
    Loads and prepares the BETA dataset for a single subject.
    This will serve as the test set for the corresponding fold.
    """
    # print(f"Loading data for test subject {subject_id}...")
    file_path: str = os.path.join(data_path, f'S{subject_id}.mat')
    mat_data: Dict[str, Any] = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    eeg_data: np.ndarray = mat_data['data'].EEG
    eeg_data: np.ndarray = np.transpose(eeg_data, (2, 3, 0, 1))
    X: np.ndarray = eeg_data.reshape(-1, CHANS, eeg_data.shape[-1])

    if X.shape[2] < SAMPLES:
        padded_X: np.ndarray = np.zeros((X.shape[0], CHANS, SAMPLES))
        padded_X[:, :, :X.shape[2]] = X
        X = padded_X
    
    return X

###############################################################################
# # 3. RE-EVALUATION LOOP
# This loop iterates through each subject, loads the corresponding pre-trained
# model, and evaluates its performance. NO TRAINING IS DONE HERE.
###############################################################################

all_subject_accuracies: List[float] = []
models_dir: str = 'models_cross_subject'

# print("\n############################################################")
# print("      Re-evaluating Saved Models from Cross-Subject Run     ")
# print("############################################################")

for test_subject_id in range(1, NUM_SUBJECTS + 1):
    
    # print(f"\n--- Evaluating Fold {test_subject_id}/{NUM_SUBJECTS} (Test Subject: {test_subject_id}) ---")
    
    # --- 1. Load the Test Data for this Fold ---
    # The test data is the complete dataset for the subject left out in this fold.
    X_test: np.ndarray = load_subject_data(test_subject_id)
    X_test: np.ndarray = X_test.reshape(X_test.shape[0], CHANS, SAMPLES, 1)
    
    # Create the corresponding labels for the test set
    y_test_labels: np.ndarray = np.array([i for i in range(CLASSES)] * 4) # 4 blocks of 40 classes
    y_test: np.ndarray = np_utils.to_categorical(y_test_labels, num_classes=CLASSES)

    # --- 2. Recreate the Model Architecture ---
    # We must first create a model with the same architecture as the one that was saved.
    model: Model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                         dropoutRate=0.5, kernLength=KERN_LENGTH,
                         F1=F1, D=D, F2=F2)
    
    # Compile the model so it can be evaluated. The optimizer doesn't matter here.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # --- 3. Load the Saved Weights ---
    model_path: str = os.path.join(models_dir, f'EEGNet_SSVEP_Test_S{test_subject_id}.h5')
    
    try:
        # print(f"Loading weights from: {model_path}")
        model.load_weights(model_path)
    except Exception as e:
        print(f"ERROR: Could not load weights for Subject {test_subject_id}. File might be missing or corrupt. Skipping.")
        print(f"Error details: {e}")
        continue # Skip to the next subject

    # --- 4. Evaluate the Loaded Model ---
    # print("Evaluating model on the test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    all_subject_accuracies.append(test_acc)
    
    print(f"--- Accuracy for Subject {test_subject_id}: {test_acc:.4f} ---")

###############################################################################
# # 4. FINAL RESULTS
# Calculate and print the overall average performance from the recovered values.
###############################################################################

if all_subject_accuracies:
    overall_mean_accuracy: float = np.mean(all_subject_accuracies)
    print("\n=========================================================")
    print("              FINAL RECOVERED RESULTS                  ")
    print("=========================================================")
    print(f"Recovered Mean Accuracy Across {len(all_subject_accuracies)} Subjects: {overall_mean_accuracy:.4f}")
    print("=========================================================")
else:
    print("\nNo accuracies were recovered. Please check the models directory and file paths.")
