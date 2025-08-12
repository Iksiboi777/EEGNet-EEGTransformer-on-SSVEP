# test_feature_extractor.py

import numpy as np
import os
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import utils as np_utils

# Import the specialized EEGNet_SSVEP model.
from EEGModels import EEGNet_SSVEP

# Import typing for explicit type hints
from typing import Tuple, List, Dict, Any

###############################################################################
# # 1. CONFIGURATION AND PARAMETERS
###############################################################################

# --- File Paths ---
DATA_PATH = '../../Data'
# Path to the directory where the TRAINED feature extractor model is saved
MODEL_DIR = '../new_feature_extractor'
MODEL_NAME = 'BETA_EEGNet_Feature_Extractor.h5'

# --- Data Parameters ---
SAMPLING_RATE = 250
MAX_TRIAL_SECS = 4.0
CHANS = 64
CLASSES = 40
SAMPLES = int(MAX_TRIAL_SECS * SAMPLING_RATE)

# --- EEGNet_SSVEP Model Parameters (must match the trained model) ---
KERN_LENGTH = 250
F1 = 96
D = 1
F2 = 96

# --- Subject Partitioning ---
# Define the subjects that were held out as the UNSEEN test set.
TEST_SUBJECT_IDS = list(range(66, 71))

###############################################################################
# # 2. DATA LOADING FUNCTION
###############################################################################

def load_single_subject_data(subject_id: int, data_path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """ Loads and prepares the BETA dataset for a single subject. """
    print(f"--- Loading data for Test Subject {subject_id} ---")
    file_path = os.path.join(data_path, f'S{subject_id}.mat')
    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found for subject {subject_id} at {file_path}")
        return None, None
        
    mat_data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    eeg_data = np.transpose(mat_data['data'].EEG, (2, 3, 0, 1))
    
    X = eeg_data.reshape(-1, CHANS, eeg_data.shape[-1])
    if X.shape[2] < SAMPLES:
        padded_X = np.zeros((X.shape[0], CHANS, SAMPLES))
        padded_X[:, :, :X.shape[2]] = X
        X = padded_X
        
    y = np.array([i for i in range(eeg_data.shape[1])] * eeg_data.shape[0])
    
    return X, y

###############################################################################
# # 3. MAIN SCRIPT: EVALUATING THE PRE-TRAINED EXTRACTOR
###############################################################################

if __name__ == "__main__":
    print("\n######################################################################")
    print("## Evaluating Pre-Trained EEGNet on the Hold-Out Test Set ##")
    print("######################################################################")

    # --- 1. Load the pre-trained model ---
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Please run training script first.\nExpected path: {model_path}")

    model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                         dropoutRate=0.5, kernLength=KERN_LENGTH,
                         F1=F1, D=D, F2=F2)
    model.load_weights(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Successfully loaded model from: {model_path}")

    # --- 2. Evaluate on each test subject individually ---
    test_subject_accuracies = []
    
    for subject_id in TEST_SUBJECT_IDS:
        X_test_sub, y_test_sub = load_single_subject_data(subject_id)
        if X_test_sub is None:
            continue

        # Prepare data for evaluation
        X_test_sub = X_test_sub.reshape(X_test_sub.shape[0], CHANS, SAMPLES, 1)
        y_test_sub_cat = np_utils.to_categorical(y_test_sub, num_classes=CLASSES)
        
        # Evaluate and store accuracy
        loss, acc = model.evaluate(X_test_sub, y_test_sub_cat, verbose=0)
        test_subject_accuracies.append(acc)
        print(f"--- Accuracy for Subject {subject_id}: {acc:.4f} ---")

    # --- 3. Calculate and Print Final Average Test Accuracy ---
    if test_subject_accuracies:
        average_test_accuracy = np.mean(test_subject_accuracies)
        
        print("\n=========================================================")
        print("              FINAL TEST SET RESULTS              ")
        print("=========================================================")
        print(f"Evaluated on subjects: {TEST_SUBJECT_IDS}")
        for i, subject_id in enumerate(TEST_SUBJECT_IDS):
            print(f"  - Subject {subject_id} Accuracy: {test_subject_accuracies[i]:.4f}")
        print("---------------------------------------------------------")
        print(f"Average Accuracy on Unseen Test Set: {average_test_accuracy:.4f}")
        print("=========================================================")
    else:
        print("\nNo test subjects were evaluated. Please check TEST_SUBJECT_IDS and data paths.")