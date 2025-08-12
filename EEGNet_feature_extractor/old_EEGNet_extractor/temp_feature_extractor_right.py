import numpy as np
import os
from scipy.io import loadmat
import tensorflow.keras as keras
from keras.models import Model
from typing import Tuple, List

# Import the specialized EEGNet_SSVEP model, which is needed to
# recreate the architecture before loading the weights.
from EEGModels import EEGNet_SSVEP

###############################################################################
# # 1. SETUP PARAMETERS (Must match the original training script)
###############################################################################

# --- Data Parameters ---
SAMPLING_RATE: int = 250
MAX_TRIAL_SECS: float = 4.0
CHANS: int = 64
CLASSES: int = 40
SAMPLES: int = int(MAX_TRIAL_SECS * SAMPLING_RATE)
NUM_SUBJECTS: int = 70

# --- EEGNet_SSVEP Model Parameters ---
KERN_LENGTH: int = 250
F1: int = 96
D: int = 1
F2: int = 96

# --- Configuration for this script ---
# Path to the directory where your best models from the LOSO training are saved
MODELS_DIR: str = '../../models_cross_subject'
# Path to your raw data directory
DATA_PATH: str = '../../Data'
# Path for the output file
OUTPUT_DIR: str = '../../features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# # 2. FEATURE EXTRACTION WITH ENSEMBLE OF MODELS
# This script now uses the corresponding Leave-One-Out model for each subject
# to create a more robust and unbiased feature set.
###############################################################################


if __name__ == "__main__":

    # --- Initialize lists to store all features, labels, and subject IDs ---
    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_subject_indices: List[np.ndarray] = []
    
    print("\n--- Starting ENSEMBLE feature extraction for all 70 subjects ---")
    
    # --- Loop through every subject to extract their features ---
    for subject_id in range(1, NUM_SUBJECTS + 1):
        print(f"Processing Subject {subject_id}...")
        
        # --- 1. Load the specific model trained with this subject held out ---
        model_filename: str = f'EEGNet_SSVEP_Test_S{subject_id}.h5'
        weights_path: str = os.path.join(MODELS_DIR, model_filename)
        
        full_model: Model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                                         dropoutRate=0.5, kernLength=KERN_LENGTH,
                                         F1=F1, D=D, F2=F2)
        try:
            full_model.load_weights(weights_path)
        except Exception as e:
            print(f"ERROR: Could not load weights for S{subject_id}. Skipping. Details: {e}")
            continue

        # --- 2. Create the feature extractor from this specific model ---
        feature_layer_name: str = 'temporal_feature_dropout'
        feature_extractor: Model = Model(
            inputs=full_model.input,
            outputs=full_model.get_layer(feature_layer_name).output,
            name=f'eegnet_feature_extractor_s{subject_id}'
        )

        # --- 3. Load the raw data for this subject ---
        file_path = os.path.join(DATA_PATH, f'S{subject_id}.mat')
        mat_data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        eeg_data = mat_data['data'].EEG
        eeg_data = np.transpose(eeg_data, (2, 3, 0, 1))
        X = eeg_data.reshape(-1, CHANS, eeg_data.shape[-1])

        if X.shape[2] < SAMPLES:
            padded_X = np.zeros((X.shape[0], CHANS, SAMPLES))
            padded_X[:, :, :X.shape[2]] = X
            X = padded_X
        
        X = X.reshape(X.shape[0], CHANS, SAMPLES, 1)

        # --- 4. Generate labels and subject IDs for this subject ---
        num_blocks = eeg_data.shape[0]
        num_conditions = eeg_data.shape[1]
        y = np.array([i for i in range(num_conditions)] * num_blocks)
        subject_ids = np.full(y.shape[0], subject_id) # Create subject ID array
        
        # --- 5. Extract features and append all data to master lists ---
        extracted_features = feature_extractor.predict(X)
        transformer_input = np.squeeze(extracted_features, axis=1)
        
        all_features.append(transformer_input)
        all_labels.append(y)
        all_subject_indices.append(subject_ids)

    # --- 6. Combine all data into single arrays ---
    print("\n--- Consolidating all features, labels, and subject indices ---")
    final_features = np.concatenate(all_features, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    final_subject_indices = np.concatenate(all_subject_indices, axis=0)
    
    print(f"Final features array shape: {final_features.shape}")
    print(f"Final labels array shape: {final_labels.shape}")
    print(f"Final subject indices array shape: {final_subject_indices.shape}")
    
    # --- 7. Save the final annotated feature set to a file ---
    output_path = os.path.join(OUTPUT_DIR, 'BETA_EEGNet_Ensemble_Features.npz')
    print(f"\nSaving final feature set to: {output_path}")
    np.savez_compressed(output_path, 
                        features=final_features, 
                        labels=final_labels, 
                        subject_indices=final_subject_indices)
    
    print("\n--- Ensemble feature extraction complete! ---")
