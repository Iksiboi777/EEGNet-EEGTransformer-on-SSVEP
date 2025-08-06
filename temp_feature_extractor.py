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
# Path to the directory where your best models were saved
MODELS_DIR: str = '../models_cross_subject'
# The ID of the subject whose model you want to use as the base for the extractor.
# A subject with high accuracy (like S3, S18, or S37) is a good choice.
# This model, trained on 69 other subjects, will serve as our single, generalized feature extractor.
SUBJECT_ID_FOR_WEIGHTS: int = 3 

# Path to your raw data directory
DATA_PATH: str = '../Data'

# Path for the output file
OUTPUT_DIR: str = '../features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# # 2. FEATURE EXTRACTOR CREATION FUNCTION
###############################################################################

def create_eegnet_feature_extractor(weights_path: str) -> Model:
    """
    Loads a pre-trained EEGNet_SSVEP model and creates a new model that
    serves as a temporal feature extractor.
    """
    print(f"--- Creating feature extractor from weights at: {weights_path} ---")
    
    full_model: Model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                                     dropoutRate=0.5, kernLength=KERN_LENGTH,
                                     F1=F1, D=D, F2=F2)
    
    try:
        full_model.load_weights(weights_path)
        print("Successfully loaded pre-trained weights.")
    except Exception as e:
        print(f"ERROR: Could not load weights. Please check the path. Details: {e}")
        return None

    feature_layer_name: str = 'dropout_1'
    
    feature_extractor_model: Model = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(feature_layer_name).output,
        name='eegnet_feature_extractor'
    )
    
    print(f"Feature extractor created. Output will be from layer: '{feature_layer_name}'")
    
    return feature_extractor_model

###############################################################################
# # 3. DATA PROCESSING AND FEATURE EXTRACTION
# This section iterates through all subjects, extracts features, and saves them.
###############################################################################

if __name__ == "__main__":
    
    # --- 1. Create the single, generalized feature extractor ---
    model_filename: str = f'EEGNet_SSVEP_Test_S{SUBJECT_ID_FOR_WEIGHTS}.h5'
    full_weights_path: str = os.path.join(MODELS_DIR, model_filename)
    feature_extractor = create_eegnet_feature_extractor(full_weights_path)

    if feature_extractor:
        
        # --- 2. Initialize lists to store all features and labels ---
        all_features: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        
        print("\n--- Starting feature extraction for all 70 subjects ---")
        
        # --- 3. Loop through every subject to extract their features ---
        for subject_id in range(1, NUM_SUBJECTS + 1):
            print(f"Processing Subject {subject_id}...")
            
            # Load the raw data for the current subject
            file_path = os.path.join(DATA_PATH, f'S{subject_id}.mat')
            mat_data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
            eeg_data = mat_data['data'].EEG
            eeg_data = np.transpose(eeg_data, (2, 3, 0, 1))
            X = eeg_data.reshape(-1, CHANS, eeg_data.shape[-1])

            # Pad the data if necessary
            if X.shape[2] < SAMPLES:
                padded_X = np.zeros((X.shape[0], CHANS, SAMPLES))
                padded_X[:, :, :X.shape[2]] = X
                X = padded_X
            
            # Reshape for model input
            X = X.reshape(X.shape[0], CHANS, SAMPLES, 1)

            # Generate labels for this subject
            num_blocks = eeg_data.shape[0]
            num_conditions = eeg_data.shape[1]
            y = np.array([i for i in range(num_conditions)] * num_blocks)
            
            # Use the feature extractor to get the features for this subject
            extracted_features = feature_extractor.predict(X)
            
            # Reshape the features into the (batch, sequence_length, feature_dimension) format
            transformer_input = np.squeeze(extracted_features, axis=1)
            
            # Append the results to our master lists
            all_features.append(transformer_input)
            all_labels.append(y)

        # --- 4. Combine all features and labels into single arrays ---
        print("\n--- Consolidating all features and labels ---")
        final_features = np.concatenate(all_features, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Final features array shape: {final_features.shape}")
        print(f"Final labels array shape: {final_labels.shape}")
        
        # --- 5. Save the final feature set to a file ---
        output_path = os.path.join(OUTPUT_DIR, 'BETA_EEGNet_features.npz')
        print(f"\nSaving final feature set to: {output_path}")
        # Use np.savez_compressed to save space
        np.savez_compressed(output_path, features=final_features, labels=final_labels)
        
        print("\n--- Feature extraction complete! ---")
        print("You can now use the 'BETA_EEGNet_features.npz' file as the input for your Transformer model.")
