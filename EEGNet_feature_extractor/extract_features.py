# extract_features_v2.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from scipy.io import loadmat

# Import the specialized EEGNet_SSVEP model.
from EEGModels import EEGNet_SSVEP

# --- CONFIGURATION ---
DATA_PATH = '../../Data'
MODEL_DIR = '../new_feature_extractor'
MODEL_NAME = 'BETA_EEGNet_Feature_Extractor.h5'
OUTPUT_DIR = '../../features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODEL PARAMETERS (must match the trained model) ---
CHANS = 64
CLASSES = 40
SAMPLES = 1000
KERN_LENGTH = 250
F1 = 96
D = 1
F2 = 96
NUM_SUBJECTS = 70

if __name__ == "__main__":
    print("\n### Extracting V2 Features (from end of Block 2) ###")

    # --- 1. Load the pre-trained full model ---
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    full_model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                              dropoutRate=0.5, kernLength=KERN_LENGTH,
                              F1=F1, D=D, F2=F2)
    full_model.load_weights(model_path)
    print(f"Loaded model from: {model_path}")

    # --- 2. Create the NEW feature extractor ---
    # MODIFIED: Target the dropout layer at the end of the SECOND block.
    feature_layer_name = 'spatial_feature_dropout'
    feature_extractor = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(feature_layer_name).output,
        name='eegnet_feature_extractor_v2'
    )
    feature_extractor.summary()

    # --- 3. Iterate through all subjects to extract features ---
    all_features, all_labels, all_subject_indices = [], [], []
    for subject_id in range(1, NUM_SUBJECTS + 1):
        print(f"Processing Subject {subject_id}...")
        file_path = os.path.join(DATA_PATH, f'S{subject_id}.mat')
        mat_data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        eeg_data = np.transpose(mat_data['data'].EEG, (2, 3, 0, 1))
        
        X = eeg_data.reshape(-1, CHANS, eeg_data.shape[-1])
        if X.shape[2] < SAMPLES:
            padded_X = np.zeros((X.shape[0], CHANS, SAMPLES))
            padded_X[:, :, :X.shape[2]] = X
            X = padded_X
        X = X.reshape(X.shape[0], CHANS, SAMPLES, 1)

        y = np.array([i for i in range(eeg_data.shape[1])] * eeg_data.shape[0])
        subject_ids = np.full(y.shape[0], subject_id)
        
        extracted_features = feature_extractor.predict(X, verbose=0)
        
        # Squeeze out the singleton dimension, shape becomes (batch, timepoints, features)
        squeezed_features = np.squeeze(extracted_features)
        
        all_features.append(squeezed_features)
        all_labels.append(y)
        all_subject_indices.append(subject_ids)

    # --- 4. Consolidate and Save the Final Feature Set ---
    final_features = np.concatenate(all_features, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    final_subject_indices = np.concatenate(all_subject_indices, axis=0)
    
    # MODIFIED: New output filename
    output_path = os.path.join(OUTPUT_DIR, 'BETA_AllSubjects_EEGNet_Block2_Features.npz')
    
    print(f"\n--- Saving final V2 feature set ---")
    print(f"Final features shape: {final_features.shape}")
    
    np.savez_compressed(output_path, 
                        features=final_features, 
                        labels=final_labels, 
                        subject_indices=final_subject_indices)

    print(f"The final feature set has been saved to: {output_path}")