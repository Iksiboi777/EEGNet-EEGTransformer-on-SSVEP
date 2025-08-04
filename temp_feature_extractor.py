import numpy as np
import os
from scipy.io import loadmat
import tensorflow.keras as keras
from keras.models import Model
from typing import Tuple

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

# --- EEGNet_SSVEP Model Parameters ---
KERN_LENGTH: int = 250
F1: int = 96
D: int = 1
F2: int = 96

# --- Configuration for this script ---
# Path to the directory where your best models were saved
MODELS_DIR: str = 'models_cross_subject'
# The ID of the subject whose model you want to use as the base for the extractor
# A subject with high accuracy (like S3) is a good choice.
SUBJECT_ID_FOR_WEIGHTS: int = 3 

# Path to your data, needed to load a sample for demonstration
DATA_PATH: str = '../Data'

###############################################################################
# # 2. FEATURE EXTRACTOR CREATION FUNCTION
# This function loads a pre-trained EEGNet_SSVEP model and creates a new
# model that outputs intermediate features.
###############################################################################

def create_eegnet_feature_extractor(weights_path: str) -> Tuple[Model, str]:
    """
    Loads a pre-trained EEGNet_SSVEP model and creates a new model that
    serves as a temporal feature extractor.

    Args:
        weights_path (str): The full path to the saved .h5 weights file.

    Returns:
        A tuple containing:
        - feature_extractor_model (keras.Model): The new model.
        - feature_layer_name (str): The name of the layer used for feature extraction.
    """
    print(f"--- Creating feature extractor from weights at: {weights_path} ---")
    
    # 1. Recreate the full model architecture
    # We must first instantiate a model with the exact same parameters as the one we saved.
    full_model: Model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                                     dropoutRate=0.5, kernLength=KERN_LENGTH,
                                     F1=F1, D=D, F2=F2)
    
    # 2. Load the pre-trained weights into this architecture
    try:
        full_model.load_weights(weights_path)
        print("Successfully loaded pre-trained weights.")
    except Exception as e:
        print(f"ERROR: Could not load weights. Please check the path. Details: {e}")
        return None, None

    # 3. Identify the desired feature layer
    # We will take the output from the layer just before the 'Flatten' operation.
    # By inspecting EEGModels.py, this layer is named 'dropout_1'.
    feature_layer_name: str = 'dropout_1'
    
    # 4. Create the new feature extractor model
    # The new model will have the same input as the original model, but its output
    # will be the output of our chosen intermediate layer.
    feature_extractor_model: Model = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(feature_layer_name).output
    )
    
    print(f"Feature extractor created. Output will be from layer: '{feature_layer_name}'")
    
    return feature_extractor_model, feature_layer_name

###############################################################################
# # 3. DEMONSTRATION
# This section shows how to use the feature extractor and how to reshape
# its output for a Transformer.
###############################################################################

if __name__ == "__main__":
    
    # --- Construct the path to the weights file ---
    model_filename: str = f'EEGNet_SSVEP_Test_S{SUBJECT_ID_FOR_WEIGHTS}.h5'
    full_weights_path: str = os.path.join(MODELS_DIR, model_filename)

    # --- Create the feature extractor ---
    feature_extractor, layer_name = create_eegnet_feature_extractor(full_weights_path)

    if feature_extractor:
        # --- Load a sample of data to test the extractor ---
        print("\n--- Running a test with sample data from Subject 1 ---")
        # We can reuse the data loading function from the training script
        from test_cross_subject import load_all_beta_data 
        
        # Load data for a single subject for this example
        X_all, _, _ = load_all_beta_data(data_path=DATA_PATH)
        # Take the first 5 trials as a sample batch
        sample_batch = X_all[:5]
        sample_batch = sample_batch.reshape(sample_batch.shape[0], CHANS, SAMPLES, 1)
        print(f"Input sample batch shape: {sample_batch.shape}")

        # --- Get the features ---
        extracted_features = feature_extractor.predict(sample_batch)
        print(f"Shape of extracted features from '{layer_name}': {extracted_features.shape}")
        
        # --- Reshape the features for a Transformer ---
        # A Transformer expects input in the format: (batch, sequence_length, feature_dimension)
        # The current output is (batch, filters, 1, time_points). We need to reshape it.
        
        # 1. Squeeze the singleton dimension
        squeezed_features = np.squeeze(extracted_features, axis=2)
        print(f"Shape after squeezing: {squeezed_features.shape}")
        
        # 2. Permute the last two dimensions to get (batch, time_points, filters)
        transformer_input = np.transpose(squeezed_features, (0, 2, 1))
        print(f"Final shape ready for Transformer input: {transformer_input.shape}")
        
        print("\n--- Demonstration Complete ---")
        print("You can now use this 'feature_extractor' model as the first stage of your MTGNet,")
        print("passing its reshaped output to your Transformer block.")

        # --- Save the feature extractor model for later use ---
        save_path = os.path.join(MODELS_DIR, 'eegnet_feature_extractor.keras')
        print(f"\nSaving the feature extractor model to: {save_path}")
        feature_extractor.save(save_path)
        print("Model saved successfully.")
