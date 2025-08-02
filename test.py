import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow.keras as keras
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
# Import the Model class for type hinting
from keras.models import Model

# Import typing for explicit type hints
from typing import Tuple, List, Dict, Any


# Import both EEGNet and the specialized EEGNet_SSVEP model.
# We will be using EEGNet_SSVEP as it is designed for this data type.
from EEGModels import EEGNet_SSVEP

# Set a seed of reproducibility
np.random.seed(42)

###############################################################################
# # 1. SETUP PARAMETERS
# This section defines the key parameters for loading the BETA dataset and
# configuring the EEGNet_SSVEP model.
###############################################################################

# Set the path to the directory containing the BETA dataset
DATA_PATH = '../Data'

# --- Data Parameters ---
NUM_SUBJECTS = 70
SAMPLING_RATE = 250  # Hz
MAX_TRIAL_SECS = 4.0

# --- EEGNet_SSVEP Model Parameters ---
# These parameters are derived from the data and the Waytowich et al. (2018) paper.
CHANS = 64
CLASSES = 40
# All trials will be padded to the max length. T = 4s * 250 Hz = 1000 samples
SAMPLES = int(MAX_TRIAL_SECS * SAMPLING_RATE)

# For SSVEP, a longer temporal kernel is used to capture the steady-state signal.
# The Waytowich paper used a 1-second kernel (256 samples at 256Hz).
# We will use a 1-second kernel for our 250Hz data.
KERN_LENGTH = 250

# These filter parameters are the defaults from the EEGNet_SSVEP implementation,
# matching the configuration used in the Waytowich paper.
F1 = 96 # Number of temporal filters
D = 1 # Number of spatial filters
F2 = 96 # Number of pointwise filters

# --- Training Parameters ---
EPOCHS = 50
BATCH_SIZE = 32

###############################################################################
# # 2. DATA LOADING AND PREPARATION FUNCTION
# This function handles loading data for a single subject and implements
# padding to handle the different trial lengths without losing data.
###############################################################################

def load_subject_data(subject_id, data_path=DATA_PATH):
    """
    Load and prepares the BETA dataset for a single subject using padding.
    
    This function loads the .mat file, extracts EEG data and labels,
    and pads the sorter trials (from S1-15) with zeros to match the length
    of the longest trials (from S16-S70).
    
    Args:
        subject_id (int): The subject ID (1-70).
        data_path (str): Path to the directory containing the BETA dataset.
        
    Returns:
        A tuple of (X, y, groups) where X is the padded EEG data, y are the labels,
        and groups are the block numbers for cross-validation.
        
    """
    print(f"\n--- Loading data for Subject {subject_id}... ---")

    file_path = os.path.join(data_path, f'S{subject_id}.mat')
    mat_data: Dict[str, any] = loadmat(file_path)

    # Access the EEG data array from the nested MATLAB struct
    eeg_data:np.ndarray = mat_data['data']['eeg'][0, 0]
    # Transpose the dimensions from (channel, sample, block, condition) to (block, condition, channel, sample) for easier iteration
    eeg_data:np.ndarray = np.transpose(eeg_data, (2, 3, 0, 1))

    num_blocks: int = eeg_data.shape[0]
    num_conditions: int = eeg_data.shape[1]

    # --- Padding to Handle Variable Lengths ---
    # Reshape the data to combine blocks and conditions into a single dimension of trials: (total_trials, channels, samples)    
    X: np.ndarray = eeg_data.reshape(-1, CHANS, eeg_data.shape[-1])

    # Check if the current subject's trial are shorter than the maximum length(i.e., subjects S1-S15)
    if X.shape[2] < SAMPLES:
        print(f"Padding trials for Subject {subject_id} from {X.shape[2]} to {SAMPLES} samples.")
        # Create a new array of zeros with the target shape (total_trials, channels, SAMPLES)
        padded_X: np.ndarray = np.zeros((X.shape[0], CHANS, SAMPLES))
        # Copy the existing data into the padded array
        padded_X[:, :, :X.shape[2]] = X
        X: np.ndarray = padded_X

    # Generate the labels (y): Conditions are 1-40, so we create labels 0-39.
    y: np.ndarray = np.array([i for i in range(num_conditions)] * num_blocks)    
    # Generate the groups array, which controls the block number for each trial. This is needed for the cross-validation.
    groups: np.ndarray = np.array([i for i in range(num_blocks) for _ in range(num_conditions)])

    # Print a status message confirming the number of loaded trials
    print(f"Loaded {X.shape[0]} trials.")
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}, Groups shape: {groups.shape}")

    return X, y, groups


###############################################################################
# # 3. TRAINING FUNCTION
# This loop iterates through each subject, performs Leave-One-Block-Out
# cross-validation using the specialized EEGNet_SSVEP model.
###############################################################################

# Initialize an empty list to store the average accuracy for each subject
all_subject_accuracies = []

# Create a directory to store the saved models if it doesn't exist
os.makedirs('models', exist_ok=True)

print("############################################################")
print("Starting EEGNet_SSVEP Training and Evaluation on BETA Dataset")
print("############################################################")

# Loop through each subject in the dataset
for subject_id in range(1, NUM_SUBJECTS + 1):

    # Load the data for the current subject
    X: np.ndarray; y: np.ndarray; groups: np.ndarray = load_subject_data(subject_id)
    # Reshape the data to add a fourth dimension for the 'kernels', as required by Keras Conv2D layers
    X: np.ndarray = X.reshape(X.shape[0], CHANS, SAMPLES, 1)
    # Convert labels to categorical format
    y_categorical: np.ndarray = np_utils.to_categorical(y, num_classes=CLASSES)

    # --- Leave One Block Out cross-validation ---
    logo: LeaveOneGroupOut = LeaveOneGroupOut()
    subject_fold_accs: List[float] = []

    # Loop through the splits generated by LeaveOneGroupOut
    fold_num: int; train_index: np.ndarray; test_index: np.ndarray

    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, y_categorical, groups)):
        print(f"\n--- Training on Subject {subject_id}, Fold {fold_num + 1}/4 ---")

        # Split the data into training and testing sets
        X_train: np.ndarray; X_test: np.ndarray = X[train_idx], X[test_idx]
        y_train: np.ndarray; y_test: np.ndarray = y_categorical[train_idx], y_categorical[test_idx]

        # --- Instantiate and Compile the specialized EEGNet_SSVEP model ---
        # We use the EEGNet_SSVEP model designed for SSVEP data.
        # The parameters are set according to the Waytowich et al. (2018) paper.
        model = EEGNet_SSVEP(nb_classes=CLASSES, Chans=CHANS, Samples=SAMPLES,
                             dropoutRate=0.5, kernLength=KERN_LENGTH,
                             F1=F1, D=D, F2=F2)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint_path = f'models/EEGNet_SSVEP_Subject{subject_id}_Fold{fold_num + 1}.h5'
        # Define a checkpoint to save the best model during training
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        # Announce the start of the training phase
        print(f"Starting training for {EPOCHS} epochs...")
        # Train the model on the training data for the specified number of epochs
        history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, 
                                epochs=EPOCHS, verbose=1, 
                                validation_data=(X_test, y_test),
                                callbacks=[checkpoint])        

        # Load the best model saved during training
        model.load_weights(checkpoint_path)
        print(f"Best model loaded from {checkpoint_path}")
        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(X-X_test, y_test, verbose=1)
        subject_fold_accs.append(test_acc)

        # Get the final training loss and accuracy from the history object
        train_loss: float = history.history['loss'][-1]
        train_accuracy: float = history.history['accuracy'][-1]

        # Print the final results for the fold
        print(f"Fold {fold_num + 1} Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Fold {fold_num + 1} Test Accuracy: {test_acc:.4f}")


    # Calculate the average accuracy across all folds for the current subject
    subject_mean_accuracy: float = np.mean(subject_fold_accs)
    # Append the subject's average accuracy to the overall list
    all_subject_accuracies.append(subject_mean_accuracy)
    # Print the average accuracy for the current subject
    print(f"--- Subject {subject_id} Average Accuracy: {subject_mean_accuracy:.4f} ---")


###############################################################################
# # 4. FINAL RESULTS
# Calculate and print the overall average performance across all subjects.
###############################################################################

# Calculate the final mean accuracy across all 70 subjects
overall_mean_accuracy: float = np.mean(all_subject_accuracies)
# Print a formatted summary of the final result
print("\n=========================================================")
print("                      FINAL RESULTS                      ")
print("=========================================================")
print(f"Overall Mean Classification Accuracy Across All {NUM_SUBJECTS} Subjects: {overall_mean_accuracy:.4f}")
print("=========================================================")

