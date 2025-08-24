# EEGNet + EEGTransformer for SSVEP Classification (Archived Baseline)

> **Note:** This repository contains the code for an earlier implementation of an SSVEP classification system and is **archived**. This project is no longer under active development. It served as the initial baseline that informed the development of the more advanced, PyTorch-based **[SSVEPformer project](https://github.com/Iksiboi777/SSVEPformer-GNN-Modal)**.

## Project Overview

This project was an initial exploration into classifying Steady-State Visually Evoked Potentials (SSVEP) from raw EEG signals. The primary goal was to establish a performance baseline using a combination of established and adapted deep learning architectures.

The pipeline developed here utilized a Convolutional Neural Network (EEGNet) for robust feature extraction, followed by a Transformer-based model for the final classification task. While this implementation was a valuable exercise in building an end-to-end EEG classification system in TensorFlow, it did not achieve the desired performance benchmarks, which led to the development of the newer SSVEPformer architecture.

### Architecture

The model follows a two-stage process:

1.  **EEGNet Feature Extractor:** A compact CNN specifically designed for EEG signals is used to learn robust spatial and temporal representations from the raw time-series data.
2.  **Transformer Classifier:** The features extracted by EEGNet are then passed as a sequence to a Transformer-based classifier. This component uses self-attention mechanisms to weigh the importance of different learned features for the final SSVEP classification.

## Tech Stack

* **Framework:** TensorFlow, Keras
* **Libraries:** Scikit-learn, NumPy, Matplotlib
* **EEG Processing:** MNE-Python
