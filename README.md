MTGNet: A Multi-Modal Transformer-Graph Network for Real-Time EEG Analysis
This repository contains the work-in-progress code for MTGNet, a novel deep learning framework for real-time EEG signal generation and cognitive state decoding, developed as part of a Master's Thesis.

Abstract
This project presents MTGNet, a novel framework for real-time EEG signal generation and cognitive state decoding that leverages a fusion of transformer encoders and graph neural networks. Using established datasets—such as the BETA dataset for steady-state visual evoked potentials and the SEED dataset for emotional state recognition—the system first preprocesses EEG signals with standard artifact removal and segmentation techniques. A 1D convolutional neural network extracts channel-wise temporal features, which are then processed by a transformer encoder to capture long-range temporal dependencies and dynamic patterns. In parallel, a graph is constructed from EEG channels using functional connectivity measures, and a graph neural network refines these spatial relationships. The resulting multi-modal features (with optional integration of complementary data like fNIRS) are fused and fed into a classifier for accurate, real-time cognitive state decoding. Visualization modules display attention maps and dynamic inter-channel connectivity to provide interpretable neurofeedback. MTGNet thus offers a robust, adaptable solution for advanced brain–computer interface applications and real-time neurofeedback systems without relying on reinforcement learning.

Core Architecture
MTGNet employs a hybrid, multi-modal architecture to simultaneously process the temporal and spatial dimensions of EEG data. This dual-path approach allows the model to capture both the dynamic patterns within individual EEG channels over time and the complex spatial relationships between different channels across the scalp.

To use the image, upload it to your repository and replace "path/to/your/diagram.png" with the actual path to the image.

Current Status
This is an active research repository for an ongoing Master's Thesis.

The code contained here is a work-in-progress and is actively being developed, tested, and refined. The primary goal of this repository is to serve as a development base for the research and to document the methodology as it evolves. As such, some components may be experimental or subject to change.

Roadmap & Future Work
The following is a planned roadmap for the completion of the MTGNet project:

[ ] Model Implementation: Finalize and validate the core Transformer-GNN architecture.

[ ] Dataset Integration: Complete the data loading and preprocessing pipelines for the BETA and SEED datasets.

[ ] Baseline Model Training: Train and benchmark the model's performance on the SSVEP (BETA) and emotion recognition (SEED) tasks.

[ ] Multi-Modal Fusion: Implement and test the fusion mechanism for integrating complementary data streams like fNIRS.

[ ] Interpretability Module: Develop the visualization components for generating Transformer attention maps and dynamic GNN connectivity graphs.

[ ] Real-Time Pipeline: Adapt the model for a simulated real-time decoding environment for BCI and neurofeedback applications.

[ ] Thesis Documentation: Complete the final thesis paper documenting the methodology, results, and conclusions.
