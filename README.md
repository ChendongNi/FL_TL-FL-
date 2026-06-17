# README

## Code and Model Availability

This repository contains the source code and scripts used for the study:

**Improving Cross-Center Generalization for Multi-modal MRI Meningioma Segmentation via Glioma-Pretrained Federated Learning**

All source code files, including scripts for data preprocessing, model training, federated learning, Glioma-pretrained FL, inference, and evaluation, are available in the GitHub repository:

https://github.com/ChendongNi/FL_TL-FL-

The final trained model weights generated in this study have been deposited in the following publicly accessible Google Drive folder:

https://drive.google.com/drive/folders/1JSSd90zJqBxJNooJM4ygnVj54WabxYT_?usp=sharing

The model-weight folder contains the trained weights corresponding to the main experimental models reported in the manuscript, including the centralized UMamba model, meningioma-pretrained FL model, and Glioma-pretrained FL model.

## Data Availability

The public BraTS2023-Men and BraTS2023-Gli datasets used in this study are available through the official BraTS 2023 Challenge data-access platform on Synapse, subject to the corresponding BraTS data-use agreements and access requirements.

The SPHS clinical MRI dataset is not included in this repository because it contains human-participant clinical imaging data and is subject to patient privacy protections, institutional data-governance policies, ethical restrictions, and the approved institutional review board protocol.

## Notes

Users should obtain access to the BraTS2023 datasets through the official BraTS/Synapse platform before running training or evaluation scripts that depend on BraTS data.

Access to SPHS-related data is not provided through this repository. Requests for de-identified SPHS-related data should be directed to the corresponding author and will be subject to institutional review, ethics approval where applicable, and data-use agreement requirements.
