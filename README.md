# DL_APT

Deep Learning Framework for Medical X-ray Image Analysis  
(CNN or Vision Transformer Hybrid Model)

---

## 📌 Overview

This project provides a deep learning pipeline for medical X-ray image analysis, including:

- DICOM data preprocessing
- Conversion to PNG format
- CNN or Vision Transformer (ViT) hybrid model
- Model training and evaluation
- Visualization (e.g., CAM)

This framework is designed for reproducible research in medical imaging.

---

## 🧠 Model Architecture

The core model integrates:

- Convolutional Neural Network (CNN) for local feature extraction  
- Vision Transformer (ViT) for image modeling  

## Implemented in:
DL_APT/
  -│
  -├── CNN_5+VIT.py # CNN + ViT model
  -├── gen_PNG.py # DICOM → PNG preprocessing
  -├── gen_text.py # Label/text generation
  -├── shuff_cam_last.py # CAM visualization
  -├── train2.py # Training script
  -├── test2.py # Testing script

## Recommended environment:

- Python 3.8.8
- PyTorch
- scikit-learn
- numpy
- pandas
- opencv-python
- pydicom
- matplotlib

## Convert DICOM X-ray images to CNN-compatible format:
python gen_PNG.py
## Training
python train2.py
## Testing
python test2.py
## Class activation maps (CAM):
python shuff_cam_last.py










