# Detecting Pneumonia in Chest X-Ray Images

**Group 17:** Oren Raz, Shaked Shabat, Itay Keinan, Evyatar Yatir  
**Data Science in the Industry Course Project**

## Overview

This project compares **anomaly detection** with **supervised transfer learning** for pneumonia detection in chest X-ray images.

The anomaly-detection approach uses a convolutional autoencoder trained only on **NORMAL** images. Its performance is compared with three supervised models trained on both classes:

- ResNet18
- EfficientNet-B0
- Vision Transformer (ViT)

## Dataset

The project uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

- 5,856 pediatric chest X-ray images
- Train: 1,341 NORMAL / 3,875 PNEUMONIA
- Test: 234 NORMAL / 390 PNEUMONIA
- The original 16-image validation set was replaced with an internal validation split from the training data

## Method

1. Download the dataset through the Kaggle API
2. Pad and resize images to 224×224
3. Apply CLAHE and training-only augmentation
4. Train an autoencoder on NORMAL images only
5. Fine-tune ResNet18, EfficientNet-B0, and ViT using transfer learning
6. Evaluate using AUC, precision, recall, F1, confusion matrices, ROC curves, and precision-recall curves

## Results

| Model | Approach | Test AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Autoencoder | Anomaly detection | 0.511 | 0.625 | 0.064 | 0.116 |
| ResNet18 | Supervised CNN | 0.949 | 0.742 | 0.997 | 0.851 |
| EfficientNet-B0 | Supervised CNN | 0.957 | 0.761 | 0.997 | 0.863 |
| Vision Transformer | Supervised ViT | **0.958** | **0.796** | 0.997 | **0.885** |

Supervised transfer learning clearly outperformed reconstruction-based anomaly detection. ViT achieved the best overall results, while EfficientNet-B0 provided very similar AUC with a lighter architecture.

## Repository Contents

- [`Detecting_Pneumonia.ipynb`](Detecting_Pneumonia.ipynb) — complete preprocessing, modeling, evaluation, and visualizations
- [`Pneumonia_Detection_Presentation.pptx`](Pneumonia_Detection_Presentation.pptx) — final project presentation
- [`requirements.txt`](requirements.txt) — Python dependencies

## Running the Notebook

The notebook is designed for **Google Colab**.

1. Open `Detecting_Pneumonia.ipynb` in Google Colab.
2. Download your personal `kaggle.json` API token from Kaggle.
3. Run the notebook and upload `kaggle.json` when prompted.
4. Run the remaining cells sequentially.
