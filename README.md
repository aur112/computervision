# MLP vs CNN Image Classification on FashionMNIST

A comparative study of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures for multi-class clothing image classification, built with PyTorch.

## Overview

This project explores why CNNs outperform MLPs on image data by training both architectures on the same task and comparing their results. A third section extends the CNN with data augmentation to improve real-world generalization.

The notebook is structured in three parts:
1. **MLP baseline** — fully connected network, used as a performance reference
2. **CNN model** — convolutional architecture trained on the same data
3. **Data augmentation** — horizontal and vertical flips applied to expand the training set

## Dataset

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) — 60,000 grayscale 28×28 images across 10 clothing categories:
T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

The dataset is downloaded automatically via `torchvision`.

## Model Architectures

**MLP**
- Flatten → Linear(784→512) → ReLU → Linear(512→64) → ReLU → Linear(64→10) → Sigmoid
- Optimizer: SGD (lr=0.001)
- Loss: CrossEntropyLoss

**CNN**
- 3× Conv2d blocks, each with BatchNorm, ReLU, Dropout(0.2), and MaxPool where applicable
- Channel progression: 1 → 16 → 64 → 256
- Fully connected head: Linear(256→64) → ReLU → Linear(64→10)
- Optimizer: SGD with momentum (lr=0.001, momentum=0.9)
- Loss: CrossEntropyLoss

## Results

| Model | Validation Accuracy | Notes |
|-------|-------------------|-------|
| MLP | ~60% | Spatial structure lost by flattening |
| CNN | ~90% | Preserves spatial features via convolutions |
| CNN + Augmentation | ~90% | Better generalization to spatial variations |

Training was run for 5 epochs with a learning rate scheduler (ReduceLROnPlateau, factor=0.5, patience=1).

## Key Findings

- MLPs flatten the image before processing, destroying spatial relationships between pixels, which is why they struggle with image tasks.
- CNNs slide learned filters across the image, capturing local patterns (edges, textures, shapes) regardless of position — leading to significantly better accuracy.
- Data augmentation (horizontal and vertical flips applied to 2/3 of the dataset) did not significantly change accuracy on this dataset, but improved the model's ability to generalize to different spatial orientations of the same object.

## Requirements

```
torch
torchvision
torchsummary
pytorch-ignite
matplotlib
numpy
```

Install with:
```bash
pip install torch torchvision torchsummary pytorch-ignite matplotlib numpy
```

## How to Run

The project is implemented as a Jupyter/Colab notebook. Open it in Google Colab or locally in Jupyter and run cells in order. The FashionMNIST dataset will be downloaded automatically on first run.

GPU is used automatically if available (`cuda`), otherwise falls back to CPU.
