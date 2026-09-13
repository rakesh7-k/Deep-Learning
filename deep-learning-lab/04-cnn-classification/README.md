# CNN Multiclass Classification

This experiment applies a small convolutional neural network to the MNIST dataset. The notebook focuses on building the model directly in Keras, training it on digit images, and evaluating the resulting classification performance.

## Overview

MNIST is a classic handwritten digits dataset. A convolutional neural network is especially effective for image recognition because it learns spatial patterns such as strokes, loops, and contours.

## Core Idea

The model architecture processes image tensors through convolution, activation, pooling, flattening, and dense layers before producing a class probability distribution.

## Architecture

```text
Input -> Conv2D -> ReLU -> MaxPooling -> Flatten -> Dense -> Softmax
```

## Mathematical Foundation

- Convolution kernels detect local patterns.
- ReLU introduces non-linearity.
- Pooling reduces spatial dimension.
- Softmax produces class probabilities.

## Dataset

MNIST is used because it is a standard and lightweight dataset for learning image classification with CNNs.

## Implementation

This experiment is primarily implemented in a notebook, which is the appropriate place for Keras model definition, training, and visualization.

## Experiments

- Explore sample digits.
- Preprocess the images.
- Train the CNN.
- Plot accuracy and loss curves.
- Inspect confusion matrix and sample predictions.

## Results

The model learns to classify digits and demonstrates typical CNN behavior on a well-known benchmark.

## Technical Insights

CNNs are effective because they preserve local spatial structure and exploit translation invariance, which makes them more suitable for image tasks than fully connected networks.

## Limitations

- Uses a relatively simple architecture.
- Notebook-focused demonstration rather than production-level optimization.
- Requires a modest amount of compute for training.

## Possible Improvements

- Add dropout and batch normalization.
- Use data augmentation.
- Explore deeper and more modern CNN designs.

## Reproducibility

Use the root requirements file and run the notebook in this folder. The model is defined directly in the notebook to match the educational Keras-first workflow.

## Technologies

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
