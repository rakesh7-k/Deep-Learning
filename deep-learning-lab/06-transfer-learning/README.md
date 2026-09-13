# Transfer Learning

This experiment uses a pretrained MobileNetV2 model trained on ImageNet and adapts it for a smaller classification problem. The notebook demonstrates how to reuse powerful visual features while keeping training efficient.

## Overview

Transfer learning leverages a model already trained on a large dataset and reuses its learned feature extraction layers for a new task. This is especially useful when data and compute are limited.

## Core Idea

A pretrained CNN acts as a fixed feature extractor. A small classification head is trained on the target dataset while the base layers remain frozen.

## Architecture

```text
Pretrained CNN -> Frozen Backbone -> Global Average Pooling -> Dense -> Softmax
```

## Mathematical Foundation

- Feature extraction from pretrained convolutional layers
- Transfer of learned representational patterns
- Dense classification head with softmax output

## Dataset

A standard image dataset such as CIFAR-10 is used for the classification task.

## Implementation

The notebook loads MobileNetV2, freezes the backbone, adds a classifier on top, and trains the head. A fine-tuning section can be included if computationally feasible.

## Experiments

- Load a pretrained model.
- Freeze the backbone.
- Train the classifier.
- Evaluate predictions.
- Optionally unfreeze selected layers for fine-tuning.

## Results

The transfer-learning pipeline shows faster convergence and often better performance than training a model from scratch on a small dataset.

## Technical Insights

Transfer learning is useful because generic visual features often transfer well across tasks. It is one of the most practical strategies in modern computer vision.

## Limitations

- Requires a pretrained model and possibly a large initial download.
- Data mismatch may reduce performance if the target domain is very different.
- Fine-tuning can be computationally expensive.

## Possible Improvements

- Add data augmentation.
- Fine-tune the top layers.
- Compare with train-from-scratch baselines.

## Reproducibility

The notebook is designed to run with the installed TensorFlow package and internet access for pretrained model weights.

## Technologies

- Python 3
- TensorFlow / Keras
- MobileNetV2
- Matplotlib
- Jupyter Notebook
