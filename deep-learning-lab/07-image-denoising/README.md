# Image Denoising Autoencoder

This experiment trains an autoencoder to reconstruct clean images from noisy versions of the same images. It is a classic example of representation learning and restoration.

## Overview

The network receives a noisy image, compresses it into a latent representation, and reconstructs a denoised version. The goal is to learn a compact internal representation that preserves the key information needed for restoration.

## Core Idea

The denoising autoencoder learns to map corrupted input back to a clean target. Training with noisy data encourages the model to focus on the underlying structure rather than the noise itself.

## Architecture

```text
Noisy Image -> Encoder -> Latent Representation -> Decoder -> Denoised Image
```

## Mathematical Foundation

- Encoder compresses the image into a lower-dimensional representation.
- Decoder reconstructs the cleaned image from that latent code.
- MSE or MAE is often used as the reconstruction loss.

## Dataset

The MNIST dataset is used because the images are simple and easy to inspect visually during training.

## Implementation

The notebook adds Gaussian noise, visualizes original and noisy examples, trains the autoencoder, and compares reconstruction quality before and after denoising.

## Experiments

- Create noisy versions of clean MNIST digits.
- Train the autoencoder.
- Compare original, noisy, and reconstructed images.
- Analyze reconstruction quality.

## Results

The autoencoder reduces noise and reconstructs the underlying digit structure while retaining the core visual content.

## Technical Insights

Autoencoders are useful for learning compact latent representations and restoring corrupted data. They are foundational for generative modeling and self-supervised representation learning.

## Limitations

- Training quality depends on dataset diversity and noise level.
- The demo is intended as a learning example rather than a production denoising system.
- Complex natural images may require deeper networks and stronger preprocessing.

## Possible Improvements

- Use larger architectures.
- Add more realistic noise models.
- Compare MSE, SSIM, and perceptual loss.

## Reproducibility

Run the notebook sequentially in this project folder after installing the dependencies in the root requirements file.

## Technologies

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
