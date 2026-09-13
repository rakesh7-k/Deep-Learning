# GAN Image Generation

This experiment introduces generative adversarial networks (GANs) using the MNIST digits dataset. The notebook focuses on the generator-discriminator interaction and the learning dynamics that make GANs both powerful and unstable.

## Overview

A GAN consists of two networks: a generator that creates fake images and a discriminator that distinguishes real from fake samples. The two models compete during training, pushing each other to improve.

## Core Idea

The generator learns to synthesize images resembling the training data, while the discriminator learns to separate real images from generated ones.

## Architecture

```text
Random Noise -> Dense -> Reshape -> Conv2DTranspose -> Generated Image
```

```text
Image -> Conv2D -> Flatten -> Dense -> Real/Fake
```

## Mathematical Foundation

- Generator loss encourages realistic outputs.
- Discriminator loss distinguishes real from fake samples.
- Min-max adversarial training drives both networks.

## Dataset

MNIST is used because it is simple, familiar, and well suited to learning the basics of GANs.

## Implementation

The notebook builds the generator and discriminator, defines their losses, and trains them in a loop while periodically visualizing generated samples.

## Experiments

- Prepare MNIST data.
- Train generator and discriminator together.
- Track sample generation over epochs.
- Analyze stability and mode collapse behavior.

## Results

The model gradually learns to generate digit-like samples from random noise, illustrating the behavior of adversarial training.

## Technical Insights

GAN training is unstable and can suffer from mode collapse or oscillating loss curves. Even when the generator produces plausible images, the training dynamics must be monitored carefully.

## Limitations

- GANs are harder to stabilize than supervised models.
- Small or simple datasets may not expose the full complexity of generative modeling.
- Quality depends heavily on architecture and training setup.

## Possible Improvements

- Use Wasserstein loss.
- Add gradient penalties.
- Explore deeper architectures and better normalization strategies.

## Reproducibility

This notebook is designed for sequential execution in a local environment. It is meant to teach the mechanics of GAN training without claiming production-quality results.

## Technologies

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
