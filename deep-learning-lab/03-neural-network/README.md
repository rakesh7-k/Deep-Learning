# One-Hidden-Layer Neural Network

This experiment extends the concept of a single neuron to a small network with one hidden layer and a binary output. The goal is to understand forward propagation, hidden activations, and backpropagation in a simple architecture.

## Overview

The model uses:

- input features
- hidden layer with tanh activation
- output layer with sigmoid activation

This allows the network to model non-linear decision boundaries while still staying small enough for educational study.

## Core Idea

$$
Z_1 = XW_1 + b_1
$$

$$
A_1 = \tanh(Z_1)
$$

$$
Z_2 = A_1W_2 + b_2
$$

$$
A_2 = \sigma(Z_2)
$$

Loss is computed with binary cross-entropy and gradients are propagated backward to update parameters.

## Architecture

```text
Input -> Hidden Layer (tanh) -> Output Layer (sigmoid)
```

## Mathematical Foundation

- Parameter initialization
- Forward propagation
- Backpropagation
- Binary cross-entropy
- Gradient descent

## Dataset

A binary classification dataset such as the Breast Cancer dataset is used for the learning example. This makes the experiment easier to understand, and the same ideas transfer to larger networks.

## Implementation

The source file contains a reusable implementation of the one-hidden-layer network. The notebook imports the implementation and focuses on training behavior, loss curves, and model evaluation.

## Experiments

- Train a small neural network.
- Track loss over epochs.
- Visualize model learning.
- Evaluate predictions and accuracy.

## Results

The network learns the target relationship over training iterations and produces a binary classification output with a measurable learning curve.

## Technical Insights

This experiment helps bridge the gap between logistic regression and deep neural networks. The hidden layer introduces non-linearity, which allows the model to learn more complex patterns than a purely linear classifier.

## Limitations

- Small network and dataset size for teaching.
- Not optimized for large-scale production use.
- Binary output only in the basic example.

## Possible Improvements

- Add regularization.
- Use more features and deeper layers.
- Compare to logistic regression as a baseline.

## Reproducibility

Use the root requirements file and run the notebook within this project folder. The custom implementation is kept in the Python source file, not duplicated inside the notebook.

## Technologies

- Python 3
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
