# Logistic Regression

This experiment introduces binary classification through a custom logistic regression implementation. The focus is on understanding how the model computes probabilities, updates weights, and learns from data.

## Overview

Logistic regression is a linear model used for binary classification. It predicts a probability between 0 and 1 by applying the sigmoid function to a linear combination of inputs.

## Core Idea

$$
z = Xw + b
$$

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

The model minimizes binary cross-entropy during training and updates parameters using gradient descent.

## Architecture

```text
Input Features -> Linear Score -> Sigmoid -> Probability -> Binary Prediction
```

## Mathematical Foundation

- Sigmoid activation
- Binary cross-entropy loss
- Gradient descent optimization
- Decision threshold at 0.5

## Dataset

The notebook uses a binary classification dataset such as the Breast Cancer dataset from scikit-learn, which is appropriate for learning the mechanics of logistic regression while avoiding unnecessary complexity.

## Implementation

The source file contains the reusable model routines, while the notebook focuses on loading data, preprocessing, training, plotting losses, and evaluating predictions.

## Experiments

- Train on a binary classification dataset.
- Plot the learning curve.
- Inspect model probabilities.
- Evaluate classification metrics.

## Results

The trained model demonstrates learning over multiple iterations and produces a practical classification baseline.

## Technical Insights

Logistic regression works well for linearly separable problems and is an ideal foundation for understanding neural networks, optimization, and probability-based decision-making.

## Limitations

- Assumes a linear decision boundary.
- Limited to binary classification without modification.
- Not suitable for highly non-linear patterns in its basic form.

## Possible Improvements

- Add feature engineering.
- Compare with a neural network baseline.
- Extend to multiclass softmax regression.

## Reproducibility

Install dependencies from the root requirements file and run the notebook in the project folder. The source file remains the reusable implementation.

## Technologies

- Python 3
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
