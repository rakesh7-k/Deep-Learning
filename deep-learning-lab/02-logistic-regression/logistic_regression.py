import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_logistic_regression(X, y, learning_rate=0.1, epochs=2000, seed=42):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    n_samples, n_features = X.shape
    weights = rng.normal(0.0, 0.01, size=(n_features, 1))
    bias = np.zeros(1)
    losses = []

    for _ in range(epochs):
        linear_output = X @ weights + bias
        probabilities = sigmoid(linear_output)
        error = probabilities - y
        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error, axis=0)

        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

        loss = binary_cross_entropy(y, probabilities)
        losses.append(loss)

    return {"weights": weights, "bias": bias, "losses": losses}


def predict_proba(X, weights, bias):
    X = np.asarray(X, dtype=float)
    return sigmoid(X @ weights + bias)


def predict(X, weights, bias, threshold=0.5):
    probabilities = predict_proba(X, weights, bias)
    return (probabilities >= threshold).astype(int)
