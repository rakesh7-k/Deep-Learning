import numpy as np


def initialize_parameters(input_size, hidden_size, seed=42):
    rng = np.random.default_rng(seed)
    parameters = {
        "W1": rng.normal(0.0, 0.1, size=(input_size, hidden_size)),
        "b1": np.zeros((1, hidden_size)),
        "W2": rng.normal(0.0, 0.1, size=(hidden_size, 1)),
        "b2": np.zeros((1, 1)),
    }
    return parameters


def tanh(z):
    return np.tanh(z)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def forward_propagation(X, parameters):
    Z1 = X @ parameters["W1"] + parameters["b1"]
    A1 = tanh(Z1)
    Z2 = A1 @ parameters["W2"] + parameters["b2"]
    A2 = sigmoid(Z2)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def binary_cross_entropy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()


def backward_propagation(X, y_true, cache, parameters):
    m = X.shape[0]
    A2 = cache["A2"]
    A1 = cache["A1"]

    dZ2 = A2 - y_true.reshape(-1, 1)
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ parameters["W2"].T
    dZ1 = dA1 * (1 - np.square(A1))
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }
    return gradients


def update_parameters(parameters, gradients, learning_rate):
    for key in ["W1", "b1", "W2", "b2"]:
        if key == "W1":
            parameters[key] -= learning_rate * gradients["dW1"]
        elif key == "b1":
            parameters[key] -= learning_rate * gradients["db1"]
        elif key == "W2":
            parameters[key] -= learning_rate * gradients["dW2"]
        elif key == "b2":
            parameters[key] -= learning_rate * gradients["db2"]
    return parameters


def train_neural_network(X, y, hidden_size=8, learning_rate=0.05, epochs=500, seed=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    parameters = initialize_parameters(X.shape[1], hidden_size, seed=seed)
    losses = []

    for _ in range(epochs):
        A2, cache = forward_propagation(X, parameters)
        loss = binary_cross_entropy(y, A2)
        gradients = backward_propagation(X, y, cache, parameters)
        parameters = update_parameters(parameters, gradients, learning_rate)
        losses.append(loss)

    return {"parameters": parameters, "losses": losses}


def predict(X, parameters, threshold=0.5):
    probabilities, _ = forward_propagation(np.asarray(X, dtype=float), parameters)
    return (probabilities >= threshold).astype(int)
