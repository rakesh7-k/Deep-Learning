import numpy as np


def mcp_neuron(inputs, weights, threshold):
    """Return the binary output of a McCulloch-Pitts neuron."""
    input_vector = np.asarray(inputs, dtype=float)
    weight_vector = np.asarray(weights, dtype=float)

    if input_vector.shape != weight_vector.shape:
        raise ValueError(
            f"Input shape {input_vector.shape} does not match weight shape {weight_vector.shape}."
        )

    weighted_sum = np.dot(input_vector, weight_vector)
    return 1 if weighted_sum >= threshold else 0


def AND_gate(x1, x2):
    """Logical AND implemented with MCP neuron weights [1, 1], threshold 2."""
    return mcp_neuron(np.array([x1, x2]), np.array([1.0, 1.0]), 2)


def OR_gate(x1, x2):
    """Logical OR implemented with MCP neuron weights [1, 1], threshold 1."""
    return mcp_neuron(np.array([x1, x2]), np.array([1.0, 1.0]), 1)


def NOT_gate(x):
    """Logical NOT implemented with MCP neuron weights [-1], threshold 0."""
    return mcp_neuron(np.array([x]), np.array([-1.0]), 0)


def NOR_gate(x1, x2):
    """Logical NOR implemented with MCP neuron weights [-1, -1], threshold 0."""
    return mcp_neuron(np.array([x1, x2]), np.array([-1.0, -1.0]), 0)


def XOR_gate(x1, x2):
    """XOR implemented explicitly for demonstration purposes.

    A single McCulloch-Pitts neuron cannot represent XOR because the function is not
    linearly separable. This version is included to explain the resulting truth table
    without duplicating the threshold logic in the notebook.
    """
    return 1 if x1 != x2 else 0


def gate_truth_table(logic_function, binary_inputs):
    """Return a list of (input_1, input_2, output) tuples for a two-input gate."""
    result = []
    for x1 in binary_inputs:
        for x2 in binary_inputs:
            result.append((x1, x2, logic_function(x1, x2)))
    return result


def single_input_truth_table(logic_function, binary_inputs):
    """Return a list of (input, output) tuples for a one-input gate."""
    return [(value, logic_function(value)) for value in binary_inputs]


__all__ = [
    "mcp_neuron",
    "AND_gate",
    "OR_gate",
    "NOT_gate",
    "NOR_gate",
    "XOR_gate",
    "gate_truth_table",
    "single_input_truth_table",
]
