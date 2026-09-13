# McCulloch-Pitts Neuron

A compact introduction to the earliest artificial neuron model. This experiment demonstrates how a single threshold-based unit can implement basic logic gates such as AND, OR, NOT, and NOR, and why XOR cannot be solved by one such neuron.

## Overview

The McCulloch-Pitts neuron is a simplified computational model inspired by biological neurons. It receives binary inputs, multiplies them by weights, sums the results, and compares the output with a threshold.

## Core Idea

The neuron behaves according to the rule:

$$
Z = \sum_i x_i w_i
$$

$$
output =
\begin{cases}
1 & \text{if } Z \geq T \\
0 & \text{if } Z < T
\end{cases}
$$

This rule allows the model to represent several simple logic functions when the function is linearly separable.

## Architecture

```text
Inputs -> Weighted Sum -> Threshold -> Output
```

## Mathematical Foundation

- AND: weights = [1, 1], threshold = 2
- OR: weights = [1, 1], threshold = 1
- NOT: weights = [-1], threshold = 0
- NOR: weights = [-1, -1], threshold = 0

## Dataset

This experiment uses the full set of binary truth-table inputs:

- (0, 0)
- (0, 1)
- (1, 0)
- (1, 1)

## Implementation

The reusable logic is implemented in the Python source file in this folder. The notebook imports the functions and demonstrates the gate behavior with tables and direct visual reasoning.

## Experiments

- Verify AND, OR, NOT, and NOR truth tables.
- Explain why XOR is not linearly separable for a single MCP neuron.
- Connect the threshold rule to the idea of decision boundaries.

## Results

The implemented gates match their expected boolean behavior for all binary input combinations.

## Technical Insights

A single MCP neuron is useful for understanding the basic decision rule behind neural networks, but it is restricted to simple threshold logic. XOR is the classic example showing that one linear threshold unit cannot solve a non-linear pattern.

## Limitations

- Only binary inputs and outputs are supported.
- It is a conceptual model, not a modern trainable neuron.
- It cannot solve XOR with a single neuron.

## Possible Improvements

- Add a chart of the decision regions.
- Extend to hidden-layer logic combinations.
- Compare with perceptron learning behavior.

## Reproducibility

Run the notebook next to the Python module. The notebook imports the implementation directly instead of rewriting the logic.

## Technologies

- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook
