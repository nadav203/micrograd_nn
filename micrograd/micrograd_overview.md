# MicroGrad System Overview

This document provides a structured explanation of the MicroGrad system, detailing the functionality of `engine.py` and `neuralnet.py`, and how they interact to construct a basic yet functional neural network.

---

## 1. engine.py – Automatic Differentiation and Backpropagation

### Auto-Differentiation (Autograd)
Automatic differentiation is a fundamental concept in modern machine learning, particularly in deep learning frameworks such as TensorFlow and PyTorch. These frameworks automate gradient computation, enabling efficient model training without manual derivative calculations.

In MicroGrad, the `Value` class provides a simplified version of this mechanism by tracking:
- **Data** – The numerical value.
- **Gradient** – The partial derivative of the final output with respect to this value.
- **Operation** – The mathematical operation that generated it (e.g., `+`, `*`, `tanh`).
- **Computation Graph** – References to child nodes involved in the computation.

For example, performing `z = x * y` generates a new `Value` instance (`z`), which maintains a computational relationship with `x` and `y`, allowing for gradient propagation.

### Backpropagation via `backward()`
- Calling `backward()` on the final output (e.g., a loss function) initiates backpropagation.
- The function automatically traverses the computational graph in reverse (using topological sorting) and applies the chain rule to accumulate gradients.
- This mechanism mirrors how mainstream deep learning libraries compute and store gradients during training.