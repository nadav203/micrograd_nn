import random
import math
import matplotlib.pyplot as plt
from engine import Value
from neuralnet import MLP, Module, Neuron, Layer

random.seed(42)


def generate_data(n_samples: int) -> list[tuple[float, float]]:
    """
    Generates a simple dataset based on a sine function with some noise.

    Args:
        n_samples (int): Number of data points to generate.

    Returns:
        List[Tuple[float, float]]: Generated dataset as (x, y) pairs.
    """
    data = []
    for _ in range(n_samples):
        x = random.uniform(-math.pi, math.pi)
        y = math.sin(x) + random.gauss(0, 0.1)  # sin function with noise
        data.append((x, y))
    return data


def main():
    data = generate_data(n_samples=100)
    # Initialize model
    model = MLP(input_num=1, output_nums=[10, 10, 1])


if __name__ == "__main__":
    main()
