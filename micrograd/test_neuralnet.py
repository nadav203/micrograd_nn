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

def mean_squared_error(predictions: list[Value], targets: list[float]) -> Value:
    """
    Computes the mean squared error loss.

    Args:
        predictions (List[Value]): Predicted values from the network.
        targets (List[float]): Actual target values.

    Returns:
        Value: The MSE loss.
    """
    loss = Value(0.0)
    for pred, target in zip(predictions, targets):
        diff = pred - target
        loss += diff * diff
    return loss * (1.0 / len(predictions))


def train(model: MLP, data: list[tuple[float, float]], epochs: int = 1000, lr: float = 0.01) -> list[float]:
    """
    Trains the MLP on the provided dataset.

    Args:
        model (MLP): The neural network model to train.
        data (List[Tuple[float, float]]): The training dataset.
        epochs (int): Number of training iterations.
        lr (float): Learning rate.

    Returns:
        List[float]: List of loss values over epochs.
    """
    losses = []
    for epoch in range(epochs):
        # Shuffle data for stochasticity
        random.shuffle(data)
        epoch_loss = Value(0.0)
        predicted_values = []
        target_values = []
        for x, y in data:
            predicted_values.append(model([x])[0])
            target_values.append(Value(y))
        epoch_loss = mean_squared_error(predicted_values, target_values)
        losses.append(epoch_loss.data)
        # Backward pass
        epoch_loss.backward()
        # Update parameters
        for param in model.parameters():
            param.data -= lr * param.gradient
        # Zero gradients
        model.zero_gradient()
        if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss.data:.4f}")
    return losses


def plot_loss(losses: list, lr: float, layers: list[int], epochs: int) -> None:
    """
    Plots the training loss over epochs, including learning rate and model info in the title.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    # Include learning rate, MLP architecture, and number of epochs in the title
    plt.title(
        f'Training Loss Over {epochs} Epochs\n'
        f'Learning Rate: {lr}, MLP Layers: {layers}'
    )
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(model: MLP, 
                     data: list[tuple[float, float]], 
                     lr: float, 
                     layers: list[int], 
                     epochs: int) -> None:
    """
    Plots the model's predictions against actual data, 
    with learning rate, model info, and epochs included in the title.
    """
    x_vals = [x for x, _ in data]
    y_true = [y for _, y in data]
    y_pred = [model([x])[0].data for x in x_vals]

    # Sort for better visualization
    sorted_indices = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
    x_sorted = [x_vals[i] for i in sorted_indices]
    y_true_sorted = [y_true[i] for i in sorted_indices]
    y_pred_sorted = [y_pred[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.scatter(x_sorted, y_true_sorted, label='Actual Data', color='blue', s=10)
    plt.plot(x_sorted, y_pred_sorted, label='Model Prediction', color='red')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    # Include learning rate, MLP architecture, and number of epochs in the title
    plt.title(
        f'Model Predictions vs. Actual Data\n'
        f'Learning Rate: {lr}, MLP Layers: {layers}, Trained for {epochs} Epochs'
    )
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Hyperparameters / model config
    lr = 0.9
    epochs = 1000
    layer_sizes = [10, 1]  # MLP hidden layers and output layer

    data = generate_data(n_samples=100)
    model = MLP(input_num=1, output_nums=layer_sizes)

    print("Starting training...")
    losses = train(model, data, epochs=epochs, lr=lr)

    plot_loss(losses, lr=lr, layers=layer_sizes, epochs=epochs)
    plot_predictions(model, data, lr=lr, layers=layer_sizes, epochs=epochs)

if __name__ == "__main__":
    main()