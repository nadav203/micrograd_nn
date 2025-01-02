from engine import Value
import random

class Module:
    """
    Base class for all neural network modules.
    """

    def parameters(self):
        return []
    
    def zero_gradient(self) -> None:
        for param in self.parameters():
            param.gradient = 0

class Neuron(Module):
    """
    Represents a single neuron in a neural network layer.
    Each neuron has its own weights and bias.
    """

    def __init__(self, input_num: int):
        """
        Initializes the neuron with a specified number of inputs.
        
        Args:
            input_num (int): The number of input connections to the neuron.
        """

        def value_gen():
            return Value(random.uniform(-1,1))
        
        if input_num <= 0:
            raise ValueError("Number of inputs must be a positive integer")
        self.w = [value_gen() for _ in range(input_num)]
        self.bias = value_gen()
    
    def __call__(self, x: list[float]) -> Value:
        """
        Performs the forward pass for the neuron.
        Computes the weighted sum of inputs, adds bias, and applies the tanh activation function.
        
        Args:
            x (list[float]): Input values to the neuron.
        
        Returns:
            Value: The activated output of the neuron.
        """

        if len(x) != len(self.w):
            raise ValueError("Input size does not match the number of neuron inputs")
        activation = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.bias
        res = activation.tanh()
        return res
    
    def parameters(self) -> list[Value]:
        """
        Returns:
            list[Value]: List containing all weights and the bias.
        """

        return self.w + [self.bias]
    
class Layer(Module):
    """
    Represents a layer in the neural network, consisting of multiple neurons.
    """

    def __init__(self, input_num: int, output_num: int):
        """
        Initializes the layer with a specified number of inputs and outputs.
        
        Args:
            input_num (int): Number of inputs to a single nueron.
            output_num (int): Number of neurons in the layer.
        """

        if input_num <= 0 or output_num <= 0:
            raise ValueError("input_num and output_num must be positive integers")
        self.neurons = [Neuron(input_num) for _ in range(output_num)]
    
    def __call__(self, x: list[float]):
        """
        Performs the forward pass for the layer.
        Passes the input through each neuron in the layer.
        
        Args:
            x (list[float]): Input values to the layer.
        
        Returns:
            Value or list[Value]: If the layer has one neuron, returns its output directly.
                Otherwise, returns a list of outputs from all neurons.
        """

        res = [n(x) for n in self.neurons]
        return res

    def parameters(self) -> list[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]
    
class MLP(Module):
    """
    Represents a Multi-Layer Perceptron (MLP) neural network.
    Composed of multiple layers stacked sequentially.
    """

    def __init__(self, input_num: int, output_nums: list[int]):
        """
        Initializes the MLP with a specified input size and layer configurations.
        
        Args:
            input_num (int): Number of input features.
            output_nums (list[int]): List specifying the number of neurons in each layer.
        """

        if input_num <= 0:
            raise ValueError("input_num must be a positive integer")
        if not output_nums:
            raise ValueError("output_nums list cannot be empty")
        if any(out <= 0 for out in output_nums):
            raise ValueError("All layer sizes in output_nums must be positive integers")
        sizes = [input_num] + output_nums
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(output_nums))]

    def __call__(self, x: list[float]) -> list[Value]:
        """
        Performs the forward pass through the entire MLP.
        Sequentially passes the input through each layer.
        
        Args:
            x (list[float]): Input data to the MLP.
        
        Returns:
            list[Value]: The output from the final layer.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> list[Value]:
        return [param for layer in self.layers for param in layer.parameters()]

if __name__ == "__main__":
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    print(n(x))
    print(n.parameters())
    n = Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)
