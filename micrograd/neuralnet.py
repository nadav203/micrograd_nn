from engine import Value
import random

class Module:
    """
    Base class for all neural network modules.
    """

    def parameters():
        return []
    
    def zero_gradient(self):
        for param in self.parameters():
            param.grad = 0

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
    
    def parameters(self):
        """
        Returns:
            list[Value]: List containing all weights and the bias.
        """
        
        return self.w + [self.bias]
    
class Layer(Module):
    
    def __init__(self, input_num: int, output_num: int):
        if input_num <= 0 or output_num <= 0:
            raise ValueError("input_num and output_num must be positive integers")
        self.neurons = [Neuron(input_num) for _ in range(output_num)]
    
    def __call__(self, x: list[float]):
        res = [n(x) for n in self.neurons]
        return res[0] if len(res) == 1 else res

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    
class MLP(Module):

    def __init__(self, input_num: int, output_nums: list[int]):
        if input_num <= 0:
            raise ValueError("input_num must be a positive integer")
        if not output_nums:
            raise ValueError("output_nums list cannot be empty")
        if any(out <= 0 for out in output_nums):
            raise ValueError("All layer sizes in output_nums must be positive integers")
        sizes = [input_num] + output_nums
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(output_nums))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

if __name__ == "__main__":
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    print(n(x))
    print(n.parameters())
