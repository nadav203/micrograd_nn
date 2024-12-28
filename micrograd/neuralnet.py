from engine import Value
import random

class Neuron:

    def __init__(self, input_num: int):
        def value_gen():
            return Value(random.uniform(-1,1))
        
        if input_num <= 0:
            raise ValueError("Number of inputs must be a positive integer")
        self.w = [value_gen() for _ in range(input_num)]
        self.bias = value_gen()
    
    def __call__(self, x: list[float]) -> Value:
        # forward pass
        if len(x) != len(self.w):
            raise ValueError("Input size does not match the number of neuron inputs")
        activation = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.bias
        res = activation.tanh()
        return res
    
class Layer:
    
    def __init__(self, input_num: int, output_num: int):
        if input_num <= 0 or output_num <= 0:
            raise ValueError("input_num and output_num must be positive integers")
        self.neurons = [Neuron(input_num) for _ in range(output_num)]
    
    def __call__(self, x: list[float]):
        return [n(x) for n in self.neurons]

class MLP:

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

if __name__ == "__main__":
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    print(n(x))
