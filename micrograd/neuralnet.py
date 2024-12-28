from engine import Value
import random

class Neuron:

    def __init__(self, input_num):
        def value_gen():
            return Value(random.uniform(-1,1))
        
        self.w = [value_gen() for _ in range(input_num)]
        self.bias = value_gen()
    
    def __call__(self, x):
        # forward pass
        activation = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.bias
        res = activation.tanh()
        return res
    
class Layer:
    
    def __init__(self, input_num, output_num):
        self.neurons = [Neuron(input_num) for _ in range(output_num)]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons]

class MLP:

    def __init__(self, input_num, output_nums):
        size = [input_num] + output_nums
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(output_nums))]

if __name__ == "__main__":
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
