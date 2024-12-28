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
    
if __name__ == "__main__":
    x = [2.0, 3.0]
    n = Neuron(2)
    print(n(x))