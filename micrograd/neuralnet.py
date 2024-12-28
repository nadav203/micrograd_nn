from micrograd.engine import Value
import random

class Neuron:

    def __init__(self, input_num):
        def value_gen():
            return Value(random.uniform(-1,1))
        
        self.w = [value_gen() for _ in range(input_num)]
        self.bias = value_gen()