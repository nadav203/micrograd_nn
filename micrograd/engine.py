import math

class Value:

    def __init__(self, data, _children=(), _operation=''):
        self.data = data
        self._children = set(_children)
        self._operation = _operation
        self.gradient = 0
    
    def __repr__(self):
        return f"Value: data = {self.data}"
    
    def __add__(self, other):
        return Value(self.data + other.data, _children=(self, other), _operation='+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, _children=(self, other), _operation='*')
    
    def tanh(self):
        data = (math.exp(self.data * 2) - 1) / (math.exp(self.data * 2) + 1)
        return Value(data, _children=(self, ), _operation='tanh')

    

if __name__ == "__main__":
    test_val = Value(5)
    other_val = Value(6)
    add = test_val + other_val
    print(add)
    print(add._children)
    print(add._operation)
    tanh = add.tanh()
    print(tanh)
    print(tanh._children)

    