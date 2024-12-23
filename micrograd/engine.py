class Value:

    def __init__(self, data, _children=(), _operation=''):
        self.data = data
        self._children = set(_children)
        self._operation = _operation
    
    def __repr__(self):
        return f"Value: data = {self.data}"
    
    def __add__(self, other):
        return Value(self.data + other.data, (self, other), 'addition')
    
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), 'multiplication')
    

if __name__ == "__main__":
    test_val = Value(5)
    other_val = Value(6)
    add = test_val + other_val
    print(add)
    print(add._children)
    print(add._operation)

    