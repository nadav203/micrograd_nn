import math

class Value:

    def __init__(self, data, _children=(), _operation=''):
        self.data = data
        self._children = set(_children)
        self._operation = _operation
        self.gradient = 0
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value: data = {self.data}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value(self.data + other.data, _children=(self, other), _operation='+')
        
        def _backward():
            self.gradient += 1.0 * res.gradient
            other.gradient += 1.0 * res.gradient

        res._backward = _backward
        return res
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value(self.data * other.data, _children=(self, other), _operation='*')

        def _backward():
            self.gradient += other.data * res.gradient
            other.gradient += self.data * res.gradient

        res._backward = _backward
        return res
    
    def __rmul__(self, other): 
        # other * self
        return self * other

    def tanh(self):
        other = other if isinstance(other, Value) else Value(other)
        data = (math.exp(self.data * 2) - 1) / (math.exp(self.data * 2) + 1)
        res =  Value(data, _children=(self, ), _operation='tanh')

        def _backward():
            self.gradient += (1 - data**2) * res.gradient

        res._backward = _backward
        return res

    def backward(self):
        topo_graph = []
        visited = set()

        def build_topo(vertex):
            if vertex not in visited:
                visited.add(vertex)
                for child in vertex._children:
                    build_topo(child)
                topo_graph.append(vertex)

        build_topo(self)
        self.gradient = 1
        for vertex in reversed(topo_graph):
            vertex._backward()
        # print(topo_graph)

    

if __name__ == "__main__":
    test_val = Value(5)
    other_val = Value(6)
    add = test_val + other_val
    print('gradient: ', add._backward)
    print(add)
    print(add._children)
    print(add._operation)
    tanh = add.tanh()
    print(tanh)
    print(tanh._children)
    tanh.backward()
    

    