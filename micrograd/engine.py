class Value:

    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return f"Value: data = {self.data}"
    
    def __add__(self, other):
        return Value(self.data + other.data)
    
    def __mul__(self, other):
        return Value(self.data * other.data)
    

if __name__ == "__main__":
    test_val = Value(5)
    other_val = Value(6)
    print(test_val * other_val)
    