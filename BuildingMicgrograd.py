import math
import numpy as np

# Check out the python operator documentation to learn which standard operators you can overload: https://docs.python.org/3/library/operator.html
   
class Value:
    def __init__(self, data, _children=(), _operation='', label=''):
        self.data = data
        self._previous = set(_children)
        self._operation = _operation
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), '+')
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        return output