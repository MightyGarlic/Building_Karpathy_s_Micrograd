from math import exp, cosh
import numpy as np

# Check out the python operator documentation to learn which standard operators you can overload: https://docs.python.org/3/library/operator.html
   
class Value:
    def __init__(self, data, _children=(), _operation='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None 
        self._previous = set(_children)
        self._operation = _operation
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # chain rule for addition
            # local derivative * global derivative of output
            self.grad = 1.0 * output.grad
            other.grad = 1.0 * output.grad
        output._backward = _backward
        
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
          # chain rule for multiplication
          self.grad = other.data * output.grad
          other.grad = self.data * output.grad   
        output._backward = _backward
        
        return output
    
    def tanh(self): 
        numerator = exp(2*self.data) - 1
        denominator = exp(2*self.data) + 1
        output = Value(numerator/denominator, (self, ), 'tanh')

        def _backward():
            # chain rule for tanh
            x = self.data
            self.grad = (1 / cosh(x)**2) * output.grad
        output._backward = _backward

        return output