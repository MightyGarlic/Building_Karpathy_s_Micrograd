from math import exp, cosh
import random
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
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = _backward
        
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
          # chain rule for multiplication
          self.grad += other.data * output.grad
          other.grad += self.data * output.grad   
        output._backward = _backward
        
        return output
    
    def tanh(self): 
        numerator = exp(2*self.data) - 1
        denominator = exp(2*self.data) + 1
        output = Value(numerator/denominator, (self, ), 'tanh')

        def _backward():
            # chain rule for tanh
            x = self.data
            self.grad += (1 / cosh(x)**2) * output.grad
        output._backward = _backward

        return output
    
    def backward(self):
        graph = Graph()
        
        list_topological_order = graph.get_topo(self)
        
        self.grad = 1.0 
        for node in reversed(list_topological_order):
            node._backward()


# topological sort    
class Graph: 
    def __init__(self):
        self.topo = []
        self.visited_nodes = set()

    def build_topo(self, vertex):
        if vertex not in self.visited_nodes:
            self.visited_nodes.add(vertex)
            for child in vertex._previous:
                self.build_topo(child)
            self.topo.append(vertex)

    def get_topo(self, vertex):
        self.build_topo(vertex)
        return self.topo
    

class Neuron: 
    def __init__(self, number_inputs):
        self.weight = [Value(random.uniform(-1,1)) for _ in range(number_inputs)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        activation = sum((wi*Value(xi) for wi, xi in zip(self.weight, x)), self.bias)
        output = activation.tanh()
        return output