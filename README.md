### Why build Micrograd? 

In 2020, Andrej Karpathy implemented a small, scalar valued Autograd engine inspired by PyTorch. To develop a better understanding of neural networks and backpropagation, we will follow in his footsteps! We will implement backpropagation over dynamically built DAG. Let's gooo!

### What do we need? 

- A basic understanding of derivatives 
- List of [standard python operators(https://docs.python.org/3/library/operator.html)] that we can overload
