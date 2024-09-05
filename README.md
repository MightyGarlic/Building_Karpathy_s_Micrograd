*Backprobagation is just a recursive application of chain rule backwards through the computational graph.*

### Why build Micrograd? 

In 2020, Andrej Karpathy implemented a small, scalar valued Autograd engine inspired by [PyTorch](https://github.com/pytorch/pytorch). To develop a better understanding of neural networks and backpropagation, we'll follow in his footsteps! We will implement backpropagation over dynamically built DAG. Let's gooo!

### What do we need? 

- A basic understanding of derivatives 
- List of [standard python operators](https://docs.python.org/3/library/operator.html) that we will overload

### Want more? 

- Original [micrograd implementation](https://github.com/karpathy/micrograd)
- Here's [Andrej Karpathy's full syllabus](https://karpathy.ai/zero-to-hero.html) which includes building makemore, and building GPT from scratch