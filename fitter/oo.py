from scipy.stats import norm
import numpy as np

class Function:
    def __add__(self, other):
        return Sum(self, other)
    
    def nll(self, args, x):
        y = self(x, *args)
        return -np.sum(np.log(y))

class Gauss(Function):
    def __init__(self):
        self.args = 2
    def __call__(self, x, mu, sigma):
        return norm.pdf(x, mu, sigma)

class Linear(Function):
    def __init__(self):
        self.args = 0
    def __call__(self, x):
        return x*0 + 1/20

class Sum(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.args = a.args + b.args + 1
        
    def __call__(self, x, *args):
        a_vals = self.a(x, *args[:self.a.args])
        b_vals = self.b(x, *args[self.a.args:self.a.args+self.b.args])
        mix = args[self.args - 1]
        return a_vals*(1-mix) + b_vals*mix