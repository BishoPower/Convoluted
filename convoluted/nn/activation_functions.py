import numpy as np
from .functions import *
from .layers import Function


class Sigmoid(Function):
    def forward(self, X):
        return sigmoid(X)

    def backward(self, dY):
        return dY * self.grad["X"]

    def local_grad(self, X):
        grads = {"X": sigmoid_prime(X)}
        return grads


class ReLU(Function):
    def forward(self, X):
        return relu(X)

    def backward(self, dY):
        return dY * self.grad["X"]

    def local_grad(self, X):
        grads = {"X": relu_prime(X)}
        return grads

class Tanh(Function):
    def forward(self, X):
        return tanh(X)

    def backward(self, dY):
        return dY * self.grad["X"]

    def local_grad(self, X):
        grads = {"X": tanh_prime(X)}
        return grads
