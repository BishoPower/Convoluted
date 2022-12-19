import numpy as np

#Activation Functions and their derivatives

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return x * (x > 0)


def relu_prime(x):
    return 1 * (x > 0)
    

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_prime(x):
    return 1-((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))^2)

def swish(x):
    return x * sigmoid(x)

def swish_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

class Rearranger:
    def __init__(self, thing):
        self.thing = thing

    def permute(self, *axis):
        self.thing = np.transpose(self.thing, axis)