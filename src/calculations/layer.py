from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers
import numpy as np

class Layer:
    learning_rate = 0.1

    def __init__(self, n: int, m: int, n_before: int, activator: Activations, init: Initializers):
        """
        n_before = n-1
        """
        self.W = np.fromfunction(init.apply, (n, n_before))
        self.b = np.fromfunction(init.apply, (n, n_before))
        self.m = m
        self.activator = activator

    def load(self, W: np.array, b: np.array):
        self.W = W
        self.b = b

    def forward(self, A: np.array) -> np.array:
        """
        A = A[c-1] with z = A[0]
        Return A[c]
        """
        self.Z = self.W @ A + self.b
        self.A = self.activator.apply(self.Z)

        return self.A

    def backward_last(self, Y: np.array):
        self.dZ = self.A - Y

    def backward(self, A_before: np.array, dZ_before: np.array, W_before: np.array):
        self.dZ = np.transpose(W_before) @ dZ_before * (self.A * (1 - self.A))

        self.dB = 1 / self.m * np.sum(self.dZ)b
        self.dW = 1 / self.m * self.dZ * A_before


    def update_gradient(self):
        """
        We actually do batch descent gradient because we are training threw the entire dataset 
        at the same time (and the gradient are leveraged be divided by m)
        """
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.dB
        return