from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers
import numpy as np

class Layer:
    learning_rate = 0.1

    def __init__(self, n: int, m: int, n_before: int, c: int, activator: Activations, init: Initializers):
        """
        n_before = n-1
        """
        self.W = np.fromfunction(init.apply, (n, n_before))
        self.b = np.fromfunction(init.apply, (n, 1)) # 1 because of broadcasting
        self.m = m
        self.n = n
        self.c = c
        self.activator = activator

    def load(self, W: np.array, b: np.array):
        self.W = W
        self.b = b

    def forward(self, A: np.array) -> np.array:
        """
        A = A[c-1] with z = A[0]
        Return A[c]
        """
        # print(np.shape(self.W))
        # print(np.shape(A))
        # print(np.shape(self.b))`
        self.Z = self.W @ A + self.b
        self.A = self.activator.apply(self.Z)

        return self.A

    def backward_last(self, Y: np.array):
        self.dZ = self.A - Y
        print(f"j'ai fini la backward pour {self.c}")

    def backward(self, A_before: np.array, dZ_before: np.array, W_before: np.array):
        print(np.shape(A_before))
        print(np.shape(dZ_before))
        print(np.shape(W_before))
        self.dZ = np.transpose(W_before) @ dZ_before * self.A * (1 - self.A)

        self.dB = 1 / self.m * np.sum(self.dZ)
        self.dW = 1 / self.m * self.dZ * A_before
        print(f"j'ai fini la backward pour {self.c}")


    def update_gradient(self):
        """
        We actually do batch descent gradient because we are training threw the entire dataset 
        at the same time (and the gradient are leveraged be divided by m)
        """
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.dB
        return