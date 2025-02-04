from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers
import numpy as np

class Layer:
    learning_rate = 0.05

    def __init__(self, n: int, m: int, n_before: int, c: int, activator: Activations, init: Initializers):
        """
        n_before = n-1
        """
        self.W = init.apply((n, n_before), n_before)
        self.b = init.apply((n, 1), n_before) # 1 because of broadcasting
        self.m = m
        self.n = n
        self.c = c
        self.activator = activator

    def load(self, W: np.array, b: np.array):
        self.W = W
        self.b = b

    def forward(self, A_before: np.array) -> np.array:
        """
        A = A[c-1] with z = A[0]
        Return A[c]
        """
        # print(f"A: {np.shape(A)}")
        # print(f"W: {np.shape(self.W)}")
        # print(f"b: {np.shape(self.b)}")
        self.Z = self.W @ A_before + self.b
        self.A = self.activator.apply(self.Z)
        # print(f"c: {self.c} -> {self.A}")
        return self.A

    def backward_last(self, Y: np.array):
        # print(f"je commence la backward pour {self.c} avec {self.n} neurons")
        # print(f"Y: {np.shape(Y)}")
        # print(f"W: {np.shape(self.W)}")
        # print(f"A: {np.shape(self.A)}")
        self.dZ = self.A - Y
        # print(f"j'ai fini la backward pour {self.c}")

    def backward(self, A_before: np.array, W_after: np.array, dZ_after: np.array, cf=False, Y=None):
        # print(f"je commence la backward pour {self.c} avec {self.n} neurons")
        # print(f"A: {np.shape(self.A)}")
        # print(f"W: {np.shape(self.W)}")
        # print(f"A-1: {np.shape(A_before)}")
        # print(f"dZ-1: {np.shape(dZ_after)}")
        # print(f"W-1: {np.shape(W_after)}")
        if cf:
            self.dZ = self.A - Y
        else:    
            self.dZ = np.transpose(W_after) @ dZ_after * self.A * (1 - self.A)

        self.dW = 1 / self.m * self.dZ @ np.transpose(A_before)
        self.dB = 1 / self.m * np.sum(self.dZ, axis=1, keepdims=True)
        # print(f"j'ai fini la backward pour {self.c}")


    def update_gradient(self):
        """
        We actually do batch descent gradient because we are training threw the entire dataset 
        at the same time (and the gradient are leveraged be divided by m)
        """
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.dB
        return