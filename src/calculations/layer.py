from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers
import numpy as np

class Layer:
    learning_rate = 0.01

    def __init__(self, n: int, m: int, n_before: int, c: int, activator: Activations, init: Initializers):
        self.W = init.apply((n, n_before), n_before)
        self.b = np.zeros((n, 1))
        self.m = m
        self.n = n
        self.c = c
        self.activator = activator

    def load(self, W: np.array, b: np.array):
        self.W = W
        self.b = b

    def forward(self, A_before: np.array):
        self.Z = self.W @ A_before + self.b
        self.A = self.activator.apply(self.Z)
        # print(f"forward propagation faite pour le layer {self.c}")

    def backward_last(self, Y, A_before):
        self.dZ = self.A - Y
        self.__backward(A_before)

    def backward(self, A_before: np.array, W_after: np.array, dZ_after: np.array):
        self.dZ = np.transpose(W_after) @ dZ_after * self.A * (1 - self.A)
        self.__backward(A_before)

    def __backward(self, A_before):
        self.dW = 1 / self.m * self.dZ @ np.transpose(A_before)
        self.dB = 1 / self.m * np.sum(self.dZ, axis=1, keepdims=True)
        # print(f"backward propagation faite pour le layer {self.c}")

    def update_gradient(self):
        """
        We actually do batch descent gradient because we are training threw the entire dataset 
        at the same time (and the gradient are leveraged be divided by m)
        """
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.dB
        return