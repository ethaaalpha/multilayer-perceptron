from processing.functions.activators import AbstractActivator, Sigmoide
from processing.functions.initializers import AbstractInitializer, AUTO
from processing.functions.optimizers import AbtractOptimizer, GradientDescent
from dataclasses import dataclass
import numpy as np

@dataclass
class LayerData:
    n: int
    n_before: int
    m: int
    c: int
    activator: AbstractActivator = Sigmoide()
    initializer: AbstractInitializer = AUTO()
    optimizer: AbtractOptimizer = GradientDescent(0.01)

    def generate_weights(self) -> tuple[np.array, np.array]:
        W = self.initializer.generate((self.n, self.n_before), self.n)
        b = np.zeros((self.n, 1))
        return (W, b)

class Layer:
    def __init__(self, data: LayerData):
        self.data = data
        if (self.data.c > 0):
            self.W, self.b = data.generate_weights()

    def forward(self, A_before: np.array):
        self.Z = self.W @ A_before + self.b
        self.A = self.data.activator.apply(self.Z)

    def backward_last(self, Y, A_before):
        self.dZ = self.A - Y
        self.__backward(A_before, self.data.m)

    def backward(self, A_before: np.array, W_after: np.array, dZ_after: np.array):
        self.dZ = np.transpose(W_after) @ dZ_after * self.A * (1 - self.A)
        self.__backward(A_before, self.data.m)

    def __backward(self, A_before, m):
        self.dW = 1 / m * self.dZ @ np.transpose(A_before)
        self.dB = 1 / m * np.sum(self.dZ, axis=1, keepdims=True)

    def update_gradient(self):
        """
        We actually do batch descent gradient because we are training threw the entire dataset 
        at the same time (and the gradient are leveraged be divided by m)
        """
        self.W = self.data.optimizer.getW(self.W, self.dW)
        self.b = self.data.optimizer.getB(self.b, self.dB)
        return