from processing.functions.activators import AbstractActivator, Sigmoide
from processing.functions.initializers import AbstractInitializer, Auto
from processing.functions.optimizers import AbtractOptimizer, GradientDescent
from processing.functions.losses import AbstractLoss, BCE
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

@dataclass
class LayerData:
    n: int
    c: int
    n_before: int = 1
    m: int = 1
    activator: AbstractActivator = Sigmoide()
    initializer: AbstractInitializer = Auto()
    optimizer: AbtractOptimizer = GradientDescent(0.03)

    def generate_weights(self) -> tuple[np.array, np.array]:
        W = self.initializer.generate((self.n, self.n_before), self.n)
        b = np.zeros((self.n, 1))
        return (W, b)

class AbstractLayer(ABC):
    def __init__(self, layerdata: LayerData):
        self.data: LayerData = layerdata

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass

class InputLayer(AbstractLayer):
    """InputLayer is just a 'placeholder'"""
    def __init__(self, size, c):
        super().__init__(LayerData(size, c))

    def forward(self):
        return

    def backward(self):
        return

class HiddenLayer(AbstractLayer):
    def __init__(self, size, c, n_before, m, **kwargs):
        super().__init__(LayerData(size, c, n_before, m, **kwargs))
        self.W, self.b = self.data.generate_weights()

    def forward(self, A_before: np.array):
        self.Z = self.W @ A_before + self.b
        self.A = self.data.activator.apply(self.Z)

    def backward(self, A_before: np.array, W_after: np.array, dZ_after: np.array):
        self.dZ = np.transpose(W_after) @ dZ_after * self.data.activator.apply_derivative(self.A)
        self.compute_gradients(A_before, self.data.m)

    def compute_gradients(self, A_before: np.array, m):
        self.dW = 1 / m * self.dZ @ np.transpose(A_before)
        self.dB = 1 / m * np.sum(self.dZ, axis=1, keepdims=True)

    def update_gradients(self):
        """
        We actually do mini_batch / batch descent gradient because we are training threw the x elements of the dataset 
        at the same time (and the gradient are leveraged be divided by m at the end of the epoch)
        """
        self.W = self.data.optimizer.getW(self.W, self.dW)
        self.b = self.data.optimizer.getB(self.b, self.dB)
        return

class OutputLayer(HiddenLayer):
    def backward(self, Y: np.array, A_before: np.array, loss: AbstractLoss):
        self.dZ = loss.apply_derivative(self.A, Y)
        self.compute_gradients(A_before, self.data.m)
