from abc import ABC, abstractmethod
import numpy as np

class AbstractActivator(ABC):
    @abstractmethod
    def apply(self, x):
        pass

    @abstractmethod
    def apply_derivative(self, x):
        pass

class Sigmoide(AbstractActivator):
    def apply(self, x):
        return 1 / (1 + np.exp(-x))

    def apply_derivative(self, x):
        """x is the value computed by apply()"""
        return x * (1 - x)

class SoftMax(AbstractActivator):
    def apply(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def apply_derivative(self, x):
        return 1