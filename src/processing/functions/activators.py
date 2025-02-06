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
        # we substract np.max for numerical stability
        # we use axis 0 because we want to perfom the calculation on the column (r, c)
        # column are the result of Z and row the number of class
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def apply_derivative(self, x):
        return 1
