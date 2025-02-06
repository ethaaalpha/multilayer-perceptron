from abc import ABC, abstractmethod
import numpy as np

class AbstractLoss(ABC):
    """Representation of a loss function, the mean of the loss is calculated in multilayer"""
    @abstractmethod
    def apply(self, A, Y):
        pass

    @abstractmethod
    def apply_derivative(self, A, Y):
        pass

class BCE(AbstractLoss):
    """Binary Cross Entropy"""
    def apply(self, A, Y):
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    def apply_derivative(self, A, Y):
        return A - Y

class CCE(AbstractLoss):
    """
    Categorical Cross Entropy

    You have to use softmax activation function for OUTPUT layer.
    Y should be hot encoded before.
    """
    def apply(self, A, Y):
        return -np.sum(Y * np.log(A))

    def apply_derivative(self, A, Y):
        return A - Y
