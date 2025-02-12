from abc import ABC, abstractmethod
import numpy as np

class AbstractLoss(ABC):
    """Representation of a loss function, the mean of the loss is calculated in multilayer"""
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply(self, A, Y):
        pass

    @abstractmethod
    def apply_derivative(self, A, Y):
        pass

    @abstractmethod
    def accuracy(self, A, Y):
        pass

class BCE(AbstractLoss):
    """Binary Cross Entropy"""
    def __init__(self):
        super().__init__("Binary Cross Entropy")

    def apply(self, A, Y):
        epsilon = 1e-15 # avoid log(0)
        A = np.clip(A, epsilon, 1 - epsilon)
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    def apply_derivative(self, A, Y):
        return A - Y

    def accuracy(self, A, Y):
        y_pred = np.astype((A > 0.5), int)
        y_true = np.astype(Y, int)
        return np.mean(y_pred == y_true)

class CCE(AbstractLoss):
    """
    Categorical Cross Entropy

    You have to use softmax activation function for OUTPUT layer.
    Y should be hot encoded before.
    """
    def __init__(self):
        super().__init__("Categorical Cross Entropy")

    def apply(self, A, Y):
        return -np.sum(Y * np.log(A))

    def apply_derivative(self, A, Y):
        return A - Y

    def accuracy(self, A, Y):
        y_pred = np.argmax(A, axis=0)
        y_true = np.argmax(Y, axis=0)
        return np.mean(y_pred == y_true)

def from_str(name) -> AbstractLoss:
    match (name):
        case "Binary Cross Entropy":
            return BCE()
        case "Categorical Cross Entropy":
            return CCE()