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

    def preprocessing(self, Y) -> np.array:
        return Y

class BCE(AbstractLoss):
    """Binary Cross Entropy"""
    def __init__(self):
        super().__init__("Binary Cross Entropy")

    def apply(self, A, Y):
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    def apply_derivative(self, A, Y):
        return A - Y

    def preprocessing(self, Y):
        return np.reshape(Y, 1, -1)

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

    def preprocessing(self, Y):
        return self.___hot_encode(Y)

    def ___hot_encode(self, Y):
        """HotEncode Y to be usable with softmax"""
        # https://fr.wikipedia.org/wiki/Encodage_one-hot
        classes = np.unique(Y)
        result = np.zeros(((len(Y), len(classes))))
        for i, value in enumerate(Y):
            result[i, np.where(classes == value)] = 1
        return result.T