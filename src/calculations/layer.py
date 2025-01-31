from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers
import numpy as np

class Layer:
    learning_rate = 0.1

    def __init__(self, n: int, n_before: int, activator: Activations, init: Initializers):
        """
        n_before = n-1
        """
        self.W = np.fromfunction(init.apply, (n, n_before))
        self.b = np.fromfunction(init.apply, (n, n_before))
        self.activator = activator

    def forward(self, A: np.array) -> tuple[np.array, np.array]:
        """
        A = A[c-1] with X = A[0]
        """
        self.Z = self.W @ A + self.b
        self.A = self.activator.apply(self.Z)

        return (self.Z, self.A)

    def backward(self, A_before: np.array, last_layer=False):
        """
        A+before = A[c-1] representing the result of the predecent layer, if it's the last layer then y
        """
        dZ: np.array

        if (last_layer):
            dZ = self.A - A_before
        else:
            dZ = 0