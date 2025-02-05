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
        sigmoide = self.apply(x)
        return sigmoide * (1 - sigmoide)
