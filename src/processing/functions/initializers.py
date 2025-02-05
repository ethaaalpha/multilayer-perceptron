from abc import ABC, abstractmethod
import numpy as np

class AbstractInitializer(ABC):
    @abstractmethod
    def generate(self, shape, n):
        pass

class Auto(AbstractInitializer):
    def generate(self, shape, n):
        return np.random.rand(*shape)

class He_Uniform(AbstractInitializer):
    def generate(self, shape, n):
        return np.random.uniform(-np.sqrt(6. / n), np.sqrt(6. / n), size=shape)

class He_Normal(AbstractInitializer):
    def generate(self, shape, n):
        return np.random.rand(*shape) * np.sqrt(2. / n)

class Xavier_Normal(AbstractInitializer):
    def generate(self, shape, n):
        return np.random.rand(*shape) * np.sqrt(1. / n)
