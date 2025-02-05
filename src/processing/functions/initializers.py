from abc import ABC, abstractmethod
import numpy as np

class AbstractInitializer(ABC):
    @abstractmethod
    def generate(self, shape, n):
        pass

class AUTO(AbstractInitializer):
    def generate(self, shape, n):
        return np.random.rand(*shape)

class HE_UNIFORM(AbstractInitializer):
    def generate(self, shape, n):
        return np.random.uniform(-np.sqrt(6. / n), np.sqrt(6. / n), size=shape)

class HE_INIT(AbstractInitializer):
    def generate(self, shape, n):
        return np.random.rand(*shape) * np.sqrt(2. / n)

class XAVIER_NORMAL(AbstractInitializer):
    def generate(self, shape, n):
        return np.random.rand(*shape) * np.sqrt(1. / n)
