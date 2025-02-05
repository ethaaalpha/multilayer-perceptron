from abc import ABC, abstractmethod
import numpy as np

class AbstractScaler(ABC):
    @abstractmethod
    def normalize_list(self, data: list[float]) -> list[float]:
        pass

class MinMax(AbstractScaler):
    """Normalization of the values between [0, 1]"""

    def normalize_list(self, data):
        x_min, x_max = min(data), max(data)
        return [self.__normalize(x_min, x_max, x) for x in data]

    def __normalize(self, min, max, x):
        return (x - min) / (max - min)

class Z_score(AbstractScaler):
    """Standardization of the values using z-score algorithm"""

    def normalize_list(self, data):
        mean = np.sum(data) / len(data)
        standard_deviation = np.sqrt(np.sum((data - mean)**2) / len(data))
        return [self.__normalize(x, mean, standard_deviation).item() for x in data]
    
    def __normalize(self, x, mean, standard_deviation):
        return (x - mean) / standard_deviation