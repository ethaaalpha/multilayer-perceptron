from abc import ABC, abstractmethod
from preprocessing.file.dataset import DataSet
from preprocessing.encoders import AbstractEncoder
import numpy as np

class AbstractScaler(ABC):
    @abstractmethod
    def normalize_list(self, data: list[float]) -> list[float]:
        pass

    def scale_data(self, dataset: DataSet, encoder: AbstractEncoder) -> tuple[np.array, np.array]:
        """Return (X_list, Y_list)"""
        X_raw = [col for col in dataset.columns()[1:]]
        X_list = [self.normalize_list(axis) for axis in X_raw]
        Y_list = dataset.column(0)

        return (np.array(X_list), encoder.encode(np.array(Y_list)))

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