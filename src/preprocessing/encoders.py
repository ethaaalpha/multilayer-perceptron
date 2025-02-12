from abc import ABC, abstractmethod
import numpy as np

class AbstractEncoder(ABC):
    @abstractmethod
    def encode(self, v):
        pass

class HotEncoder(AbstractEncoder):
    def encode(self, v):
        """HotEncode Y to be usable with softmax"""
        # https://fr.wikipedia.org/wiki/Encodage_one-hot
        classes = np.unique(v)
        result = np.zeros(((len(v), len(classes))))
        for i, value in enumerate(v):
            result[i, np.where(classes == value)] = 1
        return result.T

class Reshape(AbstractEncoder):
    def encode(self, v):
        return np.reshape(v, (1, -1))
