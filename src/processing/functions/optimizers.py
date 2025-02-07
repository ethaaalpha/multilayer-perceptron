from abc import ABC, abstractmethod

class AbtractOptimizer(ABC):
    @abstractmethod
    def getW(self, W, dW):
        pass

    @abstractmethod
    def getB(self, b, dB):
        pass

class GradientDescent(AbtractOptimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def getW(self, W, dW):
        return W - self.learning_rate * dW

    def getB(self, b, dB):
        return b - self.learning_rate * dB
