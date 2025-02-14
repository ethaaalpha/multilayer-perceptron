from abc import ABC, abstractmethod
import numpy as np

class AbtractOptimizer(ABC):
    """Abstract class for gradient descent methods"""
    @abstractmethod
    def getW(self, W, dW):
        pass

    @abstractmethod
    def getB(self, b, dB):
        pass

class SGD(AbtractOptimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate

    def getW(self, W, dW):
        return W - self.learning_rate * dW

    def getB(self, b, dB):
        return b - self.learning_rate * dB

class SGDMomentum(AbtractOptimizer):
    """SGD with Momentum"""
    def __init__(self, learning_rate = 0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.velocity_w = 0
        self.velocity_b = 0
        self.momentum = momentum

    def getW(self, W, dW):
        self.velocity_w = self.momentum * self.velocity_w + self.learning_rate * dW
        return W - self.velocity_w

    def getB(self, b, dB):
        self.velocity_b = self.momentum * self.velocity_b + self.learning_rate * dB
        return b - self.velocity_b

class SGDNesterovMomentum(AbtractOptimizer):
    """SGD with Nesterov Momentum"""
    def __init__(self, learning_rate = 0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.velocity_w = 0
        self.velocity_b = 0
        self.momentum = momentum

    def getW(self, W, dW):
        lookahead = W - self.velocity_w * self.momentum
        self.velocity_w = self.momentum * self.velocity_w + self.learning_rate * dW
        return lookahead - self.velocity_w

    def getB(self, b, dB):
        lookahead = b - self.velocity_b * self.momentum
        self.velocity_b = self.momentum * self.velocity_b + self.learning_rate * dB
        return lookahead - self.velocity_b

class RMSprop(AbtractOptimizer):
    """RMSprop
    0 <= decay_rate <= 1
    """
    def __init__(self, learning_rate = 0.001, decay_rate= 0.9):
        self.learning_rate = learning_rate
        self.velocity_w = 0
        self.velocity_b = 0
        self.decay_rate = decay_rate
        self.epsilon = 1e-15

    def getW(self, W, dW):
        self.velocity_w = self.decay_rate * self.velocity_w + (1 - self.decay_rate)*dW**2
        return W - (self.learning_rate / (self.epsilon + np.sqrt(self.velocity_w)) * dW)

    def getB(self, b, dB):
        self.velocity_b = self.decay_rate * self.velocity_b + (1 - self.decay_rate)*dB**2
        return b - (self.learning_rate / (self.epsilon + np.sqrt(self.velocity_b)) * dB)

    pass