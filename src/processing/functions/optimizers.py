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

class GD(AbtractOptimizer):
    """Gradient Descent"""
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate

    def getW(self, W, dW):
        return W - self.learning_rate * dW

    def getB(self, b, dB):
        return b - self.learning_rate * dB

class GDMomentum(AbtractOptimizer):
    """GD with Momentum"""
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

class GDNesterovMomentum(AbtractOptimizer):
    """GD with Nesterov Momentum"""
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
    def __init__(self, learning_rate = 0.001, decay_rate = 0.9):
        self.learning_rate = learning_rate
        self.velocity_w = 0
        self.velocity_b = 0
        self.decay_rate = decay_rate
        self.epsilon = 1e-8

    def getW(self, W, dW):
        self.velocity_w = self.decay_rate * self.velocity_w + (1 - self.decay_rate) * dW**2
        return W - (self.learning_rate / (self.epsilon + np.sqrt(self.velocity_w)) * dW)

    def getB(self, b, dB):
        self.velocity_b = self.decay_rate * self.velocity_b + (1 - self.decay_rate) * dB**2
        return b - (self.learning_rate / (self.epsilon + np.sqrt(self.velocity_b)) * dB)

class Adam(AbtractOptimizer):
    """Adam"""
    def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.velocity_w = 0
        self.velocity_b = 0
        self.momentum_w = 0
        self.momentum_b = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-8
        self.time_w = 0
        self.time_b = 0

    def getW(self, W, dW):
        self.time_w += 1
        self.momentum_w = self.beta_1 * self.momentum_w + (1 - self.beta_1) * dW
        self.velocity_w = self.beta_2 * self.velocity_w + (1 - self.beta_2) * dW**2

        corrected_momentum = self.momentum_w / (1 - self.beta_1 ** self.time_w)
        corrected_velocity = self.velocity_w / (1 - self.beta_2 ** self.time_w)
        return W - (self.learning_rate / (self.epsilon + np.sqrt(corrected_velocity)) * corrected_momentum)

    def getB(self, b, dB):
        self.time_b += 1
        self.momentum_b = self.beta_1 * self.momentum_b + (1 - self.beta_1) * dB
        self.velocity_b = self.beta_2 * self.velocity_b + (1 - self.beta_2) * dB**2

        corrected_momentum = self.momentum_b / (1 - self.beta_1 ** self.time_b)
        corrected_velocity = self.velocity_b / (1 - self.beta_2 ** self.time_b)
        return b - (self.learning_rate / (self.epsilon + np.sqrt(corrected_velocity)) * corrected_momentum)
