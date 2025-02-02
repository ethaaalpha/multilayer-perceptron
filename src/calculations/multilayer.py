from layer import Layer
import numpy as np

class MultiLayer:
    def __init__(self, X: np.array, Y: np.array):
        self.layers: list[Layer] = list()
        self.X = X
        self.Y = Y

    def add_layer():
        return

    def learn(self):
        number_epoch = 100
        for _ in range(number_epoch):
            self.__epoch()
            #ici afficher la function logloss

    def __epoch(self):
        # forward propagation
        A_before = self.X
        for layer in self.layers:
            A_before = layer.forward(A_before)

        # backward propagation
        layer_before = self.layers[-1]
        layer_before.backward_last(self.Y)
        for layer in list(reversed(self.layers[1:-1])): # omiting input and ouput layer
            layer.backward(layer_before.A, layer_before.dZ, layer_before.W)

        # gradient update / gradient descent
        for layer in list(self.layers[1:-1]):
            layer.update_gradient()

    def use():
        return
    def load():
        return