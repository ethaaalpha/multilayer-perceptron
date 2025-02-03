from calculations.layer import Layer
from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers
import numpy as np

class MultiLayer:
    def __init__(self, X: np.array, Y: np.array):
        self.layers: list[Layer] = list()
        self.X = X
        self.Y = Y
        self.c = 0

        print(np.shape(X))
        print(np.shape(Y))

    def add_layer(self, size, activator, init):
        # input layer is implicit
        n_before = self.layers[-1].n if self.c > 0 else len(self.X) # size feature or last layer neurons

        layer = Layer(size, len(self.X), n_before, self.c, activator, init)

        self.layers.append(layer)
        self.c += 1

    def learn(self):
        # number_epoch = 100
        # for _ in range(number_epoch):
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
        for layer in list(reversed(self.layers[:-1])): # omiting ouput layer
            layer.backward(layer_before.A, layer_before.dZ, layer_before.W)

        # gradient update / gradient descent
        for layer in list(self.layers[-1]):
            layer.update_gradient()

    def use():
        return
    def load():
        return