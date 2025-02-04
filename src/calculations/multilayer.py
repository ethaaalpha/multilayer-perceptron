from calculations.layer import Layer
from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers
import numpy as np
from matplotlib import pyplot as pp

class MultiLayer:
    def __init__(self, X: np.array, Y: np.array):
        self.layers: list[Layer] = list()
        self.X = X
        self.Y = Y.reshape(1, -1)
        self.c = 0

        # print(np.shape(X))
        # print(np.shape(Y))

    def add_layer(self, size, activator, init):
        # input layer is implicit
        n_before = self.layers[-1].n if self.c > 0 else len(self.X) # size feature or last layer neurons

        layer = Layer(size, len(self.X[0]), n_before, self.c, activator, init)

        self.layers.append(layer)
        self.c += 1

    def learn(self):
        print(f"il y a {len(self.layers)} layers")
        number_epoch = 100

        for i in range(number_epoch):
            self.__epoch()
            output_layer = self.layers[-1]
            ll = - 1 / output_layer.m * np.sum(self.Y * np.log(output_layer.A) + (1 - self.Y) * np.log(1 - output_layer.A))
            pp.plot(i, ll, 'o')
        pp.show()
            #ici afficher la function logloss

    def __epoch(self):
        # forward propagation
        A_before = self.X
        for layer in self.layers:
            A_before = layer.forward(A_before)

        # backward propagation
        layer_plus: Layer = self.layers[-1]
        layer_minus: Layer = self.layers[-2]
        for i in range(len(self.layers) - 1, -1, -1): # omiting ouput layer
            print(f"layer-1: {layer_minus.c} | layer: {self.layers[i].c} | | layer+1: {layer_plus.c}")
            if i == len(self.layers) - 1:
                self.layers[i].backward(layer_minus.A, None, None, Y=self.Y, cf=True)
            else:
                self.layers[i].backward(layer_minus.A, layer_plus.W, layer_plus.dZ)
            layer_plus = self.layers[i]
            layer_minus = self.layers[i - 2]

        # gradient update / gradient descent
        for layer in self.layers:
            layer.update_gradient()

    def use():
        return
    def load():
        return