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
        self.m = X.shape[1]
        self.c = 0

    def add_layer(self, size, activator=Activations.SIGMOIDE, init=Initializers.AUTO):
        n_before = 1

        if (self.c > 0):
            n_before = self.layers[-1].n

        self.layers.append(Layer(size, self.m, n_before, self.c, activator, init))
        print(f"le layer {self.c} a été ajouté !")
        self.c += 1

    def __epoch(self):
        # forward propagation
        for i in range(1, self.c):
            A_before = self.X if i == 1 else self.layers[i - 1].A
            self.layers[i].forward(A_before)

        # backward propagation
        for i in reversed(range(1, self.c)): # ommiting output layer
            layer = self.layers[i]
            A_before = self.X if i == 1 else self.layers[i - 1].A

            if (i == self.c - 1): # mean we are at final layer
                layer.backward_last(self.Y, A_before)
            else:
                layer_plus = self.layers[i + 1]
                layer.backward(A_before, layer_plus.W, layer_plus.dZ)

        # gradient update / gradient descent
        for i in range(1, self.c):
            self.layers[i].update_gradient()

    def learn(self):
        print(f"il y a {len(self.layers)} layers")
        number_epoch = 100

        ll_l = list()
        i_l = list()
        for i in range(number_epoch):
            self.__epoch()
            output_layer = self.layers[-1]
            ll = - 1 / output_layer.m * np.sum(self.Y * np.log(output_layer.A) + (1 - self.Y) * np.log(1 - output_layer.A))
            i_l.append(i)
            ll_l.append(ll)
        pp.plot(i_l, ll_l)
        pp.show()
            #ici afficher la function logloss

    def load():
        return