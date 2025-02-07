from processing.layer import *
import numpy as np
from matplotlib import pyplot as pp
from dataclasses import dataclass

def hot_encode(Y):
    """HotEncode Y to be usable with softmax"""
    # https://fr.wikipedia.org/wiki/Encodage_one-hot
    classes = np.unique(Y)
    result = np.zeros(((len(Y), len(classes))))
    for i, value in enumerate(Y):
        result[i, np.where(classes == value)] = 1
    return result

@dataclass
class ModelConfiguration:
    batch_size: int = 8
    number_epoch: int = 100

class MultiLayer:
    def __init__(self, X: np.array, Y: np.array, config: ModelConfiguration = ModelConfiguration()):
        self.layers: list[AbstractLayer] = list()
        self.X = X
        print(Y)
        print(np.shape(Y))
        Y = hot_encode(Y).T
        print(Y)
        print(np.shape(Y))
        self.Y = Y
        self.m = X.shape[1]
        self.c = 0
        self.config = config

    def add_input_layer(self, size):
        """This layer act as a placeholder"""
        if (self.c != 0):
            raise IndexError("You can't have multiples input layers!")
        else:
            self.__append_layer(InputLayer(size, 0))

    def add_dense_layer(self, size, **kwargs):
        """
        Optionnals: 
        - activator(AbstractActivator): function used to transform into a probability the neuron output
        - initializer(AbstractInitializer): function used to initialize the weights
        - optimizer(AbtractOptimizer): function used to achieve the update of the gradients
        - loss(AbstractLoss): function used to determine the loss of the model
        """
        if (self.c == 0):
            raise IndexError("You must add an input layer before!")
        else:
            n_before = self.layers[-1].data.n
            self.__append_layer(HiddenLayer(size, self.c, n_before, self.m, **kwargs))

    def add_output_layer(self, size, **kwargs):
        """Same optionnals than add_dense_layer"""
        if (self.c < 2):
            raise IndexError("You must add an input layer and minimum a dense layer before!")
        elif any(isinstance(layer, OutputLayer) for layer in self.layers):
            raise TypeError("You can't have multiple output layers!")
        else:
            n_before = self.layers[-1].data.n
            self.__append_layer(OutputLayer(size, self.c, n_before, self.m, **kwargs))

    def __append_layer(self, layer):
        self.layers.append(layer)
        self.c += 1

    def __epoch(self, batch_size) -> float:
        indices = np.arange(self.m)
        total_loss = 0
 
        for start in range(0, self.m, batch_size):
            end = min(start + batch_size, self.m)
            batch_indices = indices[start:end]
            X_batch = self.X[:, batch_indices]
            Y_batch = self.Y[:, batch_indices]

            total_loss += self.__mini_batch(X_batch, Y_batch)
        return total_loss / self.m

    def __mini_batch(self, X_batch, Y_batch) -> float:
        # forward propagation
        for i in range(1, self.c):
            A_before = X_batch if i == 1 else self.layers[i - 1].A
            self.layers[i].forward(A_before)

        # backward propagation
        # ommiting output layer
        for i in reversed(range(1, self.c)):
            layer = self.layers[i]
            A_before = X_batch if i == 1 else self.layers[i - 1].A

            # mean we are at final layer
            if (i == self.c - 1):
                layer.backward(Y_batch, A_before)
            else:
                layer_plus = self.layers[i + 1]
                layer.backward(A_before, layer_plus.W, layer_plus.dZ)

        # gradient update
        for i in range(1, self.c):
            self.layers[i].update_gradients()

        # local_log_loss = - np.sum(Y_batch * np.log(self.layers[-1].A) + (1 - Y_batch) * np.log(1 - self.layers[-1].A))
        local_loss = self.layers[-1].data.loss.apply(self.layers[-1].A, Y_batch)
        return local_loss

    def learn(self):
        print(f"il y a {len(self.layers)} layers")

        loss_history, index_history = list(), list()
        for i in range(self.config.number_epoch):
            loss = self.__epoch(self.config.batch_size)
            index_history.append(i)
            loss_history.append(loss)

        pp.plot(index_history, loss_history)
        pp.show()

    # def add_layer(self, size, **kwargs):
    #     """
    #     Optionnals: 
    #     - activator(AbstractActivator): function used to transform into a probability the neuron output
    #     - initializer(AbstractInitializer): function used to initialize the weights
    #     - optimizer(AbtractOptimizer): function used to achieve the update of the gradients
    #     - loss(AbstractLoss): function used to determine the loss of the model
    #     """
    #     n_before = self.layers[-1].data.n if self.c > 0 else 1
    #     data = LayerData(size, n_before, self.m, self.c, **kwargs)

    #     self.layers.append(Layer(data))
    #     self.c += 1