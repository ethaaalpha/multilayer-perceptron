from processing.layer import Layer, LayerData
import numpy as np
from matplotlib import pyplot as pp
from dataclasses import dataclass

@dataclass
class ModelConfiguration:
    batch_size: int = 8
    number_epoch: int = 84

class MultiLayer:
    def __init__(self, X: np.array, Y: np.array, config: ModelConfiguration = ModelConfiguration()):
        self.layers: list[Layer] = list()
        self.X = X
        self.Y = Y.reshape(1, -1)
        self.m = X.shape[1]
        self.c = 0
        self.config = config

    def add_layer(self, size, **kwargs):
        """
        Optionnals: 
        - activator(AbstractActivator): function used to transform into a probability the neuron output
        - initializer(AbstractInitializer): function used to initialize the weights
        - optimizer(AbtractOptimizer): function used to achieve the update of the gradients
        - loss(AbstractLoss): function used to determine the loss of the model
        """
        n_before = self.layers[-1].data.n if self.c > 0 else 1
        data = LayerData(size, n_before, self.m, self.c, **kwargs)

        self.layers.append(Layer(data))
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
        for i in reversed(range(1, self.c)): # ommiting output layer
            layer = self.layers[i]
            A_before = X_batch if i == 1 else self.layers[i - 1].A

            if (i == self.c - 1): # mean we are at final layer
                layer.backward_last(Y_batch, A_before)
            else:
                layer_plus = self.layers[i + 1]
                layer.backward(A_before, layer_plus.W, layer_plus.dZ)

        # gradient update
        for i in range(1, self.c):
            self.layers[i].update_gradient()

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
