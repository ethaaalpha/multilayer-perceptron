import numpy as np
from processing.stats import Stats
from processing.layer import *
from processing.functions.losses import *
from dataclasses import dataclass

@dataclass
class ModelConfiguration:
    batch_size: int = 8
    number_epoch: int = 84
    loss: AbstractLoss = CCE()

class MultiLayer:
    def __init__(self, X: np.array, Y: np.array, config: ModelConfiguration = ModelConfiguration()):
        self.layers: list[AbstractLayer] = list()
        self.X = X
        self.Y = config.loss.preprocessing(Y)
        self.m = X.shape[1]
        self.c = 0
        self.config = config
        self.stats = Stats(config)

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

    def learn(self, validation_data=None):
        if (self.c < 2 or not any(isinstance(layer, OutputLayer) for layer in self.layers)):
            raise ValueError("You must have an input layer, hidden layer(s) and an output layer!")
        self.__info()

        for _ in range(self.config.number_epoch):
            self.__epoch(self.config.batch_size)

        self.stats.display()

    def __info(self):
        print(f"You start a learning with {len(self.layers)} layers")
        print(f"config -> batch_size: {self.config.batch_size}, epochs_max: {self.config.number_epoch}, loss: {self.config.loss.name}")
        for layer in self.layers:
            print(f"layer {layer.data.c}/{self.c} -> size:{layer.data.n}, activator={layer.data.activator.name}")

    def __epoch(self, batch_size):
        indices = np.arange(self.m)
        number_batch_per_epoch = 0
 
        for start in range(0, self.m, batch_size):
            end = min(start + batch_size, self.m)
            batch_indices = indices[start:end]
            X_batch = self.X[:, batch_indices]
            Y_batch = self.Y[:, batch_indices]

            self.stats.register("training_loss", self.__mini_batch(X_batch, Y_batch))
            self.stats.register("training_accuracy", self.config.loss.accuracy(self.layers[-1].A, Y_batch))
            number_batch_per_epoch += 1

        self.stats.save("training_loss", self.m)
        self.stats.save("training_accuracy", number_batch_per_epoch)

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
                layer.backward(Y_batch, A_before, self.config.loss)
            else:
                layer_plus = self.layers[i + 1]
                layer.backward(A_before, layer_plus.W, layer_plus.dZ)

        # gradient update
        for i in range(1, self.c):
            self.layers[i].update_gradients()

        local_loss = self.config.loss.apply(self.layers[-1].A, Y_batch)
        return local_loss
