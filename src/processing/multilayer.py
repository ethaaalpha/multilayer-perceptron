import numpy as np
from processing.tools.stats import Stats
from processing.layer import *
from processing.functions.losses import *
from dataclasses import dataclass
from abc import ABC

@dataclass
class ModelConfiguration:
    batch_size: int = 8
    number_epoch: int = 84
    loss: AbstractLoss = CCE()

class LogicNetwork(ABC):
    def __init__(self, config: ModelConfiguration):
        self.layers: list[AbstractLayer] = list()
        self.config = config
        self.c = 0

    @property
    def dense_layers(self) -> list[HiddenLayer]:
        return self.layers[1:]

    def _append_layer(self, layer):
        self.layers.append(layer)
        self.c += 1

    def _mini_batch(self, X_batch, Y_batch) -> float:
        self._forward(X_batch)
        self._backward(X_batch, Y_batch)

        return self.config.loss.apply(self.layers[-1].A, Y_batch)

    def _forward(self, X_batch):
        for i in range(1, self.c):
            A_before = X_batch if i == 1 else self.layers[i - 1].A
            self.layers[i].forward(A_before)

    def _backward(self, X_batch, Y_batch):
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

class MultiLayer(LogicNetwork):
    def __init__(self, X: np.array, Y = np.array, config: ModelConfiguration = ModelConfiguration()):
        super().__init__(config)
        self.X = X
        self.Y = Y
        self.m = X.shape[1]
        self.stats = Stats(config)

    def add_input_layer(self, size):
        """This layer act as a placeholder"""
        if (self.c != 0):
            raise IndexError("You can't have multiples input layers!")
        else:
            self._append_layer(InputLayer(size, 0))

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
            self._append_layer(HiddenLayer(size, self.c, n_before, self.m, **kwargs))

    def add_output_layer(self, size, **kwargs):
        """Same optionnals than add_dense_layer"""
        if (self.c < 2):
            raise IndexError("You must add an input layer and minimum a dense layer before!")
        elif any(isinstance(layer, OutputLayer) for layer in self.layers):
            raise TypeError("You can't have multiple output layers!")
        else:
            n_before = self.layers[-1].data.n
            self._append_layer(OutputLayer(size, self.c, n_before, self.m, **kwargs))

    def learn(self, validation_data=None):
        if (self.c < 2 or not any(isinstance(layer, OutputLayer) for layer in self.layers)):
            raise ValueError("You must have an input layer, hidden layer(s) and an output layer!")
        self.__info()

        for _ in range(self.config.number_epoch):
            self.__epoch(self.config.batch_size, validation_data)

        self.stats.fig("Loss Evolution", "loss", ["training_loss", "validation_loss"])
        self.stats.fig("Accuracy Evolution", "accuracy %", ["training_accuracy", "validation_accuracy"])

    def __info(self):
        print(f"config -> batch_size: {self.config.batch_size}, epochs_max: {self.config.number_epoch}, loss: {self.config.loss.name}")
        for layer in self.layers:
            print(f"layer {layer.data.c}/{self.c} -> size:{layer.data.n}, activator={layer.data.activator.name}")

    def __epoch(self, batch_size, validation_data=None):
        indices = np.arange(self.m)
        number_batch_per_epoch = 0
 
        for start in range(0, self.m, batch_size):
            end = min(start + batch_size, self.m)
            batch_indices = indices[start:end]
            X_batch = self.X[:, batch_indices]
            Y_batch = self.Y[:, batch_indices]

            self.stats.register("training_loss", self._mini_batch(X_batch, Y_batch))
            self.stats.register("training_accuracy", self.config.loss.accuracy(self.layers[-1].A, Y_batch))
            number_batch_per_epoch += 1

        self.validate(validation_data[0], validation_data[1])
        self.stats.save("training_loss", self.m)
        self.stats.save("training_accuracy", number_batch_per_epoch)
        self.stats.epoch()

    def validate(self, X_val, Y_val):
        self._forward(X_val)

        A = self.layers[-1].A
        self.stats.register("validation_loss", self.config.loss.apply(A, Y_val))
        self.stats.register("validation_accuracy", self.config.loss.accuracy(A, Y_val))
        self.stats.save("validation_loss", len(X_val[0]))
        self.stats.save("validation_accuracy", 1)

