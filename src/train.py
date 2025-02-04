from file.csvmanager import CSVManager
from file.dataset import DataSet

from calculations.multilayer import MultiLayer
from matplotlib import pyplot as pl

from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers

import numpy as np

def main():
    data: DataSet = CSVManager().load("data.csv", 1, 2)

    X_list = [col for col in data.columns()[1:]]
    Y_list = data.column(0)

    mlp: MultiLayer = MultiLayer(np.array(X_list), np.array(Y_list))
    mlp.add_layer(24, Activations.RELU, Initializers.HE_INIT)
    mlp.add_layer(24, Activations.RELU, Initializers.HE_INIT)
    mlp.add_layer(24, Activations.RELU, Initializers.HE_INIT)
    mlp.add_layer(1, Activations.SIGMOIDE, Initializers.HE_INIT)
    mlp.learn()

if __name__ == "__main__":
    main()


# https://github.com/Sleleu/multilayer-perceptron