from preprocessing.file.csvmanager import CSVManager
from preprocessing.file.dataset import DataSet
from preprocessing.scalers import *

from processing.multilayer import MultiLayer, ModelConfiguration
from processing.functions.activators import Sigmoide, SoftMax
from processing.functions.optimizers import GradientDescent
from processing.functions.initializers import He_Uniform
from processing.functions.losses import *

import numpy as np

def main():
    scaler = Z_score()
    data: DataSet = CSVManager().load("data.csv", 1, 2)

    X_list_raw = [col for col in data.columns()[1:]]
    X_list = [scaler.normalize_list(axis) for axis in X_list_raw]
    Y_list = data.column(0)

    temp = 0.314
    config = ModelConfiguration()

    mlp: MultiLayer = MultiLayer(np.array(X_list), np.array(Y_list), config)
    mlp.add_layer(30)
    mlp.add_layer(24, activator=Sigmoide(), initializer=He_Uniform(), optimizer=GradientDescent(temp))
    mlp.add_layer(24, activator=Sigmoide(), initializer=He_Uniform(), optimizer=GradientDescent(temp))
    mlp.add_layer(24, activator=Sigmoide(), initializer=He_Uniform(), optimizer=GradientDescent(temp))
    mlp.add_layer(2, activator=SoftMax(), initializer=He_Uniform(), optimizer=GradientDescent(temp), loss=CCE())
    mlp.learn()

if __name__ == "__main__":
    main()
