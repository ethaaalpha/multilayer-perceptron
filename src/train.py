from preprocessing.file.csvmanager import CSVManager
from preprocessing.file.dataset import DataSet
from preprocessing.scalers import *

from calculations.multilayer import MultiLayer, ModelConfiguration
from calculations.functions.activators import Sigmoide
from calculations.functions.optimizers import GradientDescent
from calculations.functions.initializers import HE_UNIFORM

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
    mlp.add_layer(40, activator=Sigmoide(), initializer=HE_UNIFORM(), optimizer=GradientDescent(temp))
    mlp.add_layer(20, activator=Sigmoide(), initializer=HE_UNIFORM(), optimizer=GradientDescent(temp))
    mlp.add_layer(1, activator=Sigmoide(), initializer=HE_UNIFORM(), optimizer=GradientDescent(temp))
    mlp.learn()

if __name__ == "__main__":
    main()
