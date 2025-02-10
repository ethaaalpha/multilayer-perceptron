from preprocessing.file.csvmanager import CSVManager
from preprocessing.file.dataset import DataSet
from preprocessing.scalers import *
from processing.multilayer import MultiLayer, ModelConfiguration
from processing.functions.activators import Sigmoide, SoftMax
from processing.functions.optimizers import GradientDescent
from processing.functions.initializers import He_Uniform
from processing.functions.losses import *

def main():
    scaler = Z_score()
    training_dataset: DataSet = CSVManager().load("training.csv", 1, 2)
    validation_dataset: DataSet = CSVManager().load("validation.csv", 1, 2)

    training_data = scaler.scale_data(training_dataset)
    validation_data = scaler.scale_data(validation_dataset)

    temp = 0.03
    config = ModelConfiguration(number_epoch=120)

    mlp: MultiLayer = MultiLayer(training_data[0], training_data[1], config)
    mlp.add_input_layer(30)
    mlp.add_dense_layer(80, activator=Sigmoide(), initializer=He_Uniform(), optimizer=GradientDescent(temp))
    mlp.add_dense_layer(24, activator=Sigmoide(), initializer=He_Uniform(), optimizer=GradientDescent(temp))
    mlp.add_output_layer(2, activator=SoftMax(), initializer=He_Uniform(), optimizer=GradientDescent(temp))
    mlp.learn()

if __name__ == "__main__":
    main()
