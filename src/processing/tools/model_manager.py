from processing.multilayer import MultiLayer
from processing.multilayer import ModelConfiguration
import processing.functions.activators as activators
import processing.functions.losses as losses
import numpy as np
import json

def import_from_file(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)

def export_to_file(data: dict, file_path: str):
    with open(file_path, "w") as file:
        file.write(json.dumps(data))

class ModelManager:
    def import_model(file_path: str) -> MultiLayer:
        data = import_from_file(file_path)
        layers: dict = data["dense"]

        config = ModelConfiguration(loss=losses.from_str(data["loss"]))
        mlp = MultiLayer(np.random.random((1,1)), np.random.random((1, 1)), config)
        mlp.add_input_layer(data["input"])

        for i in range(len(layers)):
            activation = activators.from_str(layers[i]["activator"])
            n = layers[i]["n"]
            W = np.array(layers[i]["W"])
            b = np.array(layers[i]["b"])

            if i == len(layers) - 1:
                mlp.add_output_layer(n, activator=activation)
            else:
                mlp.add_dense_layer(n, activator=activation)
            mlp.layers[-1].W = W
            mlp.layers[-1].b = b
        return mlp
    
    def export_model(mlp: MultiLayer, file_path: str):
        data = dict()
        data["loss"] = mlp.config.loss.name
        data["input"] = mlp.layers[0].data.n
        data["dense"] = list()
        for layer in mlp.dense_layers:
            layer_data = {
                "n": layer.data.n,
                "W": layer.W.tolist(),
                "b": layer.b.tolist(),
                "activator": layer.data.activator.name
            }
            data["dense"].append(layer_data)

        export_to_file(data, file_path)
