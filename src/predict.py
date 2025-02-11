from argparse import ArgumentParser
from preprocessing.scalers import Z_score
from preprocessing.file.csvmanager import CSVManager
from processing.tools.model_manager import ModelManager
from processing.multilayer import MultiLayer
import numpy as np

def load_data(file_path: str):
    return CSVManager().load(file_path, 1, 2)

def predict(mlp: MultiLayer, X):
    mlp._forward(X)
    print(np.argmax(mlp.layers[-1].A, axis=0))

def main():
    scaler = Z_score()
    parser = ArgumentParser()
    parser.add_argument("model", help="model to use")
    parser.add_argument("data", help="data file to evaluate")

    args = parser.parse_args()
    data = load_data(args.data)
    scaled_data = scaler.scale_data(data)
    mlp = ModelManager.import_model(args.model)

    predict(mlp, scaled_data[0])

if __name__ == "__main__":
    main()