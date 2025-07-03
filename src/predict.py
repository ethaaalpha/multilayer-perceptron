from argparse import ArgumentParser
from preprocessing.scalers import Z_score
from preprocessing.file.csvmanager import CSVManager
from processing.tools.model_manager import ModelManager
from processing.multilayer import MultiLayer
from preprocessing.encoders import HotEncoder
import numpy as np

def load_data(file_path: str):
    return CSVManager().load(file_path, 1, 2)

def predict(mlp: MultiLayer, X, Y, raw):
    mlp.validate(X, Y)

    A = mlp.layers[-1].A

    print("## informations ##")
    print(f'loss: {mlp.stats.data["validation_loss"][0]}, accuracy: {mlp.stats.data["validation_accuracy"][0]}')
    print("##    result    ##")

    result = np.argmax(A, axis=0) # for softmax
    if raw:
        print(result)
    else:
        result = np.where(result == 1, 'M', result)
        result = np.where(result == '0', 'B', result)
        print(result)

def main():
    scaler = Z_score()
    parser = ArgumentParser()
    parser.add_argument("model", help="model to use")
    parser.add_argument("data", help="data file to evaluate")
    parser.add_argument("--raw", action="store_true", help="display the data in raw array if present")

    args = parser.parse_args()
    data = load_data(args.data)
    scaled_data = scaler.scale_data(data, HotEncoder())
    mlp = ModelManager.import_model(args.model)

    predict(mlp, scaled_data[0], scaled_data[1], args.raw)

if __name__ == "__main__":
    main()