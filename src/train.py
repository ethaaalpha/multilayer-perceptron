from file.csvmanager import CSVManager
from src.file.dataset import DataSet

from calculations.layer import Layer
from matplotlib import pyplot as pl

from calculations.functions.activations import Activations
from calculations.functions.initializers import Initializers

def main():
    # data: DataSet = CSVManager().load("data.csv", 1, 2)
    # print(Activations.SIGMOIDE.apply(5))
    test = Layer(2, 4, Activations.SIGMOIDE, Initializers.AUTO)

if __name__ == "__main__":
    main()