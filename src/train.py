from file.csvmanager import CSVManager
from objects.dataset import DataSet

from matplotlib import pyplot as pl

def main():
    data: DataSet = CSVManager().load("data.csv")

if __name__ == "__main__":
    main()