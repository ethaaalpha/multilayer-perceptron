import csv
from objects.dataset import DataSet

class CSVManager:
    def load(self, filepath: str) -> DataSet:
        """Load the CSV file"""
        data = list()

        with open(filepath, 'r') as file:
            data = list(csv.reader(file, delimiter=","))
        return DataSet(data)

    def export(self, filepath: str, *values: tuple[float, float]):
        """Export the values from the training into a specific file"""
        with open(filepath, 'w+') as file:
            for value in values:
                string = f"{value[0]}, {value[1]}"

                file.write(string + "\n")
                print(f"Written in {filepath}: {string}")
