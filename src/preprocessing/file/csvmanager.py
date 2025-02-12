import csv
from preprocessing.file.dataset import DataSet

class CSVManager:
    def load(self, filepath: str, y_index: int, values_index: int) -> DataSet:
        """Load the CSV file. You can put anything before y_index it won't be used !"""
        result = list()

        with open(filepath, 'r') as file:
            data = list(csv.reader(file, delimiter=","))

            for row in data:
                line = list()
                for i in range(y_index, len(row)):
                    if (i >= values_index):
                        try:
                            line.append(float(row[i]))
                        except ValueError:
                            print("Failed to convert value to float, appending 1 instead!")
                            line.append(1)
                    else:
                        line.append(row[i])
                result.append(line)
        return DataSet(result)

    def export(self, filepath: str, *values: tuple[float, float]):
        """Export the values from the training into a specific file"""
        with open(filepath, 'w+') as file:
            for value in values:
                string = f"{value[0]}, {value[1]}"

                file.write(string + "\n")
                print(f"Written in {filepath}: {string}")
