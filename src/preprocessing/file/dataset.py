class DataSet:
    def __init__(self, data: list[list]):
        print(data)
        self._data = data
        self._columns = [[row[i] for row in self._data] for i in range(len(self._data[0]))]
        self._rows = [[c[i] for c in self._columns] for i in range(len(self._columns[0]))]

    def raw_data(self):
        return self._data

    def columns(self) -> list:
        return self._columns

    def column(self, i: int) -> list:
        return self._columns[i]

    def rows(self) -> list[list]:
        return self._rows

    def row(self, i: int) -> list:
        return self._rows[i]
