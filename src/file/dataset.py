from calculations.utils.normalization import Normalizer

class DataSet:
    def __init__(self, data: list[list]):
        """By default the data set is normalized for each column starting at normalize_index column"""
        self._data = data
        self.__normalize()

    def __normalize(self):
        tool = Normalizer()

        raw_columns = [[row[i] for row in self._data] for i in range(len(self._data[0]))]
        normalized_columns = [tool.normalize_list(c) for c in raw_columns]
        normalized_rows = [[c[i] for c in normalized_columns] for i in range(len(normalized_columns[0]))]

        self._rows, self._columns = normalized_rows, normalized_columns

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

    ### useless
    def __in__(self):
        return self._data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data
