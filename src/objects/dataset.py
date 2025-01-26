class DataSet:
    def __init__(self, data: list[list]):
        self._data = data
    
    def columns(self) -> list:
        return [[row[i] for row in self._data] for i in range(len(self._data[0]))]

    def column(self, i: int) -> list:
        return [row[i] for row in self._data]

    def __in__(self):
        return self._data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data
