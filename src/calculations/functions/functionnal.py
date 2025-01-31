from enum import Enum

class Fonctionnal(Enum):
    def apply(self, *args):
        return self._value_[1](*args)