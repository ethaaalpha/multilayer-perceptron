import numpy as np
from matplotlib import pyplot as pp

class Stats:
    def __init__(self, config):
        self.config = config
        self.temp = dict()
        self.data = dict()

    def register(self, name, value):
        if self.temp.get(name) == None:
            self.temp[name] = 0
        self.temp[name] += value

    def save(self, name, ratio):
        if self.data.get(name) == None:
            self.data[name] = list()
        self.data[name].append(self.temp[name] / ratio)
        self.temp[name] = None

    def display(self):
        indices = np.arange(self.config.number_epoch)

        for k, v in self.data.items():
            pp.plot(indices, v, label=k)
        pp.ylim(0, 1.2) # it's better
        pp.legend()
        pp.show()