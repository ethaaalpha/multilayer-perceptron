import numpy as np
from matplotlib import pyplot as pp

class Stats:
    def __init__(self, config):
        self.config = config
        self.temp = dict()
        self.data = dict()
        self.counter = 1
        self.indices = np.arange(self.config.number_epoch)

    def register(self, name, value):
        if self.temp.get(name) == None:
            self.temp[name] = 0
        self.temp[name] += value

    def save(self, name, ratio=1):
        if self.data.get(name) == None:
            self.data[name] = list()
        self.data[name].append(self.temp[name] / ratio)
        self.temp[name] = None

    def epoch(self):
        data = ", ".join([f"{k}:{v[-1]:.5f}" for k, v in self.data.items()])
        print(f"epoch {self.counter}: {data}")
        self.counter += 1

    def fig(self, title: str, y_label: str, keys: list):
        fig = pp.figure(str(keys))
        fig.suptitle(title)

        for k in keys:
            pp.plot(self.indices, self.data[k], label=k)
        pp.ylabel(y_label)
        pp.xlabel("epochs")
        pp.grid()
        pp.legend()
        pp.ylim(0, 1)

    def display(self):
        pp.show()