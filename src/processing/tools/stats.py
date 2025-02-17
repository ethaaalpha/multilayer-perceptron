import numpy as np
from matplotlib import pyplot as pp

class Stats:
    def __init__(self, config):
        self.config = config
        self.temp = dict()
        self.data = dict()
        self.counter = 1

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

    def fig(self, title: str, y_label: str, keys: list, n_epochs: int):
        fig = pp.figure(str(keys))
        fig.suptitle(title)

        for k in keys:
            pp.plot(np.arange(n_epochs), self.data[k], label=k)
        pp.ylabel(y_label)
        pp.xlabel("epochs")
        pp.grid()
        pp.legend()
        pp.ylim(0, 1)

    def display(self):
        pp.show()

    def is_early_stop(self) -> bool:
        patience = 0
        max_patience = 3
        sigma = 0.01 

        def condition():
            nonlocal patience
            validation_loss = self.data.get("validation_loss")
            training_loss = self.data.get("training_loss")
            training_accuracy = self.data.get("training_accuracy")
            validation_accuracy = self.data.get("validation_accuracy")

            if (self.counter > 2):
                if (validation_loss[-1] > validation_loss[-2]):
                    return True
                elif (training_loss[-1] > training_loss[-2]):
                    patience += 1
                elif (training_accuracy[-1] > training_accuracy[-2]):
                    patience += 1
                elif (validation_accuracy[-1] > validation_accuracy[-2]):
                    patience += 1
                elif (abs(validation_accuracy[-1] - validation_accuracy[-2]) <= sigma):
                    patience += 1
                elif (abs(training_accuracy[-1] - training_accuracy[-2]) <= sigma):
                    patience += 1
                else:
                    patience = 0
                
                return patience > max_patience
            else:
                return False
        return condition()
