import numpy as np
from .functionnal import Fonctionnal

class Activations(Fonctionnal):
    SIGMOIDE = ("Sigmoide", lambda x: 1 / (1 + np.exp(-x)))
    RELU = ("ReLU", lambda x: np.maximum(0, x))


