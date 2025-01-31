import numpy
from .functionnal import Fonctionnal

class Activations(Fonctionnal):
    SIGMOIDE = ("Sigmoide", lambda x: 1 / (1 + numpy.exp(-x)))
