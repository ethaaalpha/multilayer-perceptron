import numpy as np
from .functionnal import Fonctionnal

class Initializers(Fonctionnal):
    AUTO = ("Auto", lambda x, y: np.random.rand(*x.shape))
