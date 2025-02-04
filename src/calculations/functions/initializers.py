import numpy as np
from .functionnal import Fonctionnal

class Initializers(Fonctionnal):
    AUTO = ("Auto", lambda shape: np.random.rand(*shape))
    HE_INIT = ("He Initialization", lambda shape, n: np.random.rand(*shape) * np.sqrt(2. / n))
