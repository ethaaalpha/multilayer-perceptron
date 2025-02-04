import numpy as np
from .functionnal import Fonctionnal

class Initializers(Fonctionnal):
    AUTO = ("Auto", lambda shape, n : np.random.rand(*shape))
    HE_INIT = ("He Initialization", lambda shape, n: np.random.rand(*shape) * np.sqrt(2. / n))
    XAVIER_NORMAL = ("Xavier Normal Init", lambda shape, n: np.random.rand(*shape) * np.sqrt(1. / n))

