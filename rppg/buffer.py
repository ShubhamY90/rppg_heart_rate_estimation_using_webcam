import numpy as np
from collections import deque

class SignalBuffer:
    def __init__(self, window_seconds, fps):
        self.maxlen = int(window_seconds * fps)
        self.R = deque(maxlen=self.maxlen)
        self.G = deque(maxlen=self.maxlen)
        self.B = deque(maxlen=self.maxlen)

    def append(self, r, g, b):
        self.R.append(r)
        self.G.append(g)
        self.B.append(b)

    def ready(self):
        return len(self.R) == self.maxlen

    def get(self):
        return (
            np.asarray(self.R),
            np.asarray(self.G),
            np.asarray(self.B)
        )
