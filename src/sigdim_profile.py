from sigdim import *
import numpy as np

d = 4
state = np.concatenate([np.eye(d, dtype=np.int16), -np.eye(d, dtype=np.int16)])
state = np.concatenate([state, np.ones((state.shape[0], 1), dtype=np.int16)], axis=1)

print(sigdim(state))