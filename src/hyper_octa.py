from sigdim import sigdim

import numpy as np
nf = 4
nt = 7
for d in range(nf, nt + 1):
    state = np.concatenate([np.eye(d, dtype=np.int16), -np.eye(d, dtype=np.int16)])
    state = np.concatenate([state, np.ones((state.shape[0], 1), dtype=np.int16)], axis=1)
    print("---")
    print("dimesion = " + str(d))
    sd = sigdim(state)
    print("sigdim = " + str(sd))