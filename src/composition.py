import numpy as np
from sigdim import *
from solids import *

A = cat_ones(hyper_octahedron(2))
# B = cat_ones(hyper_octahedron(2))
# S = np.kron(A, B)
# print(S.T.shape)
# print(S.T)
# print(sigdim(S))

C = cat_ones(hyper_octahedron(3))
S2 = np.kron(A, C)
print(S2.T.shape)
print(S2.T)
print(sigdim(S2))

# S3 = np.kron(C, C)
# print(S3.T.shape)