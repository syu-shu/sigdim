# Copyright 2024 Shuriku Kai

# This program is free software: you can redistribute it
# and/or modify it under the terms of the GNU General
# Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version.

# This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License
# for more details.

# You should have received a copy of the GNU General Public
# License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

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