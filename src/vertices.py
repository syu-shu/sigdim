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
from collections import deque

def vertices(P:np.ndarray, d:int) -> np.ndarray:
    m, n = P.shape
    # P[P != 0] = 1
    # A = np.zeros((m, 1), dtype=np.int16)
    AA =deque([[0 for _i in range(m)]])
    # print(AA)
    a = np.zeros((m, n), dtype=np.int16)
    l = 0
    # BFSなら並列化できるかも？
    def recfun(k = 0):
        for j in np.where(P[k, :] != 0)[0]:
            a[k] = np.eye(n)[j]
            # print(a)
            if (np.any(a[:k + 1, :] != 0, axis=0).sum() <= d):
                if (k == m - 1):
                    nonlocal AA
                    # nonlocal A
                    nonlocal l
                    temp = np.where(a.reshape(a.size) > 0)[0].T - np.array(range(0, P.size, n)).T
                    # print(temp)
                    # A = np.insert(A, l,
                    #     np.where(a.reshape(a.size) > 0)[0].T
                    #     - np.array(range(0, P.size, n)).T
                    # , axis=1)
                    # A = np.insert(A, 0, temp, axis=1)
                    AA.appendleft(temp.tolist())
                    l += 1
                else:
                    recfun(k + 1)
    recfun()
    # print(A)
    # A = A[:, :-1]
    # print(A.shape)
    # print(len(list(AA)))
    AA = np.array(list(AA), dtype=np.int32)
    AA = AA.T
    AA = AA.reshape(AA.shape[0], -1)
    AA = AA[:, :-1]
    # print(np.all(A == AA))
    # print(AA.shape)
    # A = np.array(A, dtype=np.int32)
    # AA = AA.T
    # print(A.shape)
    # print(AA.shape)
    # print(A)
    return AA

if __name__ == "__main__":
    print("vertices")