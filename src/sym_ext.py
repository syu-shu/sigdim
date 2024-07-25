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

def sym_ext (EXT: np.ndarray, SYM: np.ndarray, d:int=-1) -> np.ndarray:
    E, N = EXT.shape
    S = SYM.shape[0]
    if d == -1:
        d = SYM.shape[1]
    ext_reduced = []
    for e in range(E):
        ext = EXT[e, :][SYM[0]]
        for s in range(1, S):
            temp = EXT[e, :][SYM[s]]
            if compare(ext, temp) < 0:
                ext = temp
        # bin search
        l = 0
        r = len(ext_reduced) - 1
        idx = 0
        while l <= r:
            idx = (l + r) // 2
            cmp = compare(ext, np.array(ext_reduced[idx]))
            if cmp > 0:
                l = idx + 1
            elif cmp < 0:
                r = idx - 1
            else:
                break
        else:
            ext_reduced.insert(l, ext.tolist())
    return np.array(ext_reduced)

# A > B : 1
# A == B : 0
# A < B : -1
def compare (A: np.ndarray, B: np.ndarray) -> int:
    N = A.shape[0]
    for i in range(N):
        if A[i] > B[i]:
            return 1
        elif A[i] < B[i]:
            return -1
    return 0