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

import cdd
import numpy as np

# get input polyhedron's dual
def dual(S:np.ndarray) -> np.ndarray:
    state_mat = cdd.Matrix(S.tolist(), number_type="fraction")
    state_mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(state_mat)
    effect_mat = poly.get_generators()
    E = np.array([list(x) for x in effect_mat], dtype=np.int16)
    return E

# polytope to dual input
# concatenate zeros(n, 1) left of state
def dual_input(state: np.ndarray) -> np.ndarray:
    m = state.shape[0]
    s = np.concatenate([np.zeros((m, 1)), state], axis=1)
    return s

if __name__ == "__main__" :
    state = [
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, -1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, -1, 1]
    ]
    print(dual(dual_input(np.array(state, dtype=np.int16))))