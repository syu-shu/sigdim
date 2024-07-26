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
from solids import *

S = np.array([
    [ 1,  1,  0,  0],
    [ 1, -1,  0,  0],
    [-1,  1,  0,  0],
    [-1, -1,  0,  0],

    [ 1,  0,  1,  0],
    [ 1,  0, -1,  0],
    [-1,  0,  1,  0],
    [-1,  0, -1,  0],

    [ 1,  0,  0,  1],
    [ 1,  0,  0, -1],
    [-1,  0,  0,  1],
    [-1,  0,  0, -1],

    [ 0,  0,  1,  1],
    [ 0,  0,  1, -1],
    [ 0,  0, -1,  1],
    [ 0,  0, -1, -1],

    [ 0,  1,  0,  1],
    [ 0,  1,  0, -1],
    [ 0, -1,  0,  1],
    [ 0, -1,  0, -1],

    [ 0,  1,  1,  0],
    [ 0,  1, -1,  0],
    [ 0, -1,  1,  0],
    [ 0, -1, -1,  0],
])

state = cat_ones(S)
with open("results/hyper_4.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["state", "#vertices", "aff.dim", "cs", "#symmetries", "#ext0", "#ext", "sigdim"])
    ans = sigdim(state)
    writer.writerow([
        "hyper-4-24",
        state.shape[0],
        state.shape[1],
        ans["cs"],
        ans["sym"],
        ans["ext0"],
        ans["ext"],
        ans["sigdim"]
    ])
